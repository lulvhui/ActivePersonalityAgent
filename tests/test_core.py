import copy,importlib.util,json,math,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from amp.common import TRAITS,LEVELS,probabilities,read_json,read_jsonl,write_json
from amp.data import split_lookup,transcript_segments
from amp.scoring import OfflineTraitScorer
from amp.teca import advantages,clipped_grpo
from amp.train import validate_gate
from amp.evaluate import calibration,spearman,validate_jev
from amp.smoke import smoke
from test_record_training import trajectory as record_trajectory
ROOT=Path(__file__).resolve().parents[1]
CFG=read_json(ROOT/'configs/default.json')
def trajectory(*args,**kwargs):
    t=record_trajectory();t.update(sample_id='fixture',dataset='IELTS');return t

class DataAndEvidenceTests(unittest.TestCase):
    def test_person_overlap_rejected(self):
        with self.assertRaises(ValueError):
            split_lookup(dict(splits=dict(train=["01"], dev=["1"], test=[])))

    def test_asr_timing_not_promoted_to_truth(self):
        rows = transcript_segments('[00:01 - 00:11] Hello\n[00:12 - 00:18] World')
        self.assertEqual(rows[1]["start"], 12)
        self.assertEqual(rows[1]["timing"], "unverified_asr")
        self.assertIsNone(transcript_segments("No timestamp")[0]["start"])

    def test_raw_manifests_no_leakage_or_generated_annotations(self):
        directory = ROOT/'runs/data'
        if not directory.exists():
            self.skipTest("Run prepare first")
        rows = [r for split in ['train','dev','test'] for r in read_jsonl(directory/f'{split}.jsonl')]
        owners = {}
        for row in rows:
            key = (row['dataset'],row['person_id'])
            self.assertEqual(owners.setdefault(key,row['split']),row['split'])
            self.assertTrue(Path(row['video']).is_file())
            self.assertNotIn('emotion_response',str(row))
            self.assertNotIn('gemini_summary',str(row))
            self.assertTrue(all(0 <= y < 5 for y in row['labels'].values()))
        self.assertEqual(sum(r['dataset']=='IELTS' for r in rows),218)
        self.assertEqual(sum(r['dataset']=='RecruitView' for r in rows),1965)
        self.assertEqual(sum(len(r['segments']) for r in rows if r['dataset']=='IELTS'),2322)

class ScorerTests(unittest.TestCase):
    def test_invalid_distribution_rejected(self):
        for p in [[.1]*5,[0,0,0,0,float('nan')],[1,0,0,0,-.1]]:
            with self.assertRaises(ValueError): probabilities(p)

class CreditTests(unittest.TestCase):
    def test_group_must_share_initial_sample(self):
        a,b=trajectory(),trajectory()
        b['sample_id']='other'
        with self.assertRaises(ValueError): advantages([a,b],{t:4 for t in TRAITS},CFG)

    def test_gate_rejects_unvalidated_or_wrong_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'gate.json'
            write_json(p,dict(split='dev',model='jev-1.13.0',passed=True))
            with self.assertRaises(ValueError): validate_gate(p,CFG)

    def test_cannot_validate_using_cached_repeats(self):
        row=dict(id='fixture',dataset='IELTS',split='dev',labels={x:4 for x in TRAITS})
        t=trajectory();t['split']='dev';t['score']=OfflineTraitScorer().score(t['memory'])
        t['score']['metadata']=dict(model='jev-1.14.0',cache_hit=True)
        result=validate_jev([t,copy.deepcopy(t)],[row],[],min_samples=1)
        self.assertFalse(result['passed'])
        self.assertEqual(result['datasets']['IELTS']['stability_repeats'],0)

class FlowTests(unittest.TestCase):
    def test_three_stages_stop_budget_no_label_leakage(self):
        t=smoke(CFG)
        self.assertTrue(t['stopped'])
        self.assertEqual([d['kind'] for d in t['decisions']],['candidate_scan','memory','final'])
        self.assertTrue(CFG['max_seconds'] is None or t['cost']['active_seconds']<=CFG['max_seconds'])
        for d in t['decisions']:
            self.assertNotIn('labels',str(d['messages']))

    def test_static_and_fixed_variants(self):
        from amp.agent import ActiveAgent
        from amp.backend import OfflineBackend
        from amp.smoke import SyntheticMedia
        row=dict(id='s',dataset='SYNTHETIC',split='test',person_id='p',segments=[])
        for mode in ['static','fixed_context','generic','active_no_context']:
            cfg=dict(CFG,perception=mode,max_steps=2)
            t=ActiveAgent(OfflineBackend(),OfflineTraitScorer(),SyntheticMedia(),cfg).run(row)
            self.assertEqual(len(t['steps']),1)
            if mode=='static':
                self.assertFalse(any(d['kind'] in ('candidate_scan','action') for d in t['decisions']))
                self.assertEqual(t['cost']['active_seconds'],0.)
            if mode=='generic':
                self.assertFalse(any(d['kind']=='action' for d in t['decisions']))
                self.assertTrue(all(s['action_decision_index'] is None for s in t['steps']))

    def test_action_entropy_excludes_goal_and_json_syntax(self):
        from amp.backend import action_token_mask
        text='{"time":[1.5, 10],"modality":"video","goal":"check sociability"}'
        class Tokenizer:
            def decode(self, ids, **kw): return ''.join(chr(i) for i in ids)
            def __call__(self, value, **kw):
                return dict(input_ids=[ord(c) for c in value],offset_mapping=[(i,i+1) for i in range(len(value))])
        tokens=[ord(c) for c in text]
        mask=action_token_mask(Tokenizer(),dict(tokens=tokens))
        self.assertEqual(''.join(c for c,m in zip(text,mask) if m),'1.510video')
        class Noncanonical(Tokenizer):
            def __call__(self,value,**kw): return dict(input_ids=[],offset_mapping=[])
        self.assertEqual(action_token_mask(Noncanonical(),dict(tokens=tokens)),mask)

    def test_probability_metrics(self):
        p=[[1,0,0,0,0],[0,0,0,0,1]]
        m=calibration([0,4],p)
        self.assertEqual(m,dict(nll=0.,brier=0.,ece=0.))
        self.assertAlmostEqual(spearman([1,2,2,4],[3,4,4,7]),1.)
        self.assertIsNone(spearman([1,1],[1,2]))

    def test_observed_interval_uses_actual_source_pts_offset(self):
        from amp.media import MediaTools
        with tempfile.TemporaryDirectory() as tmp:
            tools=MediaTools(tmp)
            info=dict(source_key='source',start_pts=10.,duration=2.)
            with patch.object(tools,'probe',return_value=info),patch('amp.media.run',return_value=''):
                obs=tools.observe(dict(video='fixture.mp4',segments=[]),dict(time=[.8,1.2],modality='audio'))
            self.assertEqual(obs['time'],[.8,1.2])
            self.assertEqual(obs['source_pts'],[10.8,11.2])

class TensorLossTests(unittest.TestCase):
    def test_clipping_and_kl_gradient(self):
        import torch
        from amp.teca import clipped_grpo
        p=torch.tensor([math.log(.5)],requires_grad=True)
        old=p.detach().clone()
        loss=clipped_grpo(p,old,old,1.,beta=0)
        loss.backward()
        self.assertAlmostEqual(float(loss),-1.)
        self.assertLess(float(p.grad),0.)
        p=torch.tensor([0.],requires_grad=True)
        old=torch.tensor([-math.log(2)])
        loss=clipped_grpo(p,old,p.detach(),1.,epsilon=.2,beta=0)
        self.assertAlmostEqual(float(loss),-1.2,places=6)
        loss.backward()
        self.assertEqual(float(p.grad),0.)

