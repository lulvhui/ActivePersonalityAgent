"""Observation-record storage and a canonical model fixture for pipeline tests."""
import copy,json,tempfile,unittest
from pathlib import Path
from amp.agent import ActiveAgent
from amp.backend import OfflineBackend
from amp.common import TRAITS,read_json
from amp.state import empty_memory,save_observation,memory_changes,CandidatePool
from amp.perception import memory_view
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia
ROOT=Path(__file__).resolve().parents[1]
CFG=read_json(ROOT/'configs/local_omni3b.json');CFG['selection_policy']='model'
ROW=dict(id='SYNTHETIC_STORAGE',dataset='SYNTHETIC',split='train',person_id='fixture',segments=[])
OBS=dict(modality='audio',time=[0.,5.],timing='media_relative',media=[],segments=[],media_id='SYNTHETIC')

class Reader(OfflineBackend):
    def __init__(self,actions=None,text=None):
        self.actions=actions or [dict(candidate_id='C1'),dict(candidate_id='C2')]
        self.text=text if text is not None else json.dumps(dict(observations=['A concrete observed finding.'],candidates=[]));self.seen=[]
    def generate_text(self,messages,kind,step):
        if kind!='memory':return super().generate_text(messages,kind,step)
        self.seen.append((kind,step,copy.deepcopy(messages)))
        return self.text,dict(messages=messages,kind=kind,step=step,text=self.text,tokens=list(map(ord,self.text)),offline=True)
    def generate(self,messages,kind,step):
        self.seen.append((kind,step,copy.deepcopy(messages)))
        if kind=='candidate_scan':
            payload=json.loads(messages[-1]['content'][-1]['text'])
            clues=[s['text'] for s in payload['observation']['segments'][:2]] or ['A prescribed synthetic behavior clue.']
            obj=dict(summary='A synthetic source.',candidates=[dict(local_time=[0,payload['source_seconds']],modality=payload['observation']['modality'],hint=clue,goal='Inspect this concrete behavior') for clue in clues])
        elif kind=='action':obj=self.actions[step]
        elif kind=='final':obj={**{t:('high' if t=='E' else None) for t in TRAITS},'reasoning':'Synthetic model explanation.','contextual':[]}
        else:return super().generate(messages,kind,step)
        text=json.dumps(obj)
        return obj,dict(messages=messages,kind=kind,step=step,text=text,tokens=list(map(ord,text)),offline=True)

class SimpleMemoryTests(unittest.TestCase):
    def test_storage_preserves_raw_output_without_semantic_operations(self):
        memory=empty_memory()
        for i,text in enumerate(('Malformed raw output','',json.dumps(dict(observations=['One behavior.','Another behavior.'],candidates=[])))):
            memory,ids=save_observation(memory,text,OBS,f'active_{i}')
            self.assertEqual(ids,[f'M{i+1}']);self.assertEqual(memory['observations'][-1]['content'],text)
            self.assertNotIn('trait',memory['observations'][-1]);self.assertNotIn('atom_id',memory['observations'][-1])
        self.assertEqual(len(memory['observations']),3)

    def test_repeated_observation_text_is_saved_without_event_inference(self):
        raw=json.dumps(dict(observations=['The same finding.'],candidates=[]));memory=empty_memory()
        for i in range(2):memory,_=save_observation(memory,raw,OBS,f'active_{i}')
        self.assertEqual(len(memory['observations']),2)
        self.assertEqual(memory['observations'][0]['content'],memory['observations'][1]['content'])

    def test_final_view_marks_overlapping_reads_without_merging_behavior(self):
        sources={};memory=empty_memory()
        for index,(window,finding) in enumerate([
            ([41.,43.],'A sustained hesitation.'),
            ([42.6,43.],'The same hesitation is clearer.'),
            ([41.,43.],'Describes asking a colleague for help.'),
            ([80.,83.],'Describes asking a different colleague for help.')]):
            obs=dict(OBS,time=window,media_id='one-interview')
            obs_id=f'active_{index}';sources[obs_id]=obs
            memory,_=save_observation(memory,json.dumps(dict(observations=[finding],candidates=[])),obs,obs_id)
        original=copy.deepcopy(memory)
        view=memory_view(memory,observations=sources)['observations']
        self.assertEqual(view[1]['overlapping_media_reads'],['M1'])
        self.assertEqual(view[2]['overlapping_media_reads'],['M1','M2'])
        self.assertNotIn('overlapping_media_reads',view[3])
        self.assertEqual(len(view),4)
        self.assertIn('asking a colleague',view[2]['content'])
        self.assertEqual(memory,original)
        self.assertFalse(any('overlapping_media_reads' in r for r in memory_view(memory)['observations']))

    def test_overlap_navigation_does_not_link_other_media_or_unaligned_text(self):
        memory=empty_memory();sources={}
        for index,(media_id,modality) in enumerate([('one','audio'),('two','video'),('one','text'),(None,'video')]):
            obs=dict(OBS,time=[41.,43.],modality=modality)
            if media_id is not None:obs['media_id']=media_id
            else:obs.pop('media_id',None)
            obs_id=f'active_{index}';sources[obs_id]=obs
            memory,_=save_observation(memory,'{"observations":["A finding."],"candidates":[]}',obs,obs_id)
        view=memory_view(memory,observations=sources)['observations']
        self.assertFalse(any('overlapping_media_reads' in r for r in view))

    def test_view_keeps_every_finding_and_excludes_future_plans(self):
        findings=['A concrete finding.']*30
        raw=json.dumps(dict(observations=findings,candidates=[dict(local_time=[1,3],modality='video',hint='A clue',goal='UNOBSERVED_PLAN')]))
        memory,ids=save_observation(empty_memory(),raw,OBS,'active_0')
        view=memory_view(memory)['observations'][0]
        self.assertEqual(view['content'].count('A concrete finding.'),30)
        self.assertNotIn('UNOBSERVED_PLAN',view['content'])
        self.assertEqual(memory_changes(empty_memory(),memory)[0]['operation'],'STORE')

    def test_empty_findings_are_a_valid_observation_without_forced_retry(self):
        reader=Reader(text='{"observations":[],"candidates":[]}')
        memory,ids,d=ActiveAgent(reader,OfflineTraitScorer(),None,CFG).update(empty_memory(),{},dict(active_0=OBS),0)
        self.assertEqual(len(reader.seen),1);self.assertEqual(d['memory_update_status'],'accepted')
        self.assertEqual(memory_view(memory)['observations'][0]['content'],'')

    def test_queue_does_not_call_select_or_force_minimum_rounds(self):
        class NoSelect(Reader):
            def generate(self,messages,kind,step):
                if kind=='action':raise AssertionError('Queue must not generate Select')
                return super().generate(messages,kind,step)
        t=ActiveAgent(NoSelect(),OfflineTraitScorer(),SyntheticMedia(),dict(CFG,selection_policy='queue',max_steps=1)).run(ROW)
        self.assertEqual(len(t['steps']),1);self.assertEqual(t['stop_reason'],'candidate_pool_exhausted')
        self.assertEqual(t['decisions'][-1]['kind'],'final');self.assertEqual(t['final_status'],'generated')
        final_messages=t['decisions'][-1]['messages']
        self.assertIn('M IDs identify reads, not independent behaviors',final_messages[-1]['content'][0]['text'])

    def test_no_coarse_candidate_is_not_filled_even_with_large_budget(self):
        class Empty(Reader):
            def generate(self,messages,kind,step):
                obj=dict(summary='No useful inspection target.',candidates=[]) if kind=='candidate_scan' else {**{t:None for t in TRAITS},'reasoning':'Insufficient observed evidence.','contextual':[]}
                raw=json.dumps(obj)
                return obj,dict(kind=kind,step=step,messages=messages,text=raw,tokens=[1],offline=True)
        for budget in (32,64):
            with self.subTest(budget=budget):
                t=ActiveAgent(Empty(),OfflineTraitScorer(),SyntheticMedia(),dict(CFG,selection_policy='queue',max_steps=budget)).run(ROW)
                self.assertEqual(t['initial']['candidate_count'],0);self.assertEqual(t['steps'],[])
                self.assertEqual(t['stop_reason'],'candidate_pool_empty');self.assertEqual(t['final_status'],'generated')

    def test_candidate_windows_modalities_and_repeated_actions(self):
        pool=CandidatePool(40,30);action=dict(time=[1,5],modality='audio')
        key=pool.add(action);pool.start(action);pool.finish(key)
        with self.assertRaises(ValueError):pool.start(action)
        self.assertIsNotNone(pool.add(dict(time=[1,5],modality='video')))
        for bad in [dict(time=[-1,5],modality='audio'),dict(time=[0,41],modality='audio'),dict(time=[0,35],modality='video')]:
            with self.assertRaises(ValueError):pool.validate(bad)

if __name__=='__main__':unittest.main()
