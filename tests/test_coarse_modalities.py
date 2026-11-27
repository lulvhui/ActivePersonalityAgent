"""Three input modalities, retained coarse notes, and small-model Final shapes."""
import copy
import json
import tempfile
import unittest
from amp.agent import ActiveAgent
from amp.common import TRAITS
from amp.media import MediaTools
from amp.scoring import OfflineTraitScorer
from test_simple_memory import Reader,CFG,ROW,OBS


class CoarseMedia:
    def coarse(self,row,*args):
        source=lambda mode,time:dict(modality=mode,time=time,media=[],segments=[],timing='media_relative')
        return dict(duration=37.466667,media_id='SYNTHETIC',question='',
            transcript=[dict(start=0,end=10,text='A synthetic statement.')],
            audio=[source('audio',[0,20]),source('audio',[20,37.46])],
            video=[source('video',[0,20]),source('video',[20,37.466667])],
            chunk_seconds=20,available_modalities=['text','audio','video','audio_video'])


class SummaryReader(Reader):
    def generate(self,messages,kind,step):
        if kind=='candidate_scan':
            self.seen.append((kind,step,copy.deepcopy(messages)))
            p=json.loads(messages[-1]['content'][-1]['text']);mode=p['observation']['modality']
            obj=dict(summary=f'Actual {mode} source description.',candidates=[])
            text=json.dumps(obj)
            return obj,dict(messages=messages,kind=kind,step=step,text=text,tokens=list(map(ord,text)),offline=True)
        return super().generate(messages,kind,step)


class CoarseModalityTests(unittest.TestCase):
    def test_legal_target_does_not_require_a_goal_string(self):
        from amp.state import CandidatePool
        pool=CandidatePool(40,30)
        action=pool.validate(dict(time=[20,37.46],modality='audio'))
        self.assertEqual(action,dict(time=[20.,37.46],modality='audio',goal=''))

    def test_plain_source_description_is_retained_without_json_retries(self):
        class Plain(SummaryReader):
            def generate_text(self,messages,kind,step):
                if kind=='candidate_scan':
                    self.seen.append((kind,step,copy.deepcopy(messages)))
                    text='A factual description of this source, not JSON.'
                    return text,dict(messages=messages,kind=kind,step=step,text=text,tokens=[1],offline=True)
                return super().generate_text(messages,kind,step)
        reader=Plain(actions=[dict(stop=True)])
        t=ActiveAgent(reader,OfflineTraitScorer(),CoarseMedia(),CFG).run(ROW)
        self.assertEqual(sum(k=='candidate_scan' for k,_,_ in reader.seen),5)
        self.assertTrue(all(s['summary']=='' and s['summary_status']=='invalid' for s in t['global_scan']['coarse_summaries']))

    def test_explicit_goals_can_reference_the_actual_source_summary(self):
        class Candidates(SummaryReader):
            def generate(self,messages,kind,step):
                if kind=='candidate_scan':
                    p=json.loads(messages[-1]['content'][-1]['text']);mode=p['observation']['modality']
                    c=dict(modality=mode,goal='Inspect this source')
                    obj=dict(summary='Observed source content.',candidates=[c]);text=json.dumps(obj)
                    return obj,dict(messages=messages,kind=kind,step=step,text=text,tokens=[1],offline=True)
                return super().generate(messages,kind,step)
        t=ActiveAgent(Candidates(actions=[dict(stop=True)]),OfflineTraitScorer(),CoarseMedia(),CFG).run(ROW)
        self.assertEqual(t['initial']['candidate_count'],0)
        self.assertTrue(any(d.get('rejected_candidates') for d in t['decisions']))

    def test_each_source_summary_survives_an_empty_candidate_pool(self):
        reader=SummaryReader(actions=[dict(stop=True)])
        t=ActiveAgent(reader,OfflineTraitScorer(),CoarseMedia(),CFG).run(ROW)
        summaries=t['global_scan']['coarse_summaries']
        self.assertEqual(len(summaries),5)
        self.assertEqual({s['modality'] for s in summaries},{'text','audio','video'})
        self.assertTrue(all(s['summary'] and s['summary_status']=='provided' for s in summaries))
        self.assertEqual(t['memory']['observations'],[])
        self.assertFalse(any(k=='action' for k,_,_ in reader.seen))
        self.assertEqual(t['stop_reason'],'candidate_pool_empty')

    def test_scan_distinguishes_current_inputs_from_available_action_modes(self):
        reader=SummaryReader(actions=[dict(stop=True)])
        ActiveAgent(reader,OfflineTraitScorer(),CoarseMedia(),CFG).run(ROW)
        for kind,_,messages in reader.seen:
            if kind!='candidate_scan':continue
            p=json.loads(messages[-1]['content'][-1]['text'])
            self.assertEqual(p['input_modalities'],[p['observation']['modality']])
            self.assertIn('audio_video',p['allowed_modalities'])
            self.assertNotIn('audio_video',p['input_modalities'])

    def test_coarse_and_detailed_observation_have_distinct_tasks(self):
        reader=SummaryReader(actions=[dict(stop=True)])
        agent=ActiveAgent(reader,OfflineTraitScorer(),CoarseMedia(),CFG)
        trajectory=agent.run(ROW)
        memory,_,_=agent.update(trajectory['memory'],{'active_0':OBS},{'active_0':OBS},0)
        scan=next(messages for kind,_,messages in reader.seen if kind=='candidate_scan')
        observe=next(messages for kind,_,messages in reader.seen if kind=='memory')
        coarse={'task':scan[-1]['content'][0]['text']}
        detail={'task':observe[-1]['content'][0]['text']}
        self.assertIn('closer read',coarse['task'])
        self.assertIn('local_time',coarse['task'])
        self.assertIn('Do not infer hidden motives',coarse['task'])
        self.assertIn('Do not infer hidden motives',detail['task'])
        self.assertIn('candidates=[]',coarse['task'])
        self.assertIn('narrower window or another modality',detail['task'])
        self.assertIn('distinguishing self-reports',detail['task'])
        self.assertIn('Leave personality judgments to Final Reasoning',observe[0]['content'])
        self.assertNotIn('possible personality interpretations',detail['task'])
        self.assertTrue(memory['observations'])

    def test_fixed_video_chunks_match_audio_transport_windows(self):
        class Media(MediaTools):
            def probe(self,path):return dict(duration=37.466667,source_key='SYNTHETIC')
            def observe(self,row,action,**kw):return dict(action,media=[dict(sampled_frames=1)])
            def full_audio(self,row,seconds):return [dict(time=[0,20]),dict(time=[20,37.46])],599360,16000
        with tempfile.TemporaryDirectory() as temp:
            result=Media(temp,30,1).coarse(dict(video='not-opened',segments=[]),1,20)
        self.assertEqual(result['video_coverage'],[[0,20],[20,37.466667]])
        self.assertEqual(result['chunk_seconds'],20)

    def test_flat_levels_and_case_are_accepted_without_invented_explanations(self):
        from amp.state import normalize_final
        raw={**{t:'high' for t in TRAITS},'reasoning':'A test explanation.','contextual':[]}
        normalized=normalize_final(raw)
        self.assertTrue(all(v['level']=='high' and v['reasoning'] is None for v in normalized.values()))
        self.assertIn('reasoning',raw)

    def test_nested_legacy_results_and_optional_reasoning_are_preserved(self):
        from amp.state import normalize_final
        raw={t:dict(level=None,reasoning='Insufficient context.') for t in TRAITS}
        with self.assertRaises(ValueError):normalize_final(raw)
        raw['O']['level']='unknown-bin'
        with self.assertRaises(ValueError):normalize_final(raw)

    def test_flat_final_reaches_trajectory_with_missing_reasoning_marked(self):
        class Flat(SummaryReader):
            def generate(self,messages,kind,step):
                if kind=='final':
                    obj={t:'high' for t in TRAITS};text=json.dumps(obj)
                    return obj,dict(messages=messages,kind=kind,step=step,text=text,tokens=list(map(ord,text)),offline=True)
                return super().generate(messages,kind,step)
        t=ActiveAgent(Flat(actions=[dict(stop=True)]),OfflineTraitScorer(),CoarseMedia(),CFG).run(ROW)
        self.assertEqual(t['final_status'],'unavailable')
        self.assertFalse(t['final_prediction_available'])

if __name__=='__main__':unittest.main()
