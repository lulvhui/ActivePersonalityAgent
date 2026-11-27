"""One current JSON protocol; invalid generations are audit data, never inferred facts."""
import json,unittest
from amp.agent import ActiveAgent
from amp.state import CandidatePool,save_observation,empty_memory,contextual_final
from amp.perception import decode_observation,memory_view,bind_candidate,admit_candidates
from amp.provenance import index_transcript
from amp.scoring import minimize_state,OfflineTraitScorer,JevTraitScorer
from amp.smoke import SyntheticMedia
from test_simple_memory import Reader,CFG,ROW,OBS

class ObservationProtocolTests(unittest.TestCase):
    def test_legacy_aliases_and_truncation_are_explicit_errors(self):
        for raw in ['{"findings":["A fact"],"candidates":[]}',"{'observations':['A fact'],'candidates':[]}",'Observations: A fact\nCandidates: none','{"observations":["A complete fact"],"candidates":[']:
            decoded=decode_observation(raw)
            self.assertEqual(decoded['status'],'invalid');self.assertEqual(decoded['observations'],'')
            memory,_=save_observation(empty_memory(),raw,OBS,'active_0')
            self.assertEqual(memory['observations'][0]['content'],raw)
            self.assertFalse(any(minimize_state(memory).values()))

    def test_memory_views_and_scorer_exclude_unobserved_plans(self):
        raw=json.dumps(dict(observations=['Asked a colleague for help.'],candidates=[dict(local_time=[1,2],modality='video',hint='Inspect',goal='UNOBSERVED_PLAN')]))
        memory,_=save_observation(empty_memory(),raw,OBS,'active_0')
        self.assertIn('Asked a colleague',memory_view(memory)['observations'][0]['content'])
        self.assertNotIn('UNOBSERVED_PLAN',str(minimize_state(memory)))
        self.assertEqual(memory['observations'][0]['content'],raw)

    def test_plan_only_output_is_not_a_personality_prediction(self):
        raw=json.dumps(dict(observations=[],candidates=[dict(local_time=[1,2],modality='audio',hint='A clue',goal='Clarify')]))
        memory,_=save_observation(empty_memory(),raw,OBS,'active_0')
        self.assertFalse(any(minimize_state(memory).values()))
        self.assertFalse(OfflineTraitScorer().score(memory)['metadata']['predictive'])

    def test_text_source_does_not_overwrite_the_child_window(self):
        obs=dict(modality='text',time=[20.,40.],segments=index_transcript([dict(start=20,end=40,text='A concrete observed self-report.')],'synthetic',60))
        bound=bind_candidate(dict(local_time=[2,8],modality='audio',hint='A behavior clue',goal='Clarify'),obs,60)
        self.assertEqual(bound['time'],[22.,28.]);self.assertEqual(bound['source_fragment_time'],[20.,40.])
        self.assertEqual(bound['retrieval_timing'],'unverified_asr')

    def test_invalid_child_does_not_discard_a_valid_sibling(self):
        pool=CandidatePool(40,30);parent=pool.add(dict(time=[0,5],modality='audio'));pool.finish(parent)
        child=lambda time,mode:dict(local_time=time,modality=mode,hint='A source clue',goal='Clarify the behavior')
        ids,errors=admit_candidates(pool,dict(candidates=[child([1,3],'video'),child([100,110],'audio')]),OBS,'active_0',parent_id=parent,origin='observation')
        self.assertEqual(len(ids),1);self.assertEqual(len(errors),1)
        self.assertEqual(pool.items[ids[0]]['parent_id'],parent)

    def test_retry_corrects_output_without_interpreting_free_text(self):
        class Corrected(Reader):
            def __init__(self):super().__init__();self.count=0
            def generate_text(self,messages,kind,step):
                self.count+=1
                raw='A free text response.' if self.count==1 else '{"observations":["Asked for help."],"candidates":[]}'
                return raw,dict(kind=kind,step=step,messages=messages,text=raw,tokens=[1])
        reader=Corrected()
        memory,_,d=ActiveAgent(reader,OfflineTraitScorer(),None,dict(CFG,max_validation_attempts=2)).update(empty_memory(),{},dict(active_0=OBS),0)
        self.assertEqual(reader.count,2);self.assertEqual(len(d['failed_attempts']),1)
        self.assertEqual(len(memory['observations']),1)
        self.assertEqual(d['memory_update_status'],'accepted')

    def test_context_citations_must_use_existing_record_ids(self):
        memory,_=save_observation(empty_memory(),'{"observations":["A fact"],"candidates":[]}',OBS,'active_0')
        item=dict(trait='E',context_type='reported',context='Project',expression='The speaker reports seeking help.',citations=['M1'])
        good,bad=contextual_final(dict(contextual=[item,dict(item,citations=['M99'])]),memory)
        self.assertEqual(len(good),1);self.assertEqual(len(bad),1)
        self.assertTrue(contextual_final(dict(contextual=[dict(item,trait='Extraversion')]),memory)[1])

    def test_question_positions_and_followups_use_local_seconds(self):
        reader=Reader()
        obs=dict(OBS,time=[200.,220.],questions=[dict(time=[202,205],question='Describe the project')])
        ActiveAgent(reader,OfflineTraitScorer(),None,CFG).update(empty_memory(),{},dict(active_0=obs),0,candidate_state=[dict(time=[202,205],modality='audio',status='done')])
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertEqual(payload['question_context'][0]['local_time'],[2.,5.])
        self.assertEqual(payload['known_observation_requests'][0]['covered_local_time'],[2.,5.])

if __name__=='__main__':unittest.main()
