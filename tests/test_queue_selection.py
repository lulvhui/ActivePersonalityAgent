"""Current selection is program scheduling; only perception/final are generated."""
import json
import tempfile
import unittest
from pathlib import Path
from amp.agent import ActiveAgent
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia
from amp.perception import admit_candidates,decode_observation
from amp.provenance import index_transcript
from amp.state import CandidatePool,contextual_final,save_observation
from test_simple_memory import Reader,CFG,ROW,OBS


class NoSelectReader(Reader):
    def generate(self,messages,kind,step):
        if kind=='action':raise AssertionError('Select must not call a model')
        return super().generate(messages,kind,step)


class QueueMedia(SyntheticMedia):
    def coarse(self,row,*args):
        base=super().coarse(row,*args)
        base['audio']=[dict(modality='audio',time=[0.,5.],media=[],segments=[])]
        return base
    def observe(self,row,action):
        result=super().observe(row,action)
        if action['modality']=='text':
            indexed=index_transcript(self.coarse(row,1,20)['transcript'],'synthetic',60)
            result['segments']=[s for s in indexed if s['source_time'][0]<action['time'][1] and s['source_time'][1]>action['time'][0]]
        return result


class QueueSelectionTests(unittest.TestCase):

    def test_action_entropy_uses_canonical_actions_only(self):
        from amp.backend import action_token_mask
        class Characters:
            def decode(self,tokens,**kwargs):return ''.join(chr(t) for t in tokens)
            def __call__(self,text,**kwargs):return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
        def mask(text):return action_token_mask(Characters(),dict(tokens=list(map(ord,text))))
        for action in ['{"candidate_id":"C2"}','{"time":[2,8],"modality":"audio","goal":"Inspect"}']:
            self.assertTrue(any(mask(action)))
        for retired in ['C2','{"id":"C2"}','{"stop":true}','{"action":"stop"}']:
            with self.assertRaises(ValueError):mask(retired)

    def test_coarse_output_does_not_request_model_source_ids(self):
        reader=NoSelectReader();self.run_agent(reader,max_steps=1)
        scan=next(messages for kind,_,messages in reader.seen if kind=='candidate_scan')
        fields=scan[0]['output_schema']['properties']['candidates']['items']['properties']
        self.assertEqual(set(fields),{'hint','goal'})
    def run_agent(self,reader,**options):
        cfg=dict(CFG,selection_policy='queue',max_steps=3);cfg.update(options)
        return ActiveAgent(reader,OfflineTraitScorer(),QueueMedia(),cfg).run(ROW)

    def test_unlocated_text_clues_share_the_real_whole_source_not_guessed_times(self):
        class Unlocated(NoSelectReader):
            def generate(self,messages,kind,step):
                if kind=='candidate_scan':
                    obj=dict(summary='A real source overview',candidates=[dict(hint='A paraphrased clue',goal='Read the self-report'),dict(hint='Another clue',goal='Read the context')])
                    return obj,dict(messages=messages,kind=kind,step=step,text=json.dumps(obj),tokens=[1],offline=True)
                return super().generate(messages,kind,step)
        t=self.run_agent(Unlocated(),max_steps=1)
        self.assertEqual(t['initial']['candidate_count'],2)
        text=next(x for x in t['pool'] if x['action']['modality']=='text')
        self.assertEqual(text['action']['time'],[0.,60.])
        self.assertEqual(len(text['leads']),2)

    def test_queue_is_default_and_creates_no_policy_action_tokens(self):
        cfg=dict(CFG,max_steps=2);cfg.pop('selection_policy',None)
        t=ActiveAgent(NoSelectReader(),OfflineTraitScorer(),SyntheticMedia(),cfg).run(ROW)
        self.assertEqual(len(t['steps']),1)
        self.assertEqual(t['selection_policy'],'queue')
        self.assertFalse(any(d['kind']=='action' for d in t['decisions']))
        self.assertTrue(all(s['action_decision_index'] is None and s['action_type']=='scheduled' for s in t['steps']))
        self.assertEqual(t['final_status'],'generated')

    def test_coarse_pairs_only_the_same_audio_video_window(self):
        class Media(SyntheticMedia):
            def coarse(self,row,*args):
                base=super().coarse(row,*args)
                base.update(audio=[dict(modality='audio',time=[0,20],media=[],segments=[])],
                    video=[dict(modality='video',time=[0,20],media=[],segments=[])])
                return base
        t=ActiveAgent(NoSelectReader(),OfflineTraitScorer(),Media(),dict(CFG,selection_policy='queue',max_steps=1)).run(ROW)
        scans=[d for d in t['decisions'] if d['kind']=='candidate_scan']
        self.assertEqual(len(scans),2)  # One joint source and the complete transcript.
        payload=json.loads(scans[0]['messages'][-1]['content'][-1]['text'])
        self.assertEqual(payload['input_modalities'],['audio','video'])
        self.assertEqual(t['global_scan']['input_modalities'],['text','audio','video'])

    def test_sft_and_outcome_records_do_not_invent_scheduler_tokens(self):
        from amp.train import export_sft
        from amp.teca import advantages,memory_outcomes
        from amp.common import TRAITS,read_jsonl
        t=self.run_agent(NoSelectReader(),max_steps=2)
        t.update(offline=False,teacher_accepted=True)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sft.jsonl'
            export_sft([t],path)
            self.assertFalse(any(r['kind']=='action' for r in read_jsonl(path)))
        values=advantages([t,t],{k:0 for k in TRAITS},CFG)
        self.assertEqual(len(values[0]),len(t['decisions']))
        outcomes=memory_outcomes(t,{k:0 for k in TRAITS},CFG)
        self.assertEqual(outcomes[0]['next_action'],t['steps'][1]['action'])
        self.assertIsNone(outcomes[0]['next_action_decision_index'])

    def test_coarse_targets_are_read_before_pending_refinements(self):
        class Refining(NoSelectReader):
            def generate_text(self,messages,kind,step):
                if kind=='memory':
                    self.text=json.dumps(dict(observations=['Asked for help.'],candidates=[dict(local_time=[1,3],modality='video',hint='Inspect response',goal='Clarify visible behavior')] if step==0 else []))
                return super().generate_text(messages,kind,step)
        t=self.run_agent(Refining())
        self.assertEqual([s['candidate_id'] for s in t['steps']],['C1','C2','C3'])
        self.assertEqual(t['steps'][2]['parent_candidate_id'],'C1')
        self.assertEqual(len(t['memory']['observations']),3)

    def test_refinement_link_does_not_demote_an_existing_coarse_target(self):
        pool=CandidatePool(60,30)
        parent=pool.add(dict(time=[0,20],modality='audio'))
        root=pool.add(dict(time=[20,22],modality='audio'),source_observation_id='global_audio_1')
        linked=pool.add(dict(time=[20,22],modality='audio'),parent_id=parent,
            source_observation_id='active_0',origin='observation',hint='Hear the continuation')
        self.assertEqual(linked,root)
        self.assertEqual(pool.items[root]['origin'],'coarse')
        self.assertIsNone(pool.items[root]['parent_id'])
        self.assertEqual(pool.items[root]['source_observation_id'],'global_audio_1')
        self.assertEqual(pool.items[root]['leads'][0]['source_observation_id'],'active_0')

    def test_deeper_children_do_not_jump_ahead_of_older_refinements(self):
        class Refining(NoSelectReader):
            def generate_text(self,messages,kind,step):
                if kind=='memory':
                    targets={0:([1,3],'video'),1:([10,12],'video'),2:([0,1],'audio')}
                    proposals=[]
                    if step in targets:
                        time,modality=targets[step]
                        proposals=[dict(local_time=time,modality=modality,hint='An observed clue',goal='Clarify the response')]
                    self.text=json.dumps(dict(observations=['A source finding.'],candidates=proposals))
                return super().generate_text(messages,kind,step)
        t=self.run_agent(Refining(),max_steps=5,min_new_seconds=None)
        self.assertEqual([s['candidate_id'] for s in t['steps']],['C1','C2','C3','C4','C5'])
        self.assertEqual(t['stop_reason'],'candidate_pool_exhausted')

    def test_terminal_observation_does_not_fill_unexecutable_followups(self):
        raw=json.dumps(dict(observations=['A complete finding.'],candidates=[dict(local_time=[1,3],modality='video',hint='Visible response',goal='Clarify')]))
        t=self.run_agent(NoSelectReader(text=raw),max_steps=1)
        self.assertEqual(t['steps'][0]['followup_status'],'disabled_budget')
        self.assertEqual(t['steps'][0]['refinement_candidate_ids'],[])
        self.assertEqual(len(t['pool']),2)
        self.assertEqual(t['stop_reason'],'max_steps')
        self.assertEqual(t['final_status'],'generated')

    def test_existing_requests_use_local_seconds_and_only_relevant_context(self):
        reader=NoSelectReader()
        ActiveAgent(reader,OfflineTraitScorer(),None,CFG).update(dict(observations=[]),{},
            {'active_0':dict(OBS,time=[200.,220.])},0,candidate_state=[
                dict(time=[202,205],modality='audio',status='done'),dict(time=[0,20],modality='video',status='pending')],remaining_steps=2)
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertEqual(payload['known_observation_requests'],[dict(covered_local_time=[2.,5.],modality='audio',status='done')])

    def test_observe_reads_related_findings_but_never_previous_plans(self):
        memory,_=save_observation(dict(observations=[]),json.dumps(dict(observations=['Asked a colleague for help.'],
            candidates=[dict(goal='UNEXECUTED_PLAN')])),dict(OBS,time=[200.,220.]),'active_0')
        memory,_=save_observation(memory,json.dumps(dict(observations=['UNRELATED_SOURCE'],candidates=[])),OBS,'active_1')
        reader=NoSelectReader()
        ActiveAgent(reader,OfflineTraitScorer(),None,CFG).update(memory,{},
            {'active_2':dict(OBS,time=[202.,205.])},2,media_duration=206.)
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertEqual([r['id'] for r in payload['prior_observations']],['M1'])
        self.assertIn('Asked a colleague',payload['prior_observations'][0]['content'])
        self.assertNotIn('UNEXECUTED_PLAN',json.dumps(payload))
        self.assertEqual(payload['allowed_local_seconds'],[-202.,4.])

    def test_unaligned_asr_and_global_history_do_not_leak_into_proposal_coordinates(self):
        source=dict(OBS,time=[0.,300.])
        memory,_=save_observation(dict(observations=[]),'{"observations":["A prior finding."],"candidates":[]}',source,'active_0')
        obs=dict(OBS,time=[200.,220.],segments=[dict(text='An unaligned transcript.',question='',timing='unverified_asr',source_time=[0.,300.])])
        reader=NoSelectReader()
        ActiveAgent(reader,OfflineTraitScorer(),None,CFG).update(memory,{},dict(active_1=obs),1,
            candidate_state=[dict(time=[0.,300.],modality='text',status='done')],media_duration=300.)
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertNotIn('transcript',payload)
        self.assertNotIn('local_retrieval_time',payload['segments'][0])
        self.assertNotIn('retrieval_window',payload['prior_observations'][0])
        self.assertEqual(payload['known_observation_requests'][0]['covered_local_time'],[-5.,25.])

    def test_verified_text_positions_remain_on_the_same_local_clock(self):
        reader=NoSelectReader()
        obs=dict(OBS,time=[200.,220.],segments=[dict(text='A verified source.',timing='media_aligned',
            source_time=[0.,300.],media_time=[202.,205.])])
        ActiveAgent(reader,OfflineTraitScorer(),None,CFG).update(dict(observations=[]),{},dict(active_0=obs),0)
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertEqual(payload['segments'][0]['local_time'],[2.,5.])
        self.assertEqual(obs['segments'][0]['media_time'],[202.,205.])

    def test_parameter_repair_only_receives_failed_proposals_and_full_constraints(self):
        class Repair(NoSelectReader):
            def generate(self,messages,kind,step):
                self.repair_messages=messages
                obj=dict(candidates=[])
                return obj,dict(messages=messages,kind=kind,step=step,text=json.dumps(obj),tokens=[1])
        reader=Repair();agent=ActiveAgent(reader,OfflineTraitScorer(),None,dict(CFG,max_validation_attempts=3))
        pool=CandidatePool(300,30);parent=pool.add(dict(time=[200.,220.],modality='audio'));pool.finish(parent)
        good=dict(local_time=[1,3],modality='video',hint='Visible behavior',goal='Clarify a response')
        bad=dict(local_time=[100,110],modality='audio',hint='A clue',goal='Clarify the reply')
        obs=dict(OBS,time=[200.,220.]);raw=json.dumps(dict(observations=['A finding.'],candidates=[good,bad]))
        ids,errors=admit_candidates(pool,json.loads(raw),obs,'active_0',parent_id=parent,origin='observation')
        _,remaining,repair=agent.repair_candidates(pool,obs,'active_0',parent,dict(text=raw),errors,0)
        payload=json.loads(reader.repair_messages[-1]['content'][-1]['text'])
        self.assertEqual(payload['invalid_proposals'],[dict(proposal=bad,errors=[errors[0]['error']])]);self.assertEqual(remaining,[])
        self.assertEqual(payload['allowed_local_seconds'],[-200.,100.]);self.assertEqual(payload['max_video_window_seconds'],30)
        self.assertEqual({r['status'] for r in payload['known_requests']},{'done','pending'})
        self.assertEqual(pool.items[ids[0]]['status'],'pending')
        self.assertTrue(repair['candidate_output_valid'])

    def test_adjacent_media_request_uses_observed_text_source_not_future_asr(self):
        obs=dict(modality='text',time=[41.,51.],segments=index_transcript([dict(start=41,end=51,text='A concrete observed self-report.')],'synthetic',80))
        pool=CandidatePool(80,30);parent=pool.add(dict(time=obs['time'],modality='text'));pool.finish(parent)
        ids,errors=admit_candidates(pool,dict(candidates=[dict(local_time=[10,15],modality='audio_video',hint='Clarify the continuation',goal='Hear the rest of the answer')]),obs,'active_0',parent_id=parent,origin='observation')
        self.assertEqual(errors,[])
        self.assertEqual(pool.items[ids[0]]['action']['time'],[51.,56.])
        self.assertEqual(pool.items[ids[0]]['leads'][0]['source_fragment_time'],[41.,51.])



if __name__=='__main__':unittest.main()
