"""Current record Memory must reach source judging, teacher selection and Memory credit."""
import copy,json,tempfile,unittest
from pathlib import Path
from amp.common import TRAITS,digest,read_json
from amp.perception import decode_observation
from amp.state import memory_changes
from amp.teca import memory_quality_credit
from amp.train import filter_best

CFG=read_json(Path(__file__).resolve().parents[1]/'configs/local_omni3b.json')

def trajectory():
    source=dict(time=[0.,5.],modality='text',segments=[dict(segment_id='S1',text='I asked a colleague for help.')],media=[])
    record=dict(id='M1',observation_id='active_0',content=json.dumps(dict(observations=['Asked a colleague for help.'],candidates=[])),modality='text',retrieval_window=[0.,5.],segments=source['segments'])
    memory=dict(observations=[record])
    check=dict(supported=True,reason='The supplied transcript supports the finding.',record_fingerprint=digest(record),source_fingerprint=digest(source))
    return dict(sample_id='SYNTHETIC:record',dataset='SYNTHETIC',person_id='p',split='train',offline=False,run_status='complete',memory_protocol='observation_text',
        initial=dict(memory=dict(observations=[])),memory=memory,observations=dict(active_0=source),
        steps=[dict(index=0,observation_id='active_0',memory=memory,cost=.1,before={t:[.2]*5 for t in TRAITS},after={t:[.2]*5 for t in TRAITS})],
        decisions=[],final={t:dict(level='medium') for t in TRAITS},final_explanation='A test explanation.',contextual=[],
        judge=dict(records=dict(M1=check),final=dict(faithfulness=1.,citation_correctness=1.)),
        cost=dict(active_seconds=5.,coarse_audio_seconds=0.,coarse_frames=0,generated_tokens=1,input_tokens=1))

class RecordTrainingTests(unittest.TestCase):
    def test_credit_report_reads_records_without_fabricating_trait_gains(self):
        from amp.evaluate import credit_diagnostics
        result=credit_diagnostics([trajectory()])
        self.assertEqual(result['n_observations'],1)
        self.assertEqual(result['observations'][0]['F_memory'],1.)
        self.assertIsNone(result['trait_gain'])
        self.assertIsNone(result['context_gain'])

    def test_supported_current_record_can_pass_teacher_filter(self):
        t=trajectory();best,rejected=filter_best([t],{t:2 for t in TRAITS})
        self.assertIsNotNone(best,rejected)

    def test_supported_record_update_gets_quality_credit(self):
        self.assertEqual(memory_quality_credit(trajectory(),0)[0],1.)

    def test_retired_observation_shapes_are_not_read_as_facts(self):
        for text in ['A free text finding.','```json\n{"observations":[],"candidates":[]}\n```','{"findings":["A fact"],"candidates":[]}']:
            d=decode_observation(text)
            self.assertEqual(d['status'],'invalid')
            self.assertEqual(d['observations'],'')

if __name__=='__main__':unittest.main()

class RecordQualityTests(unittest.TestCase):
    def test_missing_or_stale_judge_checks_do_not_earn_credit(self):
        for change in ('missing','changed_record','changed_source','non_boolean'):
            t=trajectory()
            if change=='missing':t['judge']['records']={}
            elif change=='changed_record':t['memory']['observations'][0]['content']=json.dumps(dict(observations=['A different fact.'],candidates=[]))
            elif change=='changed_source':t['observations']['active_0']['segments'][0]['text']='A changed source.'
            else:t['judge']['records']['M1']['supported']='yes'
            self.assertEqual(memory_quality_credit(t,0)[0],0.,change)
            self.assertIsNone(filter_best([t],{k:2 for k in TRAITS})[0])

    def test_invalid_source_or_unsupported_record_is_negative(self):
        t=trajectory();t['judge']['records']['M1']['supported']=False
        self.assertEqual(memory_quality_credit(t,0)[0],-1.)
        t=trajectory();t['memory']['observations'][0]['retrieval_window']=[1.,4.]
        self.assertEqual(memory_quality_credit(t,0)[0],-1.)

    def test_reused_old_record_cannot_receive_current_observation_credit(self):
        t=trajectory();t['steps'][0]['observation_id']='active_other'
        self.assertEqual(memory_quality_credit(t,0)[0],-1.)
        t=trajectory();t['initial']['memory']=copy.deepcopy(t['memory'])
        self.assertEqual(memory_quality_credit(t,0)[0],0.)

    def test_true_reverse_behavior_is_not_penalized_by_personality_labels(self):
        t=trajectory();r=t['memory']['observations'][0]
        r['content']=json.dumps(dict(observations=['Usually waits for others to start conversations.'],candidates=[]))
        t['observations']['active_0']['segments'][0]['text']='I usually wait for others to start conversations.'
        check=t['judge']['records']['M1'];check.update(record_fingerprint=digest(r),source_fingerprint=digest(t['observations']['active_0']))
        self.assertEqual(memory_quality_credit(t,0)[0],1.)

    def test_empty_observation_is_not_a_positive_quality_reward(self):
        t=trajectory();t['memory']['observations'][0]['content']='{"observations":[],"candidates":[]}'
        self.assertEqual(memory_quality_credit(t,0)[0],0.)

    def test_record_memory_does_not_fabricate_trait_or_context_gain(self):
        from amp.teca import local_credit
        values,parts=local_credit(trajectory(),{k:2 for k in TRAITS},CFG)
        self.assertEqual(parts[0]['trait'],0.);self.assertEqual(parts[0]['context'],0.)
        self.assertIn('not_attributable',parts[0]['trait_status'])
        with self.assertRaises(ValueError):local_credit(trajectory(),{k:2 for k in TRAITS},CFG,enable_trait=True)

    def test_existing_judge_reads_findings_and_actual_source_not_candidate_plans(self):
        from amp.judge import MultimodalJudge
        t=trajectory();r=t['memory']['observations'][0]
        r['content']=json.dumps(dict(observations=['Asked a colleague for help.'],candidates=[dict(local_time=[1,2],modality='audio',hint='A clue',goal='UNOBSERVED_PLAN')]))
        seen=[]
        class Judge:
            def generate(self,messages,kind,step):
                payload=json.loads(messages[-1]['content'][-1]['text']);seen.append(payload)
                if 'record' in payload:return dict(supported=True,reason='Matches the transcript.'),{}
                return dict(faithfulness=1.,citation_correctness=1.),{}
        t['judge']=MultimodalJudge(Judge()).assess(t)
        self.assertEqual(set(t['judge']['records']),{'M1'})
        self.assertNotIn('UNOBSERVED_PLAN',str(seen))
        self.assertIn('I asked a colleague',str(seen))
        self.assertEqual(seen[-1]['records']['M1']['retrieval_window'],[0.,5.])
        self.assertEqual(memory_quality_credit(t,0)[0],1.)

    def test_retired_atomic_memory_is_rejected_explicitly(self):
        from amp.state import memory_records
        with self.assertRaises(ValueError):memory_records({k:dict(level=None,evidence=[]) for k in TRAITS})

class TrainingTokenTests(unittest.TestCase):
    def backend_and_decision(self):
        import torch
        from types import SimpleNamespace
        from test_output_contract import OutputContractTests
        from amp.perception import generation_schema
        b=OutputContractTests().backend();size=b._format_tokenizer_data.vocab_size
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(size))
            def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.bias.expand(1,input_ids.shape[1],size))
        b.policy=Policy();b.prepare=lambda *args:dict(input_ids=torch.tensor([[1]]),attention_mask=torch.ones(1,1,dtype=torch.long))
        b.save=lambda path:Path(path).mkdir(parents=True,exist_ok=True)
        text=trajectory()['memory']['observations'][0]['content']
        d=dict(kind='memory',step=0,text=text,messages=[dict(role='system',content='PROMPT_ONLY',output_schema=generation_schema('memory'))],tokens=b.processor.tokenizer.encode(text)+[2],format_mode='json_schema',python_result='PYTHON_RESULT_NOT_A_TARGET')
        return b,d

    def test_sft_exports_actor_tokens_only_and_performs_a_cpu_update(self):
        import torch
        from amp.train import export_sft,sft
        b,d=self.backend_and_decision();t=trajectory();t.update(teacher_accepted=True,decisions=[d])
        before=b.policy.bias.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sft.jsonl';self.assertEqual(export_sft([t],path),1)
            row=json.loads(path.read_text());target=b.processor.tokenizer.decode(row['tokens'])
            self.assertNotIn('PROMPT_ONLY',target);self.assertNotIn('PYTHON_RESULT',target)
            sft(b,path,Path(folder)/'model',dict(CFG,learning_rate=.001))
        self.assertFalse(torch.equal(before,b.policy.bias.detach()))

    def test_memory_quality_affects_segment_advantage_and_grpo_gradient(self):
        import torch
        from amp.teca import advantages,clipped_grpo
        b,d=self.backend_and_decision();t=trajectory();t['decisions']=[d]
        other=copy.deepcopy(t);other['final']={k:dict(level='very_low') for k in TRAITS}
        labels={k:2 for k in TRAITS}
        base=advantages([t,other],labels,CFG,'grpo')[0][0]
        full=advantages([t,other],labels,CFG,'teca')[0][0]
        self.assertAlmostEqual(full-base,CFG['memory_quality_beta'])
        optimizer=torch.optim.AdamW(b.policy.parameters(),lr=.001);before=b.policy.bias.detach().clone()
        logp,_=b.token_stats(d);loss=clipped_grpo(logp,logp.detach(),logp.detach(),full)
        loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss));self.assertFalse(torch.equal(before,b.policy.bias.detach()))
        t['decisions'][0]['proposal_repair']=True
        self.assertAlmostEqual(advantages([t,other],labels,CFG,'teca')[0][0],base)

    def test_retired_tool_targets_cannot_enter_sft(self):
        from amp.train import export_sft
        _,d=self.backend_and_decision();t=trajectory();d['tools']=[dict(name='retired_memory_tool')];t.update(teacher_accepted=True,decisions=[d])
        with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):export_sft([t],Path(folder)/'bad.jsonl')
