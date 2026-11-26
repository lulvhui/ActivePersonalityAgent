"""Performance changes must preserve policy probabilities, gradients and source time."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from amp.backend import QwenBackend
import test_output_contract


class PerformanceTests(unittest.TestCase):
    def test_timing_does_not_change_generated_tokens(self):
        b=test_output_contract.OutputContractTests().backend()
        b.prepare=lambda messages:dict(input_ids=torch.tensor([[1]]))
        b.policy=SimpleNamespace(generate=lambda **kw:torch.tensor([[1,2]]))
        _,d=b.generate([],'memory',0,parse=False)
        self.assertEqual(d['tokens'],[2])
        self.assertEqual(d['completion_tokens'],1)
        self.assertGreaterEqual(d['timing']['prepare_seconds'],0)
        self.assertGreaterEqual(d['timing']['generation_seconds'],0)
        self.assertEqual(d['timing']['constraint_calls'],0)
        self.assertNotIn('profile',d)

    def test_optional_profile_cleans_up_hooks_and_stage_summary_counts_repairs(self):
        from unittest.mock import patch
        from amp.backend import stage_timing
        b=test_output_contract.OutputContractTests().backend();b.profiling=True
        b.prepare=lambda messages:dict(input_ids=torch.tensor([[1]]))
        class Policy(torch.nn.Module):
            def forward(self,x):return x
            def generate(self,**kwargs):
                self(torch.tensor([1]));self(torch.tensor([2]));return torch.tensor([[1,2]])
        b.policy=Policy()
        with patch('torch.cuda.is_available',return_value=False):_,d=b.generate([],'memory',0,parse=False)
        self.assertEqual(d['profile']['forward_calls'],2)
        self.assertFalse(b.policy._forward_hooks);self.assertFalse(b.policy._forward_pre_hooks)
        d['failed_attempts']=[dict(d)]
        summary=stage_timing([d,dict(d,failed_attempts=[],proposal_repair=True)])
        self.assertEqual(summary['observe']['calls'],1)
        self.assertEqual(summary['repair']['calls'],2)
        self.assertEqual(summary['repair']['completion_tokens'],2)

    def test_coarse_context_keeps_neighbors_across_modalities_without_a_count_cap(self):
        from amp.agent import ActiveAgent
        from amp.state import CandidatePool
        from amp.scoring import OfflineTraitScorer
        from test_simple_memory import Reader,CFG
        reader=Reader();pool=CandidatePool(300,30)
        near=pool.add(dict(time=[35,40],modality='video',goal='Near'))
        far=pool.add(dict(time=[200,210],modality='audio',goal='Far'))
        agent=ActiveAgent(reader,OfflineTraitScorer(),None,dict(CFG,coarse_context_mode='nearby'))
        agent.scan_candidates(pool,'source',dict(time=[40,60],modality='audio',segments=[],media=[]),'',0)
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertEqual([c['goal'] for c in payload['existing_candidates']],['Near'])
        agent.config['coarse_context_mode']='full'
        agent.scan_candidates(pool,'source',dict(time=[40,60],modality='audio',segments=[],media=[]),'',1)
        payload=json.loads(reader.seen[-1][2][-1]['content'][-1]['text'])
        self.assertIn('Far',[c['goal'] for c in payload['existing_candidates']])

    def test_chunked_statistics_match_dense_logp_entropy_and_gradients(self):
        from transformers import RepetitionPenaltyLogitsProcessor
        from amp.perception import generation_schema
        b=test_output_contract.OutputContractTests().backend()
        vocab=b._format_tokenizer_data.vocab_size
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.linspace(-2,2,vocab))
            def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.bias.expand(1,input_ids.shape[1],vocab))
        b.policy=Policy();prompt=torch.tensor([[1,5,6,5]])
        b.prepare=lambda messages:dict(input_ids=prompt.clone())
        text='{"observations":["A fact","Another fact"],"candidates":[]}'
        tokens=b.processor.tokenizer.encode(text)+[2]
        for constrained in (False,True):
            for penalty in (1.,1.1):
                d=dict(messages=[dict(role='system',content='Read',output_schema=generation_schema('memory'))],tokens=tokens,repetition_penalty=penalty)
                if constrained:d['format_mode']='json_schema'
                rows=[];constraint=b.format_constraint(generation_schema('memory')) if constrained else None
                for i,token in enumerate(tokens):
                    history=torch.tensor([[*prompt[0].tolist(),*tokens[:i]]])
                    # The original training implementation penalizes unique IDs;
                    # scatter with duplicate indices has an incorrect duplicate gradient.
                    row=RepetitionPenaltyLogitsProcessor(penalty)(torch.unique(history)[None],b.policy.bias[None].clone())[0]
                    if constraint:
                        mask=torch.zeros(vocab,dtype=torch.bool);mask[constraint(0,history[0])]=True
                        row=row.masked_fill(~mask,float('-inf'))
                    rows.append(row)
                dense=torch.stack(rows).log_softmax(-1)
                expected=dense.gather(-1,torch.tensor(tokens)[:,None])[:,0]
                entropy=-(dense.exp()*dense.masked_fill(~dense.isfinite(),0)).sum(-1)
                for chunk in (1,7,128):
                    for need_entropy in (False,True):
                        b.policy.zero_grad();loss=-expected.mean()+(.03*entropy.mean() if need_entropy else 0)
                        reference_grad=torch.autograd.grad(loss,b.policy.bias,retain_graph=True)[0]
                        actual,h=b.token_stats(d,need_entropy=need_entropy,chunk_size=chunk)
                        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
                        if need_entropy:torch.testing.assert_close(h,entropy,atol=1e-5,rtol=0)
                        else:self.assertIsNone(h)
                        (-actual.mean()+(.03*h.mean() if need_entropy else 0)).backward()
                        torch.testing.assert_close(b.policy.bias.grad,reference_grad,atol=1e-5,rtol=0)

    def test_preprocessing_cache_is_cpu_only_scoped_and_not_mutated(self):
        from unittest.mock import patch
        from transformers.feature_extraction_utils import BatchFeature
        calls=[]
        class Processor:
            def apply_chat_template(self,*args,**kwargs):return 'A prompt'
            def __call__(self,**kwargs):
                calls.append(kwargs)
                return BatchFeature(dict(input_ids=torch.tensor([[1,3]]),attention_mask=torch.ones(1,2,dtype=torch.long)))
        b=QwenBackend.__new__(QwenBackend);b.torch=torch;b.processor=Processor();b.policy=SimpleNamespace(device=torch.device('cpu'),dtype=torch.float32)
        messages=[dict(role='system',content='A source')]
        with patch('qwen_omni_utils.process_mm_info',return_value=(None,None,None,{})):
            with b.prepared_group():
                a=b.prepare(messages);a['input_ids']=torch.tensor([[1,3,4]])
                self.assertEqual(b.prepare(messages)['input_ids'].shape[1],2)
                self.assertTrue(all(x.device.type=='cpu' for v in b._prepared_cache.values() for x in v.values()))
                self.assertEqual(len(calls),1)
            b.prepare(messages);self.assertEqual(len(calls),2)
        self.assertIsNone(b._prepared_cache)

    def test_sft_accumulates_gradients_and_normalizes_the_short_final_group(self):
        from test_record_training import TrainingTokenTests,CFG
        from amp.common import write_jsonl
        from amp.train import sft
        from unittest.mock import patch
        b,d=TrainingTokenTests().backend_and_decision()
        d.update(split='train',offline=False)
        rows=[dict(d,tokens=d['tokens'][:],text=d['text']) for _ in range(3)]
        reference=copy.deepcopy(b.policy)
        opt=torch.optim.AdamW(reference.parameters(),lr=.001)
        for size in (2,1):
            b.policy=reference;opt.zero_grad()
            for _ in range(size):
                logp,_=b.token_stats(d,need_entropy=False)
                (-logp.mean()/size).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);opt.step()
        expected=reference.bias.detach().clone()
        b.policy=type(reference)()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sft.jsonl';write_jsonl(path,rows)
            with patch.object(torch.optim.AdamW,'step',autospec=True,side_effect=torch.optim.AdamW.step) as steps:
                sft(b,path,Path(folder)/'model',dict(CFG,learning_rate=.001,grad_accum_steps=2))
                self.assertEqual(steps.call_count,2)
        torch.testing.assert_close(b.policy.bias,expected,atol=1e-6,rtol=0)

    def test_rl_update_keeps_old_reference_and_current_passes_without_unused_entropy(self):
        from test_record_training import TrainingTokenTests,trajectory,CFG
        from amp.train import rl
        from amp.common import TRAITS
        from unittest.mock import patch
        b,d=TrainingTokenTests().backend_and_decision();before=b.policy.bias.detach().clone()
        b.policy.device=torch.device('cpu')
        b.policy.load_adapter=lambda *args,**kwargs:None
        b.policy.set_adapter=lambda name:None
        b.policy.save_pretrained=lambda *args,**kwargs:None
        b.processor.save_pretrained=lambda *args:None
        calls=[];stats=b.token_stats
        def measured(decision,**kwargs):
            self.assertIsNotNone(b._prepared_cache)
            calls.append(kwargs['need_entropy']);return stats(decision,**kwargs)
        b.token_stats=measured
        class Agent:
            n=0
            def run(self,row):
                t=trajectory();t['decisions']=[copy.deepcopy(d)]
                if self.n:t['final']={k:dict(level='very_low') for k in TRAITS}
                self.n+=1
                return t
        class Judge:
            def assess(self,t):return t['judge']
        with tempfile.TemporaryDirectory() as folder,patch('peft.load_peft_weights',return_value={}),patch('peft.set_peft_model_state_dict'):
            rl(b,Agent(),Judge(),[dict(split='train',labels={k:2 for k in TRAITS})],folder,dict(CFG,learning_rate=.001),'synthetic_adapter',group_size=2)
        self.assertEqual(calls,[False]*6)
        self.assertFalse(torch.equal(before,b.policy.bias.detach()))
        self.assertIsNone(b._prepared_cache)


if __name__=='__main__':unittest.main()
