"""Length limits must not discard completed findings or the whole trajectory."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from amp.agent import ActiveAgent
from amp.backend import GenerationFormatError,QwenBackend
from amp.common import TRAITS
from amp.perception import decode_observation
from amp.scoring import OfflineTraitScorer,minimize_state
from amp.smoke import SyntheticMedia
from test_simple_memory import Reader,CFG,ROW


class CompletionTests(unittest.TestCase):

    def test_nonempty_question_objects_are_not_silently_treated_as_empty_findings(self):
        decoded=decode_observation('{"observations":[{"question":"What do you regret?"}],"candidates":[]}')
        self.assertEqual(decoded['status'],'invalid')
        self.assertEqual(decoded['errors'][0]['scope'],'observation')

    def test_native_finish_reason_is_logged_before_parsing(self):
        class Policy:
            def generate(self,**kwargs):
                if kwargs['repetition_penalty']!=1.1:raise AssertionError('Missing generation repetition control')
                return torch.tensor([[1,123,125,2]])
        with tempfile.TemporaryDirectory() as temp:
            backend=QwenBackend.__new__(QwenBackend)
            backend.torch=torch;backend.max_tokens=3;backend.temperature=0.;backend.policy=Policy()
            backend.generation_log_dir=Path(temp)
            backend.processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2,decode=lambda *a,**kw:'{}'))
            backend.prepare=lambda *args:dict(input_ids=torch.tensor([[1]]))
            _,d=backend.generate([],'memory',0)
            saved=json.loads(next(Path(temp).glob('*.json')).read_text())
            self.assertEqual(saved['finish_reason'],'eos')
            self.assertFalse(saved['truncated'])  # EOS on the last permitted token.
            self.assertEqual(d['repetition_penalty'],1.1)

    def test_repetition_penalty_log_probs_match_the_native_processor(self):
        from transformers import RepetitionPenaltyLogitsProcessor
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.tensor([-2.,-1.,0.,1.,2.,3.]))
            def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.bias.expand(1,input_ids.shape[1],6))
        backend=QwenBackend.__new__(QwenBackend);backend.torch=torch;backend.policy=Policy()
        backend.prepare=lambda *args:dict(input_ids=torch.tensor([[1,3]]))
        d=dict(messages=[],tokens=[3,4],repetition_penalty=1.1)
        logp,entropy=backend.token_stats(d)
        processor=RepetitionPenaltyLogitsProcessor(1.1)
        for index,token in enumerate(d['tokens']):
            history=torch.tensor([[1,3,*d['tokens'][:index]]])
            expected=processor(history,backend.policy.bias[None].clone()).log_softmax(-1)[0,token]
            torch.testing.assert_close(logp[index],expected)
        (-logp.mean()).backward()
        self.assertTrue(torch.isfinite(backend.policy.bias.grad).all())
        self.assertTrue(torch.isfinite(entropy).all())



    def test_observe_retries_only_when_no_usable_description_was_recovered(self):
        class Retry(Reader):
            def __init__(self):super().__init__();self.reads=[]
            def generate_text(self,messages,kind,step):
                if kind!='memory':return super().generate_text(messages,kind,step)
                self.reads.append(messages)
                raw='{"observations":"An unfinished' if len(self.reads)==1 else '{"observations":["Asked for help."],"candidates":[]}'
                return raw,dict(messages=messages,kind=kind,step=step,text=raw,tokens=[1],truncated=len(self.reads)==1,offline=True)
        reader=Retry()
        t=ActiveAgent(reader,OfflineTraitScorer(),SyntheticMedia(),dict(CFG,max_steps=1,max_validation_attempts=3)).run(ROW)
        self.assertEqual(len(reader.reads),2)
        self.assertEqual(len(t['memory']['observations']),1)
        d=t['decisions'][t['steps'][0]['memory_decision_index']]
        self.assertEqual(len(d['failed_attempts']),1)
        self.assertIn('unfinished',d['failed_attempts'][0]['text'])
        self.assertIn('Asked for help',str(minimize_state(t['memory'])))

    def test_step_budget_is_followed_by_a_real_final_call(self):
        from test_queue_selection import QueueMedia
        t=ActiveAgent(Reader(),OfflineTraitScorer(),QueueMedia(),dict(CFG,max_steps=1)).run(ROW)
        self.assertEqual(t['stop_reason'],'max_steps')
        self.assertEqual(len(t['steps']),1)
        self.assertEqual(t['decisions'][-1]['kind'],'final')
        self.assertEqual(t['final_status'],'generated')
        self.assertEqual(set(t['final']),set(TRAITS))

    def test_short_malformed_final_uses_one_bounded_protocol_correction(self):
        class Retry(Reader):
            def __init__(self):super().__init__();self.calls=0
            def generate(self,messages,kind,step):
                if kind!='final':return super().generate(messages,kind,step)
                self.calls+=1
                obj={trait:'medium' for trait in TRAITS}
                obj['reasoning']='A complete model explanation.';obj['contextual']=[]
                raw=json.dumps(obj)
                d=dict(messages=messages,kind=kind,step=step,text=raw,tokens=[1],truncated=False,offline=True)
                if self.calls==1:
                    d['text']+='\nExtra explanation outside JSON'
                    raise GenerationFormatError('Extra data',d)
                return obj,d
        reader=Retry()
        t=ActiveAgent(reader,OfflineTraitScorer(),SyntheticMedia(),dict(CFG,max_steps=1,max_validation_attempts=2)).run(ROW)
        self.assertEqual(reader.calls,2)
        self.assertEqual(t['final_status'],'generated')
        self.assertEqual(t['final_explanation'],'A complete model explanation.')


    def test_final_failure_still_returns_memory_and_explicit_unavailable_status(self):
        class Invalid(Reader):
            def generate(self,messages,kind,step):
                if kind!='final':return super().generate(messages,kind,step)
                d=dict(messages=messages,kind=kind,step=step,text='{"O":"high",',tokens=[1],truncated=True,offline=True)
                raise GenerationFormatError('Incomplete final',d)
        t=ActiveAgent(Invalid(),OfflineTraitScorer(),SyntheticMedia(),dict(CFG,max_steps=1,max_validation_attempts=2)).run(ROW)
        self.assertEqual(t['final_status'],'unavailable')
        self.assertFalse(t['final_prediction_available'])
        self.assertEqual(t['final_abstentions'],{})
        self.assertEqual(t['run_status'],'partial')
        self.assertEqual(len(t['memory']['observations']),1)
        self.assertEqual(len(t['decisions'][-1]['failed_attempts']),1)
        self.assertTrue(all(v['level'] is None for v in t['final'].values()))

if __name__=='__main__':unittest.main()
