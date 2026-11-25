"""Invalid Select correction remains a model decision within current pending IDs."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from amp.agent import ActiveAgent
from amp.backend import QwenBackend
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia
from test_simple_memory import Reader as BaseReader
class QueuedReader(BaseReader):
    def __init__(self,actions):super().__init__();self.queue=list(actions)
    def generate(self,messages,kind,step):
        if kind!='action':return super().generate(messages,kind,step)
        obj=self.queue.pop(0);text=json.dumps(obj)
        return obj,dict(messages=messages,kind=kind,step=step,text=text,tokens=list(map(ord,text)),offline=True)
from test_simple_memory import ROW,CFG


class GuidedSelectTests(unittest.TestCase):
    def test_existing_retry_uses_model_choice_over_pending_ids(self):
        class Reader(QueuedReader):
            def __init__(self):
                super().__init__([dict(candidate_id='C1'),dict(candidate_id='C1')]);self.guided=[]
            def generate_choice(self,messages,candidate_ids,step):
                self.guided.append(candidate_ids)
                obj=dict(candidate_id=candidate_ids[-1]);text=json.dumps(obj)
                return obj,dict(messages=messages,kind='action',step=step,text=text,tokens=list(map(ord,text)),offline=True,choice_texts=[json.dumps(dict(candidate_id=c)) for c in candidate_ids])
        reader=Reader()
        from test_queue_selection import QueueMedia
        trajectory=ActiveAgent(reader,OfflineTraitScorer(),QueueMedia(),dict(CFG,max_steps=5,max_validation_attempts=2)).run(ROW)
        self.assertEqual(reader.guided,[['C2']])
        self.assertEqual([s['candidate_id'] for s in trajectory['steps']],['C1','C2'])
        self.assertEqual(trajectory['stop_reason'],'candidate_pool_exhausted')
        self.assertEqual(len(trajectory['memory']['observations']),2)

    def test_native_generation_and_token_probabilities_share_the_constraint(self):
        class Tokenizer:
            eos_token_id=2
            def encode(self,text,add_special_tokens=False):return list(map(ord,text))
            def decode(self,tokens,skip_special_tokens=True):return ''.join(chr(t) for t in tokens if t!=2)
        tokenizer=Tokenizer()
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(128))
            def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.bias.expand(1,input_ids.shape[1],128))
            def generate(self,input_ids,prefix_allowed_tokens_fn,**kwargs):
                chosen=tokenizer.encode('{"candidate_id":"C9"}')+[2]
                for i,token in enumerate(chosen):
                    allowed=prefix_allowed_tokens_fn(0,torch.tensor(input_ids[0].tolist()+chosen[:i]))
                    self.assert_valid(token,allowed)
                return torch.tensor([input_ids[0].tolist()+chosen])
            def assert_valid(self,token,allowed):
                if token not in allowed:raise AssertionError('Decoder excluded the model-selected legal choice')
        backend=QwenBackend.__new__(QwenBackend)
        backend.torch=torch;backend.max_tokens=100;backend.temperature=0.;backend.generation_log_dir=None
        backend.processor=SimpleNamespace(tokenizer=tokenizer);backend.policy=Policy()
        backend.prepare=lambda *args:dict(input_ids=torch.tensor([[1,1]]),attention_mask=torch.ones(1,2,dtype=torch.long))
        obj,decision=backend.generate_choice([dict(role='user',content='choose')],['C2','C9'],1)
        self.assertEqual(obj,dict(candidate_id='C9'))
        self.assertEqual(len(decision['choice_texts']),2)
        logp,entropy=backend.token_stats(decision)
        self.assertTrue(torch.isfinite(logp).all());self.assertTrue(torch.isfinite(entropy).all())
        self.assertAlmostEqual(float(logp.sum()),-float(torch.log(torch.tensor(2.))),places=5)
        (-logp.mean()).backward()
        self.assertTrue(torch.isfinite(backend.policy.bias.grad).all())
        self.assertGreater(float(entropy.max()),0.)

if __name__=='__main__':unittest.main()
