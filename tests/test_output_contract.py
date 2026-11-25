"""Reject the recurring bad shapes during decoding, not after generation."""
import json
from types import SimpleNamespace
import unittest
import torch
from lmformatenforcer import TokenEnforcerTokenizerData
from amp.backend import QwenBackend
from amp.common import TRAITS
from amp.perception import generation_schema


class OutputContractTests(unittest.TestCase):
    def backend(self):
        chars=list(dict.fromkeys([chr(i) for i in range(9,127)]+list('独立线索中文')))+['.","',' .",]']
        vocabulary={ch:i+3 for i,ch in enumerate(chars)};reverse={v:k for k,v in vocabulary.items()}
        encode=lambda text,**kw:[vocabulary[ch] for ch in text]
        decode=lambda ids,**kw:''.join(reverse.get(int(i),'') for i in ids)
        b=QwenBackend.__new__(QwenBackend);b.torch=torch;b.structured_output=True
        b.max_tokens=1000;b.temperature=0.;b.generation_log_dir=None
        b.processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=encode,decode=decode,eos_token_id=2))
        b._format_tokenizer_data=TokenEnforcerTokenizerData([(i,ch,False) for ch,i in vocabulary.items()],decode,2,False,len(vocabulary)+3)
        return b

    def accepts(self,schema,text):
        b=self.backend();constraint=b.format_constraint(schema);prefix=[1]
        for token in b.processor.tokenizer.encode(text):
            if token not in constraint(0,torch.tensor(prefix)):return False
            prefix.append(token)
        return 2 in constraint(0,torch.tensor(prefix))

    def test_wrong_observation_field_is_not_an_allowed_generation(self):
        schema=generation_schema('memory',modalities=['text','audio','video','audio_video'])
        self.assertTrue(self.accepts(schema,'{"observations":["独立线索"],"candidates":[]}'))
        self.assertFalse(self.accepts(schema,'{"findings":["a fact"],"candidates":[]}'))
        self.assertFalse(self.accepts(schema,'{"observations":"a fact","candidates":[]}'))

    def test_number_pairs_and_modality_enum_cannot_drift_into_wrong_types(self):
        schema=generation_schema('memory',modalities=['audio','video'])
        item=dict(local_time=[1,3],modality='audio',hint='A clue',goal='Clarify')
        self.assertTrue(self.accepts(schema,json.dumps(dict(observations=['A fact'],candidates=[item]))))
        for change in [dict(local_time=[1]),dict(local_time=[1,2,3]),dict(local_time=['1','3']),dict(modality='speech')]:
            self.assertFalse(self.accepts(schema,json.dumps(dict(observations=['A fact'],candidates=[dict(item,**change)]))))

    def test_real_null_and_only_existing_memory_citations(self):
        schema=generation_schema('final',memory=dict(observations=[dict(id='M1')]))
        obj={t:None for t in TRAITS};obj.update(reasoning='Insufficient evidence.',contextual=[])
        self.assertTrue(self.accepts(schema,json.dumps(obj)))
        self.assertFalse(self.accepts(schema,json.dumps(dict(obj,O='null'))))
        item=dict(trait='E',context_type='reported',context='Friends',expression='A supported difference',citations=['M2'])
        self.assertFalse(self.accepts(schema,json.dumps(dict(obj,contextual=[item]))))

    def test_default_twenty_item_cap_is_actually_disabled(self):
        obj=dict(observations=['独立线索']*25,candidates=[])
        self.assertTrue(self.accepts(generation_schema('memory'),json.dumps(obj,ensure_ascii=False)))

    def test_valid_json_field_order_is_not_a_generation_requirement(self):
        schema=generation_schema('memory')
        self.assertTrue(self.accepts(schema,'{"candidates":[],"observations":["A fact"]}'))

    def test_action_validation_does_not_silently_cut_the_goal(self):
        from amp.state import CandidatePool
        goal='Inspect the actual response. '*30
        self.assertEqual(CandidatePool(30,30).validate(dict(time=[1,2],modality='audio',goal=goal))['goal'],goal)

    def test_qwen_merged_token_can_close_a_string_and_start_the_next_key(self):
        b=self.backend();constraint=b.format_constraint(generation_schema('candidate_scan'))
        prefix=[1]
        for token in b.processor.tokenizer.encode('{"summary":"A fact'):
            self.assertIn(token,constraint(0,torch.tensor(prefix)))
            prefix.append(token)
        merged=next(token for token,value,_ in b._format_tokenizer_data.regular_tokens if value=='.","')
        self.assertIn(merged,constraint(0,torch.tensor(prefix)))
        invalid=next(token for token,value,_ in b._format_tokenizer_data.regular_tokens if value==' .",]')
        self.assertNotIn(invalid,constraint(0,torch.tensor(prefix)))

    def test_policy_sees_the_schema_before_generating(self):
        b=self.backend();schema=generation_schema('memory',modalities=['audio'])
        def prepare(messages,tools=None):
            self.assertIn('Output JSON schema:',messages[0]['content'])
            self.assertIn(json.dumps(schema,ensure_ascii=False,separators=(',',':')),messages[0]['content'])
            return dict(input_ids=torch.tensor([[1]]))
        b.prepare=prepare
        b.policy=SimpleNamespace(generate=lambda **kwargs:torch.tensor([[1,2]]))
        _,decision=b.generate([dict(role='system',content='Describe findings.',output_schema=schema)],'memory',0,parse=False)
        self.assertIn('Output JSON schema:',decision['messages'][0]['content'])

    def test_native_generation_and_token_loss_use_the_same_schema_mask(self):
        b=self.backend();text='{"observations":["A fact"],"candidates":[]}'
        tokens=b.processor.tokenizer.encode(text)+[2];vocab=b._format_tokenizer_data.vocab_size
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(vocab))
            def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.bias.expand(1,input_ids.shape[1],vocab))
            def generate(self,input_ids,prefix_allowed_tokens_fn,**kwargs):
                prefix=input_ids[0].tolist()
                for token in tokens:
                    if token not in prefix_allowed_tokens_fn(0,torch.tensor(prefix)):raise AssertionError('Illegal generated token')
                    prefix.append(token)
                return torch.tensor([prefix])
        b.policy=Policy();b.prepare=lambda *args:dict(input_ids=torch.tensor([[1]]))
        messages=[dict(role='system',content='Read the source.',output_schema=generation_schema('memory'))]
        obj,d=b.generate(messages,'memory',0)
        self.assertEqual(d['format_mode'],'json_schema')
        self.assertEqual(obj['observations'],['A fact'])
        logp,entropy=b.token_stats(d)
        self.assertTrue(torch.isfinite(logp).all());self.assertTrue(torch.isfinite(entropy).all())
        (-logp.mean()).backward()
        self.assertTrue(torch.isfinite(b.policy.bias.grad).all())

if __name__=='__main__':unittest.main()
