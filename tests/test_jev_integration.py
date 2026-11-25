"""Official SDK integration through local mock transport; never sends personal data."""
import argparse
import copy
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx2
from typesafe_sdk import TypeSafeClient,RetryPolicy
from amp.common import TRAITS,read_json
from amp.scoring import JevTraitScorer,prediction_feedback
from amp.state import empty_memory
from amp.agent import ActiveAgent
from amp.backend import OfflineBackend
from amp.smoke import SyntheticMedia
from amp.local_test import build_plan

CFG=read_json(Path(__file__).resolve().parents[1]/'configs/local_omni3b.json')
P=[.01,.02,.07,.76,.14]

def raw_score():
    return dict(model='jev-1.14.0',answers={t:dict(type='score',score=3.,confidence=.8,
      probabilities={str(i):p for i,p in enumerate(P)},legend={str(i):str(i) for i in range(5)}) for t in TRAITS},usage=dict(input_tokens=100,output_tokens=20))

def memory():
    return dict(observations=[dict(id='M1',modality='audio',content=json.dumps(dict(observations=['Initiates a conversation.'],candidates=[])))])

class JevIntegrationTests(unittest.TestCase):
    def client(self,handler):
        return TypeSafeClient(api_key='local-test-key',transport=httpx2.MockTransport(handler),retry=RetryPolicy(max_retries=0))

    def test_empty_memory_is_not_a_live_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=JevTraitScorer(tmp).score(empty_memory())
            self.assertEqual(s['metadata']['api_calls'],0)
            self.assertIsNone(prediction_feedback(s))
            self.assertTrue(all(v['level'] is None and not v['predictive'] for v in s['traits'].values()))

    def test_official_sdk_request_probability_mapping_cache_and_unknown_mask(self):
        requests=[]
        def handler(request):
            self.assertEqual(request.url.path,'/v1/systemone')
            payload=json.loads(request.content);requests.append(payload)
            self.assertEqual(payload['model'],'jev-1.14.0')
            self.assertEqual(set(payload['questions']),set(TRAITS))
            self.assertTrue(all(q['type']=='score' and len(q['criteria'])==5 for q in payload['questions'].values()))
            self.assertNotIn('level',str(payload['state']))
            return httpx2.Response(200,json=raw_score())
        with tempfile.TemporaryDirectory() as tmp,self.client(handler) as client:
            scorer=JevTraitScorer(tmp,allow_external=True,client=client,input_price_per_million=1,output_price_per_million=2)
            first=scorer.score(memory());second=scorer.score(memory())
            self.assertEqual(len(requests),1)
            self.assertEqual(first['traits']['E']['probabilities'],P)
            self.assertEqual(first['traits']['E']['level'],'high')
            self.assertTrue(0<first['traits']['E']['uncertainty']<1)
            self.assertEqual(first['traits']['O']['level'],'high')
            self.assertEqual(prediction_feedback(first)['O']['probabilities'],P)
            self.assertEqual(second['metadata']['api_calls'],0)
            self.assertEqual(second['metadata']['billed_cost'],0)
            self.assertAlmostEqual(first['metadata']['billed_cost'],.00014)
            self.assertEqual(len(list(Path(tmp).glob('*.raw.json'))),1)

    def test_official_status_retry_and_auth_failure_are_not_uniform_fallbacks(self):
        calls=[]
        def handler(request):
            calls.append(request)
            return httpx2.Response(429,json={'error':'rate limited'}) if len(calls)==1 else httpx2.Response(200,json=raw_score())
        with tempfile.TemporaryDirectory() as tmp,self.client(handler) as client,patch('amp.scoring.time.sleep'):
            s=JevTraitScorer(tmp,allow_external=True,client=client).score(memory())
            self.assertEqual(s['metadata']['attempts'],2)
            self.assertEqual(len(calls),2)
        calls=[]
        def denied(request):
            calls.append(request);return httpx2.Response(401,json={'error':'private-detail-must-not-be-logged'})
        with tempfile.TemporaryDirectory() as tmp,self.client(denied) as client:
            with self.assertRaises(RuntimeError) as error:JevTraitScorer(tmp,allow_external=True,client=client).score(memory())
            self.assertNotIn('private-detail',str(error.exception))
            self.assertEqual(len(calls),1)
            self.assertNotIn('private-detail',''.join(p.read_text() for p in Path(tmp).glob('*.error.json')))

    def test_bad_probabilities_preserve_raw_response_without_valid_cache(self):
        raw=raw_score();del raw['answers']['E']['probabilities']['4']
        class Response:
            def model_dump(self,mode):return raw
        class Client:
            def system_one(self,**kwargs):return Response()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Invalid Jev'):
                JevTraitScorer(tmp,allow_external=True,client=Client()).score(memory())
            self.assertEqual(len(list(Path(tmp).glob('*.raw.json'))),1)
            self.assertEqual(len([p for p in Path(tmp).glob('*.json') if not p.name.endswith(('.raw.json','.error.json'))]),0)

    def test_concurrent_workers_share_one_cached_request(self):
        calls=[]
        def handler(request):
            calls.append(request);time.sleep(.05)
            return httpx2.Response(200,json=raw_score())
        with tempfile.TemporaryDirectory() as tmp,self.client(handler) as client,ThreadPoolExecutor(2) as workers:
            scorers=[JevTraitScorer(tmp,allow_external=True,client=client) for _ in range(2)]
            results=list(workers.map(lambda s:s.score(memory()),scorers))
            self.assertEqual(len(calls),1)
            self.assertEqual(sorted(s['metadata']['cache_hit'] for s in results),[False,True])



    def test_launcher_propagates_explicit_jev_choice(self):
        args=argparse.Namespace(datasets=['IELTS'],split='dev',limit=1,scorer='jev',allow_external=True)
        command=build_plan(args,Path('/tmp/jev-plan'))['jobs'][0]['command']
        self.assertEqual(command[command.index('--scorer')+1],'jev')
        self.assertIn('--allow-external',command)

if __name__=='__main__':unittest.main()
