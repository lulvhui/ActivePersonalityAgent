import copy
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from baselines.omniagent_native.expanded import select_pool,summarize
from amp.common import write_json,write_jsonl
from amp.common import read_json,read_jsonl
from baselines.omniagent_native import native_batches
from baselines.omniagent_native import run_ours


class ExpandedComparisonTests(unittest.TestCase):
    def test_selection_uses_only_dev_and_excludes_people_not_just_answers(self):
        rows=[dict(id=f'{ds}:{person}:{answer}',dataset=ds,person_id=str(person),split=split,labels={'O':person})
              for ds in ('IELTS','RecruitView') for split,person in [('dev',1),('dev',2),('dev',3),('train',4),('test',5)] for answer in (0,1)]
        excluded={('IELTS','1'),('RecruitView','1')}
        selected=select_pool(rows,excluded,2)
        self.assertEqual(len(selected),4)
        self.assertEqual(len({(r['dataset'],r['person_id']) for r in selected}),4)
        self.assertTrue(all(r['split']=='dev' and r['person_id']!='1' for r in selected))
        changed=copy.deepcopy(rows)
        for row in changed:row['labels']={'O':999}
        self.assertEqual([r['id'] for r in selected],[r['id'] for r in select_pool(changed,excluded,2)])

    def test_failed_partial_final_cannot_increase_prediction_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write_json(p/'data/offline_references.json',{'s':dict(dataset='IELTS',labels={t:0 for t in 'OCEAN'})})
            write_json(p/'data/protocol.json',{})
            write_jsonl(p/'data/manifest.jsonl',[dict(id='s',segments=[])])
            write_json(p/'current_7b/results.json',dict(records=[dict(case_id='s',trajectory=dict(final={'O':dict(level='high')},memory={},
                decisions=[dict(kind='memory',step=0,rejected_candidates=[dict(error='invalid child range')])]),
                metrics=dict(complete=False,attempted_observations=1,generated_tokens=5,input_tokens=20),error='invalid final')]))
            report=summarize(p)
            self.assertEqual(report['summaries']['current_7b']['valid_trait_coverage'],0)
            self.assertEqual(report['summaries']['current_7b']['complete'],0)
            self.assertEqual(report['cases'][0]['groups']['current_7b']['raw_final']['O']['level'],'high')
            self.assertEqual(report['cases'][0]['groups']['current_7b']['validation_errors'][0]['rejected_candidates'][0]['error'],'invalid child range')

    def test_native_restart_never_resubmits_finished_or_failed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_jsonl(root/'data/manifest.jsonl',[dict(id=x) for x in ('a','b','c')])
            write_json(root/'omniagent_native/results.json',[dict(sample_id='a',status='answered')])
            write_json(root/'omniagent_native/config.json',{})
            submitted=[]
            def execute(command,**kwargs):
                inputs=Path(command[command.index('--manifest')+1])
                folder=Path(command[command.index('--output')+1])
                pending=list(read_jsonl(inputs));submitted.append([r['id'] for r in pending])
                # First fresh engine fails one task; next engine gets only the remainder.
                sample=pending[0]['id']
                write_json(folder/'results.json',[dict(sample_id=sample,status='failed' if sample=='b' else 'answered')])
                return type('Result',(),dict(returncode=0))()
            with patch('sys.argv',['native_batches','--run-dir',str(root)]),patch.object(native_batches.subprocess,'run',side_effect=execute):
                native_batches.main()
            self.assertEqual(submitted,[['b','c'],['c']])
            self.assertEqual([r['sample_id'] for r in read_json(root/'omniagent_native/results.json')],['a','b','c'])

    def test_identical_finished_group_does_not_load_a_second_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.jsonl';output=root/'out'
            write_jsonl(manifest,[dict(id='frozen-case')])
            cfg=dict(read_json(run_ours.ROOT/'configs/local_omni3b.json'),model='LOCAL_TEST_MODEL')
            identity=dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run_ours.ROOT/'amp').glob('*.py'))})
            saved=dict(config=cfg,input_identity=identity,records=[dict(case_id='frozen-case')])
            write_json(output/'results.json',saved)
            argv=['run_ours','--manifest',str(manifest),'--output',str(output),'--model','LOCAL_TEST_MODEL']
            with patch('sys.argv',argv),patch.object(run_ours,'QwenBackend') as backend:
                run_ours.main();backend.assert_not_called()
            saved['input_identity']['manifest_sha256']='different-input'
            write_json(output/'results.json',saved)
            with patch('sys.argv',argv),patch.object(run_ours,'QwenBackend') as backend:
                with self.assertRaisesRegex(ValueError,'different inputs'):
                    run_ours.main()
                backend.assert_not_called()


if __name__=='__main__':unittest.main()
