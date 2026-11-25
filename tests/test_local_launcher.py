import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from amp.common import TRAITS, read_json, write_json
from amp.local_test import ROOT, build_plan, environment, gpu_ids, main, preflight, run_jobs, validate_result
from amp.state import empty_memory


def trajectory(sample='IELTS:4'):
    return dict(sample_id=sample,offline=False,initial=dict(memory=empty_memory()),memory=empty_memory(),
                decisions=[dict(kind='candidate_scan'),dict(kind='memory'),dict(kind='action'),dict(kind='final')],steps=[],
                cost=dict(active_seconds=0),final={t:dict(level=None,reasoning='Evidence limited.',citations=[]) for t in TRAITS})


class LocalLauncherTests(unittest.TestCase):
    def test_plan_uses_real_actor_local_profile_and_no_external_api(self):
        args=argparse.Namespace(datasets=['IELTS','RecruitView'],split='dev',limit=1)
        plan=build_plan(args,Path('/tmp/launcher plan'))
        self.assertEqual([j['dataset'] for j in plan['jobs']],args.datasets)
        for job in plan['jobs']:
            cmd=job['command']
            self.assertIn('infer',cmd)
            self.assertEqual(cmd[cmd.index('--perception')+1],'full')
            self.assertEqual(cmd[cmd.index('--scorer')+1],'offline')
            self.assertNotIn('--allow-external',cmd)
            self.assertNotIn('smoke',cmd)
        env=environment(['1'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')

    def test_rejected_or_duplicate_predictions_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):validate_result(tmp,['IELTS:4'])
            write_json(Path(tmp)/'trajectories.json',[])
            with self.assertRaises(ValueError):validate_result(tmp,['IELTS:4'])
            write_json(Path(tmp)/'trajectories.json',[trajectory(),trajectory()])
            with self.assertRaises(ValueError):validate_result(tmp,['IELTS:4','IELTS:10'])
            fake=trajectory();fake['offline']=True
            write_json(Path(tmp)/'trajectories.json',[fake])
            with self.assertRaises(ValueError):validate_result(tmp,['IELTS:4'])

    def test_stop_is_reported_and_active_coverage_can_be_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp)/'trajectories.json',[trajectory()])
            summary=validate_result(tmp,['IELTS:4'])
            self.assertFalse(summary[0]['active_observe_covered'])
            with self.assertRaisesRegex(ValueError,'active'):validate_result(tmp,['IELTS:4'],True)
            broken=trajectory();broken['decisions']=[]
            write_json(Path(tmp)/'trajectories.json',[broken])
            with self.assertRaises(ValueError):validate_result(tmp,['IELTS:4'])

    def test_current_queue_and_record_memory_are_accepted_without_action_tokens(self):
        from amp.agent import ActiveAgent
        from amp.scoring import OfflineTraitScorer
        from amp.smoke import SyntheticMedia
        from test_simple_memory import Reader,CFG,ROW
        t=ActiveAgent(Reader(),OfflineTraitScorer(),SyntheticMedia(),dict(CFG,selection_policy='queue',max_steps=1)).run(ROW)
        t['offline']=False  # Fixture for the validator only, not a real inference result.
        t['final_explanation']='Synthetic shared explanation.'
        for item in t['final'].values():item['reasoning']=None
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp)/'trajectories.json',[t])
            result=validate_result(tmp,[ROW['id']],True)
            self.assertEqual(result[0]['saved_observation_records'],1)
            t['contextual']=[dict(trait='E',context_type='reported',context='Project',expression='A test statement.',citations=['M999'])]
            write_json(Path(tmp)/'trajectories.json',[t])
            with self.assertRaises(ValueError):validate_result(tmp,[ROW['id']])

    def test_recruitview_expected_ids_match_person_level_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            def prepare(*args,**kwargs):
                path=output/'data/dev.jsonl';path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(dict(id=f'RecruitView:{i}',dataset='RecruitView',person_id='12',task_unit='answer'))+'\n' for i in (1,2)))
                return subprocess.CompletedProcess([],0)
            with patch('amp.local_test.preflight',return_value={'errors':[]}),patch('amp.local_test.subprocess.run',side_effect=prepare),patch('amp.local_test.run_jobs',return_value=[dict(status='success')]) as jobs,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--datasets','RecruitView','--limit','2','--output',str(output)]),0)
            self.assertEqual(jobs.call_args.args[0][0]['expected_ids'],['RecruitView:person:12'])

    def test_gpu_ids_are_unique_and_preflight_requires_no_jev(self):
        self.assertEqual(gpu_ids('0, 1'),['0','1'])
        for bad in ['','0,','0,0','--anything']:
            with self.assertRaises(argparse.ArgumentTypeError):gpu_ids(bad)
        cfg=json.loads((ROOT/'configs/local_omni3b.json').read_text())
        with patch('amp.local_test.importlib.util.find_spec',return_value=None),patch('amp.local_test.shutil.which',return_value=None):
            result=preflight(cfg,['0','1'])
        self.assertNotIn('typesafe_sdk',result['missing_modules'])
        self.assertNotIn('peft',result['missing_modules'])
        self.assertIn('torch',result['missing_modules'])
        self.assertTrue(result['errors'])

    def test_shell_dry_run_works_from_other_directory_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'no files yet'
            result=subprocess.run(['bash',str(ROOT/'scripts/test_local_omni3b.sh'),'--dry-run','--output',str(out)],
                                  cwd=tmp,text=True,capture_output=True,env=dict(os.environ,PYTHON_BIN=sys.executable))
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads(result.stdout)
            self.assertEqual(plan['config']['model'],str(Path(read_json(ROOT/'configs/local_omni3b.json')['model']).resolve()))
            self.assertFalse(out.exists())
            self.assertEqual(plan['config']['max_steps'],16)

    def test_scheduler_pins_separate_processes_to_each_gpu(self):
        # Real subprocesses, fake inference bodies: no model/GPU/media is used in this scheduler test.
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            jobs=[]
            for dataset in ['IELTS','RecruitView']:
                directory=output/dataset
                row=trajectory(dataset+':fixture')
                body='import os,json,pathlib; p=pathlib.Path(__import__("sys").argv[1]); p.mkdir(); (p/"gpu.txt").write_text(os.environ["CUDA_VISIBLE_DEVICES"]); (p/"trajectories.json").write_text(__import__("sys").argv[2])'
                jobs.append(dict(dataset=dataset,output=directory,expected_ids=[row['sample_id']],
                                 command=[sys.executable,'-c',body,str(directory),json.dumps([row])]))
            with contextlib.redirect_stdout(io.StringIO()):
                results=run_jobs(jobs,['0','1'],output)
            self.assertEqual([r['status'] for r in results],['success','success'])
            self.assertEqual((output/'IELTS/gpu.txt').read_text(),'0')
            self.assertEqual((output/'RecruitView/gpu.txt').read_text(),'1')
            self.assertTrue((output/'summary.json').is_file())

    def test_zero_exit_with_empty_trajectory_is_worker_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);directory=output/'IELTS';directory.mkdir()
            write_json(directory/'trajectories.json',[])
            job=dict(dataset='IELTS',output=directory,expected_ids=['IELTS:4'],command=[sys.executable,'-c','pass'])
            with contextlib.redirect_stdout(io.StringIO()):results=run_jobs([job],['0'],output)
            self.assertEqual(results[0]['exit_code'],0)
            self.assertEqual(results[0]['status'],'failed')

    def test_explicit_interpreter_bin_is_added_to_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            bindir=Path(tmp)/'chosen env/bin'
            bindir.mkdir(parents=True)
            interpreter=bindir/'python'
            interpreter.write_text('#!/bin/sh\nprintf "%s" "$PATH"\n')
            interpreter.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'scripts/test_local_omni3b.sh'),'--dry-run'],
                                  cwd=tmp,text=True,capture_output=True,
                                  env=dict(os.environ,PYTHON_BIN=str(interpreter)))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(str(bindir),result.stdout.split(os.pathsep))

    def test_preflight_failure_points_to_setup_instead_of_missing_system_pip(self):
        with patch('amp.local_test.preflight',return_value={'errors':['missing fixture dependencies']}), \
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()) as stderr:
            code=main(['--check-only','--gpus','0'])
        self.assertEqual(code,2)
        self.assertIn('setup_local_omni3b.sh',stderr.getvalue())


if __name__=='__main__':unittest.main()
