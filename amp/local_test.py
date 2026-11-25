"""Small local-model integration launcher; no training, hosted API or mock actor."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .common import TRAITS, read_json, read_jsonl, write_json
from .state import validate_final,memory_records,contextual_final
from .perception import refinement_count

ROOT = Path(__file__).resolve().parents[1]
GPU_PROBE = '''
import json, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this Python environment; check GPU access/driver/PyTorch")
print(json.dumps({"torch":torch.__version__, "cuda":torch.version.cuda,
    "devices":[{"logical_index":i,"name":torch.cuda.get_device_name(i),
                "total_gib":round(torch.cuda.get_device_properties(i).total_memory/1024**3,2)}
               for i in range(torch.cuda.device_count())]}))
'''


def gpu_ids(value):
    result = [s.strip() for s in value.split(',')]
    if not all(result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('GPU IDs must be nonempty and unique, e.g. 0,1 or 0')
    if any(not (s.isdigit() or s.startswith('GPU-') or s.startswith('MIG-')) for s in result):
        raise argparse.ArgumentTypeError('Use CUDA GPU indices or UUIDs')
    return result


def environment(gpus):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpus), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    env.setdefault('FORCE_QWENVL_VIDEO_READER','torchvision')
    return env


def preflight(cfg, gpus):
    report = dict(python=sys.executable, model=cfg['model'], visible_gpus=gpus,
                  external_api=cfg.get('scorer')=='jev' and cfg.get('allow_external',False), errors=[])
    model = Path(cfg['model'])
    if not (model/'config.json').is_file():
        report['errors'].append(f'Local model config not found: {model}')
    elif (model/'model.safetensors.index.json').is_file():
        shards = set(read_json(model/'model.safetensors.index.json')['weight_map'].values())
        absent = [name for name in sorted(shards) if not (model/name).is_file()]
        if absent:
            report['errors'].append(f'Missing local weight shards: {absent}')
    elif not any(model.glob('*.safetensors')) and not (model/'pytorch_model.bin').is_file():
        report['errors'].append(f'No local weights found in {model}')
    if not Path(cfg['data_root']).is_dir():
        report['errors'].append(f'Data root not found: {cfg["data_root"]}')
    report['missing_modules'] = [name for name in ['torch','transformers','accelerate','qwen_omni_utils',
                                                   'numpy','soundfile','PIL','torchvision','av']
                                 if importlib.util.find_spec(name) is None]
    if cfg.get('scorer')=='jev':
        if not cfg.get('allow_external'):report['errors'].append('Jev requires --allow-external after confirming data-use permission')
        if not os.environ.get('TYPESAFE_API_KEY'):report['errors'].append('Set TYPESAFE_API_KEY for the official Jev API')
        if importlib.util.find_spec('typesafe_sdk') is None:report['missing_modules'].append('typesafe_sdk')
    report['missing_tools'] = [name for name in ['ffmpeg','ffprobe'] if shutil.which(name) is None]
    if report['missing_modules']:
        report['errors'].append('Missing Python dependencies: '+', '.join(report['missing_modules']))
    if report['missing_tools']:
        report['errors'].append('Missing programs on PATH: '+', '.join(report['missing_tools']))
    if not report['missing_modules'] and not report['errors']:
        probe = subprocess.run([sys.executable,'-c',GPU_PROBE],env=environment(gpus),cwd=ROOT,
                               capture_output=True,text=True,timeout=90)
        if probe.returncode:
            report['errors'].append('Model imports/CUDA check failed: '+probe.stderr[-4000:])
        else:
            report['gpu'] = json.loads(probe.stdout.strip().splitlines()[-1])
            if len(report['gpu']['devices']) != len(gpus):
                report['errors'].append('Not all requested GPUs are visible; select available GPUs using --gpus')
    return report


def cli_command(config, *args):
    return [sys.executable,'-u','-m','amp.cli','--config',str(config),*map(str,args)]


def build_plan(args, output):
    config, data = output/'config.json', output/'data'
    jobs = []
    for dataset in args.datasets:
        scorer=getattr(args,'scorer',None) or 'offline'
        extra=['--allow-external'] if getattr(args,'allow_external',False) else []
        jobs.append(dict(dataset=dataset, output=output/dataset,
                         command=cli_command(config,'infer','--manifest',data/f'{args.split}.jsonl',
                             '--dataset',dataset,'--limit',args.limit,'--perception','full',
                             '--scorer',scorer,*extra,'--output',output/dataset)))
    return dict(prepare=cli_command(config,'prepare','--output',data),jobs=jobs)


def validate_result(directory, expected_ids, require_active=False):
    """A CLI exit code of zero alone is insufficient: infer may reject all samples."""
    path = Path(directory)/'trajectories.json'
    if not path.is_file():
        raise ValueError(f'Missing trajectory file: {path}')
    trajectories = read_json(path)
    actual = [t['sample_id'] for t in trajectories]
    if len(actual) != len(expected_ids) or set(actual) != set(expected_ids) or len(actual) != len(set(actual)):
        raise ValueError(f'Incomplete/duplicate predictions: expected {expected_ids}, got {actual}; inspect rejected.json')
    summaries = []
    for trajectory in trajectories:
        if trajectory.get('offline'):
            raise ValueError('Synthetic actor output is not a local Qwen integration result')
        kinds = [d['kind'] for d in trajectory['decisions']]
        mode=trajectory.get('config',{}).get('perception','full')
        policy=trajectory.get('selection_policy',trajectory.get('config',{}).get('selection_policy','model'))
        if (mode!='static' and 'candidate_scan' not in kinds) or (trajectory['steps'] and 'memory' not in kinds) or (trajectory['steps'] and policy=='model' and mode not in ('static','fixed_context') and 'action' not in kinds):
            raise ValueError('Missing Stage 1 candidate scan or Stage 2 observation/update')
        if 'final' not in kinds and set(trajectory.get('final_abstentions',{})) != set(TRAITS):
            raise ValueError('Missing Final generation or complete evidence abstention')
        recorded='observations' in trajectory['memory']
        final={t:dict(item,reasoning=item.get('reasoning') or trajectory.get('final_explanation')) for t,item in trajectory['final'].items()}
        validate_final(final,None if recorded else trajectory['memory'])
        if recorded:
            _,errors=contextual_final(dict(contextual=trajectory.get('contextual',[])),trajectory['memory'])
            if errors:raise ValueError(f'Invalid contextual references: {errors}')
        steps = [s for s in trajectory['steps'] if s.get('action_type')!='static']
        if require_active and not steps:
            raise ValueError('--require-active needs at least one active observation')
        summaries.append(dict(sample_id=trajectory['sample_id'], active_observations=len(steps),
            run_status=trajectory.get('run_status','complete'),
            candidate_failures=sum(d.get('candidate_status')=='rejected' for d in trajectory['decisions']),
            invalid_candidates=sum(len(d.get('rejected_candidates',[])) for d in trajectory['decisions']),
            active_observe_covered=bool(steps), modalities=[s['action']['modality'] for s in steps],
            global_scan=trajectory.get('global_scan',{}),
            saved_observation_records=len(trajectory['memory'].get('observations',[])),
            trait_scores=trajectory.get('score',{}).get('traits',{}),
            scorer_metadata={k:v for k,v in trajectory.get('score',{}).get('metadata',{}).items() if k!='raw_response'},
            semantic_evaluation='not_run', final_abstentions=trajectory.get('final_abstentions',{}), stop_reason=trajectory.get('stop_reason','legacy_unrecorded'),
            multi_round_covered=len(steps)>1,
            refinement_observations=refinement_count(steps),
            memory_failures=sum(d.get('memory_update_status')=='rejected' for d in trajectory['decisions']),
            initial_observation_records=len(memory_records(trajectory['initial']['memory'])),
            initial_candidate_count=trajectory.get('initial',{}).get('candidate_count',0),
            observation_trace=[dict(s['action'],new_ids=s.get('new_ids',[]),changed_ids=s.get('changed_ids',[]),
                memory_update_status=next((d.get('memory_update_status','legacy_unrecorded') for d in trajectory['decisions'] if d['kind']=='memory' and d.get('step')==s.get('index')),'missing')) for s in steps], evidence_status=trajectory.get('evidence_status','legacy_unrecorded'),
            predicted_traits=sum(trajectory['final'][t]['level'] is not None for t in TRAITS),
            cost=trajectory['cost'], final=trajectory['final']))
    return summaries


def run_jobs(jobs, gpus, output, require_active=False):
    """One independent model process per GPU, with bounded parallelism and owned-child cleanup."""
    pending, active, available, results = list(jobs), [], list(gpus), []
    try:
        while pending or active:
            while pending and available:
                job, gpu = pending.pop(0), available.pop(0)
                log_path = output/f'{job["dataset"]}.log'
                log = log_path.open('w')
                try:
                    process = subprocess.Popen(job['command'],cwd=ROOT,env=environment([gpu]),
                                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active.append((job,gpu,process,log,log_path))
                print(f'[{job["dataset"]}] GPU {gpu}, PID {process.pid}, log: {log_path}',flush=True)
            for item in active[:]:
                job,gpu,process,log,log_path = item
                if process.poll() is None:
                    continue
                log.close()
                result = dict(dataset=job['dataset'],gpu=gpu,exit_code=process.returncode,log=str(log_path))
                try:
                    if process.returncode:
                        raise ValueError(f'Inference process exited with {process.returncode}')
                    result['samples'] = validate_result(job['output'],job['expected_ids'],require_active)
                    result['status'] = 'partial' if any(s['run_status']=='partial' for s in result['samples']) else 'success'
                    result['validation_scope'] = 'toolchain_and_source_contract_only; trait validity not evaluated'
                except (ValueError,KeyError,TypeError) as error:
                    result.update(status='failed',error=str(error))
                results.append(result)
                external=any('--allow-external' in j['command'] and 'jev' in j['command'] for j in jobs)
                write_json(output/'summary.json',dict(mode='local_multimodal_inference',external_api=external,jobs=results))
                print(f'[{job["dataset"]}] {result["status"]}: {result.get("error", "trajectory validated")}',flush=True)
                active.remove(item)
                available.append(gpu)
            if active:
                time.sleep(.2)
    finally:
        # Stop only model processes created by this launcher, including FFmpeg descendants.
        for _,_,process,log,_ in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The child can exit between poll() and killpg().
        for _,_,process,log,_ in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            log.close()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='Test the full perception pipeline using local Qwen2.5-Omni-3B')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/local_omni3b.json')
    parser.add_argument('--model',type=Path,help='Override the local model folder')
    parser.add_argument('--scorer',choices=['offline','jev'],help='Trait probability scorer (default from configuration)')
    parser.add_argument('--allow-external',action='store_true',help='Confirm data-use permission for minimized Jev evidence summaries')
    parser.add_argument('--gpus',type=gpu_ids,default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1'))
    parser.add_argument('--datasets',nargs='+',choices=['IELTS','RecruitView'],default=['IELTS','RecruitView'])
    parser.add_argument('--split',choices=['train','dev','test'],default='dev')
    parser.add_argument('--limit',type=int,default=1,help='Samples per dataset, default 1')
    parser.add_argument('--max-steps',type=int,help='Override the active observation limit')
    parser.add_argument('--coarse-fps',type=float,help='Global video sampling FPS, covering the entire video')
    parser.add_argument('--coarse-chunk-seconds',type=float,help='Transport chunk length; all audio/video chunks are processed')
    parser.add_argument('--no-active-budget',action='store_true',help='Disable the cumulative active duration budget')
    parser.add_argument('--budget-seconds',type=float,help='Total active observation budget')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--media-cache',type=Path,help='Reuse an existing extraction cache when debugging a run')
    parser.add_argument('--require-active',action='store_true',help='Require at least one active Observe')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true',help='Check local dependencies/CUDA only; do not load weights')
    modes.add_argument('--dry-run',action='store_true',help='Print the plan without imports, GPU use or file writes')
    args=parser.parse_args(argv)
    if args.limit < 1 or len(args.datasets) != len(set(args.datasets)):
        parser.error('Use a positive --limit and unique datasets')
    cfg=read_json(args.config)
    args.scorer=args.scorer or cfg.get('scorer','offline')
    cfg.update(scorer=args.scorer,allow_external=args.allow_external)
    if args.model:
        cfg['model']=str(args.model.resolve())
    if args.max_steps is not None:
        cfg['max_steps']=args.max_steps
    for arg, key in [('coarse_fps','coarse_video_fps'),('coarse_chunk_seconds','coarse_chunk_seconds'),('budget_seconds','max_seconds')]:
        if getattr(args,arg) is not None:
            cfg[key]=getattr(args,arg)
    if args.no_active_budget:
        cfg['max_seconds']=None
    if cfg['coarse_video_fps'] <= 0 or not 0 < cfg['coarse_chunk_seconds'] <= 30 or (cfg['max_seconds'] is not None and cfg['max_seconds'] <= 0):
        parser.error('Use positive coarse FPS, a transport chunk length in (0,30], and positive active budget or --no-active-budget')
    if cfg['max_steps'] < 1:
        parser.error('Full-flow testing requires --max-steps >= 1')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output=(args.output or ROOT/'runs'/f'local_omni3b_{stamp}').resolve()
    cfg['model']=str(Path(cfg['model']).resolve())
    cfg['cache']=str(args.media_cache.resolve() if args.media_cache else ROOT/'cache/media')
    cfg['generation_log_dir']=str(output/'generations')
    plan=build_plan(args,output)
    if args.dry_run:
        print(json.dumps(dict(config=cfg,gpus=args.gpus,output=output,plan=plan),default=str,ensure_ascii=False,indent=2))
        return 0
    report=preflight(cfg,args.gpus)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if report['errors']:
        if args.scorer=='jev' and (not args.allow_external or not os.environ.get('TYPESAFE_API_KEY')):
            print('Jev startup needs TYPESAFE_API_KEY in the environment and --allow-external confirming data-use permission. No model or external request was started.',file=sys.stderr)
            return 2
        print("To create the project's CUDA environment and local FFmpeg, run:\n"
              f"  bash {ROOT/'scripts/setup_local_omni3b.sh'}\n"
              "Or use an existing CUDA environment with FFmpeg/ffprobe on PATH.\n"
              "You can select that interpreter with PYTHON_BIN=/path/to/python. No packages were installed automatically.",file=sys.stderr)
        return 2
    if args.check_only:
        print('Preflight passed. No model weights were loaded.')
        return 0
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg)
    write_json(output/'preflight.json',report)
    print(f'Run folder: {output}\nPreparing person-disjoint raw-data manifests...',flush=True)
    with (output/'prepare.log').open('w') as log:
        subprocess.run(plan['prepare'],cwd=ROOT,env=environment(args.gpus),stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=list(read_jsonl(output/'data'/f'{args.split}.jsonl'))
    for job in plan['jobs']:
        selected=[r for r in rows if r['dataset']==job['dataset']][:args.limit]
        job['expected_ids']=list(dict.fromkeys(f"RecruitView:person:{r['person_id']}" if r['dataset']=='RecruitView' and r.get('task_unit')!='person_interview' else r['id'] for r in selected))
        if not job['expected_ids']:
            raise ValueError(f'No samples for {job["dataset"]}')
    results=run_jobs(plan['jobs'],args.gpus,output,args.require_active)
    print(f'Results: {output/"summary.json"}',flush=True)
    return 0 if all(r['status']=='success' for r in results) else 1


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; stopped the child processes started by this launcher.',file=sys.stderr)
        sys.exit(130)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        print(f'Local test failed: {error}',file=sys.stderr)
        sys.exit(1)
