"""Use the unmodified official ModelManager, OTA engine and video environment."""
import argparse
import json
import inspect
import os
import subprocess
import sys
import time
import tempfile
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import read_jsonl,write_json,read_json
from baselines.omniagent_native.task import parse_levels


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--upstream',type=Path,default=ROOT.parent/'OmniAgent')
    parser.add_argument('--model',default=str(ROOT.parent/'models/OmniAgent-RL-7B'))
    parser.add_argument('--manifest',type=Path,default=ROOT/'runs/omniagent_native/data/manifest.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/omniagent_native/native')
    parser.add_argument('--max-model-len',type=int,default=65536)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stop-on-model-error',action='store_true',help='Save the failed case and end this process before reusing a damaged model engine')
    parser.add_argument('--audio-attention',choices=('official','windowed_sdpa'),default='official')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    os.environ['PYTHONPATH']=os.pathsep.join([str(args.upstream),str(args.upstream/'demo'),str(ROOT),os.environ.get('PYTHONPATH','')])
    # Empty values survive python-dotenv's default override=False in Ray workers.
    for key in ('DASHSCOPE_API_KEY','OSS_ACCESS_KEY'):os.environ[key]=''
    os.environ.update(USE_OSS_IN_VIDEOENV='False',REASONING_REWARD='False',FORMAT_REWARD='False',
        VIDEO_ENV_TMP_DIR=str(args.output/'media'),VLLM_WORKER_MULTIPROC_METHOD='spawn')
    sys.path.insert(0,str(args.upstream/'demo'));sys.path.insert(0,str(args.upstream))
    from omniagent_demo_pro import DemoConfig,ModelManager,InferenceEngine,generate_history_json
    if args.audio_attention=='windowed_sdpa':
        from functools import partial
        import omniagent_demo_pro as native_module
        native_module.LLM=partial(native_module.LLM,worker_cls='baselines.omniagent_native.audio_worker.WindowedAudioWorker')
    from agent_system.environments.env_package.video_env import SingleVideoQAEnv
    import ray
    config=DemoConfig(model_path=args.model,tensor_parallel_size=2,gpu_memory_utilization=.90,
        max_model_len=args.max_model_len,max_prompt_length=args.max_model_len-4096,max_num_batched_tokens=args.max_model_len)
    # All OTA/action/memory settings and generation defaults come from DemoConfig.
    meta=dict(upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        upstream_diff=subprocess.check_output(['git','-C',str(args.upstream),'diff','--stat'],text=True),
        config=vars(config),scope='Official OTA framework; only task question changed. TP/context/memory allocation adapted to two RTX 3090s.',
        implementation=dict(engine=inspect.getfile(InferenceEngine),model_manager=inspect.getfile(ModelManager),environment=inspect.getfile(SingleVideoQAEnv)),
        external_judge_disabled=True,raw_labels_in_prompt=False,asr_in_prompt=False,audio_attention=args.audio_attention)
    write_json(args.output/'config.json',meta)
    manager=ModelManager();success,message=manager.load(config)
    if not success:raise RuntimeError(message)
    ray_temp=tempfile.mkdtemp(prefix='oan-',dir='/tmp')
    write_json(args.output/'ray_location.json',dict(path=ray_temp))
    ray.init(address='local',num_cpus=4,include_dashboard=False,namespace='omni_eval_ws',object_store_memory=512*1024**2,_temp_dir=ray_temp)
    engine=InferenceEngine(manager,config)
    original_generate=manager.generate;model_errors=[];call_number=0
    def audited_generate(*call_args,**kwargs):
        nonlocal call_number
        call_number+=1
        try:
            result=original_generate(*call_args,**kwargs)
            write_json(args.output/f'generations/{call_number:05d}.json',dict(text=result[0],tokens=result[1],metrics=result[2] if len(result)>2 else {},messages=call_args[0]))
            return result
        except Exception as error:
            model_errors.append(dict(error=str(error),traceback=traceback.format_exc()))
            write_json(args.output/'model_errors.json',model_errors)
            raise
    manager.generate=audited_generate
    results=[]
    rows=list(read_jsonl(args.manifest))
    if args.limit:rows=rows[:args.limit]
    try:
        for row in rows:
            case_start=time.monotonic();error_start=len(model_errors);final=None;error=None
            for event in engine.run(row['video'],row['task_question'],'','FF',None):
                if event['type']=='step':
                    step=event['step'];print(row['id'],step.step,step.action_type,flush=True)
                    write_json(args.output/f"partial/{row['id']}.json",json.loads(generate_history_json(engine.steps,engine.full_messages,
                        row['video'],row['task_question'],gt_answer='',question_type='FF',include_messages=True)))
                elif event['type']=='done':final=event
                elif event['type']=='error':error=event['error']
            answer=final['final'] if final else ''
            if not isinstance(answer,str):error='Official final content was not text';answer=''
            if len(model_errors)>error_start:error='Model runtime error captured; official synthetic Error answer is not counted'
            trace=json.loads(generate_history_json(engine.steps,engine.full_messages,row['video'],row['task_question'],
                gt_answer='',final_answer=answer,question_type='FF',runtime_config=vars(config),include_messages=True))
            record=dict(sample_id=row['id'],person_id=row['person_id'],split=row['split'],status='answered' if answer and not error else 'failed',
                error=error,final_answer=answer,parsed=parse_levels(answer),trace=trace,wall_seconds=time.monotonic()-case_start,
                model_errors=model_errors[error_start:],internal_reward_is_not_personality_score=True)
            results.append(record);write_json(args.output/f"cases/{row['id']}.json",record)
            write_json(args.output/'results.json',results)
            if args.stop_on_model_error and len(model_errors)>error_start:
                break
    finally:
        manager.unload();ray.shutdown()


if __name__=='__main__':main()
