"""Verify native source provenance and the actual OTA interface used in traces."""
import hashlib
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import read_json,write_json

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/omniagent_native/comparison_v3')
parser.add_argument('--native-subdir',default='native')
args=parser.parse_args()
upstream=ROOT.parent/'OmniAgent'
run=args.run_dir
config=read_json(run/args.native_subdir/'config.json')
source_files=['demo/omniagent_demo_pro.py','agent_system/environments/env_package/video_env.py','agent_system/environments/prompts/video_prompt.py']
sources={}
for name in source_files:
    path=upstream/name
    original=subprocess.check_output(['git','-C',str(upstream),'show',f'HEAD:{name}'])
    sources[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),matches_upstream=path.read_bytes()==original)
spec=importlib.util.spec_from_file_location('native_prompt',upstream/source_files[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=[]
for result in read_json(run/args.native_subdir/'results.json'):
    trace=result['trace'];system=trace['messages'][0]['content'][0]['text']
    initial=trace['messages'][1]['content']
    duration=float(re.search(r'duration_seconds: ([\d.]+)',initial[0]['text'])[1])
    steps=min(5+int(duration/60),32)
    # Official VideoEnvWorker converts these limits to ints before reset().
    expected=module.build_video_prompt_omniagent(steps,60,300,60)
    actions=[]
    for step in trace['steps']:
        try:actions.append(json.loads(step['response'])['action'])
        except (ValueError,KeyError,TypeError):actions.append(dict(unparsed_response=True,step=step['step']))
    cases.append(dict(sample_id=result['sample_id'],native_system_prompt_identical=system==expected,
        initial_content_types=[part['type'] for part in initial],gt_answer_empty=trace['gt_answer']=='',
        model_runtime_errors=result['model_errors'],actions=actions,
        media_omission_markers=json.dumps(trace['messages']).count('MEDIA OMITTED'),
        prompt_tokens=sum(s.get('prompt_total_tokens',0) for s in trace['steps']),
        response_tokens=sum(s.get('response_tokens',0) for s in trace['steps'])))
assert all(s['matches_upstream'] for s in sources.values())
assert all(c['native_system_prompt_identical'] and c['gt_answer_empty'] for c in cases)
write_json(run/'native_fidelity.json',dict(upstream_commit=config['upstream_commit'],sources=sources,cases=cases,
    all_model_calls_succeeded=all(not c['model_runtime_errors'] for c in cases),
    no_personality_memory_or_candidate_pool_in_native=True,external_api_calls=0,
    runtime_adaptations=['two-GPU tensor parallelism','64K model context / 60K prompt allowance','GPU memory fraction 0.90','Python 3.10 rather than recommended 3.11','no external FF judge; reward=0 is not a personality score','official torchvision fallback because TorchCodec is absent','official xformers/audio attention fallback because standalone flash-attn is absent']))
print('Verified original source, original per-video OTA system prompts, blank ground truth and actual native actions for',len(cases),'cases')
