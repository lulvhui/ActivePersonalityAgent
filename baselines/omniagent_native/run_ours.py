"""Current personality method on the same raw task units as native OmniAgent."""
import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import read_json,read_jsonl,write_json
from amp.agent import ActiveAgent
from amp.backend import QwenBackend
from amp.media import MediaTools
from amp.scoring import OfflineTraitScorer
from baselines.omniagent_native.task import personality_question
from scripts.validate_panel import metrics,summarize

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'runs/omniagent_native/data/manifest.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/omniagent_native/ours')
    parser.add_argument('--model',default=str(ROOT.parent/'models/Qwen2.5-Omni-7B'))
    args=parser.parse_args()
    cfg=dict(read_json(ROOT/'configs/local_omni3b.json'),model=args.model)
    args.output.mkdir(parents=True,exist_ok=True)
    # Two launchers may schedule the same group; serialize before allocating a GPU.
    lock=(args.output/'.run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    rows=list(read_jsonl(args.manifest))
    identity=dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'amp').glob('*.py'))})
    if (args.output/'results.json').exists():
        saved=read_json(args.output/'results.json')
        if (saved.get('input_identity')==identity and saved.get('config')==cfg
                and [r['case_id'] for r in saved['records']]==[r['id'] for r in rows]):
            print('All frozen cases already finished; reusing identical saved outputs',flush=True)
            lock.close()
            return
        lock.close()
        raise ValueError('Existing output is partial or has different inputs/code; preserve it and choose a fresh directory')
    backend=QwenBackend(cfg['model'],max_tokens=cfg['max_new_tokens'],temperature=cfg['temperature'],generation_log_dir=args.output/'generations')
    agent=ActiveAgent(backend,OfflineTraitScorer(),MediaTools(args.output/'media_cache',cfg['max_clip_seconds'],cfg['video_fps']),cfg)
    records=[]
    for original in rows:
        row=dict(original,question=personality_question(original,native_format=False))
        # Metadata is retained for reports only. Neither policy gets global labels.
        row.pop('task_question',None);row.pop('source_fingerprint',None)
        try:
            trace=agent.run(row);record=dict(case_id=row['id'],trajectory=trace,metrics=metrics(trace,True))
        except (ValueError,RuntimeError,KeyError,TypeError) as error:
            trace=agent.last_partial or {};failures=getattr(error,'failures',[])
            record=dict(case_id=row['id'],trajectory=trace,metrics=metrics(trace,False,failures),error=str(error),failures=failures)
        records.append(record);write_json(args.output/f"cases/{row['id']}.json",record)
        write_json(args.output/'results.json',dict(model=cfg['model'],config=cfg,summary=summarize(records),records=records,
            asr_available=True,external_api_calls=0,training=False,input_identity=identity))
        print(row['id'],record['metrics'],record.get('error',''),flush=True)
    lock.close()

if __name__=='__main__':main()
