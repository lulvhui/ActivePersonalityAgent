"""Run each frozen task once; end a batch on model error before engine reuse."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import read_json,read_jsonl,write_json,write_jsonl


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();run=args.run_dir
    manifest=list(read_jsonl(run/'data/manifest.jsonl'))
    combined=run/'omniagent_native/results.json'
    records=read_json(combined) if combined.exists() else []
    seen={r['sample_id'] for r in records}
    if len(seen)!=len(records):raise ValueError('Duplicate completed native case')
    batch=0
    while len(seen)<len(manifest):
        pending=[r for r in manifest if r['id'] not in seen]
        folder=run/f'native_batches/{batch:02d}'
        while folder.exists():
            batch+=1;folder=run/f'native_batches/{batch:02d}'
        inputs=folder/'manifest.jsonl';write_jsonl(inputs,pending)
        command=[os.environ.get('AMP_NATIVE_PYTHON',str(Path(os.environ.get('AMP_CONDA_ROOT',str(Path.home()/'miniconda3')))/'envs/omniagent-native/bin/python')),'-u',str(ROOT/'baselines/omniagent_native/run_native.py'),
                 '--manifest',str(inputs),'--output',str(folder),'--stop-on-model-error']
        with (folder/'run.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        results=read_json(folder/'results.json') if (folder/'results.json').exists() else []
        if not results:raise RuntimeError(f'Native batch {batch} made no progress; exit={process.returncode}')
        for record in results:
            if record['sample_id'] in seen or record['sample_id'] not in {r['id'] for r in pending}:
                raise ValueError('Native batch returned an unexpected or repeated task')
            seen.add(record['sample_id']);records.append(record)
            write_json(run/f"omniagent_native/cases/{record['sample_id']}.json",record)
        write_json(combined,records)
        if not (run/'omniagent_native/config.json').exists():
            write_json(run/'omniagent_native/config.json',read_json(folder/'config.json'))
        write_json(run/'native_batch_status.json',dict(finished=len(records),expected=len(manifest),
            latest_batch=str(folder),latest_exit=process.returncode,
            policy='No repeated completed cases; fresh engine only for still-unfinished tasks after runtime failure.'))
        print(f'Native completed {len(records)}/{len(manifest)}',flush=True)


if __name__=='__main__':main()
