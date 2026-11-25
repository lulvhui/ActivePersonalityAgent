"""Offline common-label evaluation; no reward or labels are sent to either actor."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import TRAITS,read_json,write_json
from baselines.omniagent_native.task import evaluate_levels


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--ours-dir',type=Path);args=p.parse_args()
    refs=read_json(ROOT/'runs/omniagent_native/data/offline_references.json')
    native=read_json(args.run_dir/'native/results.json')
    ours_path=(args.ours_dir or args.run_dir/'ours')/'results.json'
    ours=read_json(ours_path)['records']
    records=[]
    for sample,reference in refs.items():
        n=next((r for r in native if r['sample_id']==sample),None)
        o=next((r for r in ours if r['case_id']==sample),None)
        pair=dict(sample_id=sample)
        if n:
            pair['native']=dict(status=n['status'],parsed=n['parsed'],metrics=evaluate_levels(n['parsed'],reference['labels']),
                final_answer=n['final_answer'],error=n['error'],steps=len(n['trace'].get('steps',[])),wall_seconds=n['wall_seconds'])
        if o:
            final=o['trajectory'].get('final',{})
            parsed={t:dict(status='predicted' if final.get(t,{}).get('level') else 'abstained' if t in final else 'missing',level=final.get(t,{}).get('level')) for t in TRAITS}
            pair['ours']=dict(status='complete' if o['metrics']['complete'] else 'failed',parsed=parsed,
                metrics=evaluate_levels(parsed,reference['labels']),final=final,error=o.get('error'),steps=o['metrics']['attempted_observations'])
        records.append(pair)
    write_json(args.run_dir/'comparison.json',dict(protocol=read_json(ROOT/'runs/omniagent_native/data/protocol.json'),records=records,ours_results=str(ours_path),
        semantic_evidence_quality='not_human_verified',scope='Three-case functional pilot, not a formal benchmark or matched-training ablation.'))

if __name__=='__main__':main()
