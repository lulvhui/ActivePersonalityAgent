#!/usr/bin/env python3
"""Remove confirmed failed inference outputs; retain successful samples and caches."""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]/'runs'


def failed_targets(root=ROOT):
    targets=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink():continue
        summary=directory/'summary.json'
        if not summary.is_file():continue
        try:jobs=json.loads(summary.read_text()).get('jobs',[])
        except (ValueError,OSError):continue
        if not jobs:continue
        failed=[job for job in jobs if job.get('status')=='failed']
        if len(failed)==len(jobs):
            targets.extend(child for child in directory.iterdir() if child.name!='cache')
            if not (directory/'cache').exists():targets.append(directory)
        else:
            for job in failed:
                dataset=job.get('dataset')
                if dataset not in ('IELTS','RecruitView'):continue
                log=directory/f'{dataset}.log'
                if log.is_file():
                    for name in re.findall(r'saved=([^\s]+)',log.read_text(errors='replace')):
                        target=directory/'generations'/Path(name).name
                        if target.is_file():targets.append(target)
                targets.extend(p for p in (directory/dataset,log) if p.exists())
    # A directory selected for removal already owns its children.
    unique=sorted(set(targets),key=lambda p:len(p.parts))
    return [p for p in unique if not any(parent in unique for parent in p.parents)]


def cleanup(apply=False,root=ROOT):
    root=root.resolve()
    targets=failed_targets(root)
    for path in targets:
        if root not in path.absolute().parents:
            raise ValueError('Cleanup target escaped the runs directory')
        if apply:
            if path.is_symlink() or path.is_file():path.unlink()
            elif path.is_dir():shutil.rmtree(path)
    return dict(applied=apply,removed=[str(p) for p in targets],policy='Only confirmed failed inference artifacts; caches and successful samples retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    report=cleanup(args.apply)
    if args.apply:(ROOT/'cleanup_failed_runs.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
