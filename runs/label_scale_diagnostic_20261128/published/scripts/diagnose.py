#!/usr/bin/env python3
"""Local, read-only trajectory analysis. No backend, agent, model, or network calls.
Run from a checkout at 4cdc434 with --help for source paths.
Only aggregate metrics, provenance, and ordinal prediction/label pairs are published.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from amp.common import LEVELS, TRAITS
from amp.evaluate import classification, ranks, spearman


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def corr(x, y):
    return spearman(x, y)


def permutation(items, seed=20261128):
    x, y = [a['p'] for a in items], [a['raw'] for a in items]
    observed = corr(x, y)
    if observed is None:
        return dict(n=len(items), rho=None, p_positive=None, p_two_sided=None, permutations=1000,
                    reason='constant input or fewer than two observations')
    rng = random.Random(seed)
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    rx, ry = [v-mx for v in rx], [v-my for v in ry]
    denom = math.sqrt(sum(v*v for v in rx)*sum(v*v for v in ry))
    positive = two = 0
    for _ in range(1000):
        rng.shuffle(ry)
        r = sum(a*b for a,b in zip(rx,ry))/denom
        positive += r >= observed-1e-12
        two += abs(r) >= abs(observed)-1e-12
    return dict(n=len(items), rho=observed, p_positive=(positive+1)/1001,
                p_two_sided=(two+1)/1001, permutations=1000)


def metrics(items):
    if not items:
        return None
    return classification([a['y'] for a in items], [a['p'] for a in items])


def summarize(ts, lookup, train):
    traits, pooled, calibrated, baselines = {}, [], [], []
    for trait in TRAITS:
        allrows = [dict(id=t['_id'], person=t['sample_id'],
                        p=None if t['final'][trait]['level'] is None else LEVELS.index(t['final'][trait]['level']),
                        y=lookup[t['sample_id']]['labels'][trait],
                        raw=lookup[t['sample_id']]['labels_raw'][trait]) for t in ts]
        covered = [a for a in allrows if a['p'] is not None]
        n = len(covered)
        rr = ranks([a['p'] for a in covered])
        # zero-based midrank -> percentile (rank+.5)/n -> five equal-width bins.
        # Ties stay together: output cannot be exactly equal-frequency with large ties.
        cal = [dict(a, p=min(4, int(5*(rank+.5)/n))) for a,rank in zip(covered,rr)]
        dataset = ts[0]['dataset']
        tr = [r for r in train if r['dataset']==dataset and r['split']=='train' and set(r.get('labels') or {})==set(TRAITS)]
        counts = Counter(r['labels'][trait] for r in tr)
        mode = max(range(5), key=lambda k:(counts[k],-k))
        baseline = [dict(a,p=mode) for a in allrows]
        traits[trait] = dict(n_total=len(ts), n_covered=n, abstentions=len(ts)-n,
                            predicted_counts=[sum(a['p']==k for a in allrows) for k in range(5)],
                            label_counts_all=[sum(a['y']==k for a in allrows) for k in range(5)],
                            label_counts_covered=[sum(a['y']==k for a in covered) for k in range(5)],
                            spearman=permutation(covered), original=metrics(covered),
                            accuracy_all=sum(a['p']==a['y'] for a in covered)/len(ts),
                            rank_calibrated=metrics(cal),
                            calibrated_counts=[sum(a['p']==k for a in cal) for k in range(5)],
                            train_majority_level=mode, majority_all=metrics(baseline),
                            majority_covered=metrics([dict(a,p=mode) for a in covered]))
        pooled.extend(covered);calibrated.extend(cal)
        baselines.extend(dict(a,p=mode) for a in covered)
    return dict(n_trajectories=len(ts), n_people=len({t['sample_id'] for t in ts}),traits=traits,
                pooled_raw_spearman=permutation(pooled), original=metrics(pooled),
                rank_calibrated=metrics(calibrated), majority_same_covered_pairs=metrics(baselines))


def cluster_permutation(ts, lookup, trait=None):
    people=sorted({t['sample_id'] for t in ts})
    slots=[(t['sample_id'],k,LEVELS.index(t['final'][k]['level'])) for t in ts
           for k in ([trait] if trait else TRAITS) if t['final'][k]['level'] is not None]
    x=[p for _,_,p in slots]
    y=[lookup[person]['labels_raw'][k] for person,k,_ in slots]
    observed=corr(x,y)
    if observed is None:
        return dict(rho=None,p_positive=None,p_two_sided=None,n_people=len(people),permutations=1000)
    positive=two=0;rng=random.Random(20261128)
    for _ in range(1000):
        shuffled=people[:];rng.shuffle(shuffled);mapping=dict(zip(people,shuffled))
        yy=[lookup[mapping[person]]['labels_raw'][k] for person,k,_ in slots]
        r=corr(x,yy)
        if r is None:r=0.
        positive+=r>=observed-1e-12;two+=abs(r)>=abs(observed)-1e-12
    return dict(rho=observed,p_positive=(positive+1)/1001,p_two_sided=(two+1)/1001,
                n_people=len(people),permutations=1000,
                method='shuffle entire raw-label vectors across people, retaining all repetitions and cross-trait dependence')


def consistency(ts, same_experiment=False):
    pairs = [(a,b) for a,b in itertools.combinations(ts,2)
             if a['sample_id']==b['sample_id'] and (not same_experiment or a['_experiment']==b['_experiment'])]
    out = {}
    for trait in TRAITS:
        covered = [(a['final'][trait]['level'],b['final'][trait]['level']) for a,b in pairs
                   if a['final'][trait]['level'] is not None and b['final'][trait]['level'] is not None]
        out[trait] = dict(n_pairs=len(covered), agreements=sum(a==b for a,b in covered),
                          agreement_rate=sum(a==b for a,b in covered)/len(covered) if covered else None)
    total=sum(v['n_pairs'] for v in out.values());same=sum(v['agreements'] for v in out.values())
    return dict(n_trajectory_pairs=len(pairs),traits=out,pooled_agreement=same/total if total else None)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--branch-root',type=Path,required=True)
    ap.add_argument('--main-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--unit-test-log',type=Path)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    train_path=args.main_root/'runs/data/train.jsonl'
    dev_path=args.main_root/'runs/data/dev.jsonl'
    bins_path=args.main_root/'runs/data/label_bins.json'
    train=[json.loads(s) for s in train_path.read_text().splitlines() if s]
    dev=[json.loads(s) for s in dev_path.read_text().splitlines() if s]
    lookup={r['id']:r for r in train+dev}
    known={'teacher_pilot5_bounded_20261127':'abe7c9639603ffa15d48ad966b9afe2f99ec3bd5',
           'teacher_pilot5_grounded_20261127':'3b8543dcf51505244761f1d38aa702c80f1d88f5',
           'teacher_seed_c56ec92_20261127':'c56ec929f5212319e06663d7dee31bf977d48c01'}
    histories=subprocess.check_output(['git','rev-list','--all','--','amp'],text=True).splitlines()
    commit_hashes={}
    def match_source(hashes):
        # Match the archived Python files against Git blobs, never guess a tested commit.
        paths={p:h for p,h in hashes.items() if p.startswith('amp/') and p.endswith('.py')}
        if not paths:return []
        matches=[]
        for commit in histories:
            ok=True
            for p,h in paths.items():
                key=(commit,p)
                if key not in commit_hashes:
                    r=subprocess.run(['git','show',commit+':'+p],capture_output=True)
                    commit_hashes[key]=digest(r.stdout) if r.returncode==0 else None
                if commit_hashes[key]!=h:ok=False;break
            if ok:matches.append(commit)
        return matches
    found=[];excluded=[];seen={};empty_sources=[];archive_count=loose_count=0
    def ingest(data, source, nearby):
        nonlocal loose_count
        xs=json.loads(data)
        if isinstance(xs,dict):xs=[xs]
        if not xs and 'teacher' in source:
            empty_sources.append(dict(source=source,reason='empty trajectory list; no evaluable Final'))
        for index,t in enumerate(xs):
            if not isinstance(t,dict) or 'sample_id' not in t:continue
            cfg=t.get('config',{});summary=nearby.get('summary.json',{})
            if 'qwen3.8' not in str(cfg.get('model',summary.get('model',''))).lower():continue
            key=digest(json.dumps(t,sort_keys=True).encode())
            if key in seen:
                seen[key]['source_aliases'].append(source);continue
            experiment=next((p for p in source.split('/') if p.startswith('teacher_')),source.split('/runs/')[-1].split('/')[0])
            final=t.get('final')
            reason=None
            if t.get('offline'):reason='offline fixture'
            elif not isinstance(final,dict) or set(final)!=set(TRAITS):reason='missing structurally complete Final'
            elif any(v.get('level') not in [None]+list(LEVELS) for v in final.values()):reason='invalid Final level'
            record=dict(source=source,source_aliases=[],sample_id=t['sample_id'],split=t.get('split'),
                        status=t.get('run_status'),trajectory_sha256=key,experiment=experiment)
            if reason:
                excluded.append(dict(record,reason=reason));continue
            matches=match_source(nearby.get('source_hashes.json',{})) if experiment not in known else []
            record.update(tested_commit=known.get(experiment),source_hash_matching_commits=matches,
                          tested_commit_basis='published acquisition plan' if experiment in known else 'not recorded; matching source snapshots listed without asserting runtime HEAD',
                          inferred_tested_code_commit=matches[0] if len(matches)==1 else None,
                          archived_source_hashes_sha256=digest(json.dumps(nearby.get('source_hashes.json',{}),sort_keys=True).encode()) if nearby.get('source_hashes.json') else None)
            run_name=Path(source.split('::')[0]).stem if '::' in source else Path(source).parent.name
            t['_id']=f'{experiment}/{run_name}#{index}'
            t['_experiment']=experiment;record['trajectory_id']=t['_id']
            record['evaluation_scope']='train' if t['sample_id'] in {r['id'] for r in train} else 'supplementary_dev' if t['sample_id'] in lookup else 'unscorable_person_target'
            record['pairs']=[dict(sample_id=t['sample_id'],trait=trait,predicted_level=None if final[trait]['level'] is None else LEVELS.index(final[trait]['level']),
                                 label_level=lookup[t['sample_id']]['labels'][trait] if t['sample_id'] in lookup else None) for trait in TRAITS]
            seen[key]=record;found.append((t,record))
    with tempfile.TemporaryDirectory(prefix='label-scale-') as tmp:
        temp=Path(tmp)
        for root in [args.main_root,args.branch_root]:
            # Archives first: copies and candidate packets must not count as independent runs.
            for archive in sorted((root/'runs').rglob('*.tar.xz')):
                with tarfile.open(archive) as tf:
                    members=[m for m in tf.getmembers() if m.isfile() and Path(m.name).name in ['trajectories.json','summary.json','source_hashes.json']]
                    extracted={}
                    for m in members:
                        if '..' in Path(m.name).parts or Path(m.name).is_absolute():raise ValueError('Unsafe archive path')
                        target=temp/digest(str(archive).encode())/m.name
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(tf.extractfile(m).read());extracted[m.name]=target
                    for name,p in extracted.items():
                        if p.name!='trajectories.json':continue
                        nearby={q.name:read(q) for q in p.parent.iterdir() if q.name in ['summary.json','source_hashes.json']}
                        ingest(p.read_bytes(),str(archive)+'::'+name,nearby);archive_count+=1
            for p in sorted((root/'runs').rglob('*trajector*.json')):
                if 'label_scale_diagnostic' in str(p):continue
                try:
                    nearby={n:read(p.parent/n) for n in ['summary.json','source_hashes.json'] if (p.parent/n).exists()}
                    ingest(p.read_bytes(),str(p),nearby);loose_count+=1
                except (ValueError,TypeError):continue
        train_ts=[t for t,r in found if r['evaluation_scope']=='train']
        dev_ts=[t for t,r in found if r['evaluation_scope']=='supplementary_dev']
        # Deterministic, label-blind single-run selection: prefer latest named experiment,
        # then the lexical first run (r1 before r2), regardless of completeness or score.
        unique={}
        for t in sorted(train_ts,key=lambda t:(t['_experiment'],t['_id']),reverse=True):
            if t['sample_id'] not in unique or t['_experiment']==unique[t['sample_id']]['_experiment']:
                unique[t['sample_id']]=t
        one=list(unique.values())
        report=dict(analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    inputs={str(p):digest(p.read_bytes()) for p in [train_path,dev_path,bins_path]},
                    methodology=dict(levels=list(LEVELS),null_handling='trait-wise abstention as amp.evaluate; conditional metrics exclude null, accuracy_all counts null incorrect',
                        permutations='1000 random shuffles; seed 20261128; +1 correction; one-sided positive and two-sided; repeated trajectories are dependent, use one-person results for inference',
                        rank_calibration='floor(5*(zero_based_average_rank+0.5)/n), capped at 4; ties never split; not exact equal-frequency if ties span bins; calibrated on evaluated sample, not held out',
                        pooled_warning='Raw values pooled across traits are not normalized and may reflect between-trait scale differences; per-trait results take precedence.',
                        one_person_selection='lexically latest experiment, then lexically first run; label-blind; no best-of selection',
                        baseline='per dataset/trait train majority; lower level wins ties; same covered pairs for comparison',
                        scope='Train primary; Dev supplementary; unscorable RecruitView person targets never inherit answer labels'),
                    inventory=[r for t,r in found],excluded=excluded,empty_trajectory_sources=empty_sources,
                    discovery=dict(archive_trajectory_files=archive_count,loose_trajectory_files=loose_count),
                    train_all=summarize(train_ts,lookup,train),train_one_per_person=summarize(one,lookup,train),
                    one_per_person_ids=[t['_id'] for t in one],
                    consistency_all_runs=consistency(train_ts),consistency_same_experiment=consistency(train_ts,True),
                    per_experiment={e:summarize([t for t in train_ts if t['_experiment']==e],lookup,train) for e in sorted({t['_experiment'] for t in train_ts})})
        report['train_person_cluster_permutation']={k:cluster_permutation(train_ts,lookup,k) for k in TRAITS}
        report['train_person_cluster_permutation']['pooled_raw']=cluster_permutation(train_ts,lookup)
        if args.unit_test_log:
            test_text=args.unit_test_log.read_text()
            report['validation']=dict(command='python -m unittest discover -s tests',tested_commit=report['analysis_commit'],
                tests_run=int(re.search(r'Ran (\d+) tests',test_text)[1]),
                skipped=int(re.search(r'OK \(skipped=(\d+)\)',test_text)[1]),result='OK',
                log_sha256=digest(args.unit_test_log.read_bytes()),amp_code_changes=False)
        report['evaluation_command_template']='python -m amp.cli evaluate --trajectories <temporary_merged.json> --manifest <main_root>/runs/data/train.jsonl --bins <main_root>/runs/data/label_bins.json --train-manifest <main_root>/runs/data/train.jsonl --output <temporary_evaluate.json>; result embedded in report; Dev uses dev.jsonl'
        report['supplementary_dev']={d:summarize([t for t in dev_ts if t['dataset']==d],lookup,train) for d in sorted({t['dataset'] for t in dev_ts})}
        for label,ts,manifest in [('train_all',train_ts,train_path),('train_one_per_person',one,train_path),('supplementary_dev',dev_ts,dev_path)]:
            merged=temp/(label+'.json');out=temp/(label+'_evaluate.json')
            dump(merged,[{k:v for k,v in t.items() if not k.startswith('_')} for t in ts])
            command=[sys.executable,'-m','amp.cli','evaluate','--trajectories',str(merged),'--manifest',str(manifest),'--bins',str(bins_path),'--train-manifest',str(train_path),'--output',str(out)]
            result=subprocess.run(command,capture_output=True,text=True,check=True)
            report[label+'_amp_evaluate']=read(out)
        dump(args.output/'report.json',report)
        print(json.dumps(dict(train_n=len(train_ts),people=len(one),dev_n=len(dev_ts),inventory=len(found),
                              traits={k:v['spearman'] for k,v in report['train_all']['traits'].items()},
                              unique_traits={k:v['spearman'] for k,v in report['train_one_per_person']['traits'].items()},
                              original=report['train_all']['original'],calibrated=report['train_all']['rank_calibrated'],
                              majority=report['train_all']['majority_same_covered_pairs']),indent=2))

if __name__=='__main__':
    main()
