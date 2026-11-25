"""Freeze an expanded Dev pool and compare saved outputs without actor labels."""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import TRAITS,LEVELS,read_json,read_jsonl,write_json,write_jsonl
from baselines.omniagent_native.task import personality_question,evaluate_levels

GROUPS=('current_3b','current_7b','omniagent_transfer','omniagent_native')


def select_pool(rows, excluded, per_dataset=6):
    """Hash-based selection by person, independent of labels and model results."""
    selected=[]
    for dataset in ('IELTS','RecruitView'):
        people={str(r['person_id']) for r in rows if r['dataset']==dataset
                and r['split']=='dev' and (dataset,str(r['person_id'])) not in excluded}
        ranked=sorted(people,key=lambda p:hashlib.sha256(f'expanded-v1:{dataset}:{p}'.encode()).hexdigest())
        if len(ranked)<per_dataset:raise ValueError('Not enough unexcluded Dev persons')
        for person in ranked[:per_dataset]:
            choices=[r for r in rows if r['dataset']==dataset and str(r['person_id'])==person and r['split']=='dev']
            selected.append(min(choices,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()))
    return selected


def prepare(output):
    if (output/'data/manifest.jsonl').exists():raise ValueError('Use a fresh run directory; never resample an existing pool')
    rows=[r for split in ('train','dev','test') for r in read_jsonl(ROOT/f'runs/data/{split}.jsonl')]
    by_id={r['id']:r for r in rows}
    excluded_ids={c['sample_id'] for c in read_json(ROOT/'validation/panel.json')['cases']}
    excluded_ids.update(r['id'] for r in read_jsonl(ROOT/'runs/omniagent_native/data/manifest.jsonl'))
    excluded={(by_id[s]['dataset'],str(by_id[s]['person_id'])) for s in excluded_ids}
    selected=select_pool(rows,excluded)
    manifest=[];refs={}
    for source in selected:
        if not Path(source['video']).is_file():raise FileNotFoundError(source['video'])
        row={k:source[k] for k in ('id','dataset','person_id','split','video','segments','question') if k in source}
        row['task_question']=personality_question(row)
        manifest.append(row)
        refs[source['id']]={k:source[k] for k in ('labels','labels_raw','person_id','dataset','split')}
    write_jsonl(output/'data/manifest.jsonl',manifest)
    write_json(output/'data/offline_references.json',refs)
    write_json(output/'data/protocol.json',dict(groups=GROUPS,samples=[r['id'] for r in manifest],
        excluded_people=sorted(excluded),selection='Fixed hash, six Dev persons per dataset, one full task per person; no outcome/label selection',
        exclusions_scope='Known six-case diagnostic panel and previous three-case native pilot; not an exhaustive history of all older runs',
        whole_media=True,training=False,external_api_calls=0,semantic_gold=False,
        method_config='Current groups use strict validation and default 16 active steps; native preserves official dynamic budget and generation defaults',
        caveats=['Native has raw audio/video; current-framework groups also have ASR.',
                 'Base checkpoints are untrained by this project; official checkpoint already has SFT/RL.',
                 'Native answered and current strict completion are different metrics; no causal DA-CTM claim.']))
    write_json(output/'data/source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'amp').glob('*.py'))})


def summarize(output):
    refs=read_json(output/'data/offline_references.json')
    cases={sample:dict(sample_id=sample,dataset=ref['dataset'],groups={}) for sample,ref in refs.items()}
    summaries={}
    for group in GROUPS:
        path=output/group/'results.json'
        if not path.exists():continue
        payload=read_json(path);native=group=='omniagent_native'
        rows=payload if native else payload['records']
        if native:
            by_id={r['sample_id']:r for r in rows}
            for batch in sorted((output/'native_batches').glob('*/results.json')):
                for record in read_json(batch):
                    sample=record['sample_id']
                    if sample in by_id and by_id[sample]!=record:raise ValueError('Conflicting native results for one frozen task')
                    by_id[sample]=record
            rows=list(by_id.values())
        for record in rows:
            sample=record['sample_id'] if native else record['case_id']
            if sample not in cases:raise ValueError('Output sample outside frozen pool')
            if native:
                parsed=record['parsed'];trace=record['trace']
                raw=record['final_answer'];complete=record['status']=='answered'
                observations=sum(s['action_type']!='answer' for s in trace.get('steps',[]))
                tokens=sum(s.get('response_tokens',0) for s in trace.get('steps',[]))
                input_tokens=sum(s.get('prompt_total_tokens',0) for s in trace.get('steps',[]))
                native_steps=trace.get('steps',[])
                details=[dict(step=s['step'],next_action=s['action_type'],
                    observation_of_previous_step=native_steps[i-1]['step'] if i else None,
                    observation=s.get('observation','')) for i,s in enumerate(native_steps)]
                validation_errors=record.get('model_errors',[])
            else:
                trace=record['trajectory'];final=trace.get('final',{})
                complete=record['metrics']['complete']
                parsed={t:dict(status='predicted' if final.get(t,{}).get('level') else 'abstained' if t in final else 'missing',level=final.get(t,{}).get('level')) for t in TRAITS}
                raw=final;observations=record['metrics']['attempted_observations']
                tokens=record['metrics']['generated_tokens'];input_tokens=record['metrics']['input_tokens']
                details=trace.get('memory',{}).get('observations',[])
                validation_errors=[dict(kind=d['kind'],step=d['step'],error=d.get('candidate_error') or d.get('memory_error') or d.get('validation_error'),
                                        rejected_candidates=d.get('rejected_candidates',[]))
                    for d in trace.get('decisions',[]) if d.get('candidate_error') or d.get('memory_error') or d.get('validation_error') or d.get('rejected_candidates')]
            # Failed partial finals remain inspectable, but cannot count as valid predictions.
            valid=parsed if complete else {t:dict(status='missing',level=None) for t in TRAITS}
            cases[sample]['groups'][group]=dict(complete=complete,completion_definition='native_answered' if native else 'strict_source_protocol',
                episode_returned=complete if native else record['metrics'].get('episode_returned',complete),
                parsed=parsed,metrics=evaluate_levels(valid,refs[sample]['labels']),error=record.get('error'),
                observations=observations,generated_tokens=tokens,input_tokens=input_tokens,raw_final=raw,details=details,
                validation_errors=validation_errors,
                semantic_review_status='pending',semantic_quality_verified=False)
        entries=[c['groups'][group] for c in cases.values() if group in c['groups']]
        predictions=sum(e['metrics']['predicted_traits'] for e in entries)
        summaries[group]=dict(expected=len(refs),finished=len(entries),complete=sum(e['complete'] for e in entries),
            episode_returned=sum(e['episode_returned'] for e in entries),
            partial_returned=sum(e['episode_returned'] and not e['complete'] for e in entries),
            all_fields_recognized=sum(e['complete'] and e['metrics']['all_fields_recognized'] for e in entries),
            raw_all_fields_recognized=sum(all(e['parsed'][t]['status'] in ('predicted','abstained') for t in TRAITS) for e in entries),
            raw_trait_coverage=sum(e['parsed'][t]['status']=='predicted' for e in entries for t in TRAITS)/(5*len(refs)),
            valid_trait_coverage=predictions/(5*len(refs)),mean_observations=sum(e['observations'] for e in entries)/len(entries) if entries else None,
            predicted_trait_count=predictions,ordinal_mae_on_predictions=sum(e['metrics']['ordinal_mae_on_predictions']*e['metrics']['predicted_traits'] for e in entries if e['metrics']['predicted_traits'])/predictions if predictions else None,
            generated_tokens=sum(e['generated_tokens'] for e in entries),input_tokens=sum(e['input_tokens'] for e in entries),
            semantic_quality_verified=False)
    paired=[]
    for left,right in combinations(GROUPS,2):
        common=[]
        for sample,case in cases.items():
            a=case['groups'].get(left);b=case['groups'].get(right)
            if not a or not b or not a['complete'] or not b['complete']:continue
            for trait in TRAITS:
                if a['parsed'][trait]['status']==b['parsed'][trait]['status']=='predicted':
                    common.append(dict(sample_id=sample,trait=trait,left=a['parsed'][trait]['level'],right=b['parsed'][trait]['level'],
                                       reference_bin=refs[sample]['labels'][trait]))
        paired.append(dict(left=left,right=right,common_predicted_trait_count=len(common),predictions=common,
                          left_ordinal_mae_on_common_predictions=sum(abs(LEVELS.index(v['left'])-v['reference_bin']) for v in common)/len(common) if common else None,
                          right_ordinal_mae_on_common_predictions=sum(abs(LEVELS.index(v['right'])-v['reference_bin']) for v in common)/len(common) if common else None,
                          note='Common valid predictions only; abstention/failure coverage remains separate.'))
    dataset_summaries={}
    for dataset in ('IELTS','RecruitView'):
        cohort=[c for c in cases.values() if c['dataset']==dataset]
        if not cohort:continue
        dataset_summaries[dataset]={g:dict(expected=len(cohort),finished=sum(g in c['groups'] for c in cohort),
            complete=sum(c['groups'][g]['complete'] for c in cohort if g in c['groups']),
            valid_trait_coverage=sum(c['groups'][g]['metrics']['predicted_traits'] for c in cohort if g in c['groups'])/(5*len(cohort)))
            for g in summaries}
    report=dict(protocol=read_json(output/'data/protocol.json'),summaries=summaries,dataset_summaries=dataset_summaries,
                paired_common_predictions=paired,cases=list(cases.values()))
    write_json(output/'comparison.json',report)
    lines=['# 扩大测试池：逐例输出与待审核证据','',
           '固定 Dev 开发对照；所有语义审核尚未人工批准。协议完成不等于人格正确。','',
           '| 组别 | 已结束/计划 | 严格流程/原生answer完成 | 返回结果（含partial） | 原始五维格式完整 | 严格有效预测覆盖率 | 平均观察数 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for group,s in summaries.items():
        lines.append(f"| {group} | {s['finished']}/{s['expected']} | {s['complete']} | {s['episode_returned']} | {s['raw_all_fields_recognized']} | {s['valid_trait_coverage']:.1%} | {s['mean_observations']:.2f} |")
    rows={r['id']:r for r in read_jsonl(output/'data/manifest.jsonl')}
    for sample,case in cases.items():
        lines.extend(['',f'## {sample}','',f"原始问题：{rows[sample].get('question','')}",'',
            f"完整媒体：[原视频](<{rows[sample]['video']}>)" if rows[sample].get('video') else '无媒体链接（测试用例）','',
            '来源转写（ASR 时间未核验）：','',*['> '+s['text'].replace('\n',' ') for s in rows[sample]['segments']]])
        for group,e in case['groups'].items():
            lines.extend(['',f'### {group}','',f"状态：complete={e['complete']}；error={e['error']}",'',
                f"[原始逐例轨迹]({group}/cases/{sample}.json)",'',
                '规则失败：'+json.dumps(e['validation_errors'],ensure_ascii=False),'',
                '```json',json.dumps(e['raw_final'],ensure_ascii=False,indent=2),'```','',
                '观察/证据（模型生成，待审核）：','```json',json.dumps(e['details'],ensure_ascii=False,indent=2),'```'])
    (output/'review_packet.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=('prepare','summarize'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:print(json.dumps(summarize(args.output)['summaries'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
