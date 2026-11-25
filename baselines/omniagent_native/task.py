"""Shared personality task and offline evaluation, independent of either policy."""
import argparse
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import TRAITS,LEVELS,read_jsonl,write_json,write_jsonl,digest

SAMPLES=['RecruitView:0010','RecruitView:0530','IELTS:112']


def personality_question(row,native_format=True):
    text=('Assess the interviewee\'s Big Five personality from the interview. '
          'Use actual behavior or behavioral self-reports, distinguish reported life contexts from behavior observed in the interview, '
          'and describe supported context differences rather than changes in underlying personality over time. '
          'Identity, appearance, education or job title alone is insufficient evidence. '
          'For Openness (O), Conscientiousness (C), Extraversion (E), Agreeableness (A), and Neuroticism (N), '
          'use very_low, low, medium, high, very_high, or null when evidence is insufficient. '
          'Then briefly explain the behavioral evidence for each judgment, citing the actual source labels supplied by the environment. ')
    if native_format:
        text+='Begin the final answer with O=level; C=level; E=level; A=level; N=level. '
    if row.get('question'):
        text+='The original interview question is: '+row['question']
    return text


def parse_levels(answer):
    result={}
    for trait in TRAITS:
        matches=re.findall(r'\b'+trait+r'\s*=\s*["\']?(very_low|very_high|medium|high|low|null)\b',answer,re.I)
        values={v.lower() for v in matches}
        result[trait]=dict(status='missing' if not values else 'conflicting' if len(values)>1 else 'abstained' if values=={'null'} else 'predicted',
                           level=None if len(values)!=1 or values=={'null'} else next(iter(values)))
    return result


def evaluate_levels(parsed,labels):
    valid=[t for t in TRAITS if parsed[t]['status']=='predicted']
    return dict(predicted_traits=len(valid),coverage=len(valid)/5,
        ordinal_mae_on_predictions=sum(abs(LEVELS.index(parsed[t]['level'])-labels[t]) for t in valid)/len(valid) if valid else None,
        exact_accuracy_on_predictions=sum(LEVELS.index(parsed[t]['level'])==labels[t] for t in valid)/len(valid) if valid else None,
        all_fields_recognized=all(parsed[t]['status'] in ('predicted','abstained') for t in TRAITS))


def prepare(output):
    rows={r['id']:r for r in read_jsonl(ROOT/'runs/data/dev.jsonl')}
    selected=[];references={}
    for sample in SAMPLES:
        source=rows[sample]
        if source['split']!='dev':raise ValueError('Pilot must retain original Dev membership')
        row={k:source[k] for k in ('id','dataset','person_id','split','video','segments','question') if k in source}
        row['task_question']=personality_question(row)
        row['source_fingerprint']=digest(row)
        selected.append(row)
        references[sample]=dict(labels=source['labels'],labels_raw=source['labels_raw'],person_id=source['person_id'],split='dev')
    output=Path(output)
    write_jsonl(output/'manifest.jsonl',selected)
    write_json(output/'offline_references.json',references)
    write_json(output/'protocol.json',dict(samples=SAMPLES,selection='One ordinary introduction, one behavioral event, one complete IELTS Part 1; selected before native inference.',
        task='Same five ordinal categories and abstention option; no answer labels in policy input.',
        native='Official checkpoint + official OTA environment, raw audio/video only; personality is the FF question.',
        ours='Current full personality framework + original Qwen2.5-Omni-7B, including supplied ASR.',
        caveat='Small whole-system pilot. Weights, training, observation policies, ASR availability and generation defaults differ; not a causal ablation of DA-CTM.',
        native_external_judge=False,training=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=ROOT/'runs/omniagent_native/data')
    prepare(parser.parse_args().output)
