"""Offline, train-only legacy evidence audit and source-bound Memory teacher drafts.

Legacy annotations are review references, never teacher input or automatic approval.
This prepares 观察原文保存 seeds only; it does not invent histories or train a model.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amp.common import TRAITS, digest, normalize_id, read_json, read_jsonl, write_json, write_jsonl
from amp.data import split_lookup
from amp.provenance import index_transcript
from amp.state import empty_memory


def normalized(text):
    # Formatting differences only; never fuzzy-match different utterances.
    return ' '.join(text.split()).casefold()


def prepare(old_root, manifest_dir, output, pilot_per_group=8, seed=42):
    old_root, manifest_dir, output = map(Path, (old_root, manifest_dir, output))
    all_rows = [r for split in ('train', 'dev', 'test') for r in read_jsonl(manifest_dir/f'{split}.jsonl')]
    membership = {}
    for row in all_rows:
        key = row['dataset'], normalize_id(row['person_id'])
        if key in membership and membership[key] != row['split']:
            raise ValueError('Person crosses current splits')
        membership[key] = row['split']
    inputs, references, rejected, hashes = [], [], [], {}
    counts = Counter()
    for dataset in ('IELTS', 'RecruitView'):
        paths = dict(splits=old_root/f'artifacts/training_data/{dataset}/splits/split_manifest.json',
                     turns=old_root/f'data/{dataset}/processed/video_turn_inputs/{dataset.lower()}_common_turns.jsonl',
                     annotations=old_root/f'data/{dataset}/gt/evidence_gt.jsonl')
        for path in paths.values():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        splits = split_lookup(read_json(paths['splits']))
        rows = [r for r in all_rows if r['dataset'] == dataset]
        for row in rows:
            if splits[normalize_id(row['person_id'])] != row['split']:
                raise ValueError('Legacy/current person split mismatch')
        by_person, by_video = defaultdict(list), defaultdict(list)
        for row in rows:
            by_person[normalize_id(row['person_id'])].append(row)
            by_video[Path(row['video']).name].append(row)
        turns = {}
        for turn in read_jsonl(paths['turns']):
            key = normalize_id(turn['person_id']), str(turn['turn_id'])
            if key in turns:
                raise ValueError('Ambiguous legacy turn identity')
            turns[key] = turn
        seen = set()
        for annotation in read_jsonl(paths['annotations']):
            person = normalize_id(annotation['person_id'])
            if splits.get(person) != 'train':
                counts[f'{dataset}/excluded_non_train'] += 1
                continue
            key = person, str(annotation['turn_id'])
            if key in seen:
                raise ValueError('Duplicate legacy annotation')
            seen.add(key)
            counts[f'{dataset}/train_annotations'] += 1
            turn = turns.get(key)
            candidates = (by_person[person] if dataset == 'IELTS' else
                          by_video[Path(turn['video_path']).name] if turn else [])
            candidates = [r for r in candidates if normalize_id(r['person_id']) == person and r['split'] == 'train']
            reason = None
            if turn is None or len(candidates) != 1:
                reason = 'missing_or_ambiguous_source'
            elif normalized(turn['answer_text']) != normalized(annotation['answer_text']):
                reason = 'annotation_turn_text_mismatch'
            else:
                row = candidates[0]
                matches = [i for i,s in enumerate(row['segments']) if normalized(s['text']) == normalized(annotation['answer_text'])]
                if len(matches) == 1:
                    selected = matches
                elif dataset == 'RecruitView' and normalized(' '.join(s['text'] for s in row['segments'])) == normalized(annotation['answer_text']):
                    selected = list(range(len(row['segments'])))
                else:
                    reason = 'raw_transcript_mismatch_or_ambiguity'
                if not Path(row['video']).is_file():
                    reason = 'missing_original_media'
            case_id = f'{dataset}:{person}:turn_{key[1]}'
            if reason:
                rejected.append(dict(id=case_id, dataset=dataset, person_id=person, split='train', reason=reason))
                counts[f'{dataset}/{reason}'] += 1
                continue
            media_id = digest(str(Path(row['video']).resolve()))
            duration = max([float(s['end']) for s in row['segments'] if s.get('end') is not None] or [1.])
            indexed = index_transcript(row['segments'], media_id, duration)
            segments = [indexed[i] for i in selected]
            observation = dict(modality='text', time=[0., duration], timing='transcript', media_id=media_id, segments=segments, media=[])
            question = row['segments'][selected[0]].get('question') or row.get('question') or ''
            item = dict(id=case_id, sample_id=row['id'], dataset=dataset, person_id=person, split='train',
                        question=question, observation=observation, memory_before=empty_memory())
            item['source_fingerprint'] = digest(item)
            inputs.append(item)
            refs = annotation.get('trait_labels', {})
            positive = [t for t,name in TRAITS.items() if refs.get(name,{}).get('is_evidence') is True
                        and refs[name].get('direction') in ('support_high','support_low','mixed')]
            references.append(dict(id=case_id, source_fingerprint=item['source_fingerprint'], legacy_trait_labels=refs,
                legacy_question=annotation.get('question',''), legacy_times=annotation.get('time_a'),
                annotation_origin='unknown_per_item; aggregate audit cannot recover individual decisions',
                proposed_stratum='legacy_directional' if positive else 'legacy_no_clear_unverified',
                positive_traits=positive, approved=False))
            counts[f'{dataset}/matched'] += 1
    by_id = {r['id']:r for r in references}
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in inputs:
        ref = by_id[item['id']]
        groups[item['dataset'],ref['proposed_stratum']].append(item)
    pilot = []
    for group, items in sorted(groups.items()):
        rng.shuffle(items)
        # Cover legacy trait directions where possible without exposing them to teacher.
        picked, covered = [], set()
        if group[1] == 'legacy_directional':
            for item in items:
                traits = set(by_id[item['id']]['positive_traits'])
                if traits-covered and len(picked) < pilot_per_group:
                    picked.append(item); covered.update(traits)
        ids = {r['id'] for r in picked}
        picked.extend(item for item in items if item['id'] not in ids)
        pilot.extend(picked[:pilot_per_group])
    for split in ('train','dev','test'):
        path = manifest_dir/f'{split}.jsonl'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_jsonl(output/'source_inputs.jsonl', inputs)
    write_jsonl(output/'legacy_references.jsonl', references)
    write_jsonl(output/'rejected.jsonl', rejected)
    write_jsonl(output/'pilot_inputs.jsonl', pilot)
    report = dict(counts=dict(counts), matched_total=len(inputs), rejected_total=len(rejected), pilot_size=len(pilot),
        pilot_groups=dict(Counter(f"{r['dataset']}/{by_id[r['id']]['proposed_stratum']}" for r in pilot)), source_hashes=hashes,
        all_inputs_train=True, automatically_approved=0, alignment_verified=False,
        scope='Text-only 观察原文保存 draft seeds. Legacy no_clear and provenance are unverified. No full trajectories or training.',
        schema='Actor inputs are whitelisted separately from legacy references. ASR time is never a verified media citation.')
    write_json(output/'audit.json', report)
    return report


def source_fingerprint(item):
    return digest({k:v for k,v in item.items() if k != 'source_fingerprint'})


def draft(input_path, output, model, config, backend=None):
    from amp.agent import ActiveAgent
    from amp.backend import QwenBackend
    from amp.scoring import OfflineTraitScorer
    output = Path(output)
    if (output/'reviews.json').exists():
        raise ValueError('Refusing to overwrite existing review decisions; use a fresh output directory')
    backend = backend or QwenBackend(model, max_tokens=config['max_new_tokens'], temperature=0,
                                     generation_log_dir=output/'generations')
    agent = ActiveAgent(backend, OfflineTraitScorer(), None, dict(config))
    drafts = []
    for item in read_jsonl(input_path):
        if item['split'] != 'train' or source_fingerprint(item) != item['source_fingerprint']:
            raise ValueError('Teacher drafts require immutable train sources')
        observations = {'active_0':item['observation']}
        result = dict(id=item['id'], sample_id=item['sample_id'], dataset=item['dataset'], person_id=item['person_id'], split='train',
                      source_fingerprint=item['source_fingerprint'], model=model, source=item, status='rejected', approved=False)
        try:
            memory, ids, decision = agent.update(copy.deepcopy(item['memory_before']), observations, observations, 0, question=item['question'])
            result.update(status='source_valid_draft' if decision['memory_update_status']=='accepted' else 'invalid_observation_protocol', memory_after=memory, changed_ids=ids, decision=decision)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            result.update(error=str(error), failed_generations=getattr(error,'failures',[]))
        drafts.append(result)
        write_json(output/'drafts.json', drafts)
        print(f"{item['id']}: {result['status']}", flush=True)
    reviews = [dict(id=r['id'], source_fingerprint=r['source_fingerprint'], approved=False, reviewer='',
                    source_supported=None, trait_mapping_supported=None, independent_atoms_checked=None,
                    empty_response_supported=None, corrected_target=None, notes='') for r in drafts]
    review_path = output/'reviews.json'
    if review_path.exists():
        raise ValueError('Refusing to overwrite existing review decisions; use a fresh output directory')
    write_json(review_path, reviews)
    report = dict(model=model, samples=len(drafts), source_valid=sum(r['status']=='source_valid_draft' for r in drafts),
                  drafts_with_findings=sum(bool(r.get('decision',{}).get('observation_note')) for r in drafts), automatically_approved=0,
                  semantic_quality_verified=False, external_api_calls=0, training_updates=0, modalities=['text'])
    write_json(output/'summary.json', report)
    return report


def review_packet(drafts_path, references_path, output):
    references = {r['id']:r for r in read_jsonl(references_path)}
    lines = ['# Memory 教师草稿审核包', '',
             '所有样本均未批准。旧标注只是参考，不是新方法的金标准。仅包含文本 观察原文保存 草稿。',
             '对每条检查：原文是否支持行为、Trait 是否合理、是否拆分独立行为、是否误用生活情境与访谈情境。',
             '检查原始响应是否描述了实际来源。需要修改时在 reviews.json 的 corrected_target 填写审核后的观察文本。',
             'approved 保持 false，直到完成实际审核；填写 reviewer、各项结论和修改原因。此文件不会自动导出训练数据。', '']
    for item in read_json(drafts_path):
        ref = references[item['id']]
        if ref['source_fingerprint'] != item['source_fingerprint']:
            raise ValueError('Reference/source fingerprint mismatch')
        source = item['source']
        lines += [f"## {item['id']}", '', f"状态：{item['status']}；人物：{item['person_id']}；Train；文本时间未验证。", '',
                  f"问题：{source['question']}", '', '实际来源：', '```json',
                  json.dumps([dict(segment_id=s['segment_id'],text=s['text']) for s in source['observation']['segments']],ensure_ascii=False,indent=2), '```', '',
                  '旧方向标签（仅供审核，未传给教师）：', '```json',json.dumps(ref['legacy_trait_labels'],ensure_ascii=False,indent=2),'```', '',
                  '教师原始输出：', '```json',item.get('decision',{}).get('text',item.get('error','No accepted output')),'```', '']
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--legacy-root', default=str(ROOT.parent/'WWW/TraitEvidenceGraphAgent'))
    p.add_argument('--manifest-dir', default=str(ROOT/'runs/data'))
    p.add_argument('--output', default=str(ROOT/'runs/memory_teacher_seed'))
    p.add_argument('--pilot-per-group', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('draft')
    p.add_argument('--inputs', default=str(ROOT/'runs/memory_teacher_seed/pilot_inputs.jsonl'))
    p.add_argument('--output', default=str(ROOT/'runs/memory_teacher_seed/7B'))
    p.add_argument('--model', default=str(ROOT.parent/'models/Qwen2.5-Omni-7B'))
    p.add_argument('--config', default=str(ROOT/'configs/local_omni3b.json'))
    p = sub.add_parser('packet')
    p.add_argument('--drafts',default=str(ROOT/'runs/memory_teacher_seed/7B/drafts.json'))
    p.add_argument('--references',default=str(ROOT/'runs/memory_teacher_seed/legacy_references.jsonl'))
    p.add_argument('--output',default=str(ROOT/'runs/memory_teacher_seed/review_packet.md'))
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.pilot_per_group < 1: parser.error('pilot-per-group must be positive')
        report = prepare(args.legacy_root, args.manifest_dir, args.output, args.pilot_per_group, args.seed)
    elif args.command == 'draft':
        report = draft(args.inputs,args.output,args.model,read_json(args.config))
    else:
        review_packet(args.drafts,args.references,args.output)
        print(f'Unapproved review packet: {args.output}')
        return
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
