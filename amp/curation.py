"""Teacher data curation for SFT: automatic pre-screen, human review, Final resampling, split export.

Perception targets (coarse scan and Observe outputs) are judged only for fidelity to
their source; labels never enter that review. Final targets are resampled on the
reviewed Memory, chosen by a reviewer who does not see labels, and only then gated by
labels. Rule flags and Judge verdicts prioritize review; they are not verdicts.
"""
import random
import re
from collections import Counter
from copy import deepcopy
from .common import TRAITS, digest, parse_json
from .perception import decode_observation
from .state import memory_records, record_source
from .teca import final_trait_loss

RULES = {
    'numeric_measurement': r'\b\d+(?:\.\d+)?\s?(?:Hz|kHz|ms|milliseconds?|semitones?|dB|%)(?![A-Za-z])|\bF0\b',
    'inferred_motive': r'\b(?:genuine(?:ly)?|authentic(?:ity|ally)?|performed|performative|rehears\w*|decept\w*|deceiv\w*|fake|masking|impression management|true motive)',
}
# Missing-channel words: a video-only read cannot hear, an audio-only read cannot see.
MISSING_CHANNEL = {
    'video': r'\b(?:says?|said|saying|voice|tone|pitch|hear[sd]?|audible|laugh(?:s|ed|ing)?|filler|intonation)\b',
    'audio': r'\b(?:gaze|gestur\w*|smil\w*|nod(?:s|ded|ding)?|posture|eye contact|frown\w*|lean(?:s|ed|ing)?|facial)\b',
}
ACTIONS = ('keep', 'edit', 'drop')
REASON_CODES = ('measurement', 'motive', 'modality', 'unsupported', 'timing', 'speaker', 'other')
PERCEPTION_KINDS = ('candidate_scan', 'memory')


def rule_flags(text, modality=None):
    flags = [name for name, pattern in RULES.items() if re.search(pattern, text, re.I)]
    if modality in MISSING_CHANNEL and re.search(MISSING_CHANNEL[modality], text, re.I):
        flags.append('missing_channel')
    return flags


def decision_observation(trajectory, decision):
    """The source observation a decision responded to."""
    if decision['kind'] == 'candidate_scan':
        return trajectory['observations'][decision['source_observation_id']]
    return trajectory['observations'][next(s['observation_id'] for s in trajectory['steps'] if s['index'] == decision['step'])]


def decision_source(trajectory, decision):
    source = decision_observation(trajectory, decision)
    return dict(time=list(source['time']), modality=source['modality'])


def decision_fingerprint(trajectory, index):
    decision = trajectory['decisions'][index]
    return digest(dict(sample_id=trajectory['sample_id'], index=index, kind=decision['kind'], text=decision['text'],
                       source=decision_source(trajectory, decision)))


def perception_items(trajectory):
    """Reviewable perception decisions; failed, truncated or invalid outputs are excluded with a reason."""
    items, excluded = [], []
    for index, decision in enumerate(trajectory['decisions']):
        if decision['kind'] not in PERCEPTION_KINDS:
            continue
        reason = ('generation_failed' if decision.get('generation_failed') else 'truncated' if decision.get('truncated')
                  else 'invalid_candidates' if decision.get('candidate_output_valid') is False else None)
        if reason is None and decision['kind'] == 'memory' and not decision.get('proposal_repair') and decode_observation(decision['text'])['errors']:
            reason = 'invalid_observation_protocol'
        if reason:
            excluded.append(dict(sample_id=trajectory['sample_id'], decision_index=index, reason=reason))
            continue
        record_id = (decision.get('stored_ids') or [None])[0] if decision['kind'] == 'memory' and not decision.get('proposal_repair') else None
        items.append(dict(index=index, decision=decision, record_id=record_id, source=decision_source(trajectory, decision)))
    return items, excluded


def review_text(decision):
    if decision['kind'] == 'memory' and not decision.get('proposal_repair'):
        obj = parse_json(decision['text'])
        return '\n'.join(obj['observations'])
    return decision['text']


def prescreen(trajectories, judge_checks=None, sample_rate=.2, seed=42):
    """Review template: flagged or Judge-unsupported items must be reviewed, a sample of the rest checks the Judge."""
    if not 0 <= sample_rate <= 1:
        raise ValueError('sample_rate must be in [0,1]')
    rng = random.Random(seed)
    reviews, excluded = [], []
    for trajectory in trajectories:
        if trajectory.get('split') != 'train' or trajectory.get('offline'):
            raise ValueError('Curation uses genuine train trajectories only')
        checks = (judge_checks or {}).get(trajectory['sample_id'])
        items, skipped = perception_items(trajectory)
        excluded.extend(skipped)
        for item in items:
            decision, record_id = item['decision'], item['record_id']
            flags = rule_flags(review_text(decision), item['source']['modality'])
            check = checks.get(record_id) if checks is not None and record_id else None
            judged = check.get('supported') if check else None
            priority = 'must' if flags or judged is False or (record_id and checks is None) else None
            if priority is None:
                priority = 'sample' if rng.random() < sample_rate else 'optional'
            reviews.append(dict(key=f"{trajectory['sample_id']}#{item['index']}", sample_id=trajectory['sample_id'],
                                decision_index=item['index'], kind=decision['kind'], record_id=record_id,
                                proposal_repair=bool(decision.get('proposal_repair')), source=item['source'],
                                fingerprint=decision_fingerprint(trajectory, item['index']), priority=priority, flags=flags,
                                judge=dict(supported=judged, reason=check.get('reason')) if check else None,
                                action=None, corrected_text=None, reason_code=None, reason='', reviewer=''))
    return reviews, excluded


def validate_corrected(kind, text, proposal_repair=False):
    obj = parse_json(text)
    if kind == 'memory' and not proposal_repair:
        if decode_observation(text)['errors']:
            raise ValueError('corrected_text must be {"observations":[...],"candidates":[...]}')
    elif kind == 'memory':
        if not isinstance(obj, dict) or set(obj) != {'candidates'} or not isinstance(obj['candidates'], list):
            raise ValueError('corrected_text must be {"candidates":[...]}')
    else:
        if not isinstance(obj, dict) or set(obj) != {'summary', 'candidates'} or not isinstance(obj['summary'], str) or not obj['summary'].strip():
            raise ValueError('corrected_text must be {"summary":"...","candidates":[...]}')
    candidates = obj['candidates']
    if not isinstance(candidates, list) or any(not isinstance(c, dict) or set(c) != {'local_time', 'modality', 'hint', 'goal'} for c in candidates):
        raise ValueError('Each candidate needs local_time, modality, hint and goal')


def check_reviews(trajectories, reviews):
    """Return (errors, pending): structural problems and required items that are not reviewed yet."""
    by_id = {t['sample_id']: t for t in trajectories}
    errors, pending, seen = [], [], set()
    for review in reviews:
        key = review['key']
        if key in seen:
            errors.append(dict(key=key, error='duplicate review'))
            continue
        seen.add(key)
        trajectory = by_id.get(review['sample_id'])
        try:
            if trajectory is None or decision_fingerprint(trajectory, review['decision_index']) != review['fingerprint']:
                raise ValueError('stale review: the decision or its source changed')
        except (IndexError, KeyError, StopIteration):
            errors.append(dict(key=key, error='stale review: the decision or its source changed'))
            continue
        except ValueError as error:
            errors.append(dict(key=key, error=str(error)))
            continue
        action = review.get('action')
        if action is None:
            if review['priority'] in ('must', 'sample'):
                pending.append(key)
            continue
        if action not in ACTIONS:
            errors.append(dict(key=key, error=f'action must be one of {ACTIONS}'))
            continue
        if not str(review.get('reviewer') or '').strip():
            errors.append(dict(key=key, error='reviewer is required'))
        if action in ('edit', 'drop') and review.get('reason_code') not in REASON_CODES:
            errors.append(dict(key=key, error=f'reason_code must be one of {REASON_CODES}'))
        if action == 'edit':
            try:
                validate_corrected(review['kind'], review.get('corrected_text') or '', review.get('proposal_repair'))
            except (ValueError, TypeError) as error:
                errors.append(dict(key=key, error=f'corrected_text: {error}'))
        elif review.get('corrected_text'):
            errors.append(dict(key=key, error='corrected_text is only used with action edit'))
    return errors, pending


def accepted(review):
    """Accepted perception target: a human keep/edit; or, unsampled and unflagged, a record the Judge supported
    or a scan/parameter-repair output (the Judge does not check those)."""
    if review.get('action') in ('keep', 'edit'):
        return 'human_edit' if review['action'] == 'edit' else 'human_keep'
    if review.get('action') is None and review['priority'] == 'optional':
        if review.get('record_id'):
            return 'judge' if (review.get('judge') or {}).get('supported') is True else None
        return 'unflagged'
    return None


def agreement(reviews):
    """Judge and rule flags against human decisions on items that have both."""
    pairs = [(r['judge']['supported'], r['action'] == 'keep') for r in reviews
             if r.get('action') and r.get('judge') and isinstance(r['judge'].get('supported'), bool)]
    flagged = [r['action'] == 'keep' for r in reviews if r.get('action') and r['flags']]
    count = Counter((j, h) for j, h in pairs)
    judge_ok = count[True, True]+count[True, False]
    judge_bad = count[False, True]+count[False, False]
    return dict(
        judged_and_reviewed=len(pairs),
        agreement=(count[True, True]+count[False, False])/len(pairs) if pairs else None,
        judge_miss_rate=count[True, False]/judge_ok if judge_ok else None,
        judge_false_alarm_rate=count[False, True]/judge_bad if judge_bad else None,
        flagged_reviewed=len(flagged),
        flag_precision=sum(not keep for keep in flagged)/len(flagged) if flagged else None,
        reason_codes=dict(Counter(r['reason_code'] for r in reviews if r.get('action') in ('edit', 'drop'))),
        actions=dict(Counter(r.get('action') or 'unreviewed' for r in reviews)))


def reviewed_memory(trajectory, reviews, judge_checks=None):
    """Apply record reviews to Memory; return (memory, record checks usable by Final judging, dropped record IDs).

    Records without an accepted review (dropped, invalid or not supported) leave the Memory Final reads.
    """
    by_record = {r['record_id']: r for r in reviews if r['sample_id'] == trajectory['sample_id'] and r.get('record_id')}
    memory = deepcopy(trajectory['memory'])
    kept, checks, dropped = [], {}, []
    for record in memory_records(memory).values():
        review = by_record.get(record['id'])
        basis = accepted(review) if review else None
        if basis is None:
            dropped.append(record['id'])
            continue
        if review['action'] == 'edit':
            record['content'] = review['corrected_text']
            record['semantic_review'] = 'human_edited'
        elif basis == 'human_keep':
            record['semantic_review'] = 'human_kept'
        kept.append(record)
        source = record_source(record, trajectory['observations'])
        if basis == 'judge':
            checks[record['id']] = (judge_checks or {})[record['id']]
        else:
            checks[record['id']] = dict(supported=True, reason='Human review against the source',
                                        record_fingerprint=digest(record), source_fingerprint=digest(source))
    memory['observations'] = kept
    return memory, checks, dropped


def resample_final(agent, trajectory, memory, k):
    """K Final generations on fixed (reviewed) Memory, with the trajectory's own perception end and question."""
    if k < 1:
        raise ValueError('k must be positive')
    pending = sum(x.get('status') == 'pending' for x in trajectory.get('pool', []))
    perception_end = dict(reason=trajectory['stop_reason'], observations=len(trajectory['steps']), pending_candidates=pending)
    question = next((o.get('question', '') for o in trajectory['observations'].values()), '')
    candidates = []
    for index in range(k):
        result = agent.final_stage(memory, trajectory['observations'], question, perception_end, trajectory['score'], len(trajectory['steps']))
        candidates.append(dict(index=index, **{key: result[key] for key in ('final', 'final_explanation', 'final_status', 'final_error',
                                                                             'contextual', 'contextual_errors', 'decision')}))
    return dict(sample_id=trajectory['sample_id'], memory=memory, memory_fingerprint=digest(memory), candidates=candidates)


def final_review_template(resampled):
    return dict(sample_id=resampled['sample_id'], memory_fingerprint=resampled['memory_fingerprint'],
                candidates=len(resampled['candidates']), chosen=None, reviewer='', reason='', labels_seen=False)


def export_curated(trajectories, reviews, resampled, final_reviews, labels, tokenizer, max_error=.25):
    """Return (perception rows, Final rows, report). Labels gate only the chosen Final, never perception."""
    errors, pending = check_reviews(trajectories, reviews)
    if errors or pending:
        raise ValueError(f'Reviews are incomplete: {len(errors)} errors, {len(pending)} required items unreviewed')
    if tokenizer is None or not getattr(tokenizer, 'eos_token', None):
        raise ValueError('Export requires the student tokenizer and its EOS token')
    by_id = {t['sample_id']: t for t in trajectories}

    def row(decision, text, sample_id, edited, basis, key):
        result = {k: v for k, v in decision.items() if k not in ('failed_attempts', 'format_mode', 'choice_texts')} if edited else dict(decision)
        result.update(text=text, sample_id=sample_id, split='train', curation=dict(basis=basis, human_edited=edited, review_key=key))
        if edited or not decision.get('tokens'):
            result['tokens'] = tokenizer.encode(text+tokenizer.eos_token, add_special_tokens=False)
            result['tokenization'] = 'student_tokenizer'
        return result

    perception, final_rows, dropped = [], [], []
    for review in reviews:
        basis = accepted(review)
        if basis is None:
            dropped.append(dict(key=review['key'], reason=review.get('reason_code') or 'not_accepted'))
            continue
        decision = by_id[review['sample_id']]['decisions'][review['decision_index']]
        edited = review.get('action') == 'edit'
        perception.append(row(decision, review['corrected_text'] if edited else decision['text'], review['sample_id'], edited, basis, review['key']))
    resampled_by_id = {r['sample_id']: r for r in resampled}
    for choice in final_reviews:
        sample_id = choice['sample_id']
        item = resampled_by_id.get(sample_id)
        reason = None
        if item is None or item['memory_fingerprint'] != choice['memory_fingerprint']:
            reason = 'stale_or_missing_resample'
        elif choice.get('chosen') is None:
            reason = 'no_candidate_chosen'
        elif not str(choice.get('reviewer') or '').strip():
            reason = 'reviewer_missing'
        elif choice.get('labels_seen'):
            reason = 'chosen_with_labels_visible'
        if reason is None:
            candidate = item['candidates'][choice['chosen']]
            if candidate['final_status'] != 'generated' or candidate['contextual_errors']:
                reason = 'invalid_final'
            elif set(labels.get(sample_id) or {}) != set(TRAITS):
                reason = 'no_sample_labels'
            elif final_trait_loss(labels[sample_id], candidate['final']) > max_error:
                reason = 'label_gate'
        if reason:
            dropped.append(dict(key=f'{sample_id}#final', reason=reason))
            continue
        final_rows.append(row(candidate['decision'], candidate['decision']['text'], sample_id, False, 'human_choice_label_gated', f'{sample_id}#final'))
    report = dict(perception_rows=len(perception), final_rows=len(final_rows),
                  perception_by_basis=dict(Counter(r['curation']['basis'] for r in perception)),
                  dropped=dropped, agreement=agreement(reviews), max_final_error=max_error,
                  note='Perception rows are selected without labels; Final rows are reviewer-chosen without labels, then label-gated.')
    return perception, final_rows, report


def review_packet(trajectories, reviews):
    """Markdown for reviewers: one section per item, with source window, flags, Judge verdict and original text."""
    by_id = {t['sample_id']: t for t in trajectories}
    lines = ['# Perception review (labels are not shown)', '',
             'For each item, watch the source window and set action in reviews.json: keep, edit (corrected_text in the same JSON shape) or drop.',
             f'Edits and drops need reason_code ({", ".join(REASON_CODES)}) and a reviewer. Items marked must or sample are required.', '']
    for review in sorted(reviews, key=lambda r: (r['priority'] != 'must', r['priority'] != 'sample', r['key'])):
        trajectory = by_id[review['sample_id']]
        decision = trajectory['decisions'][review['decision_index']]
        media = [m.get('video') or m.get('audio') or m.get('image') for m in decision_observation(trajectory, decision).get('media', [])]
        lines += [f"## {review['key']}  [{review['priority']}]  {review['kind']}{' ' + review['record_id'] if review['record_id'] else ''}",
                  f"- source: {review['source']['modality']} {review['source']['time']} (recording seconds){'; media ' + ', '.join(map(str, media)) if media else ''}",
                  f"- flags: {', '.join(review['flags']) or 'none'}; judge: {review['judge'] if review['judge'] else 'not run'}", '',
                  '```', review_text(decision), '```', '']
    return '\n'.join(lines)


def final_packet(resampled):
    lines = ['# Final choice (labels are not shown; choose before seeing labels)', '',
             'Choose the candidate whose levels are best supported by cited findings, that does not use setting-required or absent behavior as strong evidence, '
             'and that returns null when support is missing. Set chosen, reviewer and reason in final_reviews.json.', '']
    for item in resampled:
        lines += [f"## {item['sample_id']}", '']
        for candidate in item['candidates']:
            levels = ' '.join(f"{t}={candidate['final'][t]['level'] or 'null'}" for t in TRAITS)
            lines += [f"### candidate {candidate['index']}  {levels}  ({candidate['final_status']})", '',
                      candidate['final_explanation'] or '', '']
            lines += [f"- [{c['trait']}/{c['context_type']}] {c['context']}: {c['expression']} ({', '.join(c['citations'])})" for c in candidate['contextual']]
            lines.append('')
    return '\n'.join(lines)


def level_distribution(finals):
    """Counts per trait level, for detecting collapse to one level."""
    return {t: dict(Counter(f[t]['level'] or 'null' for f in finals)) for t in TRAITS}

