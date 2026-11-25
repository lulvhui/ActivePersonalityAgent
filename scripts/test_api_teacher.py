"""Run one hosted teacher diagnostic through the existing three-stage agent."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amp.agent import ActiveAgent
from amp.backend import QwenAPIBackend
from amp.common import TRAITS, read_json, read_jsonl, write_json, write_jsonl
from amp.media import MediaTools
from amp.scoring import OfflineTraitScorer



def diagnostic_row(row):
    result={k:row[k] for k in ('id','dataset','split','person_id','video','segments','question') if k in row}
    if 'turns' in row:
        result['turns']=[{k:turn[k] for k in ('question','time') if k in turn} for turn in row['turns']]
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default=str(ROOT/'runs/omniagent_expanded_20261117/data/manifest.jsonl'))
    parser.add_argument('--sample-id', default='RecruitView:1703')
    parser.add_argument('--config', default=str(ROOT/'configs/local_omni3b.json'))
    parser.add_argument('--output', default=str(ROOT/'runs/qwen38_api_teacher'))
    parser.add_argument('--base-url', default=os.getenv('DASHSCOPE_BASE_URL'))
    parser.add_argument('--reasoning-effort', choices=['none', 'low', 'medium', 'xhigh'], default='medium')
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--max-steps', type=int, help='Optional diagnostic observation budget; default keeps the existing config')
    parser.add_argument('--allow-external', action='store_true', help='Confirm permission to send the selected interview media to Aliyun')
    parser.add_argument('--prepare-only', action='store_true', help='Save a local run plan; no API, model or media upload')
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or (args.max_steps is not None and args.max_steps < 0):
        parser.error('Invalid generation or observation budget')
    selected = [r for r in read_jsonl(args.manifest) if r['id'] == args.sample_id]
    if len(selected) != 1:
        parser.error('Select exactly one existing manifest sample')
    # Exclude raw labels, prior annotations and all prior-model answers.
    row = diagnostic_row(selected[0])
    output = Path(args.output).resolve()
    if (output/'trajectories.json').exists() or (output/'api_calls.json').exists():
        parser.error('Choose a new output directory to preserve the previous attempt')
    cfg = read_json(args.config)
    cfg.update(model='qwen3.8-omni-flash', scorer='offline',
               max_new_tokens=args.max_new_tokens,
               generation_log_dir=str(output/'generations'))
    if args.max_steps is not None:
        cfg['max_steps'] = args.max_steps
    write_json(output/'config.json', cfg)
    write_json(output/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'amp').glob('*.py')})
    write_jsonl(output/'manifest.jsonl', [row])
    plan = dict(sample_id=row['id'], split=row['split'], model=cfg['model'], base_url=args.base_url,
                reasoning_effort=args.reasoning_effort, max_steps=cfg['max_steps'],
                max_validation_attempts=cfg.get('max_validation_attempts',1),
                credential_present=bool(os.getenv('DASHSCOPE_API_KEY')), external_permission=args.allow_external,
                teacher_accepted=False, semantic_quality_verified=False, student_tokenization='not_performed',
                jev_api_calls=0, training=False, status='prepared_not_executed')
    write_json(output/'summary.json', plan)
    if args.prepare_only:
        print(f'Prepared only; no API calls: {output}')
        return
    if not args.base_url or not os.getenv('DASHSCOPE_API_KEY') or not args.allow_external:
        parser.error('Set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL, and confirm media permission with --allow-external. No API call was made.')
    media = MediaTools(cfg['cache'], cfg['max_clip_seconds'], cfg['video_fps'])
    media.require()
    backend = QwenAPIBackend(args.base_url, max_tokens=args.max_new_tokens,
                             reasoning_effort=args.reasoning_effort, generation_log_dir=cfg['generation_log_dir'],
                             allow_external=args.allow_external)
    agent = ActiveAgent(backend, OfflineTraitScorer(), media, cfg)
    trajectories, errors = [], []
    plan['status']='running'
    write_json(output/'summary.json',plan)
    try:
        trajectory = agent.run(row)
        trajectory.update(teacher_accepted=False, teacher_status='pending_source_and_semantic_review')
        trajectories.append(trajectory)
        steps = trajectory['steps']
        memory_decisions=[trajectory['decisions'][s['memory_decision_index']] for s in steps]
        plan.update(status=trajectory['run_status'], first_step_stop=not steps and trajectory['stopped'],
                    candidate_count=trajectory['initial']['candidate_count'], active_observations=len(steps),
                    active_seconds=trajectory['cost']['active_seconds'],
                    final_trait_coverage=sum(v['level'] is not None for v in trajectory['final'].values())/5,
                    memory_protocol=trajectory.get('memory_protocol'),memory_records=len(trajectory['memory'].get('observations',[])),
                    readable_observations=sum(bool(d.get('observation_note')) for d in memory_decisions),
                    stop_reason=trajectory['stop_reason'],final_status=trajectory.get('final_status'),
                    final_prediction_available=trajectory.get('final_prediction_available'),
                    child_candidates=sum(c['origin']=='observation' for c in trajectory['pool']),
                    validation_repairs=sum(len(d.get('failed_attempts',[])) for d in trajectory['decisions']),
                    candidate_parameter_repairs=sum(bool(d.get('proposal_repair')) for d in trajectory['decisions']),
                    invalid_followup_steps=[s['index'] for s in steps if s.get('followup_status') in ('invalid','partial')])
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        errors.append(dict(sample_id=row['id'], error=str(error), partial=agent.last_partial,
                           failed_generations=getattr(error, 'failures', [])))
        plan.update(status='failed', error=str(error))
    write_json(output/'trajectories.json', trajectories)
    write_json(output/'rejected.json', errors)
    write_json(output/'api_calls.json', backend.calls)
    usage = [c['usage'] for c in backend.calls if c.get('usage')]
    plan.update(api_calls=len(backend.calls), usage_complete=all(bool(c.get('usage')) for c in backend.calls),
                prompt_tokens=sum(u.get('prompt_tokens', 0) for u in usage),
                completion_tokens=sum(u.get('completion_tokens', 0) for u in usage),
                billing_cost=None, billing_note='Usage recorded; monetary billing has not been verified')
    write_json(output/'summary.json', plan)
    print(f"Teacher diagnostic {plan['status']}; {len(backend.calls)} API calls; {output}")
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
