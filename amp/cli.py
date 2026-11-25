import argparse
import importlib.util
import json
import random
import shutil
from pathlib import Path
from .common import read_json, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="V3 Active Multimodal Personality Agent")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]/"configs/default.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Build raw-only manifests with existing person splits")
    p.add_argument("--output", default="runs/data")
    p = sub.add_parser("doctor")
    p = sub.add_parser("smoke", help="Synthetic control-flow fixture; no media/model/API")
    p.add_argument("--output", default="runs/smoke.json")
    for name in ["infer", "teacher", "rl"]:
        p = sub.add_parser(name)
        p.add_argument("--manifest", default="runs/data/train.jsonl")
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--limit", type=int, default=1)
        p.add_argument("--dataset", choices=["IELTS", "RecruitView"])
        p.add_argument("--sample-id")
        p.add_argument("--perception", choices=["full", "static", "generic", "active_no_context", "fixed_context"], default="full")
        p.add_argument("--budget-seconds", type=float)
        p.add_argument("--max-steps", type=int)
        p.add_argument("--adapter")
        p.add_argument("--scorer", choices=["offline", "jev"], default=None)
        p.add_argument("--allow-external", action="store_true", help="Confirm data-use permission and enable minimized Jev requests")
        p.add_argument("--judge-model", help="Separate Qwen2.5-Omni checkpoint; must differ from policy")
        if name == "teacher":
            p.add_argument("--best-of", type=int, default=4)
        if name == "rl":
            p.add_argument("--group-size", type=int, default=4)
            p.add_argument("--updates", type=int, default=1)
            p.add_argument("--variant", choices=["grpo", "taura_style", "teca"], default="teca")
            p.add_argument("--enable-trait-reward", action="store_true")
            p.add_argument("--calibration-report")
    p = sub.add_parser("sft")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--adapter")
    p = sub.add_parser("export-sft")
    p.add_argument("--trajectories", required=True, help="JSON list of accepted teacher trajectories")
    p.add_argument("--output", default="runs/sft.jsonl")
    p.add_argument('--tokenizer',help='Local student tokenizer required for API teacher outputs; never trains on API token IDs')
    p = sub.add_parser('review-teachers',help='Re-filter saved teacher candidates without model or API calls')
    p.add_argument('--candidates',required=True,help='Saved JSON trajectory list or candidates directory')
    p.add_argument('--manifest',default='runs/data/train.jsonl')
    p.add_argument('--output',default='runs/reviewed_teachers')
    p = sub.add_parser("evaluate")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--bins", default="runs/data/label_bins.json")
    p.add_argument("--output", default="runs/metrics.json")
    p = sub.add_parser("validate-jev")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--dev", default="runs/data/dev.jsonl")
    p.add_argument("--train", default="runs/data/train.jsonl")
    p.add_argument("--output", default="runs/jev_validation.json")
    p = sub.add_parser("credit-report")
    p.add_argument("--trajectories", required=True, help="JSON rollout group recorded by RL")
    p.add_argument("--output", default="runs/credit_diagnostics.json")
    p = sub.add_parser("rescore-jev", help="Explicit independent uncached repeats of FIXED Dev evidence")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output", default="runs/jev_repeats.json")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--allow-external", action="store_true", required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    if hasattr(args,'scorer'):
        args.scorer=args.scorer or cfg.get('scorer','offline')
        cfg['scorer']=args.scorer
    random.seed(cfg["seed"])
    if args.command == "prepare":
        from .data import build
        report = build(cfg["data_root"], args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "doctor":
        report = dict(python_modules={m: importlib.util.find_spec(m) is not None for m in
            ["torch", "transformers", "peft", "qwen_omni_utils", "typesafe_sdk"]},
            tools={m: shutil.which(m) for m in ["ffmpeg", "ffprobe", "nvidia-smi"]},
            model_exists=Path(cfg["model"]).exists(), data_exists=Path(cfg["data_root"]).exists(),
            jev_live_default=False, jev_model=cfg["jev_model"])
        print(json.dumps(report, indent=2))
        return
    if args.command == "smoke":
        from .smoke import smoke
        result = smoke(cfg)
        write_json(args.output, result)
        print(f"Synthetic smoke passed; NOT a model prediction: {args.output}")
        return
    if args.command == "export-sft":
        from .train import export_sft
        tokenizer=None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        print(export_sft(read_json(args.trajectories), args.output,tokenizer=tokenizer))
        return
    if args.command == 'review-teachers':
        from .train import filter_best
        rows={r['id']:r for r in read_jsonl(args.manifest)}
        source=Path(args.candidates)
        paths=sorted(source.glob('*.json')) if source.is_dir() else [source]
        groups={}
        for path in paths:
            for trajectory in read_json(path):
                row=rows.get(trajectory['sample_id'])
                if not row or row['split']!='train' or trajectory['split']!='train' or row['person_id']!=trajectory['person_id']:
                    raise ValueError('Reviewed teachers must match the original train sample and person')
                trajectory.pop('teacher_accepted',None)
                groups.setdefault(row['id'],[]).append(trajectory)
        accepted,rejected=[],[]
        for sample_id,group in groups.items():
            best,errors=filter_best(group,rows[sample_id]['labels'])
            rejected.extend(errors)
            if best:
                best['teacher_accepted']=True
                accepted.append(best)
        write_json(Path(args.output)/'trajectories.json',accepted)
        write_json(Path(args.output)/'rejected.json',rejected)
        print(f'Reviewed saved candidates: {len(accepted)} accepted, {len(rejected)} rejected; no generation')
        return
    if args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(read_json(args.trajectories), list(read_jsonl(args.manifest)), read_json(args.bins))
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "validate-jev":
        from .evaluate import validate_jev
        result = validate_jev(read_json(args.trajectories), list(read_jsonl(args.dev)), list(read_jsonl(args.train)), cfg["jev_model"])
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "credit-report":
        from .evaluate import credit_diagnostics
        result = credit_diagnostics(read_json(args.trajectories))
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "rescore-jev":
        from copy import deepcopy
        from .scoring import build_scorer
        scorer = build_scorer(cfg,'jev',args.allow_external)
        output = []
        for trajectory in read_json(args.trajectories):
            if trajectory["split"] != "dev" or trajectory.get("offline"):
                raise ValueError("Rescore requires real Dev trajectories")
            for _ in range(args.repeats):
                new = deepcopy(trajectory)
                new["score"] = scorer.score(new["memory"], refresh=True)
                output.append(new)
                write_json(args.output, output)
        return
    from .backend import QwenBackend
    if args.command == "sft":
        from .train import sft
        backend = QwenBackend(cfg["model"], train=True, adapter=args.adapter)
        sft(backend, args.data, args.output, cfg, args.epochs)
        return
    cfg["perception"] = args.perception
    if args.budget_seconds is not None:
        cfg["max_seconds"] = args.budget_seconds
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if (cfg["max_seconds"] is not None and cfg["max_seconds"] <= 0) or cfg["max_steps"] < 0:
        raise ValueError("Invalid observation budget")
    from .agent import ActiveAgent
    from .media import MediaTools
    from .scoring import build_scorer
    from .judge import MultimodalJudge
    from .train import filter_best, rl
    rows = [r for r in read_jsonl(args.manifest) if (not args.dataset or r["dataset"] == args.dataset)
            and (not args.sample_id or r["id"] == args.sample_id)]
    if args.limit < 1 or not rows:
        raise ValueError("No selected samples or invalid --limit")
    rows = rows[:args.limit]
    if args.command in ["teacher", "rl"] and any(r["split"] != "train" for r in rows):
        raise ValueError("Teacher generation and RL require train split")
    if args.command in ["teacher", "rl"] and not args.judge_model:
        raise ValueError("Teacher/RL requires --judge-model for independent multimodal evidence checks")
    if args.judge_model and Path(args.judge_model).resolve() == Path(cfg["model"]).resolve():
        raise ValueError("Judge must use a checkpoint different from the policy")
    if args.command == "rl" and (not args.adapter or args.group_size < 2):
        raise ValueError("RL requires --adapter and --group-size >=2")
    if args.command == 'rl' and args.enable_trait_reward:
        raise ValueError('Observation-record Memory has no verified local trait attribution; Jev is auxiliary only')
    if args.command in ('teacher','rl'):
        from .common import TRAITS
        if any(set(r.get('labels',{}))!=set(TRAITS) for r in rows):
            raise ValueError('Training requires five labels for the actual sample unit; unresolved person-level inputs cannot use answer labels')
    scorer=build_scorer(cfg,args.scorer,args.allow_external)
    media = MediaTools(cfg["cache"], cfg["max_clip_seconds"], cfg["video_fps"])
    media.require()
    if args.command=='infer':
        selected_people={r['person_id'] for r in rows if r['dataset']=='RecruitView' and r.get('task_unit')!='person_interview'}
        if selected_people:
            from .data import recruitview_people
            grouped={r['person_id']:r for r in recruitview_people(cfg['data_root'],selected_people,Path(cfg['cache'])/'recruitview_people',media)}
            expanded=[];seen=set()
            for row in rows:
                if row['dataset']=='RecruitView' and row.get('task_unit')!='person_interview':
                    if row['person_id'] in seen:continue
                    seen.add(row['person_id']);row=grouped[row['person_id']]
                expanded.append(row)
            rows=expanded
    if args.command == "rl":
        cfg["temperature"] = 1.0  # On-policy sampling for raw-logit GRPO likelihoods.
    backend = QwenBackend(cfg["model"], cfg["max_new_tokens"], cfg["temperature"], train=args.command == "rl", adapter=args.adapter,
                          generation_log_dir=cfg.get("generation_log_dir"))
    backend.torch.manual_seed(cfg["seed"])
    agent = ActiveAgent(backend, scorer, media, cfg)
    judge = MultimodalJudge(QwenBackend(args.judge_model, cfg["max_new_tokens"], 0)) if args.judge_model else None
    if args.command == "rl":
        rl(backend, agent, judge, rows, args.output, cfg, args.adapter, args.group_size,
           args.variant, args.enable_trait_reward, args.calibration_report, args.updates)
        return
    output, errors = [], []
    n = args.best_of if args.command == "teacher" else 1
    if n < 1:
        raise ValueError("best-of must be positive")
    for row in rows:
        group = []
        for attempt in range(n):
            try:
                trajectory = agent.run(row)
                if judge:
                    trajectory["judge"] = judge.assess(trajectory)
                group.append(trajectory)
            except (ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
                errors.append(dict(sample_id=row["id"], attempt=attempt, error=str(error),
                                   partial=getattr(agent, "last_partial", None),
                                   failed_generations=getattr(error, "failures", [])))
        if args.command == "teacher":
            write_json(Path(args.output)/f"candidates/{row['id'].replace(':','_')}.json", group)
            best, rejected = filter_best(group, row["labels"])
            errors.extend(rejected)
            if best:
                best["teacher_accepted"] = True
                output.append(best)
        else:
            output.extend(group)
        write_json(Path(args.output)/"trajectories.json", output)
        write_json(Path(args.output)/"rejected.json", errors)
    print(f"Saved {len(output)} trajectories, {len(errors)} rejected/error records to {args.output}")


if __name__ == "__main__":
    main()
