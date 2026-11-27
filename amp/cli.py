import argparse
import importlib.util
import json
import random
import shutil
from pathlib import Path
from .common import read_json, read_jsonl, write_json, write_jsonl


def curate(args, cfg):
    """Teacher curation commands; see amp/curation.py for the review contract."""
    from . import curation
    trajectories = read_json(args.trajectories)
    output = Path(args.output) if getattr(args, "output", None) else None
    if args.command == "curate-prescreen":
        if (output/"reviews.json").exists():
            raise ValueError("Refusing to overwrite existing review decisions; use a fresh output directory")
        checks = None
        if args.judge_model:
            from .backend import QwenBackend
            from .judge import MultimodalJudge
            if Path(args.judge_model).resolve() == Path(cfg["model"]).resolve():
                raise ValueError("Judge must use a checkpoint different from the student")
            judge = MultimodalJudge(QwenBackend(args.judge_model, cfg["max_new_tokens"], 0))
            checks = {t["sample_id"]: judge.check_records(t) for t in trajectories}
        reviews, excluded = curation.prescreen(trajectories, checks, args.sample_rate, cfg["seed"])
        write_json(output/"judge_checks.json", checks or {})
        write_json(output/"excluded.json", excluded)
        write_json(output/"reviews.json", reviews)
        (output/"review_packet.md").write_text(curation.review_packet(trajectories, reviews), encoding="utf-8")
        priorities = {k: sum(r["priority"] == k for r in reviews) for k in ("must", "sample", "optional")}
        print(json.dumps(dict(items=len(reviews), excluded=len(excluded), **priorities, judge=bool(checks)), indent=2))
        return
    reviews = read_json(args.reviews)
    errors, pending = curation.check_reviews(trajectories, reviews)
    if args.command == "curate-check":
        report = dict(errors=errors, pending=pending, agreement=curation.agreement(reviews))
        if output:
            write_json(output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if errors or pending:
            raise SystemExit(1)
        return
    if errors or pending:
        raise ValueError(f"Reviews are incomplete: {len(errors)} errors, {len(pending)} required items unreviewed; run curate-check")
    if args.command == "curate-final":
        from .agent import ActiveAgent
        if args.api:
            from .backend import QwenAPIBackend
            backend = QwenAPIBackend(args.base_url, max_tokens=cfg["max_new_tokens"], reasoning_effort=args.reasoning_effort,
                                     allow_external=args.allow_external)
        else:
            from .backend import QwenBackend
            if args.k > 1 and args.temperature <= 0:
                raise ValueError("Resampling K>1 Finals locally needs temperature > 0")
            backend = QwenBackend(cfg["model"], cfg["max_new_tokens"], args.temperature)
        checks = read_json(args.judge_checks) if args.judge_checks else {}
        agent = ActiveAgent(backend, None, None, cfg)
        resampled = []
        for trajectory in trajectories:
            memory, _, dropped = curation.reviewed_memory(trajectory, reviews, checks.get(trajectory["sample_id"]))
            item = curation.resample_final(agent, trajectory, memory, args.k)
            item["dropped_records"] = dropped
            resampled.append(item)
            write_json(output/"final_candidates.json", resampled)
        if (output/"final_reviews.json").exists():
            raise ValueError("Refusing to overwrite existing Final choices; use a fresh output directory")
        write_json(output/"final_reviews.json", [curation.final_review_template(r) for r in resampled])
        (output/"final_packet.md").write_text(curation.final_packet(resampled), encoding="utf-8")
        print(f"Resampled {args.k} Finals for {len(resampled)} samples: {output}")
        return
    from transformers import AutoTokenizer
    labels = {r["id"]: r.get("labels") for r in read_jsonl(args.manifest) if r["split"] == "train"}
    perception, final, report = curation.export_curated(trajectories, reviews, read_json(args.final_candidates), read_json(args.final_reviews),
                                                        labels, AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True),
                                                        args.max_final_error)
    write_jsonl(output/"perception.jsonl", perception)
    write_jsonl(output/"final.jsonl", final)
    write_json(output/"export_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "dropped"}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="V3 Active Multimodal Personality Agent")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]/"configs/default.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Build raw-only manifests with existing person splits")
    p.add_argument("--output", default="runs/data")
    p = sub.add_parser("doctor")
    p = sub.add_parser("smoke", help="Synthetic control-flow fixture; no media/model/API")
    p.add_argument("--output", default="runs/smoke.json")
    for name in ["infer", "rl"]:
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
        if name == "rl":
            p.add_argument("--group-size", type=int, default=4)
            p.add_argument("--updates", type=int, default=1)
            p.add_argument("--variant", choices=["grpo", "taura_style", "teca"], default="teca")
    p = sub.add_parser("sft")
    p.add_argument("--data", required=True, nargs="+", help="Curated perception.jsonl and/or final.jsonl")
    p.add_argument("--output", default="runs/sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--adapter")
    p = sub.add_parser("curate-prescreen", help="Judge/rule pre-screen of teacher perception outputs and a human review template")
    p.add_argument("--trajectories", required=True, help="Teacher trajectories.json (train split)")
    p.add_argument("--output", required=True)
    p.add_argument("--judge-model", help="Local Qwen2.5-Omni checkpoint that differs from the student; omit to review every record")
    p.add_argument("--sample-rate", type=float, default=.2, help="Share of unflagged, Judge-supported items to review for Judge calibration")
    p = sub.add_parser("curate-check", help="Validate human reviews and report Judge/rule agreement")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--reviews", required=True)
    p.add_argument("--output")
    p = sub.add_parser("curate-final", help="Resample Final K times on the reviewed Memory for a label-blind human choice")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--reviews", required=True)
    p.add_argument("--judge-checks", help="judge_checks.json from curate-prescreen")
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=.7, help="Local model sampling temperature")
    p.add_argument("--api", action="store_true", help="Use the hosted teacher instead of the local model")
    p.add_argument("--base-url")
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "xhigh"], default="medium")
    p.add_argument("--allow-external", action="store_true")
    p = sub.add_parser("curate-export", help="Export reviewed perception rows and label-gated Final rows for SFT")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--reviews", required=True)
    p.add_argument("--final-candidates", required=True)
    p.add_argument("--final-reviews", required=True)
    p.add_argument("--manifest", default="runs/data/train.jsonl")
    p.add_argument("--tokenizer", required=True, help="Local student tokenizer; API token IDs are never trained on")
    p.add_argument("--max-final-error", type=float, default=.25, help="Mean ordinal error / 4 allowed for the chosen Final")
    p.add_argument("--output", required=True)
    p = sub.add_parser("rl-final", help="Stage 3: GRPO on Final only with fixed Memory and per-trait token advantages")
    p.add_argument("--rollouts", required=True, help="Student train trajectories whose Memory stays fixed")
    p.add_argument("--manifest", default="runs/data/train.jsonl")
    p.add_argument("--adapter", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--updates", type=int, default=1)
    p.add_argument("--output", default="runs/rl_final")
    p = sub.add_parser("evaluate")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--bins", default="runs/data/label_bins.json")
    p.add_argument("--train-manifest", help="Train manifest for the majority-level baseline")
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
    if args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(read_json(args.trajectories), list(read_jsonl(args.manifest)), read_json(args.bins),
                          list(read_jsonl(args.train_manifest)) if args.train_manifest else None)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command in ("curate-prescreen", "curate-check", "curate-final", "curate-export"):
        curate(args, cfg)
        return
    if args.command == "rl-final":
        from .agent import ActiveAgent
        from .judge import MultimodalJudge
        from .backend import QwenBackend
        from .train import rl_final
        if Path(args.judge_model).resolve() == Path(cfg["model"]).resolve():
            raise ValueError("Judge must use a checkpoint different from the policy")
        labels = {r["id"]: r.get("labels") for r in read_jsonl(args.manifest)}
        items = [(t, labels.get(t["sample_id"])) for t in read_json(args.rollouts)]
        cfg["temperature"] = 1.0  # On-policy sampling for GRPO likelihoods.
        backend = QwenBackend(cfg["model"], cfg["max_new_tokens"], 1.0, train=True, adapter=args.adapter,
                              generation_log_dir=cfg.get("generation_log_dir"), profiling=cfg.get('profiling', False))
        backend.torch.manual_seed(cfg["seed"])
        judge = MultimodalJudge(QwenBackend(args.judge_model, cfg["max_new_tokens"], 0))
        rl_final(backend, ActiveAgent(backend, None, None, cfg), judge, items, args.output, cfg, args.adapter, args.group_size, args.updates)
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
    from .train import rl
    rows = [r for r in read_jsonl(args.manifest) if (not args.dataset or r["dataset"] == args.dataset)
            and (not args.sample_id or r["id"] == args.sample_id)]
    if args.limit < 1 or not rows:
        raise ValueError("No selected samples or invalid --limit")
    rows = rows[:args.limit]
    if args.command == "rl" and any(r["split"] != "train" for r in rows):
        raise ValueError("RL requires train split")
    if args.command == "rl" and not args.judge_model:
        raise ValueError("RL requires --judge-model for independent multimodal evidence checks")
    if args.judge_model and Path(args.judge_model).resolve() == Path(cfg["model"]).resolve():
        raise ValueError("Judge must use a checkpoint different from the policy")
    if args.command == "rl" and (not args.adapter or args.group_size < 2):
        raise ValueError("RL requires --adapter and --group-size >=2")
    if args.command == 'rl':
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
                          generation_log_dir=cfg.get("generation_log_dir"),profiling=cfg.get('profiling',False))
    backend.torch.manual_seed(cfg["seed"])
    agent = ActiveAgent(backend, scorer, media, cfg)
    judge = MultimodalJudge(QwenBackend(args.judge_model, cfg["max_new_tokens"], 0)) if args.judge_model else None
    if args.command == "rl":
        rl(backend, agent, judge, rows, args.output, cfg, args.adapter, args.group_size,
           args.variant, args.updates)
        return
    output, errors = [], []
    for row in rows:
        try:
            trajectory = agent.run(row)
            if judge:
                trajectory["judge"] = judge.assess(trajectory)
            output.append(trajectory)
        except (ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
            errors.append(dict(sample_id=row["id"], error=str(error),
                               partial=getattr(agent, "last_partial", None),
                               failed_generations=getattr(error, "failures", [])))
        write_json(Path(args.output)/"trajectories.json", output)
        write_json(Path(args.output)/"rejected.json", errors)
    print(f"Saved {len(output)} trajectories, {len(errors)} rejected/error records to {args.output}")


if __name__ == "__main__":
    main()
