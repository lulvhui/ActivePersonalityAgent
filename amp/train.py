import random
import re
from pathlib import Path
from .backend import action_token_mask, token_offsets
from .common import TRAITS, LEVELS, read_jsonl, write_json, parse_json, standardize
from .teca import advantages, clipped_grpo, reward, memory_outcomes


def sft(backend, paths, output, cfg, epochs=1):
    torch = backend.torch
    rows = [row for path in ([paths] if isinstance(paths, (str, Path)) else paths) for row in read_jsonl(path)]
    if not rows:
        raise ValueError("Empty SFT file")
    if any(r.get("split") != "train" or r.get("offline") or not r["tokens"] for r in rows):
        raise ValueError("SFT needs genuine train-only tokenized actor segments")
    if any(not isinstance(parse_json(r['text']),dict) for r in rows):
        raise ValueError('SFT targets must use the current actor JSON protocol')
    accumulation=cfg.get('grad_accum_steps',1)
    if isinstance(accumulation,bool) or not isinstance(accumulation,int) or accumulation<1:
        raise ValueError('grad_accum_steps must be a positive integer')
    optimizer = torch.optim.AdamW([p for p in backend.policy.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    backend.policy.train()  # Enable the checkpointing configured by the trainable backend.
    rng, log = random.Random(cfg["seed"]), []
    for epoch in range(epochs):
        rng.shuffle(rows)
        for start in range(0,len(rows),accumulation):
            group=rows[start:start+accumulation]
            optimizer.zero_grad(set_to_none=True)
            with backend.prepared_group():
                for decision in group:
                    logp, _ = backend.token_stats(decision,need_entropy=False)
                    loss = -logp.mean()
                    (loss/len(group)).backward()
                    log.append(dict(epoch=epoch, loss=float(loss.detach()), kind=decision["kind"]))
            torch.nn.utils.clip_grad_norm_(backend.policy.parameters(), 1.)
            optimizer.step()
        backend.save(str(Path(output)/f"epoch_{epoch+1}"))
    write_json(Path(output)/"training_log.json", log)


def load_reference(backend, adapter_path):
    """Load the SFT adapter as the frozen KL reference; the trained policy is the 'default' adapter."""
    from peft import load_peft_weights,set_peft_model_state_dict
    backend.policy.load_adapter(adapter_path, adapter_name="reference", is_trainable=False)
    # Restore after PEFT's BF16 creation / FP32 cast so the reference retains
    # the checkpoint's exact FP32 LoRA values rather than rounded values.
    set_peft_model_state_dict(backend.policy,load_peft_weights(adapter_path,device=str(backend.policy.device)),adapter_name='reference')
    backend.policy.set_adapter("default")
    freeze_reference(backend)


def freeze_reference(backend):
    for name, param in backend.policy.named_parameters():
        if ".reference." in name:
            param.requires_grad_(False)


def reference_logp(backend, decision):
    backend.policy.set_adapter("reference")
    with backend.torch.no_grad():
        ref, _ = backend.token_stats(decision,need_entropy=False)
    backend.policy.set_adapter("default")
    freeze_reference(backend)
    return ref.cpu().tolist()


def trait_rewards(labels, final):
    """Ordinal reward per trait; an abstention counts as the largest error."""
    return {t:(-1. if final[t]['level'] is None else -abs(labels[t]-LEVELS.index(final[t]['level']))/4) for t in TRAITS}


def group_final_advantages(rewards, faithfulness, beta):
    """Group-standardized advantage per trait, plus a shared Judge-faithfulness term, for Finals of one Memory."""
    trait={t:standardize([r[t] for r in rewards]) for t in TRAITS}
    shared=[beta*v for v in standardize(faithfulness)]
    return [dict(traits={t:trait[t][g] for t in TRAITS},shared=shared[g]) for g in range(len(rewards))]


def trait_level_masks(tokenizer, decision):
    """Per trait, the generated tokens that spell that trait's top-level level value in the Final JSON."""
    text, offsets = token_offsets(tokenizer, decision['tokens'])
    masks = {}
    for trait in TRAITS:
        match = re.search(r'"%s"\s*:\s*(null|"[a-z_]+")' % trait, text)
        if match is None:
            raise ValueError(f'Final has no top-level {trait} level')
        start, end = match.span(1)
        masks[trait] = [b > start and a < end for a, b in offsets]
    return masks


def final_token_advantages(masks, advantage, length):
    """A trait's level tokens get only that trait's advantage (its reward depends on them alone);
    other tokens (reasoning, contextual items, JSON syntax) get the mean trait advantage plus the shared term."""
    values=[sum(advantage['traits'].values())/len(TRAITS)+advantage['shared']]*length
    for trait, mask in masks.items():
        for index, inside in enumerate(mask):
            if inside:
                values[index]=advantage['traits'][trait]
    return values


def rl_final(backend, agent, judge, items, output, cfg, adapter_path, group_size=8, updates=1):
    """Stage 3: GRPO on Final only. Memory and observations stay fixed; each update samples a group of
    Finals for one train rollout and assigns per-trait token advantages."""
    if group_size < 2:
        raise ValueError("GRPO group_size must be >=2")
    if not adapter_path:
        raise ValueError("Final RL requires an SFT LoRA adapter; it is also the fixed reference policy")
    if not items:
        raise ValueError("Final RL needs train rollouts")
    for trajectory, labels in items:
        if trajectory.get('split')!='train' or trajectory.get('offline'):
            raise ValueError('Final RL uses genuine train rollouts only')
        if set(labels or {})!=set(TRAITS):
            raise ValueError(f"{trajectory['sample_id']}: Final RL needs five labels for the sample unit")
    torch = backend.torch
    load_reference(backend, adapter_path)
    optimizer = torch.optim.AdamW([p for p in backend.policy.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    tokenizer = backend.processor.tokenizer
    checks = {t['sample_id']:judge.check_records(t) for t,_ in items}  # once per fixed Memory
    output, logs = Path(output), []
    backend.policy.eval()
    for iteration in range(updates):
        trajectory, labels = items[iteration % len(items)]
        pending = sum(x.get('status') == 'pending' for x in trajectory.get('pool', []))
        perception_end = dict(reason=trajectory['stop_reason'], observations=len(trajectory['steps']), pending_candidates=pending)
        question = next((o.get('question', '') for o in trajectory['observations'].values()), '')
        group = [agent.final_stage(trajectory['memory'], trajectory['observations'], question, perception_end, trajectory['score'], len(trajectory['steps']))
                 for _ in range(group_size)]
        valid = [r['final_status']=='generated' and bool(r['decision'].get('tokens')) for r in group]
        rewards = [trait_rewards(labels, r['final']) if ok else {t:-1. for t in TRAITS} for r, ok in zip(group, valid)]
        faith = [judge.assess_final(dict(trajectory, final=r['final'], final_explanation=r['final_explanation'], contextual=r['contextual']),
                                    checks[trajectory['sample_id']])['faithfulness'] if ok else 0. for r, ok in zip(group, valid)]
        credit = group_final_advantages(rewards, faith, cfg['beta_ground'])
        optimizer.zero_grad(set_to_none=True)
        total_loss, used = 0., [g for g, ok in enumerate(valid) if ok]
        with backend.prepared_group():
            for g in used:
                d = group[g]['decision']
                with torch.no_grad():
                    d['old_logp'] = backend.token_stats(d, need_entropy=False)[0].cpu().tolist()
                d['ref_logp'] = reference_logp(backend, d)
                d['token_advantage'] = final_token_advantages(trait_level_masks(tokenizer, d), credit[g], len(d['tokens']))
            backend.policy.train()
            for g in used:
                d = group[g]['decision']
                logp, _ = backend.token_stats(d, need_entropy=False)
                old = torch.tensor(d['old_logp'], device=logp.device)
                ref = torch.tensor(d['ref_logp'], device=logp.device)
                loss = clipped_grpo(logp, old, ref, d['token_advantage'], cfg['clip_epsilon'], cfg['kl_beta'])/len(used)
                loss.backward()
                total_loss += float(loss.detach())
            if used:
                torch.nn.utils.clip_grad_norm_(backend.policy.parameters(), 1.)
                optimizer.step()
        backend.policy.eval()
        write_json(output/f"final_rollouts/update_{iteration:05d}.json",
                   [dict(sample_id=trajectory['sample_id'], final=r['final'], final_explanation=r['final_explanation'], contextual=r['contextual'],
                         rewards=rw, faithfulness=f, advantage=c) for r, rw, f, c in zip(group, rewards, faith, credit)])
        logs.append(dict(iteration=iteration, sample_id=trajectory['sample_id'], loss=total_loss, generated=len(used),
                         mean_trait_reward={t:sum(r[t] for r in rewards)/len(rewards) for t in TRAITS},
                         mean_faithfulness=sum(faith)/len(faith),
                         levels={t:{str(level):sum(r['final'][t]['level']==level for r in group) for level in [*LEVELS,None]} for t in TRAITS}))
        backend.policy.save_pretrained(str(output/"adapter"), selected_adapters=["default"])
        backend.processor.save_pretrained(str(output/"adapter"))
        write_json(output/"training_log.json", logs)


def rl(backend, agent, judge, rows, output, cfg, adapter_path, group_size=4,
       variant="teca", updates=1):
    if group_size < 2:
        raise ValueError("GRPO group_size must be >=2")
    if not adapter_path:
        raise ValueError("RL requires an SFT LoRA adapter; it is also the fixed reference policy")
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("RL must use nonempty train split")
    torch = backend.torch
    load_reference(backend, adapter_path)
    optimizer = torch.optim.AdamW([p for p in backend.policy.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    logs = []
    output = Path(output)
    backend.policy.eval()
    for iteration in range(updates):
        row = rows[iteration % len(rows)]
        trajectories = []
        for sample in range(group_size):
            trajectory = agent.run(row)
            if trajectory["offline"]:
                raise ValueError("RL cannot train on offline fixtures")
            if trajectory.get('run_status','complete')!='complete':
                raise ValueError('RL cannot optimize a rollout with rejected source operations')
            trajectory["judge"] = judge.assess(trajectory)
            trajectories.append(trajectory)
        with backend.prepared_group():
            # Snapshot old log probabilities and entropy before ANY optimizer update.
            for trajectory in trajectories:
                for d in trajectory["decisions"]:
                    with torch.no_grad():
                        logp, h = backend.token_stats(d,need_entropy=d['kind']=='action')
                        d["old_logp"] = logp.cpu().tolist()
                        if d["kind"] == "action":
                            mask = torch.tensor(action_token_mask(backend.processor.tokenizer, d), device=h.device)
                            d["action_entropy"] = float(h[mask].mean())
                        d["ref_logp"] = reference_logp(backend, d)
            for trajectory in trajectories:
                trajectory['memory_outcomes']=memory_outcomes(trajectory,row['labels'],cfg)
            credit = advantages(trajectories, row["labels"], cfg, variant)
            optimizer.zero_grad(set_to_none=True)
            total_decisions = sum(len(t["decisions"]) for t in trajectories)
            backend.policy.train()
            total_loss = 0.
            for trajectory, values in zip(trajectories, credit):
                for decision, value in zip(trajectory["decisions"], values):
                    logp, _ = backend.token_stats(decision,need_entropy=False)
                    old = torch.tensor(decision["old_logp"], device=logp.device)
                    ref = torch.tensor(decision["ref_logp"], device=logp.device)
                    loss = clipped_grpo(logp, old, ref, value, cfg["clip_epsilon"], cfg["kl_beta"])/total_decisions
                    loss.backward()
                    total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(backend.policy.parameters(), 1.)
            optimizer.step()
        backend.policy.eval()
        write_json(output/f"rollouts/update_{iteration:05d}.json", trajectories)
        logs.append(dict(iteration=iteration, loss=total_loss, variant=variant,
                         rewards=[reward(t, row["labels"], cfg) for t in trajectories],
                         memory_quality=[[x['F_memory'] for x in t['memory_outcomes']] for t in trajectories],
                         memory_utility='not_estimated_no_counterfactual'))
        backend.policy.save_pretrained(str(output/"adapter"), selected_adapters=["default"])
        backend.processor.save_pretrained(str(output/"adapter"))
        write_json(output/"training_log.json", logs)
