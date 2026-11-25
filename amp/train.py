import json
import random
from pathlib import Path
from .backend import action_token_mask
from .common import TRAITS,read_json, read_jsonl, write_json, write_jsonl,digest
from .state import memory_records
from .teca import advantages, clipped_grpo, final_trait_loss, reward, memory_outcomes,record_support
from .perception import decode_observation
from .common import parse_json


def filter_best(trajectories,labels,max_error=.25,min_ground=.8):
    accepted=[];rejected=[]
    for trajectory in trajectories:
        why=[]
        if trajectory.get('offline'):why.append('offline_fixture')
        if trajectory.get('run_status')!='complete':why.append('incomplete_protocol')
        if trajectory.get('memory_protocol')!='observation_text':why.append('retired_memory_protocol')
        if trajectory.get('split')!='train':why.append('not_train')
        if final_trait_loss(labels,trajectory['final'])>max_error:why.append('label_error')
        judge=trajectory.get('judge')
        if not judge or judge['final']['faithfulness']<min_ground:why.append('grounding_not_verified')
        try:
            records=memory_records(trajectory['memory'])
            nonempty=[r for r in records.values() if decode_observation(r['content'])['observations']]
            if any(decode_observation(r['content'])['errors'] for r in records.values()):why.append('invalid_observation_protocol')
            if not nonempty or any(record_support(trajectory,r)[0] is not True for r in nonempty):
                why.append('empty_or_unsupported_observations')
        except (ValueError,KeyError,TypeError):why.append('invalid_record_memory')
        if why:rejected.append(dict(sample_id=trajectory['sample_id'],reasons=why))
        else:accepted.append(trajectory)
    best=min(accepted,key=lambda t:(final_trait_loss(labels,t['final']),-t['judge']['final']['faithfulness'],t['cost']['active_seconds'])) if accepted else None
    return best,rejected


def export_sft(trajectories,path,tokenizer=None):
    rows=[]
    for trajectory in trajectories:
        if trajectory.get('split')!='train' or trajectory.get('offline') or not trajectory.get('teacher_accepted') or trajectory.get('run_status')!='complete':
            raise ValueError('SFT export requires accepted complete real train trajectories')
        if trajectory.get('memory_protocol')!='observation_text':
            raise ValueError('Regenerate retired Memory/tool trajectories with the current observation protocol')
        memory_records(trajectory['memory'])
        for decision in trajectory['decisions']:
            if decision['kind'] not in ('candidate_scan','action','memory','final'):continue
            if decision.get('generation_failed') or decision.get('truncated') or decision.get('candidate_output_valid') is False:continue
            if decision.get('tools') or decision.get('tool_calls') or decision.get('assistant'):
                raise ValueError('Tool-call targets are retired; supervise current actor JSON only')
            obj=parse_json(decision['text'])
            if decision['kind']=='memory' and not decision.get('proposal_repair') and decode_observation(decision['text'])['errors']:
                raise ValueError('Invalid observation target')
            row=dict(decision,sample_id=trajectory['sample_id'],split='train')
            if not row.get('tokens'):
                if tokenizer is None or not tokenizer.eos_token:raise ValueError('Teacher export requires the student tokenizer and EOS')
                row['tokens']=tokenizer.encode(decision['text']+tokenizer.eos_token,add_special_tokens=False)
                row['tokenization']='student_tokenizer_from_teacher_output'
            rows.append(row)
    write_jsonl(path,rows)
    return len(rows)


def sft(backend, path, output, cfg, epochs=1):
    torch = backend.torch
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("Empty SFT file")
    if any(r.get("split") != "train" or r.get("offline") or not r["tokens"] for r in rows):
        raise ValueError("SFT needs genuine train-only tokenized actor segments")
    if any(r.get('tools') or r.get('tool_calls') or not isinstance(parse_json(r['text']),dict) for r in rows):
        raise ValueError('SFT targets must use the current actor JSON protocol')
    optimizer = torch.optim.AdamW([p for p in backend.policy.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    backend.policy.train()  # Enable the checkpointing configured by the trainable backend.
    rng, log = random.Random(cfg["seed"]), []
    for epoch in range(epochs):
        rng.shuffle(rows)
        for decision in rows:
            optimizer.zero_grad(set_to_none=True)
            logp, _ = backend.token_stats(decision)
            loss = -logp.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backend.policy.parameters(), 1.)
            optimizer.step()
            log.append(dict(epoch=epoch, loss=float(loss.detach()), kind=decision["kind"]))
        backend.save(str(Path(output)/f"epoch_{epoch+1}"))
    write_json(Path(output)/"training_log.json", log)


def validate_gate(path, cfg):
    report = read_json(path)
    if report.get("split") != "dev" or report.get("model") != cfg["jev_model"] or not report.get("passed"):
        raise ValueError("Trait reward requires passing Dev calibration/stability report for the configured Jev version")
    return report


def rl(backend, agent, judge, rows, output, cfg, adapter_path, group_size=4,
       variant="teca", enable_trait=False, calibration_report=None, updates=1):
    if group_size < 2:
        raise ValueError("GRPO group_size must be >=2")
    if not adapter_path:
        raise ValueError("RL requires an SFT LoRA adapter; it is also the fixed reference policy")
    if enable_trait:
        raise ValueError('Record Memory does not support verified local trait attribution; Jev remains auxiliary feedback')
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("RL must use nonempty train split")
    torch = backend.torch
    # The frozen reference is the SFT policy, not the original pretraining model.
    from peft import load_peft_weights,set_peft_model_state_dict
    backend.policy.load_adapter(adapter_path, adapter_name="reference", is_trainable=False)
    # Restore after PEFT's BF16 creation / FP32 cast so the reference retains
    # the checkpoint's exact FP32 LoRA values rather than rounded values.
    set_peft_model_state_dict(backend.policy,load_peft_weights(adapter_path,device=str(backend.policy.device)),adapter_name='reference')
    backend.policy.set_adapter("default")
    for name, param in backend.policy.named_parameters():
        if ".reference." in name:
            param.requires_grad_(False)
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
        # Snapshot old log probabilities and entropy before ANY optimizer update.
        for trajectory in trajectories:
            for d in trajectory["decisions"]:
                with torch.no_grad():
                    logp, h = backend.token_stats(d)
                    d["old_logp"] = logp.cpu().tolist()
                    if d["kind"] == "action":
                        mask = torch.tensor(action_token_mask(backend.processor.tokenizer, d), device=h.device)
                        d["action_entropy"] = float(h[mask].mean())
                    backend.policy.set_adapter("reference")
                    ref, _ = backend.token_stats(d)
                    d["ref_logp"] = ref.cpu().tolist()
                    backend.policy.set_adapter("default")
                    for name, param in backend.policy.named_parameters():
                        if ".reference." in name:
                            param.requires_grad_(False)
        for trajectory in trajectories:
            trajectory['memory_outcomes']=memory_outcomes(trajectory,row['labels'],cfg)
        credit = advantages(trajectories, row["labels"], cfg, variant, enable_trait)
        optimizer.zero_grad(set_to_none=True)
        total_decisions = sum(len(t["decisions"]) for t in trajectories)
        backend.policy.train()
        total_loss = 0.
        for trajectory, values in zip(trajectories, credit):
            for decision, value in zip(trajectory["decisions"], values):
                logp, _ = backend.token_stats(decision)
                old = torch.tensor(decision["old_logp"], device=logp.device)
                ref = torch.tensor(decision["ref_logp"], device=logp.device)
                loss = clipped_grpo(logp, old, ref, value, cfg["clip_epsilon"], cfg["kl_beta"])/total_decisions
                loss.backward()
                total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(backend.policy.parameters(), 1.)
        optimizer.step()
        backend.policy.eval()
        write_json(output/f"rollouts/update_{iteration:05d}.json", trajectories)
        logs.append(dict(iteration=iteration, loss=total_loss, variant=variant, trait_reward=enable_trait,
                         rewards=[reward(t, row["labels"], cfg) for t in trajectories],
                         memory_quality=[[x['F_memory'] for x in t['memory_outcomes']] for t in trajectories],
                         memory_utility='not_estimated_no_counterfactual'))
        backend.policy.save_pretrained(str(output/"adapter"), selected_adapters=["default"])
        backend.processor.save_pretrained(str(output/"adapter"))
        write_json(output/"training_log.json", logs)
