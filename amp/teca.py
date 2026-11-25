"""TECA group credit and clipped token-level policy objective."""
import math
from .common import TRAITS, LEVELS, standardize, digest
from .state import memory_records,memory_changes,record_source
from .perception import decode_observation


def trait_loss(labels, distributions):
    return sum(-math.log(max(distributions[t][labels[t]], 1e-8)) for t in TRAITS)/5


def final_trait_loss(labels, final):
    # Label-supervised ordinal loss of the agent's own final categories, independent of Jev.
    return sum((1.0 if final[t]["level"] is None else abs(labels[t]-LEVELS.index(final[t]["level"]))/4) for t in TRAITS)/5


def reward(trajectory, labels, cfg):
    judge = trajectory.get("judge")
    ground = judge["final"]["faithfulness"] if judge else 0.
    # Account for all generated text as well as active/coarse sensing; normalized units documented.
    cost = ((trajectory["cost"]["active_seconds"] + trajectory["cost"]["coarse_audio_seconds"])/(cfg.get("max_seconds") or cfg.get("cost_scale_seconds",300.))
            + trajectory["cost"]["coarse_frames"]/100 + trajectory["cost"]["generated_tokens"]/10000
            + trajectory["cost"].get("input_tokens", 0)/50000)
    return -final_trait_loss(labels, trajectory["final"]) + cfg["beta_ground"]*ground-cfg["lambda_cost"]*cost


def step_changes(trajectory,step_index):
    before=trajectory['initial']['memory'] if step_index==0 else trajectory['steps'][step_index-1]['memory']
    return memory_changes(before,trajectory['steps'][step_index]['memory'])


def record_support(trajectory,record):
    try:source=record_source(record,trajectory['observations'])
    except (ValueError,KeyError,TypeError):return False,'invalid_observation_source'
    decoded=decode_observation(record['content'])
    if decoded['errors']:return False,'invalid_observation_protocol'
    if not decoded['observations']:return None,'no_recorded_findings'
    check=trajectory.get('judge',{}).get('records',{}).get(record['id'],{})
    if check.get('record_fingerprint')!=digest(record) or check.get('source_fingerprint')!=digest(source):
        return None,'missing_or_stale_independent_check'
    if not isinstance(check.get('supported'),bool):return None,'unverified_support_type'
    return check['supported'],'independent_source_check'


def local_credit(trajectory,labels,cfg,enable_trait=False):
    if enable_trait:
        raise ValueError('Record Memory has no verified trait-level attribution; local Jev gain is unsupported')
    values=[-cfg['lambda_cost']*step['cost'] for step in trajectory['steps']]
    parts=[dict(trait=0.,context=0.,cost=step['cost'],
        trait_status='not_attributable_from_observation_records',context_status='not_estimated_without_context_changes') for step in trajectory['steps']]
    return values,parts


def memory_quality_credit(trajectory,step_index):
    terms=[];details=[];step=trajectory['steps'][step_index]
    for change in step_changes(trajectory,step_index):
        record=change['after'];value=0.
        if record is None or change['before'] is not None or record.get('observation_id')!=step['observation_id']:
            value=-1.;reason='not_a_new_current_observation_record'
        else:
            support,reason=record_support(trajectory,record)
            if support is not None:value=1. if support else -1.
        terms.append(value);details.append(dict(operation='STORE',record_id=(record or change['before'])['id'],value=value,reason=reason))
    return (sum(terms)/len(terms) if terms else 0.),details


def memory_outcomes(trajectory,labels,cfg):
    """Observed downstream associations only; U remains unestimated, never causal."""
    result=[]
    for index,step in enumerate(trajectory['steps']):
        value,details=memory_quality_credit(trajectory,index)
        next_index=step.get('next_action_decision_index')
        next_action=trajectory['decisions'][next_index].get('action') if next_index is not None else None
        if next_action is None and index+1<len(trajectory['steps']):next_action=trajectory['steps'][index+1]['action']
        result.append(dict(step=index,F_memory=value,F_details=details,U_memory=None,
            U_status='not_estimated_no_counterfactual',memory_changes=step_changes(trajectory,index),
            next_action=next_action,next_action_decision_index=next_index,
            subsequent_observation_cost=sum(s['cost'] for s in trajectory['steps'][index+1:]),
            score_before=step['before'],score_after=step['after'],
            score_valid_before=step.get('score_valid_before'),score_valid_after=step.get('score_valid_after'),
            trajectory_reward=reward(trajectory,labels,cfg),
            final_citations={t:list(dict.fromkeys(ref for item in trajectory.get('contextual',[]) if item['trait']==t for ref in item['citations'])) for t in TRAITS}))
    return result


def normalize_local(values):
    """RMS scaling preserves zero/sign: ineligible edits cannot gain positive credit by centering."""
    if not values:
        return []
    scale = math.sqrt(sum(v*v for v in values)/len(values))
    return [v/(scale+1e-8) for v in values]


def advantages(trajectories, labels, cfg, variant="teca", enable_trait=False):
    if len(trajectories) < 2:
        raise ValueError("GRPO needs at least two trajectories from the SAME initial sample")
    if len({t["sample_id"] for t in trajectories}) != 1:
        raise ValueError("GRPO group contains different initial samples")
    if variant not in ["grpo", "taura_style", "teca"]:
        raise ValueError("Unknown credit variant")
    terminal = standardize([reward(t, labels, cfg) for t in trajectories])
    raw = [local_credit(t, labels, cfg, enable_trait=enable_trait)[0] for t in trajectories]
    flat = normalize_local([g for values in raw for g in values])
    cursor, result = 0, []
    for i, trajectory in enumerate(trajectories):
        local = flat[cursor:cursor+len(raw[i])]
        cursor += len(raw[i])
        action_decisions = [d for d in trajectory["decisions"] if d["kind"] == "action"]
        entropies = [d.get("action_entropy", 0.) for d in action_decisions]
        mean_h = sum(entropies)/len(entropies) if entropies else 0.
        weights = [min(max(h/(mean_h+1e-8), .25), 4.) if mean_h > 1e-8 else 1. for h in entropies]
        per_decision = []
        for d in trajectory["decisions"]:
            value = terminal[i]
            step = d["step"]
            if d["kind"] == "action":
                omega = weights[action_decisions.index(d)]
                if variant == "taura_style":
                    value *= omega  # Named style baseline, not a claim to reproduce TAURA.
                elif variant != "grpo" and 0 <= step < len(local):
                    value += cfg["alpha"]*(omega if variant == "teca" else 1.)*local[step]
            elif d["kind"] == "memory" and variant == "teca" and 0 <= step < len(trajectory["steps"]):
                faithful,_=memory_quality_credit(trajectory,step) if not d.get('proposal_repair') and d.get('candidate_output_valid') is not False else (0.,[])
                value += cfg.get('memory_quality_beta',cfg['alpha'])*faithful
            per_decision.append(value)
        result.append(per_decision)
    return result


def clipped_grpo(logp, old_logp, ref_logp, advantage, epsilon=.2, beta=.04):
    import torch
    ratio = torch.exp(logp-old_logp)
    adv = torch.as_tensor(advantage, device=logp.device)
    surrogate = torch.minimum(ratio*adv, torch.clamp(ratio, 1-epsilon, 1+epsilon)*adv)
    # Positive k3 estimator of reverse KL, with a fixed reference policy.
    delta = ref_logp-logp
    kl = torch.exp(delta)-delta-1
    return (-surrogate + beta*kl).mean()
