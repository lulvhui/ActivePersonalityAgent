import math
from collections import defaultdict
from .common import TRAITS, LEVELS, read_json, standardize
from .teca import trait_loss
from .state import memory_records


def classification(labels, predictions):
    per_class = []
    for k in range(5):
        tp = sum(y == k and p == k for y, p in zip(labels, predictions))
        fp = sum(y != k and p == k for y, p in zip(labels, predictions))
        fn = sum(y == k and p != k for y, p in zip(labels, predictions))
        per_class.append(2*tp/max(1, 2*tp+fp+fn))
    return dict(macro_f1=sum(per_class)/5, accuracy=sum(y == p for y, p in zip(labels, predictions))/len(labels),
                ordinal_mae=sum(abs(y-p) for y, p in zip(labels, predictions))/len(labels))


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result, i = [0.]*len(values), 0
    while i < len(order):
        j = i+1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i+j-1)/2
        i = j
    return result


def spearman(a, b):
    if len(a) < 2:
        return None
    x, y = standardize(ranks(a)), standardize(ranks(b))
    if not any(x) or not any(y):
        return None
    return sum(i*j for i, j in zip(x, y))/len(x)


def calibration(labels, predictions):
    n = len(labels)
    nll = sum(-math.log(max(p[y], 1e-8)) for y, p in zip(labels, predictions))/n
    brier = sum(sum((v-float(k == y))**2 for k, v in enumerate(p)) for y, p in zip(labels, predictions))/n
    ece = 0.
    for i in range(10):
        indexes = [k for k, p in enumerate(predictions) if i/10 <= max(p) < (i+1)/10 or (i == 9 and max(p) == 1)]
        if indexes:
            acc = sum(max(range(5), key=predictions[k].__getitem__) == labels[k] for k in indexes)/len(indexes)
            conf = sum(max(predictions[k]) for k in indexes)/len(indexes)
            ece += len(indexes)/n*abs(acc-conf)
    return dict(nll=nll, brier=brier, ece=ece)


def evaluate(trajectories, rows, bins):
    lookup = {r["id"]: r for r in rows}
    grouped = defaultdict(list)
    for t in trajectories:
        if t.get("offline"):
            raise ValueError("Offline fixtures cannot produce research metrics")
        grouped[t["dataset"]].append(t)
    report = {}
    for dataset, ts in grouped.items():
        metrics = {}
        for trait in TRAITS:
            labels = [lookup[t["sample_id"]]["labels"][trait] for t in ts]
            covered = [t for t in ts if t["final"][trait]["level"] is not None]
            known_labels = [lookup[t["sample_id"]]["labels"][trait] for t in covered]
            pred = [LEVELS.index(t["final"][trait]["level"]) for t in covered]
            raw = [lookup[t["sample_id"]]["labels_raw"][trait] for t in covered]
            mapped = [bins[dataset][trait]["centers"][p] for p in pred]
            metrics[trait] = dict(prediction_coverage=len(covered)/len(ts), abstentions=len(ts)-len(covered),
                accuracy_all=sum(y == p for y,p in zip(known_labels,pred))/len(ts),
                macro_f1=None, accuracy=None, ordinal_mae=None, raw_mae=None, spearman=None)
            if covered:
                metrics[trait].update(classification(known_labels, pred),
                    raw_mae=sum(abs(a-b) for a,b in zip(raw,mapped))/len(covered), spearman=spearman(raw,mapped))
        judged = [t["judge"]["final"] for t in ts if t.get("judge")]
        valid_refs=[ref in memory_records(t['memory']) for t in ts for item in t.get('contextual',[]) for ref in item['citations']]
        report[dataset] = dict(n=len(ts), traits=metrics,
            macro_f1=sum(m["macro_f1"] or 0 for m in metrics.values())/5,
            mean_prediction_coverage=sum(m["prediction_coverage"] for m in metrics.values())/5,
            classification_scope="conditional on non-abstained predictions; accuracy_all counts abstentions as incorrect",
            active_seconds=sum(t["cost"]["active_seconds"] for t in ts)/len(ts),
            mean_observations=sum(sum(s.get('action_type')!='static' for s in t['steps']) for t in ts)/len(ts),
            structural_citation_validity=sum(valid_refs)/len(valid_refs) if valid_refs else None,
            automatic_faithfulness=sum(j["faithfulness"] for j in judged)/len(judged) if judged else None,
            automatic_citation_correctness=sum(j["citation_correctness"] for j in judged)/len(judged) if judged else None,
            judge_coverage=len(judged)/len(ts), continuous_prediction="train-bin centroid of final class")
    return report


def validate_jev(trajectories, rows, train_rows, model="jev-1.14.0", min_samples=30, max_ece=.15, max_stability_l1=.1):
    """Conservative experiment gate. User can inspect, never treats Jev as ground truth."""
    lookup = {r["id"]: r for r in rows}
    groups = defaultdict(list)
    for t in trajectories:
        if t["split"] != "dev" or t.get("offline") or t["score"]["metadata"]["model"] != model or not t['score']['metadata'].get('predictive',True):
            raise ValueError("Calibration requires real Dev-only results for the pinned Jev model")
        groups[t["dataset"]].append(t)
    report = dict(split="dev", model=model, passed=True, datasets={},
                  thresholds=dict(min_samples=min_samples, max_ece=max_ece, max_stability_l1=max_stability_l1),
                  note="Automatic gate against train prior. Passing does not establish psychological validity.")
    for dataset in ["IELTS", "RecruitView"]:
        ts = groups[dataset]
        repeats = defaultdict(list)
        for t in ts:
            repeats[t["sample_id"]].append(t)
        stable, metrics = [], {}
        for same in repeats.values():
            # Same evidence, independent uncached refreshes only. Repeated cached values prove nothing.
            if len(same) > 1 and len({str(x["memory"]) for x in same}) == 1 and all(not x["score"]["metadata"].get("cache_hit") for x in same):
                for t in same[1:]:
                    stable.append(sum(abs(a-b) for trait in TRAITS for a, b in zip(
                        same[0]["score"]["traits"][trait]["probabilities"], t["score"]["traits"][trait]["probabilities"]))/5)
        unique = [v[0] for v in repeats.values()]
        if not unique:
            report["datasets"][dataset] = {"passed": False, "reason": "no_dev_results"}
            report["passed"] = False
            continue
        train = [r for r in train_rows if r["dataset"] == dataset and r["split"] == "train"]
        for trait in TRAITS:
            labels = [lookup[t["sample_id"]]["labels"][trait] for t in unique]
            p = [t["score"]["traits"][trait]["probabilities"] for t in unique]
            prior = [(1+sum(r["labels"][trait] == k for r in train))/(len(train)+5) for k in range(5)]
            predicted = [max(range(5), key=values.__getitem__) for values in p]
            metrics[trait] = dict(calibration(labels, p), classification=classification(labels, predicted),
                                  prior=calibration(labels, [prior]*len(labels)))
        passed = len(unique) >= min_samples and len(stable) >= min_samples and max(stable, default=1) <= max_stability_l1
        passed = passed and all(m["ece"] <= max_ece and m["nll"] < m["prior"]["nll"] and m["brier"] < m["prior"]["brier"] for m in metrics.values())
        report["datasets"][dataset] = dict(passed=passed, n=len(unique), stability_repeats=len(stable),
            max_stability_l1=max(stable) if stable else None, traits=metrics)
        report["passed"] = report["passed"] and passed
    return report


def credit_diagnostics(trajectories):
    """Record fidelity diagnostics only; no fabricated trait or contextual attribution."""
    from .teca import memory_quality_credit
    updates=[]
    for trajectory in trajectories:
        if trajectory.get('offline'):
            raise ValueError('Credit diagnostics require genuine trajectories')
        memory_records(trajectory['memory'])
        for index,step in enumerate(trajectory['steps']):
            quality,details=memory_quality_credit(trajectory,index)
            updates.append(dict(sample_id=trajectory['sample_id'],step=index,
                F_memory=quality,details=details,cost=step['cost']))
    return dict(n_observations=len(updates),observations=updates,trait_gain=None,context_gain=None,
        attribution_status='not_attributable_from_observation_records',
        note='Source fidelity only; missing independent checks yield no positive credit. Not causal Memory utility.')
