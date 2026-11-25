"""Fixed trait evaluators. Live Jev calls are explicit opt-in; tests never call it."""
import json
import fcntl
import tempfile
import math
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from .common import TRAITS, LEVELS, digest, probabilities, read_json, write_json
from .state import memory_records
from .perception import decode_observation

BEHAVIORS = {
    "O": ["Consistently rejects novelty and prefers strictly familiar ideas or activities",
          "Usually prefers familiar routines; little curiosity about alternatives",
          "Balances familiar activities with occasional curiosity and exploration",
          "Frequently explores new ideas, experiences and perspectives",
          "Consistently seeks diverse novel experiences and sustained imaginative or intellectual exploration"],
    "C": ["Repeatedly abandons commitments and shows pervasive disorganization",
          "Often acts without planning and inconsistently follows commitments",
          "Shows a mixture of planning and spontaneous behavior with ordinary follow-through",
          "Usually plans carefully, acts dependably and follows commitments",
          "Consistently shows exceptional organization, sustained self-discipline and meticulous follow-through"],
    "E": ["Consistently prefers solitude, avoids social engagement and speaks very reservedly across contexts",
          "Usually prefers quiet settings and limited social engagement",
          "Balances sociability and solitude; engagement varies by context",
          "Frequently initiates social interaction and expresses social energy",
          "Consistently seeks intensive social stimulation and actively leads interactions across contexts"],
    "A": ["Repeatedly shows hostility, exploitation or disregard for others' needs",
          "Frequently expresses distrust, competitive antagonism or low concern for others",
          "Balances cooperation and self-interest; interpersonal warmth varies by context",
          "Usually shows empathy, trust and willingness to cooperate",
          "Consistently prioritizes compassion, cooperation and consideration across difficult interpersonal contexts"],
    "N": ["Consistently remains calm and recovers readily under meaningful stress",
          "Usually emotionally steady with limited worry under stress",
          "Shows ordinary context-dependent worry and emotional recovery",
          "Frequently experiences worry, emotional distress or heightened stress sensitivity",
          "Consistently reports or displays intense worry, emotional volatility and difficulty recovering under stress"],
}


class TraitScorer(ABC):
    @abstractmethod
    def score(self, memory):
        """Return {traits: {O: {level, probabilities, score, confidence}, ...}, metadata}."""


def scored_result(distributions, metadata):
    traits = {}
    for t in TRAITS:
        p = probabilities(distributions[t])
        traits[t] = dict(level=LEVELS[max(range(5), key=p.__getitem__)], probabilities=p,
                         score=sum(i*x for i, x in enumerate(p)), confidence=max(p))
    for item in traits.values():
        h=-sum(x*math.log(x) for x in item["probabilities"] if x>0)
        item.update(entropy=h,uncertainty=h/math.log(5))
    return dict(traits=traits, metadata=metadata)


def retain_unknown_levels(result, memory):
    """A scoring distribution does not turn missing evidence into a known level."""
    for trait in TRAITS:
        known=any(decode_observation(r["content"])["observations"] for r in memory.get("observations",[]))
        item=result["traits"][trait]
        item["predictive"]=known and result["metadata"].get("predictive",True) and not result["metadata"].get("offline",False)
        item["status"]="scored" if item["predictive"] else "insufficient_evidence"
        if not item["predictive"]:
            item["level"] = None
            item["uncertainty"] = 1.0
    return result


def minimize_state(memory):
    """Only decoded findings and modality; plans, IDs, paths, labels and times stay local.

    This is data minimization, not a guarantee of anonymity. Review free text before enabling.
    """
    def redact(text):
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", str(text))
        return re.sub(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)", "[PHONE]", text)
    notes=[dict(observation=redact(content),source=r['modality']) for r in memory_records(memory).values()
           if (content:=decode_observation(r['content'])['observations'])]
    return {t:notes for t in TRAITS}


def _cache_write(path, value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,suffix=".tmp",delete=False) as stream:
        temp=Path(stream.name)
        try:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.flush()
            os.replace(temp,path)
        finally:
            temp.unlink(missing_ok=True)


class JevTraitScorer(TraitScorer):
    def __init__(self, cache, model="jev-1.14.0", allow_external=False, client=None,
                 input_price_per_million=None, output_price_per_million=None, timeout=60, max_retries=3):
        self.cache,self.model=Path(cache),model
        self.allow_external,self.client=allow_external,client
        self.prices=input_price_per_million,output_price_per_million
        self.timeout,self.max_retries=timeout,max_retries
        if timeout<=0 or max_retries<0 or any(p is not None and (not math.isfinite(p) or p<0) for p in self.prices):
            raise ValueError("Invalid Jev timeout, retry count or pricing")
        self.cache.mkdir(parents=True,exist_ok=True)

    def check_live_ready(self):
        if not self.allow_external:
            raise PermissionError("Live Jev requires --allow-external after confirming data-use permission")
        if self.client is None:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError("Set TYPESAFE_API_KEY in the process environment")
            from typesafe_sdk import TypeSafeClient,RetryPolicy
            self.client=TypeSafeClient(retry=RetryPolicy(max_retries=0),timeout=self.timeout)

    def score(self,memory,refresh=False):
        state=minimize_state(memory)
        if not any(state.values()):
            return retain_unknown_levels(scored_result({t:[.2]*5 for t in TRAITS},dict(
                model=self.model,requested_model=self.model,predictive=False,status="no_observed_evidence",
                api_calls=0,cache_hit=False,billed_cost=0.,usage={},raw_response=None)),memory)
        questions={t:dict(type="score",instructions=(
            f"Assess {name} using only actually observed contextual behavior in state. "
            "State entries are observed cue summaries, not personality labels. Distinguish situational behavior from stable tendencies. "
            "Entries may be repeated or complementary reads of the same behavior. Base support on distinct behaviors in context; additional reads refine a report rather than multiply independent support. "
            "Missing information is unknown; do not infer from identity, accent, appearance or demographic stereotypes."),
            criteria=BEHAVIORS[t]) for t,name in TRAITS.items()}
        key=digest(dict(schema_version=4,state=state,questions=questions,model=self.model))
        path=self.cache/f"{key}.json"
        # One request per identical memory even when two GPU workers score concurrently.
        with (self.cache/f"{key}.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists() and not refresh:
                result=read_json(path)
                result["metadata"].update(cache_hit=True,api_calls=0,billed_cost=0.)
                return retain_unknown_levels(result,memory)
            self.check_live_ready()
            from typesafe_sdk import Score
            started=time.monotonic()
            for attempt in range(self.max_retries+1):
                try:
                    response=self.client.system_one(model=self.model,state=state,
                        questions={t:Score(instructions=q["instructions"],criteria=q["criteria"]) for t,q in questions.items()})
                    raw=response.model_dump(mode="json")
                    break
                except Exception as error:
                    status=getattr(error,"status",getattr(error,"status_code",None))
                    details=dict(type=type(error).__name__,status=status,attempt=attempt+1,model=self.model,cache_key=key)
                    _cache_write(self.cache/f"{key}.error.json",details)
                    transient=status in [429,500,502,503,504,529] or isinstance(error,(ConnectionError,TimeoutError))
                    if not transient or attempt==self.max_retries:
                        raise RuntimeError(f"Jev request failed: {details}") from None
                    time.sleep(min(2**attempt,8))
            _cache_write(self.cache/f"{key}.raw.json",raw)
            try:
                if raw.get("model",self.model)!=self.model:
                    raise ValueError("Response model does not match the pinned Jev version")
                distributions={}
                for t in TRAITS:
                    answer=raw["answers"][t]
                    if answer.get("type","score")!="score":raise ValueError("Expected Score answer")
                    values=answer["probabilities"]
                    if {str(k) for k in values}!={str(i) for i in range(5)}:
                        raise ValueError("Expected exactly five probability levels 0..4")
                    distributions[t]=[values[str(i)] if str(i) in values else values[i] for i in range(5)]
                    for field,maximum in [("confidence",1),("score",4)]:
                        v=float(answer[field])
                        if not math.isfinite(v) or not 0<=v<=maximum:raise ValueError(f"Invalid {field}")
                usage=raw.get("usage") or {}
                known_usage=all(isinstance(usage.get(k),(int,float)) and math.isfinite(usage[k]) and usage[k]>=0 for k in ('input_tokens','output_tokens'))
                cost=None if any(p is None for p in self.prices) or not known_usage else (
                    usage.get("input_tokens",0)*self.prices[0]+usage.get("output_tokens",0)*self.prices[1])/1e6
                result=scored_result(distributions,dict(model=raw.get("model",self.model),requested_model=self.model,
                    predictive=True,status="scored_uncalibrated",level_order=LEVELS,cache_hit=False,usage=usage,estimated_cost=cost,billed_cost=cost,
                    cost_note="null means pricing not configured; retries may incur additional provider charges",
                    api_calls=attempt+1,attempts=attempt+1,latency_seconds=time.monotonic()-started,
                    called_at_unix=time.time(),cache_key=key,raw_response=raw))
                for t in TRAITS:
                    result["traits"][t].update(confidence=float(raw["answers"][t]["confidence"]),score=float(raw["answers"][t]["score"]))
            except (ValueError,KeyError,TypeError) as error:
                _cache_write(self.cache/f"{key}.error.json",dict(type="InvalidScoreResponse",model=self.model,cache_key=key))
                raise ValueError("Invalid Jev Score response; raw response retained in cache") from error
            retain_unknown_levels(result,memory)
            _cache_write(path,result)
            return result


def build_scorer(config,mode="offline",allow_external=False):
    if mode=="offline":return OfflineTraitScorer()
    if mode!="jev":raise ValueError("Unknown trait scorer")
    scorer=JevTraitScorer(config.get("jev_cache","cache/jev"),config.get("jev_model","jev-1.14.0"),allow_external,
        timeout=config.get("jev_timeout_seconds",60),max_retries=config.get("jev_max_retries",3),
        input_price_per_million=config.get("jev_input_price_per_million"),output_price_per_million=config.get("jev_output_price_per_million"))
    scorer.check_live_ready()
    return scorer


def prediction_feedback(score):
    """Hide unknown/no-evidence distributions from the actor, retain them in audit history."""
    if score["metadata"].get("offline") or not score["metadata"].get("predictive",True):return None
    return {t:(dict(item) if item.get("predictive",True) else dict(level=None,probabilities=None,
                uncertainty=1.,status="insufficient_evidence")) for t,item in score["traits"].items()}


class OfflineTraitScorer(TraitScorer):
    """Uniform plumbing fixture. Not Jev, not a predictive model or research baseline."""
    def score(self, memory):
        result = scored_result({t: [.2]*5 for t in TRAITS}, dict(model="OFFLINE_UNIFORM_TEST_ONLY", offline=True,predictive=False,api_calls=0,billed_cost=0.,
            status='offline_test_only' if any(decode_observation(r['content'])['observations'] for r in memory_records(memory).values()) else 'no_observed_evidence'))
        return retain_unknown_levels(result,memory)
