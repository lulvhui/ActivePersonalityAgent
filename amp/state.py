"""Program-owned observation records, executable candidates and Final references."""
import math
import re
from copy import deepcopy
from .common import TRAITS, LEVELS, MODALITIES, time_pair


def empty_memory():
    return dict(observations=[])


def memory_records(memory):
    if not isinstance(memory,dict) or set(memory)!={'observations'} or not isinstance(memory['observations'],list):
        raise ValueError('Memory must contain observations only; regenerate retired atomic-evidence trajectories')
    records={}
    for record in memory['observations']:
        key=record.get('id')
        if not isinstance(key,str) or not re.fullmatch(r'M[1-9]\d*',key) or key in records:
            raise ValueError('Observation records require unique M IDs')
        records[key]=record
    return records


def record_source(record, observations):
    source=observations.get(record.get('observation_id'))
    if source is None or source.get('time')!=record.get('retrieval_window') or source.get('modality')!=record.get('modality'):
        raise ValueError('Stored record does not match its actual observation source')
    return source


def save_observation(memory, content, observation, observation_id):
    """Store one complete model response. Never interpret, split, merge or deduplicate it."""
    memory_records(memory)
    result=deepcopy(memory)
    records=result['observations']
    record=dict(id=f'M{len(records)+1}',observation_id=observation_id,content=content,
        modality=observation['modality'],retrieval_window=deepcopy(observation['time']),
        segments=deepcopy(observation.get('segments',[])),semantic_review='unreviewed')
    if observation.get('questions'):record['questions']=deepcopy(observation['questions'])
    records.append(record)
    return result,[record['id']]

class CandidatePool:
    def __init__(self, duration, max_clip, available_modalities=None):
        self.duration, self.max_clip = duration, max_clip
        self.available_modalities = list(MODALITIES if available_modalities is None else available_modalities)
        self.items = {}

    def validate(self, action):
        start,end=time_pair(action['time'])
        modality=str(action['modality']).strip().casefold()
        if not all(math.isfinite(x) for x in [start, end]):
            raise ValueError("Nonfinite action time")
        if start < 0 or end <= start or end > self.duration+1e-3 or (modality in ("video","audio_video") and end-start > self.max_clip+1e-3):
            raise ValueError("Invalid action window")
        if modality not in MODALITIES or modality not in self.available_modalities:
            raise ValueError("Action requires a supported observation mode")
        return dict(time=[start, end], modality=modality, goal=str(action.get("goal") or ""))

    def matching(self, action):
        return next((item['id'] for item in self.items.values()
                     if item['action']['modality']==action['modality'] and
                     all(abs(a-b)<=.005 for a,b in zip(item['action']['time'],action['time']))),None)

    def select_action(self, obj, allowed_ids=None):
        """Normalize equivalent envelopes, then enforce one consumption per action."""
        if not isinstance(obj,dict) or set(obj)-{'candidate_id','time','modality','goal'}:
            raise ValueError('Return candidate_id or time/modality/goal')
        candidate_id=obj.get('candidate_id')
        if candidate_id is not None:
            if candidate_id not in self.items:raise ValueError('Unknown candidate_id; use an id from the pending pool')
            item=self.items[candidate_id]
            if item['status']!='pending':raise ValueError(f"Candidate {candidate_id} already consumed: {item['action']['modality']} {item['action']['time']}; choose a pending target")
            action=self.validate({k:obj.get(k,item['action'][k]) for k in ('time','modality','goal')})
            if self.matching(action)!=candidate_id:raise ValueError('candidate_id conflicts with time or modality')
        else:
            if not all(k in obj for k in ('time','modality')):
                raise ValueError('An observation requires candidate_id or time and modality')
            action=self.validate({k:obj[k] for k in ('time','modality','goal') if k in obj})
            candidate_id=self.matching(action)
        if candidate_id is not None:
            item=self.items[candidate_id]
            if item['status']!='pending':raise ValueError(f"Candidate {candidate_id} already consumed: {item['action']['modality']} {item['action']['time']}; choose a pending target")
            if allowed_ids is not None and candidate_id not in allowed_ids:raise ValueError('Candidate is outside remaining observation budget')
            action=dict(item['action'],goal=action['goal'])
        return action,candidate_id

    def add(self, action, priority=0, parent_id=None, source_observation_id=None, origin="coarse", hint=None, source_modality=None,source_fragment=None):
        action = self.validate(action)
        if not math.isfinite(float(priority)):
            raise ValueError("Nonfinite candidate priority")
        if parent_id is not None and parent_id not in self.items:
            raise ValueError("Candidate parent does not exist")
        key=self.matching(action)
        if key is not None and self.items[key]['status']!='pending':return None
        if key is not None:action=dict(self.items[key]['action'],goal=action['goal'])
        if key is None:key=f'C{len(self.items)+1}'
        if key==parent_id:raise ValueError('Refinement repeats its parent observation')
        ancestor=parent_id
        while ancestor is not None:
            if ancestor==key:raise ValueError('Refinement would create a candidate ancestry cycle')
            ancestor=self.items[ancestor]['parent_id']
        if key not in self.items:
            self.items[key] = dict(id=key, action=action, priority=float(priority), status="pending",
                                   parent_id=parent_id, source_observation_id=source_observation_id, origin=origin,leads=[])
        elif self.items[key]["status"] == "pending":
            self.items[key].update(priority=float(priority), action=action)
        if hint:
            lead=dict(source_observation_id=source_observation_id,source_modality=source_modality,hint=hint,goal=action['goal'])
            lead.update(source_fragment or {})
            if lead not in self.items[key]['leads']:
                self.items[key]['leads'].append(lead)
        return key

    def start(self, action):
        action = self.validate(action)
        key=self.matching(action)
        if key is not None and self.items[key]['status']!='pending':raise ValueError('Observation already consumed')
        if key is None:key=self.add(action,origin='active')
        self.items[key]["status"] = "running"
        return key

    def finish(self, key, success=True):
        self.items[key]["status"] = "done" if success else "failed"

    def snapshot(self):
        return sorted(self.items.values(), key=lambda x: -x["priority"])


def memory_changes(before,after):
    previous,current=memory_records(before),memory_records(after)
    return [dict(operation='STORE',before=deepcopy(previous.get(key)),after=deepcopy(current.get(key)))
            for key in dict.fromkeys([*previous,*current]) if previous.get(key)!=current.get(key)]


def validate_final(final, memory=None):
    if not isinstance(final,dict) or set(final)!=set(TRAITS):
        raise ValueError('Final must contain all five traits')
    valid=set(memory_records(memory)) if memory is not None else set()
    for trait,item in final.items():
        if not isinstance(item,dict) or item.get('level') not in [None,*LEVELS] or not isinstance(item.get('reasoning'),str) or not item['reasoning'].strip():
            raise ValueError(f'{trait}: require a level or null and a nonempty explanation')
        if memory is not None and not set(item.get('citations',[]))<=valid:
            raise ValueError('Final citations must refer to saved observation records')
    return final


def normalize_final(final):
    if not isinstance(final,dict) or set(final)!=set(TRAITS)|{'reasoning','contextual'}:
        raise ValueError('Return O,C,E,A,N as class strings or null')
    if not isinstance(final['reasoning'],str) or not final['reasoning'].strip() or not isinstance(final['contextual'],list):
        raise ValueError('Final requires reasoning:string and contextual:array')
    if any(final[t] not in [None,*LEVELS] for t in TRAITS):
        raise ValueError('Use the five allowed levels or null')
    return {t:dict(level=final[t],reasoning=None) for t in TRAITS}

def contextual_final(final, memory):
    """Check record references, not personality semantics; retain valid siblings."""
    values=final.get('contextual',[])
    valid_ids={r['id'] for r in memory.get('observations',[])}
    accepted=[];rejected=[]
    if not isinstance(values,list):return [],[dict(error='contextual must be an array')]
    for index,value in enumerate(values):
        try:
            if not isinstance(value,dict):raise ValueError('Contextual description must be an object')
            trait=value.get('trait')
            if trait not in TRAITS:raise ValueError('Use a Big Five trait key: O/C/E/A/N')
            if value.get('context_type') not in ('reported','observed','unknown'):
                raise ValueError('context_type must be reported, observed or unknown')
            for key in ('context','expression'):
                if not isinstance(value.get(key),str) or not value[key].strip():raise ValueError(f'Missing {key}')
            refs=value.get('citations')
            if not isinstance(refs,list) or not refs or not all(isinstance(ref,str) and ref in valid_ids for ref in refs):
                raise ValueError('Citations must reference saved observation records')
            item={key:deepcopy(value[key]) for key in ('trait','context_type','context','expression','citations')}
            item['trait']=trait
            accepted.append(item)
        except ValueError as error:rejected.append(dict(index=index,error=str(error)))
    return accepted,rejected
