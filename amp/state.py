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

# Channels a read gives the model; text is the transcript span of the window.
CHANNELS = {'audio': ('audio',), 'video': ('video',), 'audio_video': ('audio', 'video'), 'text': ('text',)}


def is_still(time, fps):
    """MediaTools returns one raw frame taken at the window start when the window spans under two sampled frames."""
    return (time[1]-time[0])*fps < 2


def video_frames(time, fps):
    """Frame indices on the global sampling grid used for windows of at least two sampled frames."""
    if is_still(time, fps):
        return set()
    start, end = time
    return set(range(int(math.ceil(start*fps-1e-9)), int(math.ceil(end*fps-1e-9))))


def uncovered_seconds(interval, covered):
    start, end = interval
    left = end-start
    for a, b in covered:
        left -= max(0., min(end, b)-max(start, a))
    return max(0., left)


def merge_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]+1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


class CandidatePool:
    def __init__(self, duration, max_clip, available_modalities=None):
        self.duration, self.max_clip = duration, max_clip
        self.available_modalities = list(MODALITIES if available_modalities is None else available_modalities)
        self.items = {}
        self.novelty_rule = None
        self.read_spans, self.read_frames, self.read_stills = dict(audio=[], text=[]), set(), set()
        self.focus_used, self.filtered = {}, []

    def configure_novelty(self, video_fps, min_new_seconds, max_focus):
        """Refinements must add recording content the model has not been given.

        A same-channel re-read of read content returns the same samples, transcript
        or sampled frames, so by the data processing inequality it adds no
        information about the person. A refinement is executed only if, at execution
        time, it removes at least min(min_new_seconds, unread) seconds of unread
        content from some channel, or it is one of `max_focus` closer looks inside
        its original coarse region. A still (one raw frame at a short window's
        start) is a different frame from the grid, but removes no unread grid time,
        so it is executed only as a focus. Hence executed reads are bounded by
        roots * (1 + max_focus) + sum over channels of ceil(unread / min_new_seconds),
        whatever the model proposes.
        """
        if min_new_seconds is None:
            self.novelty_rule = None
            return
        video_fps, min_new_seconds, max_focus = float(video_fps), float(min_new_seconds), int(max_focus)
        if not math.isfinite(video_fps) or video_fps <= 0 or not math.isfinite(min_new_seconds) or min_new_seconds <= 0 or max_focus < 0:
            raise ValueError('Novelty rule needs positive video_fps and min_new_seconds and max_focus_per_region >= 0')
        self.novelty_rule = dict(video_fps=video_fps, min_new_seconds=min_new_seconds, max_focus=max_focus)

    def record_read(self, action):
        """Register the content of a completed active read."""
        for channel in CHANNELS[action['modality']]:
            if channel in self.read_spans:
                self.read_spans[channel] = merge_intervals(self.read_spans[channel]+[list(action['time'])])
        if 'video' in CHANNELS[action['modality']] and self.novelty_rule:
            if is_still(action['time'], self.novelty_rule['video_fps']):
                self.read_stills.add(round(action['time'][0], 6))
            else:
                self.read_frames |= video_frames(action['time'], self.novelty_rule['video_fps'])

    def new_seconds(self, action):
        """Per channel: unread seconds this action removes, and unread seconds left in the recording."""
        channels, result = CHANNELS[action['modality']], {}
        for channel in channels:
            if channel in self.read_spans:
                result[channel] = (uncovered_seconds(action['time'], self.read_spans[channel]),
                                   uncovered_seconds([0., self.duration], self.read_spans[channel]))
        if 'video' in channels:
            fps = self.novelty_rule['video_fps']
            frames = video_frames(action['time'], fps)
            result['video'] = (len(frames-self.read_frames)/fps,
                               len(video_frames([0., self.duration], fps)-self.read_frames)/fps)
        return result

    def new_still(self, action):
        """True when a short video window returns a raw frame not given to the model before."""
        fps = self.novelty_rule['video_fps']
        if 'video' not in CHANNELS[action['modality']] or not is_still(action['time'], fps):
            return False
        start = round(action['time'][0], 6)
        on_grid = abs(start*fps-round(start*fps)) < 1e-6 and int(round(start*fps)) in self.read_frames
        return start not in self.read_stills and not on_grid

    def unread_intervals(self, modality):
        """Unread spans of a channel in recording seconds; video spans follow sampled frame times."""
        if modality in self.read_spans:
            spans, cursor = [], 0.
            for a, b in self.read_spans[modality]:
                if a > cursor+1e-9:
                    spans.append([cursor, a])
                cursor = max(cursor, b)
            return spans+([[cursor, self.duration]] if cursor < self.duration-1e-9 else [])
        fps = self.novelty_rule['video_fps']
        frames = sorted(video_frames([0., self.duration], fps)-self.read_frames)
        spans = []
        for f in frames:
            if spans and f == spans[-1][1]:
                spans[-1][1] = f+1
            else:
                spans.append([f, f+1])
        return [[a/fps, min(self.duration, b/fps)] for a, b in spans]

    def root(self, key):
        while self.items[key]['parent_id'] is not None:
            key = self.items[key]['parent_id']
        return key

    def refinement_check(self, action, parent_id):
        """Pure check against completed reads: basis 'unread_content', 'focus' or None."""
        new = self.new_seconds(action)
        detail = dict(new_seconds={c: round(v[0], 6) for c, v in new.items()}, new_still=self.new_still(action))
        if any(added > 1e-9 and added >= min(self.novelty_rule['min_new_seconds'], unread)-1e-9 for added, unread in new.values()):
            return 'unread_content', None, detail
        root = self.root(parent_id)
        left, right = self.items[root]['action']['time']
        detail['region_root'] = root
        if not (left-1e-6 <= action['time'][0] and action['time'][1] <= right+1e-6):
            return None, 'no_unread_content_outside_region', detail
        if self.focus_used.get(root, 0) >= self.novelty_rule['max_focus']:
            return None, 'focus_used', detail
        return 'focus', None, detail

    def admit_refinement(self, action, parent_id, record):
        """Proposal-time screen; it consumes no focus. Execution repeats the check (commit_refinement)."""
        if self.novelty_rule is None:
            return True
        basis, reason, detail = self.refinement_check(action, parent_id)
        if basis is None:
            self.filtered.append(dict(record, stage='proposal', parent_id=parent_id, time=list(action['time']),
                                      modality=action['modality'], reason=reason, **detail))
        return basis is not None

    def prune_refinements(self):
        """Before each selection, retire pending refinements whose content has since been read."""
        if self.novelty_rule is None:
            return
        for item in self.items.values():
            if item['status'] != 'pending' or item['origin'] != 'observation' or item['parent_id'] is None:
                continue
            basis, reason, detail = self.refinement_check(item['action'], item['parent_id'])
            if basis is None:
                item['status'] = 'filtered'
                self.filtered.append(dict(stage='execution', candidate_id=item['id'], parent_id=item['parent_id'],
                                          time=list(item['action']['time']), modality=item['action']['modality'],
                                          reason=reason, **detail))

    def commit_refinement(self, key):
        """Consume a focus when a pruned-valid refinement is executed as one."""
        item = self.items.get(key)
        if self.novelty_rule is None or not item or item['origin'] != 'observation' or item['parent_id'] is None:
            return
        basis, reason, _ = self.refinement_check(item['action'], item['parent_id'])
        if basis is None:
            raise ValueError(f'Refinement {key} no longer adds unread content: {reason}')
        if basis == 'focus':
            root = self.root(item['parent_id'])
            self.focus_used[root] = self.focus_used.get(root, 0)+1

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


def validate_final(final):
    if not isinstance(final,dict) or set(final)!=set(TRAITS):
        raise ValueError('Final must contain all five traits')
    for trait,item in final.items():
        if not isinstance(item,dict) or item.get('level') not in [None,*LEVELS] or not isinstance(item.get('reasoning'),str) or not item['reasoning'].strip():
            raise ValueError(f'{trait}: require a level or null and a nonempty explanation')
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
