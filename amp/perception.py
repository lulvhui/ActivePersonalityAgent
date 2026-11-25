"""Observed-source prompts and descriptive coverage; never synthetic trait scores."""
from .common import TRAITS,LEVELS,MODALITIES
from .state import memory_records
from .common import parse_json,time_pair

SOURCE_FOCUS = {
    'text': 'Read self-reported behavior and situational context in the transcript.',
    'audio': 'Listen to the interviewee’s account of actions and context, including vocal delivery.',
    'video': 'Observe the interviewee’s communication and changes in expressions, gaze, posture and gestures in context.',
    'audio_video': 'Listen to the interviewee’s account of actions and context, alongside vocal delivery and visible behavior.',
}


def generation_schema(kind,modalities=None,memory=None,candidates_only=False):
    """One canonical wire format; source IDs/times remain program-owned."""
    def obj(fields,required=None):
        return dict(type='object',properties=fields,required=list(fields) if required is None else required,additionalProperties=False)
    text=dict(type='string',minLength=1)
    if kind=='candidate_scan':
        return obj(dict(summary=text,candidates=dict(type='array',items=obj(dict(hint=text,goal=text)))))
    if kind=='memory':
        target=obj(dict(local_time=dict(type='array',items=dict(type='number'),minItems=2,maxItems=2),
            modality=dict(type='string',enum=list(modalities or MODALITIES)),hint=text,goal=text))
        fields={} if candidates_only else dict(observations=dict(type='array',items=text))
        fields['candidates']=dict(type='array',items=target)
        return obj(fields)
    if kind=='final':
        level=dict(anyOf=[dict(type='string',enum=LEVELS),dict(type='null')])
        refs=[r['id'] for r in (memory or {}).get('observations',[])]
        context=obj(dict(trait=dict(type='string',enum=list(TRAITS)),context_type=dict(type='string',enum=['reported','observed','unknown']),
            context=text,expression=text,citations=dict(type='array',minItems=1,items=dict(type='string',enum=refs))))
        contextual=dict(type='array',items=context) if refs else dict(type='array',maxItems=0)
        return obj({**{t:level for t in TRAITS},'reasoning':text,'contextual':contextual})
    return None


def decode_observation(text):
    """Read the current JSON protocol only; failed output remains in the raw audit."""
    result=dict(observations='',candidates=[],status='structured',errors=[])
    try:
        obj=parse_json(text)
        if not isinstance(obj,dict) or set(obj)!={'observations','candidates'}:
            raise ValueError('Return observations and candidates arrays')
        if not isinstance(obj['observations'],list) or not all(isinstance(v,str) and v.strip() for v in obj['observations']):
            raise ValueError('observations must be an array of nonempty finding strings')
        result['observations']='\n'.join(obj['observations'])
    except (ValueError,TypeError) as error:
        result.update(status='invalid',errors=[dict(scope='observation',index=None,error=str(error))])
        return result
    if not isinstance(obj['candidates'],list):
        result.update(status='invalid',errors=[dict(scope='candidate',index=None,error='candidates must be an array')])
    else:result['candidates']=obj['candidates']
    return result


def observation_view(observation):
    start,end=observation['time']
    view = dict(modality=observation['modality'],time=[start,end],clip_seconds=end-start,
                time_rule='time is the global retrieval window; playback starts at local zero. ASR retrieval times are not verified alignment.',
                segments=[dict(segment_id=s['segment_id'],text=s['text']) for s in observation.get('segments',[])],
                event_segments=[dict(segment_id=s['segment_id'],text=s['text']) for s in observation.get('event_segments',[])],
                transcript_status='provided' if observation['modality']=='text' else 'not_requested; supplied media is the observation, empty text segments are not an ASR result')
    video = next((m for m in observation.get('media',[]) if m['type']=='video'), None)
    if video:
        view['video_sampling']={k:video[k] for k in ('fps','sampled_frames') if k in video}
    return view


def media_coverage(duration,modalities,steps):
    """Action-coordinate coverage, not transcript alignment or a personality gap."""
    result={}
    for modality in modalities:
        unseen=[[0.,float(duration)]]
        for step in steps:
            action=step['action']
            if action['modality']!=modality:continue
            left,right=action['time'];remaining=[]
            for start,end in unseen:
                if right<=start or left>=end:remaining.append([start,end]);continue
                if start<left:remaining.append([start,left])
                if right<end:remaining.append([right,end])
            unseen=remaining
        result[modality]=unseen
    return result


def response_items(obj,field):
    if isinstance(obj,dict):
        if isinstance(obj.get(field),list):return obj[field]
    raise ValueError(f'Return a JSON object with a {field} array')


def bind_candidate(candidate,observation,extension_seconds=0):
    """Observe proposals use seconds from the parent clip, with a context margin."""
    if not isinstance(candidate,dict) or set(candidate)!={'local_time','modality','hint','goal'}:
        raise ValueError('A follow-up requires local_time, modality, hint and goal')
    left,right=time_pair(candidate['local_time']);start,end=observation['time']
    if left < -extension_seconds-1e-6 or right > end-start+extension_seconds+1e-6:
        raise ValueError(f'local_time must stay within {-extension_seconds:.6g} to {end-start+extension_seconds:.6g} seconds from clip start')
    return dict(candidate,time=[start+left,start+right],source_fragment_time=list(observation['time']),
                retrieval_timing=observation.get('timing','unverified_asr' if observation['modality']=='text' else 'media_relative'))


def admit_candidates(pool,obj,observation,observation_id,parent_id=None,origin='coarse',extension_seconds=0):
    proposals=response_items(obj,'candidates');ids=[];rejected=[]
    if isinstance(obj,dict) and any(obj.get(k) for k in ('cues','evidence','traits')):
        raise ValueError('Candidate discovery cannot write Trait Memory')
    for index,candidate in enumerate(proposals):
        try:
            if origin=='coarse':
                if not isinstance(candidate,dict) or set(candidate)!={'hint','goal'}:
                    raise ValueError('A coarse candidate requires hint and goal only; Python binds its source')
                bound=dict(candidate,time=list(observation['time']),modality=observation['modality'])
            else:
                bound=bind_candidate(candidate,observation,extension_seconds)
                bound['time']=[max(0.,bound['time'][0]),min(pool.duration,bound['time'][1])]
            if not isinstance(bound['hint'],str) or not bound['hint'].strip() or not isinstance(bound['goal'],str) or not bound['goal'].strip():
                raise ValueError('Candidate hint and goal must describe the clue and inspection goal')
            if origin=='observation' and parent_id is not None:
                action=pool.validate(bound)
                parent=pool.items[parent_id]['action']
                if action['time']==parent['time'] and action['modality']==parent['modality'] and isinstance(bound.get('hint'),str) and bound['hint'].strip():
                    continue  # Already observed: deduplication, not a new task or protocol failure.
            ids.extend(add_candidate_proposals(pool,[bound],observation['time'],observation_id,parent_id,origin,observation['modality'],extension_seconds))
        except (ValueError,KeyError,TypeError) as error:
            rejected.append(dict(index=index,error=str(error)))
    return list(dict.fromkeys(ids)),rejected


def add_candidate_proposals(pool, proposals, scope, source_observation_id, parent_id=None, origin="coarse", source_modality=None, extension_seconds=0):
    """Validate model-proposed regions and bind their lineage in program state."""
    if not isinstance(proposals, list):
        raise ValueError("Candidate proposals must be a list")
    start, end = map(float, scope)
    added = []
    for candidate in proposals:
        if not isinstance(candidate,dict):
            raise ValueError('Each candidate must be an object')
        hint=candidate.get('hint')
        if source_modality is not None and (not isinstance(hint,str) or not hint.strip()):
            raise ValueError('Each candidate needs a hint describing information from the observed source')
        action = pool.validate(candidate)
        left, right = action["time"]
        if left < start-extension_seconds-1e-6 or right > end+extension_seconds+1e-6:
            raise ValueError("Candidate must stay inside the source observation")
        if parent_id is not None:
            parent = pool.items[parent_id]["action"]
            pleft, pright = parent["time"]
            if not pleft-extension_seconds-1e-6<=left<right<=pright+extension_seconds+1e-6:
                raise ValueError('Refinement exceeds its parent context margin')
            narrower = pleft-1e-6 <= left and right <= pright+1e-6 and right-left < pright-pleft-1e-6
            modality_change = action["modality"] != parent["modality"]
            extends = left<pleft-1e-6 or right>pright+1e-6
            if not narrower and not modality_change and not extends:
                raise ValueError("Refinement must narrow the current candidate or change modality")
        candidate_id = pool.add(action, candidate.get("priority", 0), parent_id=parent_id,
                                source_observation_id=source_observation_id, origin=origin,hint=hint,source_modality=source_modality,
                                source_fragment={k:candidate[k] for k in ('segment_id','segment_ids','quote','source_fragment_time','retrieval_timing') if k in candidate})
        if candidate_id:
            added.append(candidate_id)
    return added


def memory_view(memory):
    records=[]
    for record in memory_records(memory).values():
        decoded=decode_observation(record['content'])
        view={k:record[k] for k in ('id','observation_id','modality','retrieval_window','questions') if k in record}
        view.update(content=decoded['observations'],content_status=decoded['status'])
        records.append(view)
    return dict(observations=records)


def refinement_count(steps):
    count = 0
    for index, step in enumerate(steps):
        action = step['action']
        start, end = action['time']
        parent_id=step.get('parent_candidate_id')
        if parent_id is not None and any(previous.get('candidate_id')==parent_id for previous in steps[:index]):
            count+=1
            continue
        for previous in steps[:index]:
            parent = previous['action']
            left, right = parent['time']
            if left <= start < end <= right and (end-start < right-left-1e-6 or action['modality'] != parent['modality']):
                count += 1
                break
    return count
