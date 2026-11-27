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
    target=obj(dict(local_time=dict(type='array',items=dict(type='number'),minItems=2,maxItems=2),
        modality=dict(type='string',enum=list(modalities or MODALITIES)),hint=text,goal=text))
    if kind=='candidate_scan':
        return obj(dict(summary=text,candidates=dict(type='array',items=target)))
    if kind=='memory':
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


def response_items(obj,field):
    if isinstance(obj,dict):
        if isinstance(obj.get(field),list):return obj[field]
    raise ValueError(f'Return a JSON object with a {field} array')


# Requests may round past an end by up to this much; they are cut to the end and the cut is recorded.
BOUNDARY_TOLERANCE_SECONDS = .5


def clamp_end(left, right, low, high):
    """Clamp a window to [low, high] when it overshoots by at most the tolerance; return (left, right, cut seconds)."""
    if left < low-BOUNDARY_TOLERANCE_SECONDS or right > high+BOUNDARY_TOLERANCE_SECONDS:
        return None
    cut = max(0., low-left)+max(0., right-high)
    return max(low, left), min(high, right), cut


def bind_candidate(candidate,observation,duration):
    """Observe proposals use seconds on the current clip's playback clock and may point anywhere in the recording."""
    if not isinstance(candidate,dict) or set(candidate)!={'local_time','modality','hint','goal'}:
        raise ValueError('A follow-up requires local_time, modality, hint and goal')
    left,right=time_pair(candidate['local_time']);start=observation['time'][0]
    window=clamp_end(left,right,-start,duration-start)
    if window is None:
        raise ValueError(f'local_time must stay within {-start:.6g} to {duration-start:.6g} seconds from clip start (the recording)')
    left,right,cut=window
    bound=dict(candidate,time=[start+left,start+right],source_fragment_time=list(observation['time']),
               retrieval_timing=observation.get('timing','unverified_asr' if observation['modality']=='text' else 'media_relative'))
    if cut:bound['boundary_cut_seconds']=round(cut,6)
    return bound


def admit_candidates(pool,obj,observation,observation_id,parent_id=None,origin='coarse'):
    proposals=response_items(obj,'candidates');ids=[];rejected=[]
    for index,candidate in enumerate(proposals):
        try:
            bound=bind_candidate(candidate,observation,pool.duration)
            if origin=='coarse':
                start,end=observation['time']
                window=clamp_end(*bound['time'],start,end)
                if window is None:
                    raise ValueError(f'A coarse candidate marks a moment inside this source: local_time within 0 to {end-start:.6g}')
                bound['time']=list(window[:2])
                if window[2]:bound['boundary_cut_seconds']=round(bound.get('boundary_cut_seconds',0.)+window[2],6)
            if not isinstance(bound['hint'],str) or not bound['hint'].strip() or not isinstance(bound['goal'],str) or not bound['goal'].strip():
                raise ValueError('Candidate hint and goal must describe the clue and inspection goal')
            ids.extend(add_candidate_proposals(pool,[bound],observation_id,parent_id,origin,observation['modality']))
        except (ValueError,KeyError,TypeError) as error:
            rejected.append(dict(index=index,error=str(error)))
    return list(dict.fromkeys(ids)),rejected


def add_candidate_proposals(pool, proposals, source_observation_id, parent_id=None, origin="coarse", source_modality=None):
    """Validate model-proposed regions and bind their lineage in program state.

    A completed read is never re-queued (CandidatePool.add); a new refinement must
    add unread recording content or be the single focus inside its original region.
    """
    added = []
    for candidate in proposals:
        hint=candidate.get('hint')
        if source_modality is not None and (not isinstance(hint,str) or not hint.strip()):
            raise ValueError('Each candidate needs a hint describing information from the observed source')
        action = pool.validate(candidate)
        if origin=='observation' and pool.matching(action) is None and not pool.admit_refinement(
                action,parent_id,dict(source_observation_id=source_observation_id,hint=hint,goal=action['goal'])):
            continue
        candidate_id = pool.add(action, candidate.get("priority", 0), parent_id=parent_id,
                                source_observation_id=source_observation_id, origin=origin,hint=hint,source_modality=source_modality,
                                source_fragment={k:candidate[k] for k in ('segment_id','segment_ids','quote','source_fragment_time','retrieval_timing','boundary_cut_seconds') if k in candidate})
        if candidate_id:
            added.append(candidate_id)
    return added


OBSERVABLE_ONLY = ('Describe only what is seen or heard: no hidden motives, authenticity or inner states, '
    'and no numeric measurements of voice or movement.')


READ_SUPPORT_RULE = ('M IDs identify reads, not independent behaviors. Assess support from distinct behaviors in context; '
    'repeated or complementary reads clarify a behavior rather than multiply independent support. '
    'overlapping_media_reads marks shared recording regions only: different behaviors may share a window, '
    'and disjoint reads may describe the same event. Keep uncertain event relations unconfirmed and cite the reads used.')


def memory_view(memory, observations=None):
    records=[]
    located=[]
    for record in memory_records(memory).values():
        decoded=decode_observation(record['content'])
        view={k:record[k] for k in ('id','observation_id','modality','retrieval_window','questions') if k in record}
        view.update(content=decoded['observations'],content_status=decoded['status'])
        source=(observations or {}).get(record.get('observation_id'),{})
        window=record.get('retrieval_window')
        if (record.get('modality') in ('audio','video','audio_video') and source.get('media_id')
                and source.get('time')==window and source.get('modality')==record.get('modality')
                and isinstance(window,list) and len(window)==2):
            overlaps=[key for key,media_id,previous in located if media_id==source['media_id']
                      and max(previous[0],window[0])<min(previous[1],window[1])]
            if overlaps:view['overlapping_media_reads']=overlaps
            located.append((record['id'],source['media_id'],window))
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
