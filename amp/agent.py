from copy import deepcopy
import math
from .backend import SYSTEM, message
from .common import TRAITS, LEVELS, digest, parse_json
from .state import CandidatePool, memory_changes, normalize_final, save_observation, contextual_final
from .provenance import index_transcript
from .scoring import prediction_feedback, minimize_state
from .perception import admit_candidates,memory_view,SOURCE_FOCUS,decode_observation,generation_schema

def dist(score):
    return {t:(score['traits'][t]['probabilities'] if score['traits'][t].get('predictive',True) else [.2]*5) for t in TRAITS}


class RequestValidationError(ValueError):
    def __init__(self, text, failures):
        super().__init__(text)
        self.failures = failures


class ActiveAgent:
    def __init__(self, backend, scorer, media, config):
        self.backend, self.scorer, self.media, self.config = backend, scorer, media, config

    def request(self, messages, kind, step, validate,choices=None,attempts=None):
        """Bounded format/contract repair; never fabricate an action or a cue.

        The research defaults remain one attempt. The local integration profile
        allows retries, saving failed generations and including their token cost.
        """
        original_messages = list(messages)
        messages = list(messages)
        failures = []
        attempts = max(1, int(self.config.get("max_validation_attempts", 1))) if attempts is None else attempts
        for attempt in range(attempts):
            decision = None
            try:
                if kind=='action' and attempt and choices and hasattr(self.backend,'generate_choice'):
                    obj,decision=self.backend.generate_choice(messages,choices,step)
                elif kind=='action':
                    text,decision=self.backend.generate_text(messages,kind,step)
                    obj=parse_json(text)
                else:obj,decision=self.backend.generate(messages,kind,step)
                value = validate(obj)
            except (ValueError, KeyError, TypeError) as error:
                failed = decision or getattr(error, "decision", None)
                if failed:
                    failures.append(dict(failed, validation_error=str(error)))
                if attempt + 1 == attempts:
                    raise RequestValidationError(f'{kind} step={step} failed validation after {attempts} attempt(s): {error}', failures) from error
                # Keep bad output in the audit record, not in the next attempt's
                # context: echoing invalid assistant examples invites copying them.
                messages = [dict(m) for m in original_messages]
                contract = {
                    "action": 'Return one executable observation using candidate_id or time and modality.',
                    "final": 'Put O,C,E,A,N first as five-class strings or null. Then give concise reasoning; finish the JSON without repeating conclusions.'
                }.get(kind, "Return only the requested JSON shape.")
                messages[0] = dict(messages[0], content=messages[0]["content"] + " Protocol correction: " + str(error)
                                   + " " + contract)
                messages.append(dict(role="user",content=[dict(type="text",text=f'Retry the original task. Your previous response was rejected because: {error}\n{contract}\nUse valid JSON with double-quoted keys. Do not invent evidence or timestamps.')]))
                continue
            if failures:
                decision["failed_attempts"] = failures
            return value, decision

    def scan_candidates(self, pool, observation_id, observation, question, step):
        if observation.get('questions'):question='\n'.join(q['question'] for q in observation['questions'])
        payload=dict(task=SOURCE_FOCUS[observation['modality']]+' Summarize this source. Optionally propose closer inspection of a concrete behavioral clue relevant to later personality analysis: hint states what was observed; goal states what remains unclear and how closer inspection could clarify it. Related details can share one goal. A summary with candidates=[] is sufficient when this source offers no useful follow-up. Use existing candidates as context for complementary requests.',
            question=question,question_context=observation.get('questions',[]),
            input_modalities=['audio','video'] if observation['modality']=='audio_video' else [observation['modality']],
            source_window=observation['time'],visual_representation=observation.get('visual_representation'),
            existing_candidates=[dict(modality=x['action']['modality'],time=x['action']['time'],goal=x['action']['goal'],
                hints=[l['hint'] for l in x['leads']]) for x in pool.snapshot()],
            allowed_modalities=pool.available_modalities,
            source_binding='Python binds candidates to this source window and modality. Finer windows and modality changes belong to Observe.',
            observation=dict(modality=observation['modality'],time=observation['time'],
                segments=[dict(text=v['text'],question=v.get('question','')) for v in observation.get('segments',[])]))
        messages=[dict(role='system',content='Use the supplied interview source to perform the current task.',
            output_schema=generation_schema('candidate_scan')),
            message(payload,observation.get('media',[]))]
        text,decision=self.backend.generate_text(messages,'candidate_scan',step)
        ids=[];rejected=[];summary='';summary_status='invalid'
        try:
            obj=parse_json(text)
            if not isinstance(obj,dict) or set(obj)!={'summary','candidates'} or not isinstance(obj['summary'],str) or not obj['summary'].strip() or not isinstance(obj['candidates'],list):
                raise ValueError('Return summary:string and candidates:array')
            summary=obj['summary'];summary_status='provided'
            ids,rejected=admit_candidates(pool,obj,observation,observation_id)
        except (ValueError,KeyError,TypeError) as error:rejected=[dict(index=None,error=str(error))]
        decision.update(candidate_status='partial' if rejected and ids else 'rejected' if rejected else 'accepted',
            candidate_ids=ids,rejected_candidates=rejected,source_observation_id=observation_id,
            coarse_summary=summary,summary_status=summary_status)
        observation.update(coarse_summary=summary,summary_status=summary_status)
        return decision

    def update(self, memory, observations, new_observations, step, question="", followup_goal=None, available_modalities=None,candidate_state=None,remaining_steps=None,media_duration=None):
        obs_id,obs=next(iter(new_observations.items()))
        if obs.get('questions'):question='\n'.join(q['question'] for q in obs['questions'])
        start,end=obs['time']
        margin=float(self.config.get('refinement_context_seconds',5))
        bounds=[round(max(-margin,-start),6),round(min(end+margin,media_duration if media_duration is not None else end+margin)-start,6)]
        payload=dict(task=SOURCE_FOCUS[obs['modality']]+' Describe the actions and communication found in this source, grouping related details into coherent findings and distinguishing self-reports from observed behavior. Use the question and prior observations as context. Optional candidates investigate a specific unresolved detail: hint states the current finding; goal states what additional information the requested window and modality can provide. Choose a window with enough context to answer that question, rather than subdividing the clip for coverage. Keep findings in observations and future requests in candidates; either array may be empty.',
            input_modalities=['audio','video'] if obs['modality']=='audio_video' else [obs['modality']],
            question=question,question_context=[dict(local_time=[max(0.,q['time'][0]-start),min(end-start,q['time'][1]-start)],question=q['question']) for q in obs.get('questions',[])],
            visual_representation=obs.get('visual_representation'),goal=followup_goal,modality=obs['modality'],clip_seconds=end-start,
            allowed_local_seconds=bounds,
            time_rule='Playback starts at 0 and ends at clip_seconds. local_time=[start,end] uses increasing seconds on this playback clock within allowed_local_seconds. covered_local_time is existing coverage on the same clock. Text marked unverified_asr has no precise media position.',
            available_modalities=available_modalities or [obs['modality']],
            max_video_window_seconds=self.config['max_clip_seconds'],
            prior_observations=[{k:r[k] for k in ('id','modality','content','content_status')} for r in memory_view(memory)['observations']
                if r['retrieval_window'][0]<end+margin and r['retrieval_window'][1]>start-margin],
            known_observation_requests=[dict(covered_local_time=[round(max(x['time'][0]-start,bounds[0]),6),round(min(x['time'][1]-start,bounds[1]),6)],modality=x['modality'],status=x['status'])
                for x in candidate_state or [] if x['time'][0]<start+bounds[1] and x['time'][1]>start+bounds[0]],remaining_observations=remaining_steps,
            segments=[dict(text=s['text'],question=s.get('question',''),timing=s.get('timing','unverified_asr'),
                **(dict(local_time=[round(v-start,6) for v in s['media_time']]) if s.get('media_time') is not None and s.get('timing')=='media_aligned' else {})) for s in obs.get('segments',[])])
        payload['task']+=' For closer inspection choose a narrower window or another modality. Known requests are already pending or saved. With zero remaining observations, candidates is empty.'
        messages=[dict(role='system',content='Use the supplied interview source to perform the current task. Leave personality judgments to Final Reasoning.'),message(payload,obs.get('media',[]))]
        messages[0]['output_schema']=generation_schema('memory',modalities=available_modalities or [obs['modality']])
        failures=[]
        for attempt in range(max(1,int(self.config.get('max_validation_attempts',1)))):
            content,decision=self.backend.generate_text(messages,'memory',step)
            decoded=decode_observation(content)
            if not any(e.get('scope')=='observation' for e in decoded['errors']):break
            if attempt+1>=max(1,int(self.config.get('max_validation_attempts',1))):break
            failures.append(dict(decision,validation_error='No complete observation description'))
            messages=[dict(messages[0],content=messages[0]['content']+' Return concise findings as separate strings in observations; complete the JSON.'),*messages[1:]]
        if failures:decision['failed_attempts']=failures
        after,ids=save_observation(memory,content,obs,obs_id)
        decision.update(observation_note=decoded['observations'],observation_format_status=decoded['status'],
            followup_proposals=decoded['candidates'],candidate_parse_errors=[e for e in decoded['errors'] if e.get('scope')!='observation'],
            observation_parse_errors=[e for e in decoded['errors'] if e.get('scope')=='observation'],
            memory_update_status='partial' if decoded['errors'] or decision.get('truncated') else 'accepted',stored_ids=ids)
        return after,ids,decision

    def repair_candidates(self,pool,observation,obs_id,parent_id,decision,errors,step):
        """Correct proposal parameters only, within the existing Observe attempt budget."""
        remaining=max(0,int(self.config.get('max_validation_attempts',1))-1-len(decision.get('failed_attempts',[])))
        if not remaining:return [],errors,None
        decoded=decode_observation(decision['text'])
        invalid_indices={e['index'] for e in errors if isinstance(e.get('index'),int)}
        proposals=[dict(proposal=c,errors=[e['error'] for e in errors if e.get('index') in (None,i)])
            for i,c in enumerate(decoded['candidates']) if not invalid_indices or i in invalid_indices]
        length=observation['time'][1]-observation['time'][0]
        margin=float(self.config.get('refinement_context_seconds',5))
        start=observation['time'][0]
        bounds=[round(max(-margin,-start),6),round(min(length+margin,pool.duration-start),6)]
        messages=[dict(role='system',content='Correct only the proposed observation parameters. Return {"candidates": []}, with each candidate containing local_time=[start,end] in seconds from clip start, modality, hint and goal. Preserve actual findings; do not generate new observations. An empty array is valid when no further inspection is useful.'),
            message(dict(observations=decoded['observations'],invalid_proposals=proposals,
                clip_seconds=length,allowed_local_seconds=bounds,available_modalities=pool.available_modalities,
                max_video_window_seconds=self.config['max_clip_seconds'],
                known_requests=[dict(covered_local_time=[round(max(item['action']['time'][0]-start,bounds[0]),6),round(min(item['action']['time'][1]-start,bounds[1]),6)],modality=item['action']['modality'],status=item['status'])
                    for item in pool.snapshot() if item['action']['time'][0]<start+bounds[1] and item['action']['time'][1]>start+bounds[0]],
                time_rule='Use two local seconds within the allowed range, not a point, global seconds or percentages. Adjacent continuation inside this context margin is allowed.'))]
        messages[0]['output_schema']=generation_schema('memory',modalities=pool.available_modalities,candidates_only=True)
        def validate(obj):
            if not isinstance(obj,dict) or set(obj)!= {'candidates'} or not isinstance(obj['candidates'],list):
                raise ValueError('Return only an object containing a candidates array')
            _,invalid=admit_candidates(deepcopy(pool),obj,observation,obs_id,parent_id=parent_id,origin='observation',extension_seconds=margin)
            if invalid:raise ValueError(str(invalid))
            return obj
        try:
            obj,repair=self.request(messages,'memory',step,validate,attempts=remaining)
            ids,invalid=admit_candidates(pool,obj,observation,obs_id,parent_id=parent_id,origin='observation',extension_seconds=margin)
            repair.update(candidate_output_valid=True,proposal_repair=True,refinement_candidate_ids=ids)
            return ids,invalid,repair
        except RequestValidationError as error:
            repair=dict(error.failures[-1],failed_attempts=error.failures[:-1]) if error.failures else None
            if repair:repair.update(candidate_output_valid=False,proposal_repair=True)
            return [],errors,repair

    def run(self, row):
        self.last_partial = None
        mode = self.config.get("perception", "full")
        selection_policy=self.config.get('selection_policy','queue')
        if selection_policy not in ('queue','model'):raise ValueError('selection_policy must be queue or model (historical comparison)')
        # No raw labels, generated annotations, file IDs or GT reach the policy/scorer.
        scan = self.media.coarse(row, self.config["coarse_video_fps"], self.config["coarse_chunk_seconds"])
        duration = scan["duration"]
        media_id = scan.get("media_id", digest(row.get("video", row["id"])))
        question = row.get("question", scan.get("question", ""))
        transcript = index_transcript(scan["transcript"], media_id, duration)
        available=scan.get('available_modalities')
        if available is None:
            available=(['text'] if transcript else [])+(['audio'] if scan['audio'] else [])+(['video'] if scan.get('video') else [])
            if 'audio' in available and 'video' in available:available.append('audio_video')
        pool = CandidatePool(duration, duration if mode=='static' else self.config["max_clip_seconds"],available)
        memory, observations, decisions, steps = dict(observations=[]), {}, [], []
        # Preserve chronological source order across coarse modalities.
        coarse_sources = []
        for modality, entries in (("video", scan.get("video", [])), ("audio", scan["audio"])):
            for i, entry in enumerate(entries):
                coarse_sources.append((entry.get("time", [0])[0], modality, i, entry))
        for _, modality, i, entry in sorted(coarse_sources, key=lambda item: (item[0], item[1])):
            paired=next((key for key,obs in observations.items() if selection_policy=='queue' and mode not in ('static','fixed_context')
                and obs['modality'] in ('audio','video') and obs['modality']!=modality
                and all(abs(a-b)<=1e-3 for a,b in zip(obs['time'],entry['time']))),None)
            if paired is not None:
                previous=observations.pop(paired)
                observations[f'global_audio_video_{i}']=dict(entry,modality='audio_video',
                    component_times={previous['modality']:previous['time'],modality:entry['time']},
                    media=previous.get('media',[])+entry.get('media',[]),media_id=media_id,question=question)
            else:observations[f"global_{modality}_{i}"] = dict(entry, media_id=media_id, question=question)
        observations["global_text"] = dict(time=[0, duration], modality="text", segments=transcript,
                                            media=[], timing="unverified_asr", media_id=media_id, question=question)
        # Stage 1 fills the candidate pool only. Coarse regions never become evidence.
        fixed_sources=[deepcopy(obs) for obs in observations.values() if obs['modality']!='text' or obs.get('segments')]
        if mode!='static':
            for index,(obs_id,observation) in enumerate(observations.items()):
                if observation['modality']=='text' and not observation.get('segments'):
                    observation.update(coarse_summary='No transcript supplied.',summary_status='unavailable')
                    continue
                decision=self.scan_candidates(pool,obs_id,observation,question,index)
                decisions.append(decision)
        coarse_summaries=[dict(time=obs['time'],modality=obs['modality'],summary=obs.get('coarse_summary',''),summary_status=obs.get('summary_status','not_scanned')) for obs in observations.values()]
        if selection_policy=='model':
            global_text_context=[dict(segment_id=s['segment_id'],time=s['media_time'] or s['source_time'],timing=s['timing'],text=s['text'],question=s.get('question','')) for s in transcript]
            global_media_overview=[s for s in coarse_summaries if s['modality']!='text']
            global_text_overview=next(s for s in coarse_summaries if s['modality']=='text')
        score = self.scorer.score(memory)
        initial = dict(memory=deepcopy(memory), score=score, new_ids=[], candidate_count=len(pool.items))
        scoring_history = [score]
        self.last_partial = dict(sample_id=row["id"], dataset=row["dataset"], initial=initial,
            memory=memory, observations=observations, decisions=decisions, steps=steps,
            pool=pool.snapshot(), status="incomplete")
        spent = sum(x["time"][1]-x["time"][0] for x in scan["audio"])
        active_spent = 0.
        budget = self.config.get("max_seconds")
        budget = math.inf if budget is None else float(budget)
        cost_normalizer = self.config.get("cost_scale_seconds",300.) if math.isinf(budget) else budget
        stopped = False
        stop_reason = "max_steps"
        for step in range(len(fixed_sources) if mode == "static" else self.config["max_steps"]):
            remaining = budget - active_spent
            if remaining <= 1e-6:
                stop_reason = "budget_exhausted"
                stopped = True
                break
            selectable=[]
            for item in pool.snapshot():
                if item["status"] != "pending" or item["action"]["time"][1]-item["action"]["time"][0] > remaining+1e-6:
                    continue
                selectable.append(item)
            if not selectable and mode!='static':
                stop_reason="candidate_pool_empty" if not pool.snapshot() else "candidate_pool_exhausted"
                if any(x['status']=='pending' for x in pool.items.values()):stop_reason='remaining_budget_insufficient'
                stopped=True
                break
            if selection_policy=='model':
                candidate_view=[dict(id=x['id'],status=x['status'],action=x['action'],parent_id=x.get('parent_id'),
                                     leads=[dict(source_modality=l['source_modality'],hint=l['hint']) for l in x['leads']]) for x in selectable]
                payload = dict(task='Choose the next observation using the current task state. pool lists pending executable candidates; completed_observations lists finished reads and saved Memory IDs. Reuse saved results for completed reads, then select one pending pool ID or a new time/modality pair. To clarify a completed read, choose a narrower window or another modality.',
                    memory=memory_view(memory),duration=duration,question=question,
                    global_text_context=global_text_context,global_media_overview=global_media_overview,
                    global_text_summary=global_text_overview['summary'],global_text_summary_status=global_text_overview['summary_status'],
                    available_modalities=available,max_clip=self.config['max_clip_seconds'],
                    remaining_seconds=None if math.isinf(budget) else budget-active_spent,
                    completed_observations=[dict(id=s['candidate_id'],status='done',action=s['action'],
                        parent_id=s.get('parent_candidate_id'),observation_id=s['observation_id'],memory_ids=s['changed_ids'],
                        followup_status=s.get('followup_status','not_proposed'),followup_errors=s.get('followup_errors',[])) for s in steps],
                    pool=candidate_view)
                feedback=prediction_feedback(score)
                if feedback is not None:payload['scoring_feedback']=feedback
                if mode=='generic':
                    payload.pop('memory');payload.pop('scoring_feedback',None)
                    for entry,s in zip(payload['completed_observations'],steps):
                        entry['note']=decisions[s['memory_decision_index']].get('observation_note','')
            selected={}
            action_decision_index=None
            if mode=='static':
                fixed=fixed_sources[step]
                action=dict(time=fixed['time'],modality=fixed['modality'],goal='Interpret the fixed global context')
            elif mode == "fixed_context":
                action = selectable[0]['action']
            elif selection_policy=='queue':
                selected=min(selectable,key=lambda x:(x['origin']!='coarse',int(x['id'][1:])))
                action=deepcopy(selected['action'])
            else:
                def validate_action(obj):
                    selected.clear()
                    action,key=pool.select_action(obj,{x['id'] for x in selectable})
                    if key is not None:selected.update(next(x for x in selectable if x['id']==key))
                    if active_spent + action["time"][1]-action["time"][0] > budget+1e-6:
                        raise ValueError("Observation exceeds remaining budget")
                    return action
                action_system=SYSTEM+' This is SELECT. Use pool and completed_observations as the current execution state; saved Memory and coarse descriptions provide content. Return one JSON action object. '+('For a pool choice use candidate_id equal to an available identifier shown under id in the pool. ' if candidate_view else '')+'For a new observation return {"time":[start,end],"modality":"..."}, optionally adding goal.'
                action_system+=f" Current pending IDs: {', '.join(x['id'] for x in selectable)}. Completed IDs: {', '.join(s['candidate_id'] for s in steps) or 'none'}."
                try:
                    action, decision = self.request([{"role": "system", "content": action_system}, message(payload)], "action", step, validate_action,choices=[x["id"] for x in selectable])
                except RequestValidationError as error:
                    failure=dict(error.failures[-1] if error.failures else dict(kind='action',step=step,tokens=[]),
                        action_status='rejected',generation_failed=True,failed_attempts=error.failures[:-1])
                    decisions.append(failure)
                    stop_reason='action_format_error'
                    break  # Preserve the observations and still produce Final; no fabricated STOP.
                decision['action']=deepcopy(action)
                decisions.append(decision)
                action_decision_index=len(decisions)-1
            action = pool.validate(action)
            cost = 0. if mode=='static' else action["time"][1]-action["time"][0]
            if active_spent + cost > budget+1e-6:
                raise ValueError("Agent exceeded observation budget")
            key = pool.start(action)
            candidate_id = selected.get("id", key)
            selected_candidate = pool.items.get(candidate_id, pool.items.get(key, {}))
            try:
                observation = deepcopy(fixed_sources[step]) if mode=='static' else self.media.observe(row, action)
                if observation.get('modality')!=action['modality'] or observation.get('time')!=action['time']:
                    raise ValueError('Observation tool returned a different modality or interval than requested')
                if observation.get('media_id',media_id)!=media_id:
                    raise ValueError('Observation tool returned a different source media')
                pool.finish(key)
            except Exception:
                pool.finish(key, False)
                raise
            obs_id = f"static_{step}" if mode=='static' else f"active_{step}"
            if pool.items[key]['origin']=='active':pool.items[key]['source_observation_id']=obs_id
            observations[obs_id] = dict(observation, media_id=media_id, question=question)
            previous_memory = deepcopy(memory)
            before = score
            memory, changed_ids, decision = self.update(memory, observations, {obs_id: observations[obs_id]}, step, question,
                followup_goal=([l['goal'] for l in selected_candidate.get('leads',[]) if l.get('goal')] or action['goal']) if mode not in ('generic','fixed_context','static') else None,available_modalities=available,
                candidate_state=[dict(time=x['action']['time'],modality=x['action']['modality'],status=x['status']) for x in pool.snapshot()],
                remaining_steps=self.config['max_steps']-step-1 if selection_policy=='queue' else None,media_duration=duration)
            followup_ids=[];rejected=list(decision.get('candidate_parse_errors',[]))
            proposals=decision.get('followup_proposals',[])
            followups_enabled=mode not in ('static','fixed_context') and not (selection_policy=='queue' and (step+1>=self.config['max_steps'] or active_spent+cost>=budget-1e-6))
            followup_status='disabled' if mode in ('static','fixed_context') else 'disabled_budget' if not followups_enabled else 'not_proposed'
            if followups_enabled:
                if proposals:
                    try:
                        followup_ids,errors=admit_candidates(pool,dict(candidates=proposals),
                            observations[obs_id],obs_id,parent_id=candidate_id,origin='observation',extension_seconds=float(self.config.get('refinement_context_seconds',5)))
                        rejected.extend(errors)
                    except (ValueError,KeyError,TypeError) as error:rejected.append(dict(index=None,error=str(error)))
                followup_status=('partial' if followup_ids else 'invalid') if rejected else 'added' if followup_ids else 'no_new_target' if proposals else 'not_proposed'
            decision.update(refinement_candidate_ids=followup_ids,rejected_candidates=rejected,followup_status=followup_status)
            memory_decision_index=len(decisions)
            memory_decision_indices=[memory_decision_index]
            decisions.append(decision)
            if rejected and followups_enabled:
                decision['candidate_output_valid']=False
                repaired,remaining,repair=self.repair_candidates(pool,observations[obs_id],obs_id,candidate_id,decision,rejected,step)
                if repair:
                    memory_decision_indices.append(len(decisions));decisions.append(repair)
                followup_ids=list(dict.fromkeys([*followup_ids,*repaired]));rejected=remaining
                if not rejected:followup_status='corrected'
                decision.update(refinement_candidate_ids=followup_ids,followup_status=followup_status)
            self.last_partial.update(memory=memory,pool=pool.snapshot())
            try:
                scoring_changed=minimize_state(previous_memory)!=minimize_state(memory)
                if scoring_changed:
                    score = self.scorer.score(memory)
                else:
                    score=deepcopy(score)
                    score['metadata'].update(reused=True,api_calls=0,billed_cost=0)
            except (RuntimeError,ValueError) as error:
                self.last_partial['scoring_error']=str(error)
                raise
            scoring_history.append(score)
            self.last_partial.update(memory=memory, score=score, pool=pool.snapshot())
            active_spent += cost
            steps.append(dict(index=step, action=action, observation_id=obs_id, before=dist(before), after=dist(score),
                              action_type='static' if mode=='static' else 'fixed' if mode=='fixed_context' else 'scheduled' if selection_policy=='queue' else 'candidate' if selected else 'free',action_decision_index=action_decision_index,
                              memory_decision_index=memory_decision_index,memory_changed=bool(changed_ids),
                              memory_decision_indices=memory_decision_indices,
                              scoring_memory_changed=scoring_changed,
                              score_valid_before={t:not before['metadata'].get('offline',False) and before['metadata'].get('predictive',True) and before['traits'][t].get('predictive',False) for t in TRAITS},
                              score_valid_after={t:not score['metadata'].get('offline',False) and score['metadata'].get('predictive',True) and score['traits'][t].get('predictive',False) for t in TRAITS},
                              candidate_id=candidate_id,parent_candidate_id=selected_candidate.get("parent_id"),
                              refinement_candidate_ids=followup_ids,followup_status=followup_status,followup_errors=rejected,
                              new_ids=list(changed_ids),
                              changed_ids=changed_ids, changes=memory_changes(previous_memory, memory),
                              cost=cost / cost_normalizer, memory=deepcopy(memory)))
        if not stopped and active_spent >= budget - 1e-6:
            stop_reason = "budget_exhausted"
            stopped = True
        elif not stopped and mode!='static' and not any(x['status']=='pending' for x in pool.items.values()):
            stop_reason = 'candidate_pool_exhausted' if pool.items else 'candidate_pool_empty'
            stopped = True
        if not stopped and stop_reason=='max_steps' and mode!='static':
            stopped=True
        if mode == "static":
            stop_reason = "static_mode"
        for item in steps:
            later=[i for i,d in enumerate(decisions) if d['kind']=='action' and i>item['memory_decision_index']]
            item['next_action_decision_index']=later[0] if later else None
        final_payload=dict(task='Observation is finished. Return one JSON object, putting O,C,E,A,N class strings or null first, then concise reasoning and contextual expressions. State each conclusion once and finish the JSON. Each contextual item has trait, context_type (reported/observed/unknown), context, expression and citations (saved M IDs). Explain supported situation differences and their relation to the overall judgment, without averaging context directions or claiming personality changed over time. Proposed candidates and goals are plans, not observed facts.',
            perception_end=dict(reason=stop_reason,observations=len(steps),pending_candidates=sum(x['status']=='pending' for x in pool.items.values())),
            question=question,traits=TRAITS,levels=LEVELS,memory=memory_view(memory))
        feedback=prediction_feedback(score)
        if feedback is not None:final_payload['scoring_feedback']=feedback
        final_status='generated';final_error=None
        try:
            (final,final_explanation,raw_final),decision=self.request([dict(role='system',content='Use saved observations to perform the final personality reasoning task.',output_schema=generation_schema('final',memory=memory)),message(final_payload)],
                'final',len(steps),lambda obj:(normalize_final(obj),obj.get('reasoning'),obj))
        except RequestValidationError as error:
            final_status='unavailable';final_error=str(error)
            raw_final={t:None for t in TRAITS};final={t:dict(level=None,reasoning=None) for t in TRAITS};final_explanation=None
            decision=dict(error.failures[-1],failed_attempts=error.failures[:-1]) if error.failures else dict(kind='final',step=len(steps),text='',tokens=[])
            decision.update(generation_failed=True,final_error=final_error)
        contextual,contextual_errors=contextual_final(raw_final,memory)
        contextual_status='partial' if contextual_errors else 'provided' if contextual else 'empty' if 'contextual' in raw_final else 'not_provided'
        decision.update(contextual_status=contextual_status,contextual_errors=contextual_errors)
        decisions.append(decision)
        final_reasoning_status=('provided' if final_explanation or all(v.get('reasoning') for v in final.values()) else 'not_provided')
        final_abstentions={t:final[t]['reasoning'] for t in TRAITS if final[t]['level'] is None} if final_status!='unavailable' else {}
        return dict(sample_id=row["id"], dataset=row["dataset"], split=row["split"], person_id=row["person_id"],
                    selection_policy=selection_policy,
                    run_status='partial' if final_status!='generated' or contextual_errors or any(s.get('followup_status') in ('partial','invalid') for s in steps) or any(d.get('candidate_status') in ('partial','rejected') or d.get('action_status')=='rejected' or d.get('memory_update_status') in ('partial','rejected') for d in decisions) else 'complete',
                    final_status=final_status,final_prediction_available=final_status!='unavailable',final_error=final_error,
                    initial=initial, memory=memory, observations=observations, steps=steps, final=final, final_explanation=final_explanation,contextual=contextual,contextual_status=contextual_status,contextual_errors=contextual_errors,contextual_semantic_review="unreviewed",final_reasoning_status=final_reasoning_status,final_abstentions=final_abstentions,
                    global_scan=dict(coarse_summaries=coarse_summaries,input_modalities=[m for m,present in [('text',bool(transcript)),('audio',bool(scan['audio'])),('video',bool(scan.get('video')))] if present],observation_modes=available,**{k:scan[k] for k in ("sampling_fps","full_audio_coverage","audio_coverage","video_coverage","duration","sampled_frames","audio_samples","audio_sample_rate","chunk_seconds","text_segmentation","video_segmentation","audio_segmentation") if k in scan}),
                    score=score, scoring_history=scoring_history, decisions=decisions, pool=pool.snapshot(), stopped=stopped, stop_reason=stop_reason,
                    evidence_status="recorded_unreviewed" if memory.get("observations") else "insufficient",
                    memory_protocol='observation_text',
                    cost=dict(active_seconds=active_spent, active_budget_seconds=self.config.get("max_seconds"), coarse_audio_seconds=spent, coarse_frames=scan.get("sampled_frames",0),
                              jev_api_calls=sum(s['metadata'].get('api_calls',0) for s in scoring_history),
                              jev_estimated_cost=(sum(s['metadata'].get('billed_cost',0) for s in scoring_history) if all(s['metadata'].get('billed_cost',0) is not None for s in scoring_history) else None),
                              coarse_video_segments=len(scan.get("video",[])),coarse_video_fps=scan.get("sampling_fps"),
                              generated_tokens=sum(attempt.get("completion_tokens",len(attempt["tokens"])) for d in decisions for attempt in [d,*d.get("failed_attempts",[])]),
                              input_tokens=sum(attempt.get("prompt_tokens",0) for d in decisions for attempt in [d,*d.get("failed_attempts",[])])),
                    offline=any(d.get("offline",False) for d in decisions), config=deepcopy(self.config))
