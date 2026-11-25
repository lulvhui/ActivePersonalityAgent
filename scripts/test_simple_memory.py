"""Local perception and append-only Memory smoke test. No hosted APIs or training."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amp.agent import ActiveAgent
from amp.backend import QwenBackend
from amp.common import read_json,read_jsonl,write_json,TRAITS,parse_json
from amp.media import MediaTools
from amp.scoring import OfflineTraitScorer
from amp.state import empty_memory
from amp.state import CandidatePool
from amp.perception import admit_candidates


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-id',default='RecruitView:1703')
    parser.add_argument('--max-steps',type=int,default=4,help='Smoke-test maximum; production config is unchanged')
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--coarse-only',action='store_true',help='Verify coarse source summaries without repeating active inference')
    scope.add_argument('--observe-only',action='store_true',help='Inspect a whole coarse-sized segment and save every finding in the response')
    args=parser.parse_args()
    if (args.output/'summary.json').exists():raise ValueError('Use a fresh output directory')
    row=next(r for r in read_jsonl(ROOT/'runs/omniagent_expanded_20261117/data/manifest.jsonl') if r['id']==args.sample_id)
    row={k:row[k] for k in ('id','dataset','split','person_id','video','segments','question') if k in row}
    cfg=read_json(ROOT/'configs/local_omni3b.json')
    cfg.update(model=args.model,temperature=0,scorer='offline',max_steps=args.max_steps)
    write_json(args.output/'config.json',cfg)
    report=dict(model=args.model,sample_id=row['id'],external_api_calls=0,training=False,
                memory_semantics='append complete raw observation text without parsing, deduplication, event inference or semantic filtering',status='loading')
    write_json(args.output/'summary.json',report)
    backend=QwenBackend(args.model,max_tokens=cfg['max_new_tokens'],temperature=0,generation_log_dir=args.output/'generations')
    media=MediaTools(str(args.output/'media'),cfg['max_clip_seconds'],cfg['video_fps'])
    agent=ActiveAgent(backend,OfflineTraitScorer(),media,cfg)
    duration=media.probe(row['video'])['duration']
    if args.observe_only:
        obs=media.observe(row,dict(time=[0.,min(20.,duration)],modality='audio_video'))
        content,ids,d=agent.update(dict(observations=[]),{'active_0':obs},{'active_0':obs},0,
            row.get('question',''),followup_goal='Inspect how the speaker handled an unfamiliar project',
            available_modalities=['text','audio','video','audio_video'])
        pool=CandidatePool(duration,cfg['max_clip_seconds'])
        parent=pool.add(dict(time=obs['time'],modality=obs['modality']))
        pool.finish(parent)
        try:proposal=parse_json(d['text'])
        except ValueError:proposal=None
        children,rejected=[],[]
        if isinstance(proposal,dict) and 'candidates' in proposal:
            try:children,rejected=admit_candidates(pool,dict(candidates=proposal['candidates']),obs,'active_0',parent_id=parent,origin='observation',extension_seconds=cfg.get('refinement_context_seconds',5))
            except (ValueError,KeyError,TypeError) as error:rejected=[dict(error=str(error))]
        d.update(refinement_candidate_ids=children,rejected_candidates=rejected)
        write_json(args.output/'pool.json',pool.snapshot())
        write_json(args.output/'observation.json',dict(observation=obs,decision=d,memory=content))
        report.update(status='observe_only_complete',observation_mode='audio_video',time=obs['time'],
            saved_observation_records=len(ids),response_characters=len(d['text']),truncated=d.get('truncated',False),
            raw_output_preserved=content['observations'][0]['content']==d['text'],native_media=d.get('native_media'),
            refinement_candidate_ids=children,rejected_candidates=rejected,
            clue_count='not_computed; one record contains the full multi-finding response')
        write_json(args.output/'summary.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return
    if args.coarse_only:
        scan=media.coarse(row,cfg['coarse_video_fps'],cfg['coarse_chunk_seconds'])
        pool=CandidatePool(duration,cfg['max_clip_seconds'])
        sources=[(f'global_{mode}_{i}',obs) for mode in ('audio','video') for i,obs in enumerate(scan[mode])]
        sources.sort(key=lambda item:item[1]['time'][0])
        sources.append(('global_text',dict(modality='text',time=[0.,duration],segments=scan['transcript'],media=[])))
        summaries=[]
        for i,(oid,obs) in enumerate(sources):
            d=agent.scan_candidates(pool,oid,obs,row.get('question',''),i)
            summaries.append(dict(modality=obs['modality'],time=obs['time'],summary=d['coarse_summary'],
                summary_status=d['summary_status'],candidate_status=d['candidate_status'],native_media=d.get('native_media')))
            write_json(args.output/f'{oid}.json',dict(observation=obs,decision=d))
        report.update(status='coarse_only_complete',input_modalities=['text','audio','video'],
            coarse_summaries=summaries,audio_coverage=scan['audio_coverage'],video_coverage=scan['video_coverage'],
            chunk_seconds=scan['chunk_seconds'],video_fps=scan['sampling_fps'],candidate_count=len(pool.items))
        write_json(args.output/'summary.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return
    memory=empty_memory();results=[]
    for i,modality in enumerate(('text','audio','video','audio_video')):
        # Direct full-interview reading checks input access without Select/STOP.
        # Coarse=True permits full media at the configured coarse FPS.
        obs=media.observe(row,dict(time=[0.,duration],modality=modality),coarse=True,fps=cfg['coarse_video_fps'])
        before=len(memory.get('observations',[]))
        memory,ids,d=agent.update(memory,{f'probe_{i}':obs},{f'probe_{i}':obs},i,row.get('question',''))
        result=dict(modality=modality,status=d['memory_update_status'],saved=len(ids),
            total_saved=len(memory.get('observations',[])),native_media=d.get('native_media'),
            raw_output_preserved=memory['observations'][-1]['content']==d['text'],truncated=d.get('truncated',False),semantic_quality='not_evaluated')
        assert result['total_saved']==before+len(ids)
        write_json(args.output/f'read_{modality}.json',dict(result=result,observation=obs,decision=d,memory=memory))
        results.append(result);report.update(status='reading',perception=results)
        write_json(args.output/'summary.json',report)
    write_json(args.output/'saved_memory.json',memory)
    # Separate autonomous run through the same production implementation.
    try:
        t=agent.run(row);write_json(args.output/'trajectory.json',t)
        report['active_run']=dict(status=t['run_status'],candidates=t['initial']['candidate_count'],
            observations=len(t['steps']),saved=len(t['memory'].get('observations',[])),
            stop_reason=t['stop_reason'],final=t['final'],final_reasoning_status=t.get('final_reasoning_status'),
            observation_modes=[s['action']['modality'] for s in t['steps']])
        write_json(args.output/'coarse_scan.json',t['global_scan'])
    except (ValueError,RuntimeError,KeyError,TypeError) as error:
        write_json(args.output/'failed_trajectory.json',dict(error=str(error),partial=agent.last_partial,failures=getattr(error,'failures',[])))
        report['active_run']=dict(status='failed',error=str(error))
    report.update(status='complete',semantic_quality='not_evaluated',teacher_accepted=False)
    write_json(args.output/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
