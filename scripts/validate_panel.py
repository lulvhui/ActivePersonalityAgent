"""Prepare a fixed diagnostic panel and run complete native episodes.

Reference annotations remain separate from inference rows and are NOT human
gold labels. A failed/empty episode remains in the denominator. No live API.
"""
import argparse,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amp.common import TRAITS,read_json,read_jsonl,write_json,write_jsonl
from amp.state import memory_records
from amp.perception import decode_observation

def prepare(panel_path,output):
    panel=read_json(panel_path)
    source_rows={r['id']:r for part in ('train','dev','test') for r in read_jsonl(ROOT/f'runs/data/{part}.jsonl')}
    rows=[];references=[]
    for case in panel['cases']:
        original=source_rows[case['sample_id']]
        row={k:copy.deepcopy(original[k]) for k in ('dataset','split','person_id','video','segments')}
        row.update(id=case['id'],question=original.get('question') or '')
        ref=copy.deepcopy(case);ref.update(person_id=original['person_id'],split=original['split'],alignment_verified=False)
        if 'segment_index' in case:
            segment=original['segments'][case['segment_index']]
            start=max(0.,segment['start']-2);end=segment['end']+2
            clip=output/'media'/f"{case['id']}.mp4";clip.parent.mkdir(parents=True,exist_ok=True)
            if not clip.exists():
                subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',row['video'],'-t',str(end-start),'-c:v','libx264','-c:a','aac','-y',str(clip)],check=True)
            row.update(video=str(clip),segments=[dict(start=segment['start']-start,end=segment['end']-start,text=segment['text'],timing='unverified_asr')])
            ref['raw_media_window']=[start,end]
        ref['source_text']=[s['text'] for s in row['segments']]
        rows.append(row);references.append(ref)
    # Original person split is retained; the panel creates no new train/dev split.
    splits={}
    for row in rows:
        key=(row['dataset'],str(row['person_id']))
        if key in splits and splits[key]!=row['split']:raise ValueError('Person crosses dataset split')
        splits[key]=row['split']
    write_jsonl(output/'manifest.jsonl',rows)
    write_json(output/'references.json',dict(annotation_status=panel['annotation_status'],cases=references,scope=panel['scope']))
    write_json(output/'alignment_review.json',dict(scope='Unverified template; never imported into provenance automatically',cases=[dict(case_id=r['id'],verified=False,speaker=None,verified_media_time=None,reviewer='') for r in references]))

def metrics(trajectory,complete,failures=()):
    steps=trajectory.get('steps',[])
    active=[v for k,v in trajectory.get('observations',{}).items() if k.startswith('active_')]
    produced=sum(bool(decode_observation(r['content'])['observations']) for r in trajectory.get('memory',{}).get('observations',[]))
    actions=[d for d in trajectory.get('decisions',[]) if d['kind']=='action']
    decisions=trajectory.get('decisions',[])
    attempts=[a for d in decisions for a in [d,*d.get('failed_attempts',[])]]+list(failures)
    observations=trajectory.get('observations',{})
    first=actions[0].get('action',{}) if actions else None
    return dict(complete=complete and trajectory.get('run_status','complete')=='complete',episode_returned=complete,first_stop=first.get('stop') is True if first is not None else None,
        candidate_count=trajectory.get('initial',{}).get('candidate_count',0),attempted_observations=len(active),accepted_observations=sum(s.get('action_type')!='static' for s in steps),
        readable_observation_records=produced,active_seconds=sum(o['time'][1]-o['time'][0] for o in active),
        final_trait_coverage=sum(x.get('level') is not None for x in trajectory.get('final',{}).values())/5,
        rejected_candidate_count=sum(len(d.get('rejected_candidates',[])) for d in decisions),
        generated_tokens=sum(len(a.get('tokens',[])) for a in attempts),input_tokens=sum(a.get('prompt_tokens',0) for a in attempts),
        coarse_audio_seconds=sum(o['time'][1]-o['time'][0] for k,o in observations.items() if k.startswith('global_audio_')),
        coarse_video_frames=sum(m.get('sampled_frames',0) for k,o in observations.items() if k.startswith('global_video_') for m in o.get('media',[]) if m['type']=='video'),
        semantic_quality_verified=False)

def summarize(records):
    n=len(records);m=[r['metrics'] for r in records];actions=[v for v in m if v['first_stop'] is not None]
    observed=sum(v['attempted_observations'] for v in m)
    return dict(samples=n,completed=sum(v['complete'] for v in m),first_stop_rate=sum(v['first_stop'] for v in actions)/len(actions) if actions else None,
        candidate_availability=sum(v['candidate_count']>0 for v in m)/n,mean_active_observations=observed/n,
        readable_record_rate=sum(v['readable_observation_records'] for v in m)/observed if observed else 0.,
        mean_active_seconds=sum(v['active_seconds'] for v in m)/n,final_trait_coverage=sum(v['final_trait_coverage'] for v in m)/n,
        semantic_quality_verified=False)

def run(args,backend=None):
    from amp.agent import ActiveAgent
    from amp.backend import QwenBackend
    from amp.media import MediaTools
    from amp.scoring import OfflineTraitScorer
    write_json(args.output/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'amp').glob('*.py'))})
    cfg=dict(read_json(ROOT/'configs/local_omni3b.json'),model=args.model,perception=args.mode)
    if args.max_steps is not None:cfg['max_steps']=args.max_steps
    backend=backend or QwenBackend(args.model,max_tokens=cfg['max_new_tokens'],temperature=0,generation_log_dir=args.output/'generations')
    backend.generation_log_dir=args.output/'generations'
    media=MediaTools(args.prepared/'media_cache',cfg['max_clip_seconds'],cfg['video_fps'])
    rows=list(read_jsonl(args.prepared/'manifest.jsonl'));records=[];probes=[]
    if args.limit:rows=rows[:args.limit]
    if getattr(args,'case_ids',None):rows=[r for r in rows if r['id'] in args.case_ids]
    for row in rows:
        # Only whitelisted manifest fields reach Agent. No reference/GT attached.
        agent=ActiveAgent(backend,OfflineTraitScorer(),media,cfg)
        try:
            t=agent.run(row);record=dict(case_id=row['id'],trajectory=t,metrics=metrics(t,True))
        except (ValueError,RuntimeError,KeyError,TypeError) as error:
            t=agent.last_partial or {};record=dict(case_id=row['id'],trajectory=t,metrics=metrics(t,False,getattr(error,'failures',[])),error=str(error),failures=getattr(error,'failures',[]))
        records.append(record);write_json(args.output/f"cases/{row['id']}.json",record)
        write_json(args.output/'results.json',dict(scope='Complete native diagnostic episodes; source/schema validity is not semantic accuracy',model=args.model,mode=args.mode,config=cfg,summary=summarize(records),records=records,external_api_calls=0))
        print(row['id'],record['metrics'],record.get('error',''),flush=True)
    # Review worksheet intentionally has no approved semantic verdicts.
    worksheet=[dict(case_id=record['case_id'],record_id=r['id'],observation_id=r['observation_id'],
        findings=decode_observation(r['content'])['observations'],modality=r['modality'],retrieval_window=r['retrieval_window'],
        source_supported=None,reviewer='') for record in records for r in record['trajectory'].get('memory',{}).get('observations',[])]
    write_json(args.output/'semantic_review_template.json',dict(status='unreviewed',items=worksheet))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--panel',type=Path,default=ROOT/'validation/panel.json')
    p.add_argument('--prepared',type=Path,default=ROOT/'runs/panel_validation/prepared');p.add_argument('--output',type=Path,default=ROOT/'runs/panel_validation/native')
    p.add_argument('--model',default=str(ROOT.parent/'models/Qwen2.5-Omni-3B'));p.add_argument('--mode',choices=['full','static','generic'],default='full')
    p.add_argument('--case-id',dest='case_ids',action='append')
    p.add_argument('--max-steps',type=int);p.add_argument('--limit',type=int);args=p.parse_args()
    os.environ['PATH']=str(ROOT/'.local-tools/ffmpeg/bin')+os.pathsep+os.environ.get('PATH','')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(args.panel,args.prepared)
    else:run(args)

if __name__=='__main__':main()
