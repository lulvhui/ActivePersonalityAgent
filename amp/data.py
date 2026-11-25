"""Build raw-only manifests; never import TEG evidence or generated summaries."""
import bisect
import math
import re
from collections import Counter
from pathlib import Path
from .common import TRAITS, normalize_id, read_json, read_jsonl, write_json, write_jsonl


def split_lookup(manifest):
    result = {}
    for split in ["train", "dev", "test"]:
        for person in manifest["splits"][split]:
            person = normalize_id(person)
            if person in result:
                raise ValueError(f"Person leakage: {person}")
            result[person] = split
    return result


def transcript_segments(text):
    pattern = r"\[(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)\]\s*([^\n]*)"
    rows = []
    for m in re.finditer(pattern, text):
        start = 60 * int(m[1]) + float(m[2])
        end = 60 * int(m[3]) + float(m[4])
        if end > start:
            rows.append(dict(start=start, end=end, text=m[5].strip(), timing="unverified_asr"))
    return rows or [dict(start=None, end=None, text=text, timing="unlocalized")]


def build(root, output):
    root, output = Path(root), Path(output)
    rows, audit = [], {}
    for dataset in ["IELTS", "RecruitView"]:
        split_path = root / "artifacts/training_data" / dataset / "splits/split_manifest.json"
        splits = split_lookup(read_json(split_path))
        base = root / "data" / dataset / "raw"
        excluded, missing, filtered_answers = [], [], []
        if dataset == "IELTS":
            raw = read_json(base / "asr/ielts_part1_emotion.json")
            dirs = {normalize_id(p.name): p for p in (base / "videos").iterdir() if p.is_dir()}
            for person, value in raw.items():
                pid = normalize_id(person)
                if pid not in splits:
                    excluded.append(pid)
                    continue
                directory = dirs.get(pid)
                candidates = list(directory.glob("*.mp4")) if directory else []
                # The provided transcript is explicitly part1; do not attach other parts.
                videos = [p for p in candidates if re.search(r"part[ _-]*1\b", p.stem, re.I)]
                if len(videos) != 1:
                    missing.append(dict(person=pid, reason="part1 missing/ambiguous"))
                    continue
                labels = {t: float(v) for t, v in zip(TRAITS, value["label"])}
                if len(value["label"]) != 5:
                    raise ValueError(f"Invalid IELTS labels: {pid}")
                segments = [dict(start=t.get("time_a", [None, None])[0],
                                 end=t.get("time_a", [None, None])[1], text=t.get("A", ""),
                                 question=t.get("Q", ""), timing="unverified_asr") for t in value["turns"]]
                rows.append(dict(id=f"IELTS:{pid}", dataset=dataset, person_id=pid, split=splits[pid],
                                 video=str(videos[0].resolve()), segments=segments, labels_raw=labels,
                                 task_unit="interview_part1", label_unit="person"))
        else:
            # Reuse the established 1,965-answer membership, never its timestamps or text.
            membership_path = root / "data/RecruitView/processed/video_turn_inputs/recruitview_common_turns.jsonl"
            membership = {Path(v["video_path"]).name: normalize_id(v["person_id"])
                          for v in read_jsonl(membership_path)}
            for value in read_jsonl(base / "asr/metadata.jsonl"):
                pid = normalize_id(value["user_no"])
                if pid not in splits:
                    excluded.append(pid)
                    continue
                filename = Path(value["file_name"]).name
                if filename not in membership:
                    filtered_answers.append(filename)
                    continue
                if membership[filename] != pid:
                    raise ValueError(f"RecruitView person mapping mismatch: {filename}")
                video = base / "videos" / filename
                if not video.is_file():
                    missing.append(dict(person=pid, video=str(video)))
                    continue
                labels = {t: float(value[name.lower()]) for t, name in TRAITS.items()}
                rows.append(dict(id=f"RecruitView:{value['id']}", dataset=dataset, person_id=pid,
                                 split=splits[pid], video=str(video.resolve()),
                                 segments=transcript_segments(value.get("transcript", "")),
                                 question=value.get("question", ""), labels_raw=labels,
                                 task_unit="answer", label_unit="answer"))
        audit[dataset] = dict(split_source=str(split_path.resolve()), missing=missing,
                              excluded_people=sorted(set(excluded)), filtered_answers=filtered_answers)
    # Discretization is fitted on train only, independently by dataset and trait.
    # Quintiles are an explicit experimental choice, not dataset-provided classes.
    bins = {}
    for dataset in ["IELTS", "RecruitView"]:
        train = [r for r in rows if r["dataset"] == dataset and r["split"] == "train"]
        if not train:
            raise ValueError(f"No training rows for {dataset}")
        bins[dataset] = {}
        for trait in TRAITS:
            values = sorted(r["labels_raw"][trait] for r in train)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite labels")
            def quantile(q):
                i = (len(values)-1)*q
                lo = int(i)
                hi = min(lo+1, len(values)-1)
                return values[lo] + (values[hi]-values[lo])*(i-lo)
            edges = [quantile(q) for q in [.2, .4, .6, .8]]
            centers = []
            for k in range(5):
                group = [v for v in values if bisect.bisect_right(edges, v) == k]
                centers.append(sum(group)/len(group) if group else quantile((k+.5)/5))
            bins[dataset][trait] = dict(edges=edges, centers=centers)
    for row in rows:
        row["labels"] = {t: bisect.bisect_right(bins[row["dataset"]][t]["edges"], v)
                         for t, v in row["labels_raw"].items()}
    for split in ["train", "dev", "test"]:
        write_jsonl(output / f"{split}.jsonl", (r for r in rows if r["split"] == split))
    write_json(output / "label_bins.json", bins)
    audit["counts"] = dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows))
    audit["people"] = {d: len({r["person_id"] for r in rows if r["dataset"] == d}) for d in bins}
    audit["policy"] = "Existing person-disjoint splits; raw ASR text only; train-fitted quintiles; no generated annotations"
    write_json(output / "audit.json", audit)
    return audit


def recruitview_people(root, person_ids, output, media):
    """Inference inputs: concatenate every raw answer for each selected person.

    Answer labels are intentionally not promoted to a person-level target.
    Ordering is by answer ID, not a claim about real interview chronology.
    """
    from .common import digest
    from .media import run
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    wanted = {normalize_id(person) for person in person_ids}
    splits = split_lookup(read_json(root/'artifacts/training_data/RecruitView/splits/split_manifest.json'))
    groups = {person: [] for person in wanted}
    for answer in read_jsonl(root/'data/RecruitView/raw/asr/metadata.jsonl'):
        person = normalize_id(answer['user_no'])
        if person in groups:groups[person].append(answer)
    rows = []
    for person in sorted(wanted, key=int):
        answers = sorted(groups[person], key=lambda answer: int(answer['id']))
        if not answers or person not in splits:raise ValueError(f'Missing RecruitView person or split: {person}')
        paths = [root/'data/RecruitView/raw/videos'/Path(answer['file_name']).name for answer in answers]
        infos = [media.probe(str(path)) for path in paths]
        video_streams = [next(stream for stream in info['raw']['streams'] if stream['codec_type']=='video') for info in infos]
        width = 2*math.ceil(max(stream['width'] for stream in video_streams)/2)
        height = 2*math.ceil(max(stream['height'] for stream in video_streams)/2)
        durations = [math.ceil(info['duration']*30-1e-6)/30 for info in infos]
        fingerprint = digest([(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths])
        target = output/f'person_{person}_{fingerprint}.mp4'
        if not target.exists():
            command = ['ffmpeg','-nostdin','-v','error']
            filters = []
            for index,(path,duration) in enumerate(zip(paths,durations)):
                command.extend(['-i',str(path)])
                filters.extend([
                    f'[{index}:v]setpts=PTS-STARTPTS,fps=30,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},setpts=PTS-STARTPTS[v{index}]',
                    f'[{index}:a]aresample=48000,asetpts=PTS-STARTPTS,apad,atrim=duration={duration}[a{index}]'])
            filters.append(''.join(f'[v{i}][a{i}]' for i in range(len(paths)))+f'concat=n={len(paths)}:v=1:a=1[v][a]')
            command.extend(['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
                            '-c:v','libx264','-threads','2','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
                            '-c:a','aac','-b:a','128k','-movflags','+faststart',str(target)])
            run(command)
        turns, segments, offset = [], [], 0.
        for answer,path,duration,info in zip(answers,paths,durations,infos):
            question = answer.get('question','')
            turns.append(dict(answer_id=str(answer['id']),question=question,time=[offset,offset+duration],
                              video=str(path.resolve()),source_duration=info['duration']))
            for segment in transcript_segments(answer.get('transcript','')):
                left,right=segment['start'],segment['end']
                left=0. if left is None else min(max(0.,left),duration)
                right=duration if right is None else min(max(left,right),duration)
                if right<=left:left,right=0.,duration
                segments.append(dict(segment,start=offset+left,end=offset+right,question=question,
                                     answer_id=str(answer['id']),timing='unverified_asr'))
            offset += duration
        actual = media.probe(str(target))['duration']
        if abs(actual-offset)>.1:raise ValueError(f'Concatenated timeline mismatch: {person}: {actual} vs {offset}')
        row = dict(id=f'RecruitView:person:{person}',dataset='RecruitView',person_id=person,split=splits[person],
                   video=str(target.resolve()),segments=segments,turns=turns,
                   question='\n'.join(f"[{turn['time'][0]:.2f}-{turn['time'][1]:.2f}s] {turn['question']}" for turn in turns),
                   task_unit='person_interview',label_unit='person_unresolved',
                   ordering='answer_id; assembled interview, not original chronological continuity')
        write_json(target.with_suffix('.json'),row)
        rows.append(row)
    return rows
