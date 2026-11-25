"""Bounded extraction with source PTS provenance; no shell command interpolation."""
import json
import math
import shutil
import subprocess
from pathlib import Path
from .common import MODALITIES, digest, read_json, write_json
from .provenance import index_transcript


def run(args):
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


class MediaTools:
    def __init__(self, cache, max_clip=20, fps=1):
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_clip, self.fps = max_clip, fps

    def require(self):
        for name in ["ffmpeg", "ffprobe"]:
            if not shutil.which(name):
                raise RuntimeError(f"Missing {name}; install FFmpeg and add it to PATH")

    def probe(self, video):
        self.require()
        stat = Path(video).stat()
        key = digest([str(video), stat.st_size, stat.st_mtime_ns])
        path = self.cache / f"{key}.probe.json"
        if path.exists():
            result=read_json(path)
            # Audio codec padding may extend a few milliseconds beyond video.
            result['duration']=max(result['duration'],float(result['raw'].get('format',{}).get('duration',0)))
            return result
        data = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", video]))
        stream = next(s for s in data["streams"] if s["codec_type"] == "video")
        duration = max(float(stream.get("duration",0)),float(data["format"].get("duration",0)))
        if duration <= 0:
            raise ValueError("Media duration must be positive")
        result = dict(duration=duration, start_pts=float(stream.get("start_time", 0)),
                      time_base=stream["time_base"], raw=data, source_key=key)
        write_json(path, result)
        return result

    @staticmethod
    def questions_in_window(row, start, end):
        return [dict(question=turn['question'],time=turn['time']) for turn in row.get('turns',[])
                if turn['time'][0]<end and turn['time'][1]>start]

    def observe(self, row, action, coarse=False, fps=None):
        info = self.probe(row["video"])
        start, end = map(float, action["time"])
        modality = action["modality"]
        if modality not in MODALITIES or not all(math.isfinite(x) for x in [start, end]):
            raise ValueError("Invalid modality/time")
        if start < 0 or end <= start or end > info["duration"] + 1e-3 or (not coarse and modality in ("video","audio_video") and end-start > self.max_clip+1e-3):
            raise ValueError("Observation outside media or clip budget")
        sample_fps = self.fps if fps is None else float(fps)
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("Video sampling FPS must be positive and finite")
        # Content is transcribed ASR; the nominal time alignment is not certified.
        indexed = index_transcript(row["segments"], info["source_key"], info["duration"])
        segments = [s for s in indexed if self._text_in_window(s, start, end)]
        output = dict(time=[start, end], source_pts=[info["start_pts"]+start, info["start_pts"]+end],
                      modality=modality, timing="unverified_asr" if modality == "text" else "media_relative",
                      segments=segments if modality == "text" else [], media=[],
                      media_id=info["source_key"], question=row.get("question", ""))
        output['questions']=self.questions_in_window(row,start,end)
        output['event_segments']=[s for s in indexed if s['timing']=='media_aligned' and start<=s['media_time'][0]<s['media_time'][1]<=end+1e-6]
        key = digest([info["source_key"], start, end, modality, coarse, sample_fps])
        if modality in ["audio", "audio_video"]:
            audio = self.cache / f"{key}.wav"
            if not audio.exists():
                run(["ffmpeg", "-v", "error", "-i", row["video"], "-ss", str(start), "-t", str(end-start),
                     "-vn", "-ac", "1", "-ar", "16000", "-y", str(audio)])
            output["media"].append(dict(type="audio", audio=str(audio)))
        if modality in ["video", "audio_video"] and (end-start)*sample_fps<2:
            image = self.cache/f"{key}.png"
            if not image.exists():
                run(['ffmpeg','-v','error','-i',row['video'],'-ss',str(start),'-frames:v','1',
                     '-vf',f'scale={320 if coarse else 480}:-2','-y',str(image)])
            output['media'].append(dict(type='image',image=str(image),sampled_frames=1))
            output['visual_representation']='one sampled frame; no motion observed'
            return output
        if modality in ["video", "audio_video"]:
            video = self.cache / f"{key}.mp4"
            if not video.exists():
                run(["ffmpeg", "-v", "error", "-i", row["video"], "-ss", str(start), "-t", str(end-start),
                     "-an", "-vf", f"fps={sample_fps},scale={320 if coarse else 480}:-2", "-c:v", "libx264", "-y", str(video)])
            sampled = self.probe(str(video))
            stream = next(x for x in sampled["raw"]["streams"] if x["codec_type"]=="video")
            output["media"].append(dict(type="video", video=str(video), fps=sample_fps, sampled_frames=int(stream.get("nb_frames",0))))
        return output

    def coarse(self, row, fps=1., chunk_seconds=20.):
        """Cover all audio and video; chunk size bounds requests, never coverage.

        Low-FPS video is decoded by the native processor. Unlike an isolated
        coarse image, its trusted citation unit is the observed source interval.
        """
        info = self.probe(row["video"])
        if not all(math.isfinite(float(x)) and x > 0 for x in (fps, chunk_seconds)):
            raise ValueError("Coarse FPS and transport chunk length must be positive and finite")
        count = max(1, math.ceil(info["duration"]/chunk_seconds))
        windows = [(i*chunk_seconds,min((i+1)*chunk_seconds,info["duration"])) for i in range(count)]
        video = []
        for i,(start,end) in enumerate(windows):
            print(f"[Coarse] video {i+1}/{len(windows)}: {start:.2f}-{end:.2f}s, {fps} fps",flush=True)
            video.append(self.observe(row, dict(time=[start,end],modality="video"), coarse=True, fps=fps))
        audio, audio_samples, sample_rate = self.full_audio(row, chunk_seconds)
        return dict(duration=info["duration"], video=video, audio=audio,chunk_seconds=chunk_seconds,
                    text_segmentation='full_transcript',video_segmentation='fixed_seconds',audio_segmentation='decoded_samples',
                    media_id=info["source_key"], question=row.get("question", ""),
                    transcript=index_transcript(row["segments"], info["source_key"], info["duration"]),
                    sampling_fps=fps, sampled_frames=sum(m["sampled_frames"] for obs in video for m in obs["media"]), full_audio_coverage=True,
                    video_coverage=[list(x) for x in windows], audio_coverage=[obs["time"] for obs in audio],
                    audio_samples=audio_samples,audio_sample_rate=sample_rate,
                    timing_warning="ASR timestamps remain retrieval hints; AV citations use real observed source intervals")

    def full_audio(self, row, chunk_seconds):
        """Decode the whole audio stream once, then partition every decoded sample."""
        import soundfile as sf
        info = self.probe(row["video"])
        full = self.cache/f"{info['source_key']}.full.wav"
        if not full.exists():
            run(["ffmpeg","-v","error","-i",row["video"],"-vn","-ac","1","-ar","16000","-y",str(full)])
        stream = next(s for s in info["raw"]["streams"] if s["codec_type"]=="audio")
        offset = max(0.,float(stream.get("start_time",0))-info["start_pts"])
        observations = []
        with sf.SoundFile(full) as source:
            total,rate = len(source),source.samplerate
            while source.tell() < total:
                start = source.tell()
                samples = source.read(max(1,int(chunk_seconds*rate)),dtype="int16")
                end = source.tell()
                path = self.cache/f"{info['source_key']}.audio.{start}-{end}.wav"
                if not path.exists():sf.write(path,samples,rate,subtype="PCM_16")
                observations.append(dict(time=[offset+start/rate,offset+end/rate],source_samples=[start,end],
                    source_pts=[info["start_pts"]+offset+start/rate,info["start_pts"]+offset+end/rate],
                    modality="audio",timing="media_relative",segments=[],media=[dict(type="audio",audio=str(path))],
                    media_id=info["source_key"],question=row.get("question",""),
                    questions=self.questions_in_window(row,offset+start/rate,offset+end/rate)))
        print(f"[Coarse] full audio: {total} samples at {rate} Hz, {len(observations)} continuous chunks",flush=True)
        return observations,total,rate

    @staticmethod
    def _text_in_window(segment, start, end):
        if segment["media_time"] is not None:
            a, b = segment["media_time"]
            # No word-level alignment: do not expose a whole aligned segment
            # through a tiny partial window and then claim it was fully observed.
            return start <= a < b <= end + 1e-6
        hint = segment["source_time"]
        return hint is None or hint[1] > start and hint[0] < end

