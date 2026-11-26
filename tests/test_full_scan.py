import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from amp.media import MediaTools
from amp.agent import ActiveAgent
from amp.backend import OfflineBackend
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia
from amp.common import read_json

ROOT=Path(__file__).resolve().parents[1]
CFG=read_json(ROOT/'configs/local_omni3b.json')
FFMPEG=shutil.which('ffmpeg') or str(ROOT/'.local-tools/ffmpeg/bin/ffmpeg')


class FullScanTests(unittest.TestCase):
    @unittest.skipUnless(Path(FFMPEG).is_file(),'Requires local FFmpeg')
    def test_actual_media_coverage_is_complete_and_frame_count_scales_with_fps(self):
        import soundfile
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,PATH=str(Path(FFMPEG).parent)+os.pathsep+os.environ['PATH']):
            video=Path(tmp)/'source.mp4'
            subprocess.run([FFMPEG,'-v','error','-f','lavfi','-i','color=c=blue:s=224x224:r=24:d=12',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=12',
                '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(video)],check=True)
            row=dict(video=str(video),segments=[],question='Introduce yourself')
            tools=MediaTools(Path(tmp)/'cache',max_clip=2)
            low=tools.coarse(row,.5,5)
            high=tools.coarse(row,1,5)
            self.assertEqual(low['sampled_frames']*2,high['sampled_frames'])
            self.assertGreater(high['sampled_frames'],low['sampled_frames'])
            for scan in (low,high):
                self.assertAlmostEqual(sum(b-a for a,b in scan['audio_coverage']),scan['audio_samples']/scan['audio_sample_rate'])
                self.assertAlmostEqual(sum(b-a for a,b in scan['video_coverage']),scan['duration'])
                self.assertEqual(scan['audio_coverage'][0][0],0)
                self.assertEqual(scan['audio_coverage'][-1][1],scan['audio_samples']/scan['audio_sample_rate'])
                actual_audio=sum(soundfile.info(obs['media'][0]['audio']).duration for obs in scan['audio'])
                self.assertAlmostEqual(actual_audio,scan['audio_samples']/scan['audio_sample_rate'],places=6)
                import numpy as np
                full,_=soundfile.read(next((Path(tmp)/'cache').glob('*.full.wav')),dtype='int16')
                pieces=[soundfile.read(o['media'][0]['audio'],dtype='int16')[0] for o in scan['audio']]
                self.assertTrue(np.array_equal(full,np.concatenate(pieces)))
                self.assertTrue(all(a[1]==b[0] for a,b in zip(scan['audio_coverage'],scan['audio_coverage'][1:])))
            with self.assertRaises(ValueError):tools.observe(row,dict(time=[0,4],modality='video'))

    def test_audio_actions_have_no_video_clip_duration_cap(self):
        from amp.state import CandidatePool
        pool=CandidatePool(600,30)
        pool.validate(dict(time=[0,332],modality='audio',goal='Listen to the complete answer'))
        with self.assertRaises(ValueError):pool.validate(dict(time=[0,332],modality='video',goal='Inspect high-resolution movement'))
        with tempfile.TemporaryDirectory() as tmp:
            tools=MediaTools(tmp,max_clip=30)
            import numpy as np,soundfile as sf
            sf.write(Path(tmp)/'m.full.wav',np.zeros(332*16000,dtype='int16'),16000,subtype='PCM_16')
            with patch.object(tools,'probe',return_value=dict(duration=600,start_pts=0,source_key='m',raw=dict(streams=[dict(codec_type='audio',start_time=0)]))),patch('amp.media.run',return_value=''):
                observation=tools.observe(dict(video='fixture.mp4',segments=[]),dict(time=[0,332],modality='audio'))
            self.assertEqual(observation['time'],[0.,332.])

    def test_long_transcript_candidates_are_not_lost_to_video_clip_cap(self):
        from amp.state import CandidatePool
        from amp.perception import add_candidate_proposals
        pool=CandidatePool(400,30)
        ids=add_candidate_proposals(pool,[dict(time=[56.24,95.88],modality='text',goal='Long-term exercise habit'),
            dict(time=[280.36,330.9],modality='text',goal='Housework preferences')],[0,400],'global_text')
        self.assertEqual(len(ids),2)
        with tempfile.TemporaryDirectory() as tmp:
            tools=MediaTools(tmp,max_clip=30)
            row=dict(video='fixture.mp4',segments=[dict(start=56.24,end=95.88,text='I have exercised since ninth grade')])
            with patch.object(tools,'probe',return_value=dict(duration=400,start_pts=0,source_key='m')):
                observed=tools.observe(row,pool.items[ids[0]]['action'])
            self.assertEqual(len(observed['segments']),1)

    def test_empty_stage_one_pool_means_no_minimum_observation_floor(self):
        class Backend(OfflineBackend):
            action_calls=0
            def generate(self,messages,kind,step):
                if kind=='candidate_scan':obj=dict(candidates=[])
                elif kind=='action':
                    self.action_calls+=1
                    obj=dict(stop=True,reason='No useful target remains')
                else:return super().generate(messages,kind,step)
                return obj,dict(kind=kind,step=step,messages=messages,text=json.dumps(obj),tokens=[],offline=True)
        backend=Backend()
        tr=ActiveAgent(backend,OfflineTraitScorer(),SyntheticMedia(),CFG).run(dict(id='s',dataset='SYNTHETIC',split='dev',person_id='p',segments=[]))
        self.assertEqual(len(tr['steps']),0)
        self.assertEqual(backend.action_calls,0)
        self.assertEqual(tr['stop_reason'],'candidate_pool_empty')
        self.assertTrue(tr['stopped'])
        self.assertIsNone(tr['cost']['active_budget_seconds'])

    def test_cleanup_keeps_caches_and_successful_dataset_outputs(self):
        from scripts.cleanup_failed_runs import cleanup
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            failed=root/'failed';(failed/'cache').mkdir(parents=True)
            (failed/'cache'/'clip.wav').write_text('cache')
            (failed/'output.txt').write_text('failed')
            (failed/'summary.json').write_text(json.dumps(dict(jobs=[dict(dataset='IELTS',status='failed')])))
            mixed=root/'mixed';(mixed/'IELTS').mkdir(parents=True);(mixed/'RecruitView').mkdir()
            (mixed/'summary.json').write_text(json.dumps(dict(jobs=[dict(dataset='IELTS',status='failed'),dict(dataset='RecruitView',status='success')])))
            cleanup(True,root)
            self.assertTrue((failed/'cache'/'clip.wav').exists())
            self.assertFalse((failed/'output.txt').exists())
            self.assertFalse((mixed/'IELTS').exists())
            self.assertTrue((mixed/'RecruitView').exists())

if __name__=='__main__':unittest.main()
