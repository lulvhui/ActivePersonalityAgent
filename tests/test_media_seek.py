"""Cold-cache fast seeking must preserve frames and the audio/video clock."""
import shutil,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from amp.media import MediaTools,run


@unittest.skipUnless(shutil.which('ffmpeg'),'Requires FFmpeg')
class MediaSeekTests(unittest.TestCase):
    def test_input_seek_preserves_global_sampling_and_audio_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mkv'
            run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=160x120:r=24:d=10',
                 '-itsoffset','0.375','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=9',
                 '-c:v','libx264','-g','48','-pix_fmt','yuv420p','-c:a','pcm_s16le','-output_ts_offset','5',str(source)])
            tools=MediaTools(root/'cache',max_clip=30,fps=2)
            row=dict(video=str(source),segments=[])
            info=tools.probe(source)
            # Produce the established full.wav through the unchanged coarse path.
            tools.full_audio(row,20)
            full,rate=sf.read(tools.cache/f"{info['source_key']}.full.wav",dtype='int16')
            for index,(start,end) in enumerate([(0.,2.),(2.125,4.375),(2.25,4.75),(7.1,8.6)]):
                obs=tools.observe(row,dict(time=[start,end],modality='audio_video'))
                self.assertAlmostEqual(obs['audio_offset'],.375,places=3)
                self.assertEqual(obs['source_pts'],[info['start_pts']+start,info['start_pts']+end])
                actual,sr=sf.read(obs['media'][0]['audio'],dtype='int16')
                left=round((start-obs['audio_offset'])*rate);right=round((end-obs['audio_offset'])*rate)
                expected=np.pad(full[max(0,left):min(len(full),right)],(max(0,-left),max(0,right-len(full))))
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(len(actual),round((end-start)*sr))
                old=root/f'old_{index}.mp4'
                run(['ffmpeg','-v','error','-i',str(source),'-ss',str(start),'-t',str(end-start),'-an',
                     '-vf','fps=2,scale=480:-2','-c:v','libx264',str(old)])
                new=Path(obs['media'][1]['video'])
                def frames(path):
                    return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'])
                a=np.frombuffer(frames(old),dtype=np.uint8).astype(np.float32)
                b=np.frombuffer(frames(new),dtype=np.uint8).astype(np.float32)
                self.assertEqual(a.shape,b.shape)
                rmse=float(np.sqrt(np.mean((a-b)**2)))
                print('seek comparison',dict(window=[start,end],rmse=rmse),flush=True)
                self.assertEqual(rmse,0.)
                old_info=tools.probe(old);new_info=tools.probe(new)
                old_frames=next(s['nb_frames'] for s in old_info['raw']['streams'] if s['codec_type']=='video')
                new_frames=next(s['nb_frames'] for s in new_info['raw']['streams'] if s['codec_type']=='video')
                self.assertEqual(old_frames,new_frames)
            # The one-frame path also keeps the original selected frame.
            obs=tools.observe(row,dict(time=[2.125,2.375],modality='video'))
            old=root/'old.png'
            run(['ffmpeg','-v','error','-i',str(source),'-ss','2.125','-frames:v','1','-vf','scale=480:-2',str(old)])
            self.assertEqual(old.read_bytes(),Path(obs['media'][0]['image']).read_bytes())


if __name__=='__main__':unittest.main()
