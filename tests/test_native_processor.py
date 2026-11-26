"""Real local Processor and FFmpeg integration; no model weights or GPU required."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest

from amp.backend import QwenBackend, SYSTEM, message
from amp.common import read_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path(os.environ.get('AMP_TEST_MODEL_DIR',read_json(ROOT/'configs/local_omni3b.json')['model']))
NATIVE=all(importlib.util.find_spec(m) for m in ['torch','transformers','qwen_omni_utils','av']) and MODEL.exists()


@unittest.skipUnless(NATIVE,'Native Qwen processor environment not installed')
class NativeProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen2_5OmniProcessor
        cls.backend=QwenBackend.__new__(QwenBackend)
        cls.backend.processor=Qwen2_5OmniProcessor.from_pretrained(str(MODEL),local_files_only=True)
        cls.backend.policy=types.SimpleNamespace(device=torch.device('cpu'),dtype=torch.float32)

    def test_string_system_prompt_is_normalized_without_mutation(self):
        messages=[dict(role='system',content=SYSTEM),message({'task':'Return JSON'})]
        inputs=self.backend.prepare(messages)
        self.assertGreater(inputs['input_ids'].shape[1],0)
        self.assertIsInstance(messages[0]['content'],str)

    def test_incremental_json_masks_match_full_prefix_for_real_qwen_tokens(self):
        import json,torch
        from amp.perception import generation_schema
        b=self.backend;schema=generation_schema('memory')
        text=json.dumps(dict(observations=['我主动邀请同事讨论。','He said "let us work together".'],candidates=[]),ensure_ascii=False)
        tokens=b.processor.tokenizer.encode(text,add_special_tokens=False)+[b.processor.tokenizer.eos_token_id]
        full=b.format_constraint(schema);incremental=b.format_constraint(schema).token_enforcer.iter_allowed(tokens)
        prefix=[1]
        for token in tokens:
            allowed=next(incremental)
            self.assertEqual(set(allowed),set(full(0,torch.tensor(prefix))))
            self.assertIn(token,allowed)
            prefix.append(token)

    def test_real_audio_video_sampling_rate_reaches_processor(self):
        ffmpeg=shutil.which('ffmpeg') or str(ROOT/'.local-tools/ffmpeg/bin/ffmpeg')
        if not Path(ffmpeg).is_file():self.skipTest('FFmpeg not available')
        with tempfile.TemporaryDirectory() as tmp:
            video,audio=Path(tmp)/'test.mp4',Path(tmp)/'test.wav'
            subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i','color=c=blue:s=224x224:r=24:d=4',
                            '-an','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
            subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=4',
                            '-ac','1',str(audio)],check=True)
            os.environ.setdefault('FORCE_QWENVL_VIDEO_READER','torchvision')
            messages=[dict(role='system',content=SYSTEM),message({'task':'Return observed cues'},[
                dict(type='video',video=str(video),fps=1),dict(type='audio',audio=str(audio))])]
            inputs=self.backend.prepare(messages)
            self.assertIn('pixel_values_videos',inputs)
            self.assertIn('input_features',inputs)
            self.assertAlmostEqual(float(inputs['video_second_per_grid'][0]),2.)


    def test_audio_longer_than_thirty_seconds_is_not_silently_truncated(self):
        import numpy as np
        import soundfile
        with tempfile.TemporaryDirectory() as tmp:
            audio=Path(tmp)/'long.wav'
            soundfile.write(audio,np.zeros(35*16000,dtype=np.float32),16000)
            inputs=self.backend.prepare([dict(role='system',content=SYSTEM),message(dict(task='Describe audio'),[dict(type='audio',audio=str(audio))])])
            self.assertGreater(int(inputs['feature_attention_mask'].sum()),3000)

if __name__=='__main__':unittest.main()
