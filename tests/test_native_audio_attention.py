"""Run numerics in the isolated native venv; optional in the project venv."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('vllm'),'requires the isolated native inference environment')
class NativeAudioAttentionTests(unittest.TestCase):
    def test_windowed_sdpa_matches_eager_masked_attention_and_does_not_mix_windows(self):
        import torch
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioAttention
        from baselines.omniagent_native.audio_worker import WindowedAudioAttention
        torch.manual_seed(17)
        cfg=Qwen2_5OmniAudioEncoderConfig(d_model=32,encoder_attention_heads=4,attention_dropout=0.)
        for dtype,tolerance in ((torch.float32,1e-6),(torch.bfloat16,.012)):
            with self.subTest(dtype=dtype),torch.inference_mode():
                eager=Qwen2_5OmniAudioAttention(cfg).eval().to(dtype=dtype)
                efficient=WindowedAudioAttention(cfg).eval().to(dtype=dtype)
                efficient.load_state_dict(eager.state_dict())
                # Includes a short tail and one-token window.
                source=torch.randn(19,32,dtype=dtype)
                windows=torch.tensor([0,8,17,18,19],dtype=torch.int32)
                expected=eager(source,windows);actual=efficient(source,windows)
                gap=(expected.float()-actual.float()).abs().max().item()
                self.assertLess(gap,tolerance)
                changed=source.clone();changed[8:17]+=5
                isolated=efficient(changed,windows)
                self.assertTrue(torch.equal(actual[:8],isolated[:8]))
                self.assertTrue(torch.equal(actual[17:],isolated[17:]))
                self.assertEqual(set(eager.state_dict()),set(efficient.state_dict()))
                print('audio_attention_numeric',str(dtype),'max_abs_gap',gap,flush=True)


if __name__=='__main__':unittest.main()
