"""Stage 3 Final-only GRPO: per-trait level tokens, group advantages and one CPU update."""
import importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest import mock
from amp.common import TRAITS, LEVELS
from amp.train import final_token_advantages, group_final_advantages, trait_level_masks, trait_rewards


class CharTokenizer:
    def decode(self, ids, **kw):
        return ''.join(chr(i) for i in ids)

    def __call__(self, value, **kw):
        return dict(input_ids=[ord(c) for c in value], offset_mapping=[(i, i+1) for i in range(len(value))])


FINAL = json.dumps({'O': 'low', 'C': None, 'E': 'high', 'A': 'medium', 'N': 'very_low', 'reasoning': 'O is "low" because ...',
                    'contextual': [dict(trait='E', context_type='reported', context='c', expression='e', citations=['M1'])]})


class AdvantageTests(unittest.TestCase):
    def test_level_masks_cover_only_each_top_level_value(self):
        masks = trait_level_masks(CharTokenizer(), dict(tokens=[ord(c) for c in FINAL]))
        spelled = {t: ''.join(c for c, m in zip(FINAL, masks[t]) if m) for t in TRAITS}
        self.assertEqual(spelled, dict(O='"low"', C='null', E='"high"', A='"medium"', N='"very_low"'))
        with self.assertRaises(ValueError):
            trait_level_masks(CharTokenizer(), dict(tokens=[ord(c) for c in '{"O":"low"}']))

    def test_trait_rewards_and_group_advantages(self):
        labels = {t: 1 for t in TRAITS}
        finals = [{t: dict(level='low') for t in TRAITS}, {t: dict(level='high') for t in TRAITS}]
        finals[1]['N'] = dict(level=None)
        rewards = [trait_rewards(labels, f) for f in finals]
        self.assertEqual(rewards[0]['O'], 0.);self.assertEqual(rewards[1]['O'], -.5);self.assertEqual(rewards[1]['N'], -1.)
        credit = group_final_advantages(rewards, [1., 0.], beta=.5)
        self.assertGreater(credit[0]['traits']['O'], 0);self.assertLess(credit[1]['traits']['O'], 0)
        self.assertAlmostEqual(credit[0]['shared'], .5, places=5)

    def test_level_tokens_get_only_their_trait_advantage(self):
        tokens = [ord(c) for c in FINAL]
        masks = trait_level_masks(CharTokenizer(), dict(tokens=tokens))
        advantage = dict(traits=dict(O=1., C=-1., E=2., A=0., N=3.), shared=.5)
        values = final_token_advantages(masks, advantage, len(tokens))
        for trait in TRAITS:
            self.assertTrue(all(v == advantage['traits'][trait] for v, m in zip(values, masks[trait]) if m))
        other = [v for i, v in enumerate(values) if not any(masks[t][i] for t in TRAITS)]
        self.assertTrue(all(v == 1.+.5 for v in other))  # mean trait advantage + shared


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is not installed')
class FinalUpdateTests(unittest.TestCase):
    def test_one_cpu_update_on_fixed_memory(self):
        import torch
        from types import SimpleNamespace
        from amp import train
        tokenizer = CharTokenizer()
        texts = [json.dumps({**{t: level for t in TRAITS}, 'reasoning': 'r', 'contextual': []}) for level in ('low', 'high')]

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bias = torch.nn.Parameter(torch.zeros(256))
            def set_adapter(self, name):pass
            def save_pretrained(self, path, **kw):Path(path).mkdir(parents=True, exist_ok=True)

        class Backend:
            def __init__(self):
                self.torch, self.policy = torch, Policy()
                self.processor = SimpleNamespace(tokenizer=tokenizer, save_pretrained=lambda path: None)
            def prepared_group(self):
                import contextlib
                return contextlib.nullcontext()
            def token_stats(self, decision, need_entropy=False):
                logp = self.policy.bias.log_softmax(-1)[torch.tensor(decision['tokens'])]
                return logp, None

        class Agent:
            def __init__(self):self.calls = 0
            def final_stage(self, memory, observations, question, perception_end, score, step):
                text = texts[self.calls % 2];self.calls += 1
                obj = json.loads(text)
                final = {t: dict(level=obj[t], reasoning=None) for t in TRAITS}
                return dict(final=final, final_explanation='r', final_status='generated', contextual=[],
                            decision=dict(kind='final', text=text, tokens=[ord(c) for c in text], messages=[]))

        class Judge:
            def check_records(self, trajectory):return {}
            def assess_final(self, trajectory, checks):return dict(faithfulness=1., citation_correctness=1.)

        trajectory = dict(sample_id='IELTS:1', split='train', offline=False, memory=dict(observations=[]), observations={},
                          steps=[], pool=[], stop_reason='candidate_pool_exhausted', score={})
        backend = Backend();before = backend.policy.bias.detach().clone()
        cfg = dict(learning_rate=.01, beta_ground=.5, clip_epsilon=.2, kl_beta=.04)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(train, 'load_reference', lambda b, path: None):
            train.rl_final(backend, Agent(), Judge(), [(trajectory, {t: LEVELS.index('low') for t in TRAITS})], folder, cfg, 'adapter', group_size=2)
            log = json.loads((Path(folder)/'training_log.json').read_text())
        self.assertFalse(torch.equal(before, backend.policy.bias.detach()))
        self.assertEqual(log[0]['generated'], 2)
        self.assertEqual(log[0]['levels']['O'], {**{level: 0 for level in LEVELS}, 'low': 1, 'high': 1, 'None': 0})
        # The correct 'low' Final is reinforced: probability of its level characters rises.
        low = [ord(c) for c in '"low"'];high = [ord(c) for c in '"high"']
        probs = backend.policy.bias.detach().softmax(-1)
        self.assertGreater(float(probs[low].sum()-before.softmax(-1)[low].sum()), float(probs[high].sum()-before.softmax(-1)[high].sum()))

    def test_rejects_rollouts_without_labels(self):
        from amp import train
        trajectory = dict(sample_id='RecruitView:person:1', split='train', offline=False)
        with self.assertRaises(ValueError):
            train.rl_final(None, None, None, [(trajectory, None)], 'out', {}, 'adapter', group_size=2)


if __name__ == '__main__':
    unittest.main()
