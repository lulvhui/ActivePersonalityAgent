"""Novelty-gated refinement: same-channel sub-window re-reads add no new recording content."""
import json, math, unittest
from pathlib import Path
from amp.agent import ActiveAgent
from amp.backend import OfflineBackend
from amp.common import TRAITS, read_json
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia
from amp.state import CandidatePool, video_frames

ROOT = Path(__file__).resolve().parents[1]
CFG = read_json(ROOT/'configs/local_omni3b.json')
ROW = dict(id='SYNTHETIC_NOVELTY', dataset='SYNTHETIC', split='train', person_id='fixture', segments=[])
ALL = ['text', 'audio', 'video', 'audio_video']


def pool_with_read(modality='audio_video', window=(100., 120.), fps=2., duration=300.):
    pool = CandidatePool(duration, 30., ALL)
    pool.configure_novelty(fps, 5., 1)
    root = pool.add(dict(time=list(window), modality=modality))
    pool.start(pool.items[root]['action']);pool.finish(root);pool.record_read(pool.items[root]['action'])
    return pool, root


class NoveltyMathTests(unittest.TestCase):
    def test_frames_follow_global_grid_and_short_windows_are_one_still(self):
        self.assertEqual(video_frames([1., 3.], 2.), {2, 3, 4, 5})
        self.assertEqual(video_frames([1.2, 2.8], 2.), {3, 4, 5})
        self.assertEqual(video_frames([117.2, 117.7], 2.), set())  # a still, not a grid frame

    def test_new_seconds_per_channel(self):
        pool, _ = pool_with_read()
        self.assertEqual(pool.new_seconds(dict(time=[115., 119.5], modality='video'))['video'][0], 0.)
        self.assertEqual(pool.new_seconds(dict(time=[115., 125.], modality='audio'))['audio'], (5., 280.))
        video_only, _ = pool_with_read('video')
        self.assertEqual(video_only.new_seconds(dict(time=[105., 110.], modality='audio'))['audio'][0], 5.)

    def test_unread_intervals(self):
        pool, _ = pool_with_read(duration=150.)
        self.assertEqual(pool.unread_intervals('audio'), [[0., 100.], [120., 150.]])
        self.assertEqual(pool.unread_intervals('video'), [[0., 100.], [120., 150.]])


class AdmissionTests(unittest.TestCase):
    def test_one_focus_inside_region_then_only_unread_content(self):
        pool, root = pool_with_read()
        focus = dict(time=[115., 119.5], modality='video')
        self.assertTrue(pool.admit_refinement(focus, root, {}))
        child = pool.add(focus, parent_id=root, origin='observation')
        pool.start(focus);pool.commit_refinement(child);pool.finish(child);pool.record_read(focus)
        self.assertFalse(pool.admit_refinement(dict(time=[118.5, 119.5], modality='video'), child, {}))
        self.assertEqual(pool.filtered[-1]['reason'], 'focus_used')
        # Another situation: unread content is accepted, read content is compared in Memory instead.
        self.assertTrue(pool.admit_refinement(dict(time=[200., 210.], modality='audio_video'), child, {}))
        self.assertTrue(pool.admit_refinement(dict(time=[118., 126.], modality='audio'), child, {}))
        self.assertFalse(pool.admit_refinement(dict(time=[118., 122.], modality='audio'), child, {}))

    def test_rereading_another_read_situation_is_filtered_even_with_focus_left(self):
        pool, first = pool_with_read()
        second = pool.add(dict(time=[200., 220.], modality='audio_video'))
        pool.start(pool.items[second]['action']);pool.finish(second);pool.record_read(pool.items[second]['action'])
        self.assertFalse(pool.admit_refinement(dict(time=[205., 210.], modality='audio_video'), first, {}))
        self.assertEqual(pool.filtered[-1]['reason'], 'no_unread_content_outside_region')

    def test_transcript_rereads_are_gated_like_media(self):
        pool, root = pool_with_read('text', window=(0., 300.))
        focus = dict(time=[10., 20.], modality='text')
        self.assertTrue(pool.admit_refinement(focus, root, {}))
        key = pool.add(focus, parent_id=root, origin='observation')
        pool.start(focus);pool.commit_refinement(key);pool.finish(key)
        self.assertFalse(pool.admit_refinement(dict(time=[30., 40.], modality='text'), root, {}))
        self.assertTrue(pool.admit_refinement(dict(time=[30., 40.], modality='audio'), root, {}))

    def test_small_remainder_counts_as_new(self):
        pool, root = pool_with_read(window=(0., 20.), duration=22.)
        self.assertTrue(pool.admit_refinement(dict(time=[18., 22.], modality='audio'), root, {}))

    def test_read_count_bound_holds_for_any_proposals(self):
        import random
        rng = random.Random(0)
        for trial in range(30):
            pool = CandidatePool(120., 30., ALL);pool.configure_novelty(2., 5., 1)
            roots = []
            for start in (0., 20., 40., 60., 80.):  # 100-120 left unread
                key = pool.add(dict(time=[start, start+20.], modality='audio_video'))
                pool.start(pool.items[key]['action']);pool.finish(key);pool.record_read(pool.items[key]['action'])
                roots.append(key)
            unread = {c: sum(b-a for a, b in pool.unread_intervals(c)) for c in ('text', 'audio', 'video')}
            executed = 0
            for _ in range(300):
                # A burst of overlapping proposals, then the queue executes what survives pruning.
                for _ in range(rng.randint(1, 6)):
                    a = rng.uniform(0, 119.);b = min(120., a+rng.choice([rng.uniform(.1, .9), rng.uniform(1., 20.)]))
                    action = pool.validate(dict(time=[a, b], modality=rng.choice(['text', 'audio', 'video', 'audio_video'])))
                    parent = rng.choice([k for k, v in pool.items.items() if v['status'] == 'done'])
                    if pool.matching(action) is None and pool.admit_refinement(action, parent, {}):
                        pool.add(action, parent_id=parent, origin='observation')
                pool.prune_refinements()
                pending = [k for k, v in pool.items.items() if v['status'] == 'pending']
                if not pending:
                    continue
                key = rng.choice(pending)
                pool.start(pool.items[key]['action']);pool.commit_refinement(key)
                pool.finish(key);pool.record_read(pool.items[key]['action'])
                executed += 1
            bound = len(roots)*1+sum(math.ceil(v/5.) for v in unread.values())
            self.assertLessEqual(executed, bound)

    def test_overlapping_pending_refinements_are_rechecked_at_execution(self):
        pool, root = pool_with_read(window=(0., 20.), duration=120.)
        keys = []
        for i in range(6):
            action = dict(time=[40.+i, 46.+i], modality='audio')
            self.assertTrue(pool.admit_refinement(action, root, {}))
            keys.append(pool.add(action, parent_id=root, origin='observation'))
        pool.prune_refinements()
        pool.start(pool.items[keys[0]]['action']);pool.commit_refinement(keys[0])
        pool.finish(keys[0]);pool.record_read(pool.items[keys[0]]['action'])
        pool.prune_refinements()
        # [45,51] still removes 5 s of unread audio ([46,51]); the others add 1-4 s.
        self.assertEqual([pool.items[k]['status'] for k in keys], ['done']+['filtered']*4+['pending'])
        self.assertEqual({f['stage'] for f in pool.filtered}, {'execution'})

    def test_two_pending_focus_candidates_use_one_focus(self):
        pool, root = pool_with_read()
        keys = [pool.add(dict(time=t, modality='video'), parent_id=root, origin='observation') for t in ([105., 110.], [112., 118.])]
        pool.prune_refinements()
        self.assertEqual([pool.items[k]['status'] for k in keys], ['pending', 'pending'])
        pool.start(pool.items[keys[0]]['action']);pool.commit_refinement(keys[0]);pool.finish(keys[0])
        pool.prune_refinements()
        self.assertEqual(pool.items[keys[1]]['status'], 'filtered')

    def test_distinct_short_windows_are_distinct_raw_frames(self):
        pool, root = pool_with_read()
        first, second = dict(time=[117.2, 117.7], modality='video'), dict(time=[117.4, 117.9], modality='video')
        self.assertTrue(pool.new_still(first));self.assertTrue(pool.new_still(second))
        self.assertTrue(pool.admit_refinement(first, root, {}))  # executed as the region's focus
        key = pool.add(first, parent_id=root, origin='observation')
        pool.start(first);pool.commit_refinement(key);pool.finish(key);pool.record_read(first)
        self.assertFalse(pool.new_still(first));self.assertTrue(pool.new_still(second))
        self.assertFalse(pool.admit_refinement(second, key, {}))
        self.assertTrue(pool.filtered[-1]['new_still'])  # a different frame, refused by the bound, not as a duplicate
        self.assertEqual(pool.filtered[-1]['reason'], 'focus_used')

    def test_disabled_rule_admits_everything(self):
        pool, root = pool_with_read()
        pool.configure_novelty(2., None, 1)
        for _ in range(3):
            self.assertTrue(pool.admit_refinement(dict(time=[118.5, 120.5], modality='video'), root, {}))

    def test_invalid_configuration(self):
        pool = CandidatePool(10., 5., ALL)
        for args in [(0, 5, 1), (2, 0, 1), (2, float('inf'), 1), (2, 5, -1)]:
            with self.assertRaises(ValueError):
                pool.configure_novelty(*args)


class Narrowing(OfflineBackend):
    """Every Observe proposes a narrower same-modality window: the spiral seen in the API pilot."""
    def __init__(self):
        self.calls = []

    def generate_text(self, messages, kind, step):
        self.calls.append(kind)
        if kind != 'memory':
            return super().generate_text(messages, kind, step)
        payload = json.loads(messages[-1]['content'][-1]['text'])
        length = payload['clip_seconds']
        modality = 'video' if payload['modality'] == 'text' else payload['modality']
        local = [0., 6.] if payload['modality'] == 'text' else [round(.1*length, 6), round(.9*length, 6)]
        text = json.dumps(dict(observations=['A concrete finding.'],
                               candidates=[dict(local_time=local, modality=modality, hint='A clue', goal='Inspect closer')]))
        return text, dict(messages=messages, kind=kind, step=step, text=text, tokens=[1], offline=True)

    def generate(self, messages, kind, step):
        self.calls.append(kind)
        if kind == 'candidate_scan':
            payload = json.loads(messages[-1]['content'][-1]['text'])
            obj = dict(summary='A synthetic source.', candidates=[dict(local_time=[0, payload['source_seconds']], modality=payload['observation']['modality'], hint='A clue', goal='Inspect this behavior')])
        elif kind == 'final':
            obj = {**{t: None for t in TRAITS}, 'reasoning': 'Synthetic.', 'contextual': []}
        else:
            return super().generate(messages, kind, step)
        text = json.dumps(obj)
        return obj, dict(messages=messages, kind=kind, step=step, text=text, tokens=[1], offline=True)


def run(**cfg):
    backend = Narrowing()
    trajectory = ActiveAgent(backend, OfflineTraitScorer(), SyntheticMedia(),
                             dict(CFG, selection_policy='queue', max_steps=12, video_fps=2, **cfg)).run(ROW)
    return trajectory, backend


class SpiralTests(unittest.TestCase):
    def test_rule_stops_the_narrowing_spiral_without_repair_calls(self):
        gated, backend = run(min_new_seconds=5., max_focus_per_region=1)
        free, _ = run(min_new_seconds=None)
        self.assertEqual(len(free['steps']), 12)
        self.assertEqual(free['steps'][-1]['followup_status'], 'disabled_budget')
        self.assertLess(free['steps'][-1]['action']['time'][1]-free['steps'][-1]['action']['time'][0], 1.)
        # text root -> video (new channel) -> one same-channel focus -> filtered
        self.assertEqual([s['action']['modality'] for s in gated['steps']], ['text', 'video', 'video'])
        self.assertEqual(gated['stop_reason'], 'candidate_pool_exhausted')
        self.assertEqual(len(gated['novelty_filtered']), 1)
        self.assertEqual(gated['novelty_filtered'][0]['reason'], 'focus_used')
        self.assertEqual(gated['steps'][-1]['followup_status'], 'no_new_media_content')
        self.assertNotIn('candidate_repair', backend.calls)
        self.assertEqual(gated['run_status'], 'complete')
        self.assertEqual(gated['novelty_rule'], dict(video_fps=2., min_new_seconds=5., max_focus=1))

    def test_prompt_states_the_rule_only_when_enabled(self):
        gated, _ = run(min_new_seconds=5.)
        free, _ = run(min_new_seconds=None)
        memory_prompt = lambda t: next(d for d in t['decisions'] if d['kind'] == 'memory')['messages'][-1]['content'][-2]['text']
        self.assertIn('at least 5.0 s of unread_local_time', memory_prompt(gated))
        self.assertNotIn('unread_local_time', memory_prompt(free))


if __name__ == '__main__':
    unittest.main()
