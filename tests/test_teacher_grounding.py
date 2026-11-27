"""Targeted coarse candidates, explicit clocks and capabilities in Observe, and Final evidence rules."""
import json, unittest
from amp.agent import ActiveAgent
from amp.perception import admit_candidates, generation_schema
from amp.scoring import OfflineTraitScorer
from amp.state import CandidatePool, empty_memory, save_observation
from test_simple_memory import Reader, CFG, OBS

ALL = ['text', 'audio', 'video', 'audio_video']


def coarse(local_time, modality='audio_video'):
    return dict(local_time=local_time, modality=modality, hint='Describes organizing a trip', goal='Hear how the plan is described')


class CoarseTargetTests(unittest.TestCase):
    def test_coarse_candidate_marks_a_moment_inside_the_long_source(self):
        pool = CandidatePool(300., 30., ALL)
        source = dict(time=[60., 120.], modality='audio_video')
        ids, errors = admit_candidates(pool, dict(candidates=[coarse([10, 25]), coarse([50, 70]), coarse([0, 60])]), source, 'global_audio_video_1')
        self.assertEqual(len(ids), 1)
        self.assertEqual(pool.items[ids[0]]['action']['time'], [70., 85.])
        self.assertEqual(pool.items[ids[0]]['leads'][0]['source_fragment_time'], [60., 120.])
        self.assertIn('inside this source', errors[0]['error'])
        self.assertIn('Invalid action window', errors[1]['error'])  # a whole 60 s source exceeds the 30 s video clip

    def test_coarse_schema_requires_local_time_and_modality(self):
        item = generation_schema('candidate_scan', modalities=ALL)['properties']['candidates']['items']
        self.assertEqual(set(item['required']), {'local_time', 'modality', 'hint', 'goal'})
        self.assertEqual(item['properties']['modality']['enum'], ALL)

    def test_default_coarse_chunk_is_longer_than_a_closer_read(self):
        self.assertGreater(CFG['coarse_chunk_seconds'], CFG['max_clip_seconds'])


class ObserveClockTests(unittest.TestCase):
    def observe(self, obs, memory=None, goal=None):
        reader = Reader()
        ActiveAgent(reader, OfflineTraitScorer(), None, CFG).update(memory or empty_memory(), {}, {'active_9': obs}, 0,
                                                                  followup_goal=goal, media_duration=300.)
        messages = reader.seen[-1][2]
        return json.loads(messages[-1]['content'][-1]['text']), messages[-1]['content'][-2]['text']

    def test_goal_times_carry_the_proposing_clip_clock(self):
        # IELTS:183 C27: proposed on [160,180] with "17-19 s"; read as [176,180], where it is 1-3 s.
        goal = [dict(goal='Resolve gaze in the 17-19 s window', hint='Calls the question tricky', proposed_on_local=[-16., 4.])]
        payload, task = self.observe(dict(OBS, time=[176., 180.], modality='video'), goal=goal)
        self.assertEqual(payload['goal'], goal)
        self.assertEqual(payload['source_window_recording_seconds'], [176., 180.])
        self.assertIn('proposed_on_local', task)

    def test_capabilities_state_missing_channels_and_no_measurement_tools(self):
        payload, task = self.observe(dict(OBS, time=[10., 20.], modality='video'))
        self.assertEqual(payload['capabilities'], dict(sound=False, picture=True, video_frames_per_second=CFG['video_fps'],
                                                       transcript_only=False, measurement_tools='none'))
        self.assertIn('no numeric acoustic or kinematic measurements', task)
        audio, _ = self.observe(dict(OBS, time=[10., 20.], modality='audio'))
        self.assertEqual((audio['capabilities']['sound'], audio['capabilities']['picture']), (True, False))

    def test_prior_observations_keep_their_time_on_the_current_clock(self):
        memory, _ = save_observation(empty_memory(), json.dumps(dict(observations=['Mentions baking for her parents.'], candidates=[])),
                                     dict(OBS, time=[160., 180.]), 'active_0')
        payload, _ = self.observe(dict(OBS, time=[176., 180.]), memory=memory)
        self.assertEqual(payload['prior_observations'][0]['local_time'], [-16., 4.])

    def test_run_passes_lead_clocks_to_observe(self):
        class Scan(Reader):
            def generate(self, messages, kind, step):
                if kind == 'candidate_scan':
                    payload = json.loads(messages[-1]['content'][-1]['text'])
                    obj = dict(summary='A source.', candidates=[dict(local_time=[20, 30], modality='text', hint='A clue', goal='Read it')])
                    return obj, dict(messages=messages, kind=kind, step=step, text=json.dumps(obj), tokens=[1], offline=True)
                return super().generate(messages, kind, step)
        from amp.smoke import SyntheticMedia
        reader = Scan()
        ActiveAgent(reader, OfflineTraitScorer(), SyntheticMedia(), dict(CFG, selection_policy='queue', max_steps=1)).run(
            dict(id='S', dataset='SYNTHETIC', split='train', person_id='p', segments=[]))
        payload = json.loads(next(m for k, _, m in reader.seen if k == 'memory')[-1]['content'][-1]['text'])
        self.assertEqual(payload['goal'], [dict(goal='Read it', hint='A clue', proposed_on_local=[-20., 40.])])


class FinalRuleTests(unittest.TestCase):
    def test_final_accounts_for_setting_language_and_abstention(self):
        reader = Reader()
        from amp.smoke import SyntheticMedia
        t = ActiveAgent(reader, OfflineTraitScorer(), SyntheticMedia(), dict(CFG, selection_policy='queue', max_steps=1)).run(
            dict(id='S', dataset='SYNTHETIC', split='train', person_id='p', segments=[]))
        task = t['decisions'][-1]['messages'][-1]['content'][0]['text']
        for phrase in ('second language', 'Behavior the setting requires', 'keep evidence that points the other way', 'return null'):
            self.assertIn(phrase, task)


if __name__ == '__main__':
    unittest.main()
