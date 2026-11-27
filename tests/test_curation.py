"""Teacher curation: pre-screen, human review contract, reviewed Memory, Final resampling and split export."""
import copy, json, unittest
from amp import curation
from amp.agent import ActiveAgent
from amp.common import TRAITS, LEVELS
from amp.evaluate import majority_baseline
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia
from test_simple_memory import Reader, CFG

ROW = dict(id='IELTS:1', dataset='IELTS', split='train', person_id='p1', segments=[])


class Tokenizer:
    eos_token = '<eos>'

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FinalReader(Reader):
    """Final levels follow a counter so resampled candidates differ."""
    def __init__(self):
        super().__init__(text=json.dumps(dict(observations=['Says she bakes cakes for her parents.'], candidates=[])))
        self.finals = 0

    def generate(self, messages, kind, step):
        if kind != 'final':
            return super().generate(messages, kind, step)
        level = LEVELS[self.finals % 5];self.finals += 1
        obj = {**{t: level for t in TRAITS}, 'reasoning': f'Candidate {self.finals}.',
               'contextual': [dict(trait='A', context_type='reported', context='family', expression='bakes for parents', citations=['M1'])]}
        text = json.dumps(obj)
        return obj, dict(messages=messages, kind=kind, step=step, text=text, tokens=[], offline=True)


def teacher(reader=None):
    reader = reader or FinalReader()
    t = ActiveAgent(reader, OfflineTraitScorer(), SyntheticMedia(), dict(CFG, selection_policy='queue', max_steps=2)).run(ROW)
    t['offline'] = False
    return t, reader


def review_all(reviews, action='keep'):
    for r in reviews:
        r.update(action=action, reviewer='rater1')
    return reviews


class RuleTests(unittest.TestCase):
    def test_rule_flags(self):
        self.assertEqual(curation.rule_flags('Pitch drops by 20 Hz on "good".'), ['numeric_measurement'])
        self.assertEqual(curation.rule_flags('A genuine smile, not a performed one.'), ['inferred_motive'])
        self.assertEqual(curation.rule_flags('She says she likes music.', 'video'), ['missing_channel'])
        self.assertEqual(curation.rule_flags('She says she likes music.', 'audio_video'), [])
        self.assertEqual(curation.rule_flags('Leans forward and smiles.', 'audio'), ['missing_channel'])


class PrescreenTests(unittest.TestCase):
    def test_without_judge_every_record_must_be_reviewed(self):
        t, _ = teacher()
        reviews, excluded = curation.prescreen([t], sample_rate=0.)
        records = [r for r in reviews if r['record_id']]
        self.assertTrue(records)
        self.assertTrue(all(r['priority'] == 'must' for r in records))
        self.assertTrue(any(r['kind'] == 'candidate_scan' for r in reviews))
        self.assertEqual(excluded, [])

    def test_judge_support_and_flags_set_priority(self):
        t, _ = teacher()
        checks = {t['sample_id']: {m['id']: dict(supported=True, reason='ok') for m in t['memory']['observations']}}
        reviews, _ = curation.prescreen([t], checks, sample_rate=0.)
        self.assertTrue(all(r['priority'] == 'optional' for r in reviews if r['record_id']))
        checks[t['sample_id']]['M1']['supported'] = False
        reviews, _ = curation.prescreen([t], checks, sample_rate=0.)
        self.assertEqual(next(r for r in reviews if r['record_id'] == 'M1')['priority'], 'must')
        sampled, _ = curation.prescreen([t], checks, sample_rate=1.)
        self.assertTrue(all(r['priority'] in ('must', 'sample') for r in sampled))

    def test_non_train_or_offline_is_refused(self):
        t, _ = teacher()
        for change in (dict(split='dev'), dict(offline=True)):
            with self.assertRaises(ValueError):
                curation.prescreen([dict(t, **change)])

    def test_packet_lists_items_without_labels(self):
        t, _ = teacher()
        t['labels'] = {k: 4 for k in TRAITS}
        reviews, _ = curation.prescreen([t])
        packet = curation.review_packet([t], reviews)
        self.assertIn('labels are not shown', packet)
        self.assertNotIn('"labels"', packet)
        self.assertIn('Says she bakes cakes', packet)


class ReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.t, _ = teacher()
        self.reviews, _ = curation.prescreen([self.t], sample_rate=0.)

    def test_required_items_must_be_reviewed(self):
        errors, pending = curation.check_reviews([self.t], self.reviews)
        self.assertEqual(errors, [])
        self.assertTrue(pending)
        errors, pending = curation.check_reviews([self.t], review_all(self.reviews))
        self.assertEqual((errors, pending), ([], []))

    def test_review_errors(self):
        reviews = review_all(self.reviews)
        record = next(r for r in reviews if r['record_id'])
        record.update(action='edit', corrected_text='not json')
        errors, _ = curation.check_reviews([self.t], reviews)
        self.assertEqual({e['error'].split(':')[0] for e in errors}, {'reason_code must be one of (\'measurement\', \'motive\', \'modality\', \'unsupported\', \'timing\', \'speaker\', \'other\')', 'corrected_text'})
        record.update(reason_code='motive', corrected_text=json.dumps(dict(observations=['Says she bakes cakes.'], candidates=[])), reviewer='')
        errors, _ = curation.check_reviews([self.t], reviews)
        self.assertEqual([e['error'] for e in errors], ['reviewer is required'])
        record['action'] = 'approve'
        self.assertIn('action must be', curation.check_reviews([self.t], reviews)[0][0]['error'])

    def test_changed_decision_makes_review_stale(self):
        reviews = review_all(self.reviews)
        changed = copy.deepcopy(self.t)
        index = next(r['decision_index'] for r in reviews if r['record_id'])
        changed['decisions'][index]['text'] = json.dumps(dict(observations=['Something else.'], candidates=[]))
        errors, _ = curation.check_reviews([changed], reviews)
        self.assertEqual([e['error'] for e in errors], ['stale review: the decision or its source changed'])

    def test_agreement_against_judge_and_flags(self):
        reviews = [dict(action='keep', judge=dict(supported=True), flags=[], reason_code=None),
                   dict(action='edit', judge=dict(supported=True), flags=['inferred_motive'], reason_code='motive'),
                   dict(action='drop', judge=dict(supported=False), flags=['numeric_measurement'], reason_code='measurement'),
                   dict(action='keep', judge=dict(supported=False), flags=[], reason_code=None)]
        a = curation.agreement(reviews)
        self.assertEqual(a['judged_and_reviewed'], 4)
        self.assertEqual((a['agreement'], a['judge_miss_rate'], a['judge_false_alarm_rate'], a['flag_precision']), (.5, .5, .5, 1.))
        self.assertEqual(a['reason_codes'], dict(motive=1, measurement=1))


class ReviewedMemoryTests(unittest.TestCase):
    def test_edit_drop_and_unaccepted_records(self):
        t, _ = teacher()
        reviews, _ = curation.prescreen([t], sample_rate=0.)
        review_all(reviews)
        record = next(r for r in reviews if r['record_id'] == 'M1')
        fixed = json.dumps(dict(observations=['Says she bakes for her parents.'], candidates=[]))
        record.update(action='edit', reason_code='unsupported', corrected_text=fixed)
        memory, checks, dropped = curation.reviewed_memory(t, reviews)
        self.assertEqual(memory['observations'][0]['content'], fixed)
        self.assertEqual(memory['observations'][0]['semantic_review'], 'human_edited')
        self.assertTrue(checks['M1']['supported'])
        self.assertEqual(t['memory']['observations'][0]['content'], t['decisions'][record['decision_index']]['text'])  # original untouched
        record.update(action='drop', corrected_text=None)
        memory, checks, dropped = curation.reviewed_memory(t, reviews)
        self.assertNotIn('M1', [r['id'] for r in memory['observations']])
        self.assertIn('M1', dropped)


class FinalAndExportTests(unittest.TestCase):
    def setUp(self):
        self.t, reader = teacher()
        self.reviews = review_all(curation.prescreen([self.t], sample_rate=0.)[0])
        memory, _, _ = curation.reviewed_memory(self.t, self.reviews)
        agent = ActiveAgent(reader, None, None, CFG)
        self.resampled = [curation.resample_final(agent, self.t, memory, 3)]

    def test_resample_uses_reviewed_memory_and_differs(self):
        item = self.resampled[0]
        self.assertEqual(len(item['candidates']), 3)
        self.assertEqual(len({c['final']['O']['level'] for c in item['candidates']}), 3)
        payload = json.loads(item['candidates'][0]['decision']['messages'][-1]['content'][-1]['text'])
        self.assertIn('Says she bakes cakes', json.dumps(payload['memory']))
        self.assertNotIn('labels', json.dumps(payload))
        self.assertIn('choose before seeing labels', curation.final_packet(self.resampled))

    def choose(self, index, **change):
        choice = curation.final_review_template(self.resampled[0])
        choice.update(chosen=index, reviewer='rater1', reason='best supported', **change)
        return [choice]

    def export(self, choices, labels):
        return curation.export_curated([self.t], self.reviews, self.resampled, choices, labels, Tokenizer())

    def test_export_splits_perception_and_label_gated_final(self):
        near = {t: LEVELS.index('low') for t in TRAITS}  # candidate 1 is 'low' everywhere
        perception, final, report = self.export(self.choose(1), {'IELTS:1': near})
        self.assertEqual(len(perception), len(self.reviews))
        self.assertTrue(all(r['split'] == 'train' and r['tokens'] for r in perception))
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]['curation']['basis'], 'human_choice_label_gated')
        self.assertEqual(final[0]['tokens'][-5:], [ord(c) for c in '<eos>'])
        self.assertEqual(report['final_rows'], 1)
        far = {t: 4 for t in TRAITS}
        for choices, labels, reason in [(self.choose(0), {'IELTS:1': far}, 'label_gate'),
                                        (self.choose(1, labels_seen=True), {'IELTS:1': near}, 'chosen_with_labels_visible'),
                                        (self.choose(1), {}, 'no_sample_labels'),
                                        (self.choose(None), {'IELTS:1': near}, 'no_candidate_chosen')]:
            _, final, report = self.export(choices, labels)
            self.assertEqual(final, [])
            self.assertEqual(report['dropped'][-1]['reason'], reason)

    def test_edited_perception_target_is_retokenized(self):
        record = next(r for r in self.reviews if r['record_id'] == 'M1')
        fixed = json.dumps(dict(observations=['Says she bakes for her parents.'], candidates=[]))
        record.update(action='edit', reason_code='unsupported', corrected_text=fixed)
        perception, _, report = self.export(self.choose(1), {'IELTS:1': {t: 1 for t in TRAITS}})
        row = next(r for r in perception if r['curation']['review_key'] == record['key'])
        self.assertEqual(row['text'], fixed)
        self.assertEqual(row['tokens'], [ord(c) for c in fixed+'<eos>'])
        self.assertTrue(row['curation']['human_edited'])
        self.assertNotIn('format_mode', row)
        self.assertEqual(report['perception_by_basis'].get('human_edit'), 1)

    def test_incomplete_reviews_block_export(self):
        self.reviews[0]['action'] = None
        self.reviews[0]['priority'] = 'must'
        with self.assertRaises(ValueError):
            self.export(self.choose(1), {})


class MajorityBaselineTests(unittest.TestCase):
    def test_majority_level_per_trait(self):
        train = [dict(dataset='IELTS', split='train', labels={t: level for t in TRAITS}) for level in (1, 1, 2)]
        lookup = {'a': dict(labels={t: 1 for t in TRAITS}), 'b': dict(labels={t: 3 for t in TRAITS})}
        base = majority_baseline([dict(sample_id='a'), dict(sample_id='b')], lookup, train, 'IELTS')
        self.assertEqual(base['traits']['O']['level'], 'low')
        self.assertEqual((base['mean_accuracy'], base['mean_ordinal_mae']), (.5, 1.))
        self.assertIsNone(majority_baseline([], lookup, train, 'RecruitView'))


if __name__ == '__main__':
    unittest.main()
