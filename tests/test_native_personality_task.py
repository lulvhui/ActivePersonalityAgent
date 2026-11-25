import unittest
from baselines.omniagent_native.task import personality_question,parse_levels,evaluate_levels

class NativeTaskTests(unittest.TestCase):
    def test_question_does_not_expose_labels_ids_or_transcript(self):
        row=dict(id='PRIVATE_ID',person_id='PRIVATE_PERSON',labels={'O':'SECRET_LABEL'},
                 segments=[dict(text='PRIVATE_TRANSCRIPT')],question='How did you resolve the problem?')
        question=personality_question(row)
        for value in ('PRIVATE_ID','PRIVATE_PERSON','SECRET_LABEL','PRIVATE_TRANSCRIPT'):self.assertNotIn(value,question)
        self.assertIn(row['question'],question)
        self.assertNotIn('Candidate Pool',question);self.assertNotIn('Trait Memory',question)
        self.assertIn('O=level',question);self.assertNotIn('O=level',personality_question(row,native_format=False))

    def test_missing_conflicting_and_abstained_predictions_are_distinct(self):
        parsed=parse_levels('O=high; C=null; E=low; A=high; A=low; N=unknown')
        self.assertEqual(parsed['O']['status'],'predicted')
        self.assertEqual(parsed['C']['status'],'abstained')
        self.assertEqual(parsed['A']['status'],'conflicting')
        self.assertEqual(parsed['N']['status'],'missing')

    def test_no_prediction_has_no_fabricated_accuracy(self):
        scores=evaluate_levels(parse_levels('O=null; C=null; E=null; A=null; N=null'),dict(O=4,C=2,E=1,A=3,N=0))
        self.assertEqual(scores['coverage'],0);self.assertTrue(scores['all_fields_recognized'])
        self.assertIsNone(scores['ordinal_mae_on_predictions'])
        self.assertIsNone(scores['exact_accuracy_on_predictions'])

    def test_unspecified_numeric_scale_is_not_silently_mapped(self):
        parsed=parse_levels('C=4, O=3, E=2, A=3, N=2')
        self.assertTrue(all(item['status']=='missing' for item in parsed.values()))

if __name__=='__main__':unittest.main()
