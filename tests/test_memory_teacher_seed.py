"""Legacy annotations stay outside source-only teacher input and person splits."""
import json
import tempfile
import unittest
from pathlib import Path

from amp.common import TRAITS, read_json, read_jsonl, write_json, write_jsonl
from amp.backend import OfflineBackend
from scripts.prepare_memory_teachers import prepare, draft, source_fingerprint


class MemoryTeacherSeedTests(unittest.TestCase):
    def fixture(self, root):
        legacy, manifests = root/'legacy', root/'manifests'
        rows = dict(train=[],dev=[],test=[])
        for dataset in ('IELTS','RecruitView'):
            media = root/f'{dataset}.mp4';media.write_text('SYNTHETIC_TEXT_FIXTURE_NOT_MEDIA')
            write_json(legacy/f'artifacts/training_data/{dataset}/splits/split_manifest.json',
                       dict(splits=dict(train=['01'],dev=['02'],test=[])))
            turns, labels = [], []
            for person in ('1','2'):
                turns.append(dict(person_id=person,turn_id=1,answer_text='I help my team.',video_path=str(media)))
                labels.append(dict(person_id=person,turn_id=1,answer_text='I help my team.',question='LEGACY_PRIVATE_QUESTION',
                    trait_labels={name:dict(is_evidence=code=='A',direction='support_high' if code=='A' else 'no_clear') for code,name in TRAITS.items()}))
                row=dict(id=f'{dataset}:{person}',dataset=dataset,person_id=person,split='train' if person=='1' else 'dev',
                         video=str(media),question='What did you do?',segments=[dict(start=0,end=5,text='I help my team.')],labels=dict(O=4))
                rows[row['split']].append(row)
            write_jsonl(legacy/f'data/{dataset}/processed/video_turn_inputs/{dataset.lower()}_common_turns.jsonl',turns)
            write_jsonl(legacy/f'data/{dataset}/gt/evidence_gt.jsonl',labels)
        for split,values in rows.items():write_jsonl(manifests/f'{split}.jsonl',values)
        return legacy,manifests

    def test_source_mapping_train_boundary_and_reference_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root);output=root/'out'
            report=prepare(legacy,manifests,output,2)
            items=list(read_jsonl(output/'source_inputs.jsonl'))
            self.assertEqual(report['matched_total'],2)
            self.assertEqual(report['counts']['IELTS/excluded_non_train'],1)
            self.assertEqual(report['counts']['RecruitView/excluded_non_train'],1)
            for item in items:
                self.assertEqual(item['split'],'train');self.assertEqual(item['person_id'],'1')
                self.assertEqual(item['source_fingerprint'],source_fingerprint(item))
                self.assertNotIn('trait_labels',json.dumps(item));self.assertNotIn('LEGACY_PRIVATE_QUESTION',json.dumps(item))
                self.assertIsNone(item['observation']['segments'][0]['media_time'])
            refs=list(read_jsonl(output/'legacy_references.jsonl'))
            self.assertIn('unknown_per_item',refs[0]['annotation_origin'])
            self.assertFalse(refs[0]['approved'])

    def test_mismatched_text_is_rejected_instead_of_fuzzy_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root)
            p=legacy/'data/RecruitView/gt/evidence_gt.jsonl';rows=list(read_jsonl(p));rows[0]['answer_text']='I hurt my team.';write_jsonl(p,rows)
            report=prepare(legacy,manifests,root/'out')
            self.assertEqual(report['matched_total'],1)
            self.assertEqual(list(read_jsonl(root/'out/rejected.jsonl'))[0]['reason'],'annotation_turn_text_mismatch')

    def test_current_legacy_split_disagreement_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root)
            p=manifests/'train.jsonl';rows=list(read_jsonl(p));rows[0]['split']='dev';write_jsonl(p,rows)
            with self.assertRaises(ValueError):prepare(legacy,manifests,root/'out')

    def test_raw_source_difference_cannot_inherit_legacy_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root)
            p=manifests/'train.jsonl';rows=list(read_jsonl(p));rows[1]['segments'][0]['text']='I hurt my team.';write_jsonl(p,rows)
            report=prepare(legacy,manifests,root/'out')
            self.assertEqual(report['matched_total'],1)
            self.assertEqual(list(read_jsonl(root/'out/rejected.jsonl'))[0]['reason'],'raw_transcript_mismatch_or_ambiguity')

    def test_source_change_invalidates_draft_and_review_packet_binding(self):
        from scripts.prepare_memory_teachers import review_packet
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root);prepare(legacy,manifests,root/'out')
            cfg=read_json(Path(__file__).resolve().parents[1]/'configs/default.json')
            draft(root/'out/pilot_inputs.jsonl',root/'draft','SYNTHETIC',cfg,OfflineBackend())
            review_packet(root/'draft/drafts.json',root/'out/legacy_references.jsonl',root/'packet.md')
            self.assertIn('未批准',(root/'packet.md').read_text())
            rows=list(read_jsonl(root/'out/pilot_inputs.jsonl'));rows[0]['question']='Changed after preparation'
            write_jsonl(root/'changed.jsonl',rows)
            with self.assertRaises(ValueError):draft(root/'changed.jsonl',root/'invalid','SYNTHETIC',cfg,OfflineBackend())
            refs=list(read_jsonl(root/'out/legacy_references.jsonl'));refs[0]['source_fingerprint']='wrong'
            write_jsonl(root/'refs.jsonl',refs)
            with self.assertRaises(ValueError):review_packet(root/'draft/drafts.json',root/'refs.jsonl',root/'bad_packet.md')

    def test_drafts_are_unapproved_and_actual_prompt_has_no_legacy_answers(self):
        class Backend(OfflineBackend):
            def __init__(self):self.prompts=[]
            def generate_text(self,messages,kind,step):
                self.prompts.append(json.dumps(messages));return super().generate_text(messages,kind,step)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root);prepare(legacy,manifests,root/'out')
            backend=Backend();cfg=read_json(Path(__file__).resolve().parents[1]/'configs/default.json')
            report=draft(root/'out/pilot_inputs.jsonl',root/'draft','SYNTHETIC_NOT_MODEL',cfg,backend)
            self.assertEqual(report['source_valid'],2);self.assertEqual(report['automatically_approved'],0)
            self.assertNotIn('support_high',' '.join(backend.prompts))
            self.assertTrue(all(r['approved'] is False for r in read_json(root/'draft/reviews.json')))
            with self.assertRaises(ValueError):draft(root/'out/pilot_inputs.jsonl',root/'draft','SYNTHETIC',cfg,backend)
            self.assertEqual(len(backend.prompts),2)

    def test_plain_text_drafts_need_no_tool_schema_and_remain_unapproved(self):
        class Backend(OfflineBackend):
            def generate_tools(self,*args):raise AssertionError('No Memory tools')
            def generate_text(self,messages,kind,step):
                text="An ordinary observation, not JSON."
                return text,dict(kind=kind,step=step,messages=messages,tokens=[],text=text)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy,manifests=self.fixture(root)
            prepare(legacy,manifests,root/'out')
            cfg=read_json(Path(__file__).resolve().parents[1]/'configs/default.json')
            report=draft(root/'out/pilot_inputs.jsonl',root/'draft','SYNTHETIC',cfg,Backend())
            self.assertEqual(report['source_valid'],0)
            self.assertEqual(report['automatically_approved'],0)
            self.assertTrue(all(r['memory_after']['observations'][0]['content']=='An ordinary observation, not JSON.'
                                for r in read_json(root/'draft/drafts.json')))


if __name__=='__main__':unittest.main()
