"""Person-level RecruitView inputs retain all questions on one media timeline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from amp.common import read_json,write_json,write_jsonl
from amp.data import recruitview_people
from amp.media import MediaTools
from amp.state import save_observation
from amp.perception import memory_view


class PersonInputTests(unittest.TestCase):
    def test_all_answers_are_joined_in_stable_order_without_importing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=root/'data/RecruitView/raw'
            (raw/'videos').mkdir(parents=True)
            for name in ('a.mp4','b.mp4'):(raw/'videos'/name).write_bytes(b'fixture')
            write_json(root/'artifacts/training_data/RecruitView/splits/split_manifest.json',dict(splits=dict(train=[],dev=['7'],test=[])))
            answers=[dict(id='2',user_no=7,file_name='b.mp4',question='Second question',transcript='[00:00 - 00:02] Second answer',openness=99,summary='DO NOT IMPORT'),
                     dict(id='1',user_no=7,file_name='a.mp4',question='First question',transcript='[00:00 - 00:01] First answer',openness=-99)]
            write_jsonl(raw/'asr/metadata.jsonl',answers)
            class Media:
                def probe(self,path):
                    duration=2 if Path(path).name=='a.mp4' else 3 if Path(path).name=='b.mp4' else 5
                    return dict(duration=duration,raw=dict(streams=[dict(codec_type='video',width=640,height=480)]))
            commands=[]
            def encode(command):
                commands.append(command);Path(command[-1]).write_bytes(b'synthetic encoded video')
            with patch('amp.media.run',side_effect=encode):
                rows=recruitview_people(root,['7'],root/'assembled',Media())
            self.assertEqual(len(rows),1)
            row=rows[0]
            self.assertEqual(row['id'],'RecruitView:person:7')
            self.assertEqual(row['split'],'dev')
            self.assertEqual([t['answer_id'] for t in row['turns']],['1','2'])
            self.assertEqual([t['time'] for t in row['turns']],[[0.,2.],[2.,5.]])
            self.assertEqual([(s['start'],s['end']) for s in row['segments']],[(0.,1.),(2.,4.)])
            self.assertEqual(row['segments'][1]['question'],'Second question')
            self.assertNotIn('labels',row);self.assertNotIn('labels_raw',row)
            self.assertNotIn('DO NOT IMPORT',str(row))
            self.assertEqual(commands[0].count('-i'),2)
            self.assertEqual(read_json(Path(row['video']).with_suffix('.json'))['person_id'],'7')

    def test_window_questions_include_both_sides_of_a_join(self):
        row=dict(turns=[dict(time=[0,20],question='First'),dict(time=[20,40],question='Second')])
        self.assertEqual([q['question'] for q in MediaTools.questions_in_window(row,18,23)],['First','Second'])
        self.assertEqual([q['question'] for q in MediaTools.questions_in_window(row,20,30)],['Second'])
        self.assertEqual(MediaTools.questions_in_window({},0,20),[])

    def test_short_visual_request_uses_one_real_frame_without_fake_video(self):
        with tempfile.TemporaryDirectory() as directory:
            media=MediaTools(directory,30,1)
            info=dict(duration=1.,source_key='synthetic',start_pts=0.)
            with patch.object(media,'probe',return_value=info),patch('amp.media.run') as encode:
                observation=media.observe(dict(video='fixture.mp4',segments=[]),dict(time=[0,.5],modality='video'))
            self.assertEqual(observation['media'][0]['type'],'image')
            self.assertEqual(observation['media'][0]['sampled_frames'],1)
            self.assertIn('no motion',observation['visual_representation'])
            self.assertEqual(encode.call_args.args[0].count('-frames:v'),1)

    def test_question_mapping_survives_memory_storage_and_reading(self):
        questions=[dict(time=[20,40],question='Describe a team experience')]
        observation=dict(time=[25,30],modality='audio',segments=[],questions=questions)
        memory,_=save_observation(dict(observations=[]),'A concrete behavior.',observation,'active_0')
        self.assertEqual(memory_view(memory)['observations'][0]['questions'],questions)


if __name__=='__main__':unittest.main()
