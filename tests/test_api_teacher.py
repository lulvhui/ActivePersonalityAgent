"""Hosted transport and existing-agent integration; no external requests."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from amp.agent import ActiveAgent
from amp.backend import GenerationFormatError, QwenAPIBackend, message
from amp.scoring import OfflineTraitScorer
from amp.smoke import SyntheticMedia


class APITeacherTests(unittest.TestCase):
    def test_person_questions_survive_input_sanitizing_without_labels(self):
        from scripts.test_api_teacher import diagnostic_row
        row=dict(id='synthetic',turns=[dict(question='Describe a project',time=[0,20],labels={'E':1},answer='OLD_MODEL_ANSWER')],labels={'E':1})
        result=diagnostic_row(row)
        self.assertEqual(result['turns'],[dict(question='Describe a project',time=[0,20])])
        self.assertNotIn('labels',result)
        self.assertNotIn('OLD_MODEL_ANSWER',str(result))

    def backend(self, **kwargs):
        with patch.dict(os.environ, DASHSCOPE_API_KEY='fixture-only-secret'):
            return QwenAPIBackend('https://dashscope.aliyuncs.com/compatible-mode/v1', allow_external=True, **kwargs)

    def response(self, content, finish='stop'):
        return io.BytesIO(json.dumps(dict(id='fixture-response', model='qwen3.8-omni-flash',
            choices=[dict(message=dict(content=content), finish_reason=finish)],
            usage=dict(prompt_tokens=123, completion_tokens=17, total_tokens=140))).encode())

    def test_requires_permission_credentials_and_official_https_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'permission'):
            QwenAPIBackend('https://dashscope.aliyuncs.com/compatible-mode/v1')
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'DASHSCOPE_API_KEY'):
            QwenAPIBackend('https://dashscope.aliyuncs.com/compatible-mode/v1', allow_external=True)
        with patch.dict(os.environ, DASHSCOPE_API_KEY='fixture-only-secret'):
            for url in ['http://dashscope.aliyuncs.com/v1', 'https://aliyuncs.com.evil.example/v1',
                        'https://user@dashscope.aliyuncs.com/v1', 'https://dashscope.aliyuncs.com/v1?key=x']:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    QwenAPIBackend(url, allow_external=True)

    def test_response_usage_and_json_instruction_preserved_without_fake_token_ids(self):
        backend = self.backend()
        messages = [dict(role='system', content='Locate leads.'), message(dict(task='Find leads.'))]
        with patch('urllib.request.urlopen', return_value=self.response('{"candidates":[]}')) as call:
            obj, decision = backend.generate(messages, 'candidate_scan', 0)
        request = json.loads(call.call_args.args[0].data)
        self.assertIn('JSON', request['messages'][0]['content'])
        self.assertEqual(request['messages'], decision['messages'])
        self.assertEqual(messages[0]['content'], 'Locate leads.')
        self.assertEqual(obj, dict(candidates=[]))
        self.assertEqual(decision['tokens'], [])
        self.assertEqual(decision['completion_tokens'], 17)
        self.assertEqual(decision['tokenization'], 'pending_student_tokenizer')
        self.assertEqual(request['response_format'], dict(type='json_object'))

    def test_raw_text_reader_still_requests_json_mode_for_actor_schema(self):
        backend=self.backend()
        schema=dict(type='object',properties=dict(observations=dict(type='array')),required=['observations'])
        messages=[dict(role='system',content='Describe this source.',output_schema=schema)]
        with patch('urllib.request.urlopen',return_value=self.response('{"observations":[]}')) as call:
            text,decision=backend.generate_text(messages,'memory',0)
        request=json.loads(call.call_args.args[0].data)
        self.assertEqual(request['response_format'],dict(type='json_object'))
        self.assertIn('Output JSON schema',request['messages'][0]['content'])
        self.assertEqual(text,'{"observations":[]}')
        self.assertEqual(decision['tokens'],[])

    def test_invalid_and_truncated_response_is_saved_before_validation(self):
        for text, finish in [('not json', 'stop'), ('{"stop":true}', 'length')]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                backend = self.backend(generation_log_dir=folder)
                with patch('urllib.request.urlopen', return_value=self.response(text, finish)):
                    with self.assertRaises(GenerationFormatError) as error:
                        backend.generate([message(dict(task='Select'))], 'action', 0)
                self.assertEqual(error.exception.decision['text'], text)
                self.assertEqual(error.exception.decision['truncated'],finish=='length')
                log = next(Path(folder).glob('*.json')).read_text()
                self.assertNotIn('fixture-only-secret', log)
                self.assertIn('response', json.loads(log))
                self.assertEqual(len(backend.calls), 1)

    def test_audio_encoding_and_size_limit_do_not_silently_trim(self):
        backend = self.backend()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.wav'
            path.write_bytes(b'RIFF_fixture_audio')
            converted = backend.prepare([message(dict(task='Listen'), [dict(type='audio', audio=str(path))])])
            audio = converted[0]['content'][0]
            self.assertEqual(audio['type'], 'input_audio')
            self.assertEqual(audio['input_audio']['format'], 'wav')
            self.assertTrue(audio['input_audio']['data'].startswith('data:audio/wav;base64,'))
            with patch.object(backend, 'data_url', return_value='x'*10_000_000):
                with self.assertRaisesRegex(ValueError, 'not truncated'):
                    backend.prepare([message(dict(task='Listen'), [dict(type='audio', audio=str(path))])])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required for local frame transport check')
    def test_video_preserves_all_existing_frames_and_fps_without_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=1',
                            '-t', '3', '-an', '-c:v', 'libx264', str(path)], check=True, capture_output=True)
            backend = self.backend()
            content = backend.prepare([message(dict(task='Inspect'), [dict(type='video', video=str(path), fps=1)])])[0]['content']
            self.assertEqual(content[0]['type'], 'video')
            self.assertEqual(content[0]['fps'], 1)
            self.assertEqual(len(content[0]['video']), 3)
            self.assertTrue(all(x.startswith('data:image/jpeg;base64,') for x in content[0]['video']))
            self.assertNotIn('input_audio', str(content))

    def test_http_error_has_one_attempt_and_redacted_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.backend(generation_log_dir=folder)
            error = HTTPError(backend.url, 401, 'Unauthorized', {}, io.BytesIO(b'fixture-only-secret invalid'))
            with patch('urllib.request.urlopen', side_effect=error) as call, self.assertRaisesRegex(RuntimeError, '401'):
                backend.generate([message(dict(task='Select'))], 'action', 0)
            self.assertEqual(call.call_count, 1)
            self.assertNotIn('fixture-only-secret', next(Path(folder).glob('*.json')).read_text())
            self.assertEqual(backend.calls[0]['status'], 'api_error')



if __name__ == '__main__':
    unittest.main()
