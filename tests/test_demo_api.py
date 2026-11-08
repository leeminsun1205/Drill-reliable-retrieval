"""Verify demo boundaries with hand-checkable evidence, no models or network."""
import json

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from drill_ir.demo.api import create_app


class Backend:
    def __init__(self):
        self.questions = []

    def search(self, question):
        self.questions.append(question)
        return [{'aid': 123, 'content_Article': 'Nguyên văn Điều luật.', 'law_id': 'L1', 'score': 0.5}]


def parse(response):
    return [json.loads(line) for line in response.text.splitlines()]


def test_lifecycle_evidence_stream_and_consecutive_questions(tmp_path):
    backend = Backend()
    loads = []
    def factory():
        loads.append(1)
        return backend
    def answer(question, evidence):
        assert evidence[0].aid == '123'
        assert evidence[0].text == 'Nguyên văn Điều luật.'
        yield 'Căn cứ '
        yield '[aid=123].'
    (tmp_path / 'index.html').write_text('<h1>Demo</h1>')
    with TestClient(create_app(backend_factory=factory, answer_stream=answer, static_dir=tmp_path)) as client:
        assert client.get('/').text == '<h1>Demo</h1>'
        assert client.get('/api/health').json()['ready']
        for question in ['  Câu một?  ', 'Câu hai?']:
            response = client.post('/api/chat', json={'question': question})
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('application/x-ndjson')
            events = parse(response)
            assert [e['type'] for e in events] == ['status', 'evidence', 'status', 'delta', 'delta', 'done']
            assert events[1]['evidence'][0]['aid'] == '123'
            assert ''.join(e['text'] for e in events if e['type'] == 'delta') == 'Căn cứ [aid=123].'
        assert loads == [1]
        assert backend.questions == ['Câu một?', 'Câu hai?']


def test_invalid_questions_and_failed_startup():
    def broken():
        raise RuntimeError('secret internal path')
    with TestClient(create_app(backend_factory=broken)) as client:
        assert not client.get('/api/health').json()['ready']
        assert client.post('/api/chat', json={'question': 'Câu hỏi'}).status_code == 503
        for question in ['', '   ', 'x' * 10001]:
            assert client.post('/api/chat', json={'question': question}).status_code == 422


def test_stream_error_retains_evidence_and_releases_lock():
    def broken(question, evidence):
        yield 'Một phần'
        raise RuntimeError('API_KEY=must-not-leak')
    with TestClient(create_app(backend_factory=Backend, answer_stream=broken)) as client:
        for _ in range(2):
            response = client.post('/api/chat', json={'question': 'Câu hỏi'})
            events = parse(response)
            assert events[1]['type'] == 'evidence'
            assert events[-1]['type'] == 'error'
            assert 'must-not-leak' not in response.text


def test_overlapping_requests_are_rejected():
    def answer(question, evidence):
        # TestClient dispatches this nested request while the first holds lock.
        response = client.post('/api/chat', json={'question': 'Câu khác'})
        assert response.status_code == 429
        yield 'OK'
    with TestClient(create_app(backend_factory=Backend, answer_stream=answer)) as client:
        assert parse(client.post('/api/chat', json={'question': 'Câu hỏi'}))[-1]['type'] == 'done'
