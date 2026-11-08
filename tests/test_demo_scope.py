"""Local classifier contract checks, not measured semantic accuracy."""
import sys
from types import SimpleNamespace

import pytest

from drill_ir.demo.scope import Intent, classify_question
from drill_ir.demo.memory import HistoryTurn, history_messages


@pytest.mark.parametrize('question', ['Bạn là ai?', 'Bạn làm được gì?', 'ban la ai', 'Xin chào!', 'Who are you?'])
def test_simple_identity_and_greeting_need_no_api(question, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert classify_question(question) == Intent.INTRO


@pytest.mark.parametrize('with_history', [False, True])
@pytest.mark.parametrize('question,label', [
    ('Cách nấu phở?', 'OUT_OF_SCOPE'),
    ('Thời tiết hôm nay thế nào?', 'OUT_OF_SCOPE'),
    ('Tôi bị công ty nợ lương thì phải làm sao?', 'LEGAL'),
    ('Bạn là ai và tôi được hoãn nhập ngũ khi nào?', 'LEGAL'),
    ('Giúp tôi với', 'CLARIFY'),
])
def test_semantic_route_uses_configured_model_and_only_question(question, label, monkeypatch, with_history):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['api_key'] == 'test-key'
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=label))])
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=Client))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'configured-model')
    history = [HistoryTurn(question='Xe máy vượt đèn đỏ?', answer='Trả lời cũ')] if with_history else []
    assert classify_question(question, history) == Intent(label)
    assert calls[0]['model'] == 'configured-model'
    assert calls[0]['messages'][1:-1] == history_messages(history)
    assert calls[0]['messages'][-1] == {'role': 'user', 'content': question}
