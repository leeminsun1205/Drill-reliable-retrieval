"""Check bounded conversation context and question resolution without API calls."""
import sys
from types import SimpleNamespace

from drill_ir.demo.memory import HistoryTurn, history_messages, resolve_question


def test_history_pairs_are_bounded_and_ordered():
    history = [HistoryTurn(question=f'Q{i}', answer=f'A{i}') for i in range(7)]
    messages = history_messages(history)
    assert len(messages) == 10
    assert messages[0] == {'role': 'user', 'content': 'Q2'}
    assert messages[-1] == {'role': 'assistant', 'content': 'A6'}


def test_question_without_history_does_not_need_api(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert resolve_question('Câu hỏi độc lập', []) == 'Câu hỏi độc lập'


def test_followup_resolution_uses_history_and_configured_model(monkeypatch):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Ô tô vượt đèn đỏ?'))])
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=Client))
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key')
    monkeypatch.setenv('OPENAI_MODEL', 'configured-model')
    history = [HistoryTurn(question='Xe máy vượt đèn đỏ?', answer='Trả lời cũ')]
    assert resolve_question('Còn ô tô?', history) == 'Ô tô vượt đèn đỏ?'
    assert calls[0]['model'] == 'configured-model'
    assert calls[0]['messages'][1:3] == history_messages(history)
    assert calls[0]['messages'][-1] == {'role': 'user', 'content': 'Còn ô tô?'}


def test_answer_receives_recent_context_and_current_evidence(monkeypatch):
    calls = []
    class Stream:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='Trả lời mới'))])
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            calls.append(kwargs)
            return Stream()
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=Client))
    from drill_ir.demo import answering
    from drill_ir.demo.backend import Evidence
    monkeypatch.setattr(answering, 'OpenAI', Client)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key')
    monkeypatch.setenv('OPENAI_MODEL', 'configured-model')
    history = [HistoryTurn(question='Xe máy?', answer='Câu trả lời cũ [aid=old]')]
    result = ''.join(answering.stream_answer('Còn ô tô?', [Evidence(aid='new', text='Bằng chứng mới')], history))
    assert result == 'Trả lời mới'
    messages = calls[0]['messages']
    assert messages[1:3] == history_messages(history)
    assert 'aid=new' in messages[-1]['content']
    assert 'aid=old' not in messages[-1]['content']
    assert 'Lịch sử hội thoại chỉ giúp hiểu ngữ cảnh, không phải căn cứ pháp lý.' in messages[0]['content']
