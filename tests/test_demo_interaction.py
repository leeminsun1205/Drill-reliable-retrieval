"""Validate model JSON contracts, without claiming semantic accuracy."""
import pytest
from drill_ir.demo import interaction
from drill_ir.demo.backend import Evidence


def test_clarification_uses_only_required_details(monkeypatch):
    monkeypatch.setattr(interaction, '_request_json', lambda *args: {
        'question': 'Bạn hỏi về phương tiện nào?', 'options': ['Xe máy vượt đèn đỏ bị phạt thế nào?'],
    })
    result = interaction.clarify_question('Vượt đèn đỏ bị phạt thế nào?', [])
    assert result.question == 'Bạn hỏi về phương tiện nào?'
    assert result.options == ['Xe máy vượt đèn đỏ bị phạt thế nào?']
    monkeypatch.setattr(interaction, '_request_json', lambda *args: {'question': None, 'options': []})
    assert interaction.clarify_question('Xe máy vượt đèn đỏ?', []) is None


@pytest.mark.parametrize('questions', [[], ['A?', 'A?', 'B?'], ['A?', 'B?'], ['A?', 'B?', 42], ['A?', 'B?', 'x' * 301]])
def test_invalid_suggestion_lists_are_rejected(monkeypatch, questions):
    monkeypatch.setattr(interaction, '_request_json', lambda *args: {'questions': questions})
    with pytest.raises(ValueError):
        interaction.related_questions('Q', 'A', [Evidence(aid='123', text='Điều luật')])


def test_exactly_three_questions_and_independent_disable_flags(monkeypatch):
    questions = ['Tôi cần chuẩn bị giấy tờ gì?', 'Thủ tục được thực hiện ở đâu?', 'Tôi có quyền khiếu nại không?']
    monkeypatch.setattr(interaction, '_request_json', lambda *args: {'questions': questions})
    assert interaction.related_questions('Q', 'A', [Evidence(aid='123', text='Điều luật')]) == questions
    monkeypatch.setenv('DRILL_DEMO_SUGGESTIONS', '0')
    assert interaction.related_questions('Q', 'A', [Evidence(aid='123', text='Điều luật')]) == []
    monkeypatch.setenv('DRILL_DEMO_CLARIFICATION', '0')
    assert interaction.clarify_question('Q', []) is None


@pytest.mark.parametrize('text', [
    '{"question": null, "options": []}',
    '\ufeff {"question": null, "options": []}',
    '```json\n{"question": null, "options": []}\n```',
    'Đây là kết quả:\n```json\n{"question": null, "options": []}\n```',
    'Kết quả kiểm tra: {"question": null, "options": []}\nĐã hoàn tất.',
])
def test_parse_valid_json_wrappers(text):
    assert interaction._parse_response(text) == {'question': None, 'options': []}


@pytest.mark.parametrize('text', [None, '', '   ', 'Tôi cần thêm thông tin.',
    '{"question":', '[]', '[{"question": null}]', '{"question": null} {"question": "Khác"}',
    '```json\n{}\n```\n```json\n{}\n```',
    '```json\n{}\n```\n{"question": null}'])
def test_invalid_json_is_rejected_without_echoing_provider_text(text):
    with pytest.raises(interaction.InteractionResponseError):
        interaction._parse_response(text)


def test_invalid_clarification_format_requests_details(monkeypatch):
    def broken(*args):
        raise interaction.InteractionResponseError('SECRET provider content')
    monkeypatch.setattr(interaction, '_request_json', broken)
    result = interaction.clarify_question('Câu hỏi', [])
    assert result is not None and result.options == []
    assert 'nêu rõ' in result.question
    assert 'SECRET' not in result.question


@pytest.mark.parametrize('payload', [{}, {'question': ''}, {'question': 42},
    {'question': 'Câu hỏi?', 'options': ['lặp', 'lặp']}])
def test_invalid_clarification_schema_requests_details(monkeypatch, payload):
    monkeypatch.setattr(interaction, '_request_json', lambda *args: payload)
    assert interaction.clarify_question('Câu hỏi', []) is not None


def test_provider_failure_is_not_misreported_as_missing_user_details(monkeypatch):
    def broken(*args):
        raise ConnectionError('API disconnected')
    monkeypatch.setattr(interaction, '_request_json', broken)
    with pytest.raises(ConnectionError):
        interaction.clarify_question('Câu hỏi', [])
