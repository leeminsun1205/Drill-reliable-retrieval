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
