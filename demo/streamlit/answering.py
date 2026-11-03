"""Evidence-grounded answer generation through the OpenAI Responses API."""

from __future__ import annotations

import os
from typing import Sequence

from openai import OpenAI

# The launcher runs Streamlit with this directory as its working directory.
from backend import Evidence


INSTRUCTIONS = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.
Chỉ trả lời dựa trên các đoạn Điều luật được cung cấp trong yêu cầu.
Không suy diễn thêm quy định, thời hạn, mức phạt hoặc ngoại lệ không có trong bằng chứng.
Nếu bằng chứng chưa đủ để kết luận, nói rõ thông tin còn thiếu.
Trả lời ngắn gọn bằng tiếng Việt và dẫn aid tương ứng trong câu trả lời, ví dụ [aid=123].
Đây là phần hỗ trợ tra cứu, không khẳng định thay cho tư vấn pháp lý chuyên môn."""


def generate_answer(question: str, evidence: Sequence[Evidence]) -> str:
    """Generate a short answer grounded only in the retrieved articles."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key:
        raise RuntimeError("Missing Kaggle Secret OPENAI_API_KEY")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL in the Kaggle launcher")
    if not evidence:
        return "Chưa tìm thấy Điều luật phù hợp trong các ứng viên truy xuất."

    evidence_text = []
    for article in evidence:
        heading = f"aid={article.aid}"
        if article.law_id:
            heading += f" | law_id={article.law_id}"
        if article.title:
            heading += f" | {article.title}"
        # Keep the API request bounded even if one article is unusually long.
        text = article.text[:5000]
        evidence_text.append(f"[{heading}]\n{text}")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=(
            f"Câu hỏi:\n{question.strip()}\n\n"
            "Bằng chứng pháp lý đã truy xuất:\n\n"
            + "\n\n---\n\n".join(evidence_text)
        ),
    )
    return response.output_text.strip()
