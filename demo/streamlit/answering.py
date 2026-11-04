"""Evidence-grounded answer generation through an OpenAI-compatible API."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

from openai import OpenAI

# The launcher runs Streamlit with this directory as its working directory.
from backend import Evidence


INSTRUCTIONS = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.
Chỉ trả lời dựa trên các đoạn Điều luật được cung cấp trong yêu cầu.
Không suy diễn thêm quy định, thời hạn, mức phạt hoặc ngoại lệ không có trong bằng chứng.
Nếu bằng chứng chưa đủ để kết luận, nói rõ thông tin còn thiếu.
Với mỗi căn cứ pháp lý được dùng, trích nguyên văn một câu hoặc đoạn ngắn từ phần bằng chứng,
đặt trong dấu ngoặc kép và ghi mã nguồn ngay sau đó, ví dụ: “...” [aid=123].
Chỉ trích nguyên văn nội dung xuất hiện trong bằng chứng; không tự sửa chữ hoặc ghép câu.
Sau trích dẫn, giải thích ngắn gọn câu đó liên quan thế nào đến câu hỏi.
Nếu không tìm thấy đoạn nguyên văn đủ căn cứ, nói rõ bằng chứng chưa đủ và không tự tạo trích dẫn.
Đây là phần hỗ trợ tra cứu, không khẳng định thay cho tư vấn pháp lý chuyên môn."""


def stream_answer(question: str, evidence: Sequence[Evidence]) -> Iterator[str]:
    """Yield answer text as the OpenAI-compatible API streams it."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not api_key:
        raise RuntimeError("Missing Kaggle Secret OPENAI_API_KEY")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL in the Kaggle launcher")
    if not evidence:
        yield "Chưa tìm thấy Điều luật phù hợp trong các ứng viên truy xuất."
        return

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

    client_options = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Câu hỏi:\n{question.strip()}\n\n"
                    "Bằng chứng pháp lý đã truy xuất:\n\n"
                    + "\n\n---\n\n".join(evidence_text)
                ),
            },
        ],
        stream=True,
    )

    received_content = False
    with stream:
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if isinstance(piece, str) and piece:
                received_content = True
                yield piece
    if not received_content:
        raise RuntimeError("The chat-completions API returned an empty answer")
