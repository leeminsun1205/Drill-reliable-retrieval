"""Kaggle-friendly Streamlit shell for evidence-based Vietnamese legal QA."""

from __future__ import annotations

import streamlit as st

from answering import stream_answer
from backend import Evidence, load_backend, normalize_evidence


st.set_page_config(
    page_title="Tra cứu pháp luật Việt Nam",
    page_icon="⚖️",
    layout="centered",
)
st.title("Tra cứu pháp luật Việt Nam")
st.caption("Đặt câu hỏi để tìm các Điều luật liên quan và câu trả lời có dẫn nguồn.")


@st.cache_resource(show_spinner="Đang khởi tạo bộ truy xuất…")
def get_backend():
    return load_backend()


def show_sources(evidence: list[Evidence]) -> None:
    if not evidence:
        return
    with st.expander(
        f"Văn bản nguồn được truy xuất — trích nguyên văn ({len(evidence)} Điều)",
        expanded=False,
    ):
        for index, article in enumerate(evidence, start=1):
            header = f"{index}. aid {article.aid}"
            if article.law_id:
                header += f" · {article.law_id}"
            if article.title:
                header += f" · {article.title}"
            st.markdown(f"**{header}**")
            st.write(article.text)
            if article.score is not None:
                st.caption(f"Điểm xếp hạng: {article.score:.4f}")


if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with st.spinner("Đang nạp pipeline truy xuất lần đầu; có thể mất vài phút nếu chưa có cache…"):
        retrieval_backend = get_backend()
    backend_error = None
except Exception as error:  # show a clear setup message inside the demo
    retrieval_backend = None
    backend_error = str(error)

if backend_error:
    st.info(
        "Giao diện vẫn nhận câu hỏi, nhưng chưa thể trả lời cho đến khi pipeline "
        "truy xuất khởi tạo thành công. Kiểm tra cấu hình bên dưới."
    )
    with st.expander("Thông tin cấu hình"):
        st.code(backend_error)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("evidence"):
            show_sources([normalize_evidence(item) for item in message["evidence"]])

question = st.chat_input(
    "Ví dụ: Trường hợp nào được tạm hoãn gọi nhập ngũ?",
    key="legal_qa_chat_input",
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if retrieval_backend is None:
            answer = (
                "Chưa thể truy xuất vì pipeline chưa khởi tạo được. "
                "Xem phần Thông tin cấu hình phía trên."
            )
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "error": True}
            )
        else:
            evidence: list[Evidence] = []
            progress = st.empty()
            try:
                progress.caption("🔎 Đang truy xuất thông tin…")
                evidence = [
                    normalize_evidence(item)
                    for item in retrieval_backend.search(question)
                ]
                progress.caption("✍️ Đang soạn câu trả lời…")

                def answer_pieces():
                    first_piece = True
                    for piece in stream_answer(question, evidence):
                        if first_piece:
                            progress.empty()
                            first_piece = False
                        yield piece

                answer = st.write_stream(answer_pieces())
                progress.empty()
                show_sources(evidence)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": str(answer),
                        "evidence": [
                            {
                                "aid": item.aid,
                                "text": item.text,
                                "law_id": item.law_id,
                                "title": item.title,
                                "score": item.score,
                            }
                            for item in evidence
                        ],
                    }
                )
            except Exception as error:
                progress.empty()
                answer = f"Chưa xử lý được câu hỏi: {error}"
                st.error(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "error": True}
                )
