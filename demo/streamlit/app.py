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
    with st.expander(f"Điều luật được dùng làm căn cứ ({len(evidence)})", expanded=True):
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
        "Giao diện đã khởi động. Notebook Kaggle cần nạp pipeline truy xuất và "
        "đặt `DRILL_RETRIEVER_FACTORY` trước khi có thể nhận câu hỏi."
    )
    with st.expander("Thông tin cấu hình"):
        st.code(backend_error)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("evidence", []))

question = st.chat_input(
    "Ví dụ: Trường hợp nào được tạm hoãn gọi nhập ngũ?",
    disabled=retrieval_backend is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang truy xuất và tổng hợp căn cứ…"):
                evidence = [
                    normalize_evidence(item)
                    for item in retrieval_backend.search(question)
                ]
            answer = st.write_stream(stream_answer(question, evidence))
            show_sources(evidence)
            st.session_state.messages.append(
                {"role": "assistant", "content": str(answer), "evidence": evidence}
            )
        except Exception as error:
            st.error(f"Chưa xử lý được câu hỏi: {error}")
