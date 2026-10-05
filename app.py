"""TaxBuddy — RAG chatbot for Thai personal income tax (Streamlit app)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from groq import Groq

from rag.llm import AVAILABLE_MODELS, LLMError, answer, rewrite_query
from rag.loader import build_chunks, load_documents
from rag.prompts import NOT_FOUND
from rag.retriever import SearchResult, VectorStore, load_embedder
from ui.components import (
    ERROR_MESSAGE,
    compact_header,
    hero,
    render_assistant,
    sidebar_brand,
    sidebar_help,
    topic_cards,
)
from ui.styles import inject_css

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TOP_K = 4
# Cosine score below which the question is treated as off-topic (calibrated with eval.py).
SCORE_THRESHOLD = 0.83

st.set_page_config(
    page_title="TaxBuddy · ผู้ช่วยภาษีเงินได้",
    page_icon="🧾",
    layout="centered",
    initial_sidebar_state="auto",
)


@st.cache_resource(show_spinner="กำลังเตรียมข้อมูลภาษี ครั้งแรกอาจใช้เวลาประมาณ 1 นาที ...")
def get_store() -> VectorStore:
    chunks = build_chunks(load_documents(DATA_DIR))
    return VectorStore(chunks, load_embedder())


@st.cache_resource
def get_client() -> Groq | None:
    api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else None
    return Groq(api_key=api_key) if api_key else None


def to_source_dicts(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "title": r.chunk.title,
            "heading": r.chunk.heading,
            "file": r.chunk.file,
            "text": r.chunk.text,
            "score": r.score,
        }
        for r in results
    ]


def generate_reply(client: Groq, store: VectorStore, question: str, top_k: int, model: str) -> dict:
    history = st.session_state.messages[:-1]  # exclude the question just appended
    search_query = rewrite_query(client, question, history)
    results = store.search(search_query, k=top_k)
    reply = {
        "role": "assistant",
        "sources": to_source_dicts(results),
        "query": search_query if search_query != question else None,
    }
    if not results or results[0].score < SCORE_THRESHOLD:
        return {**reply, "content": NOT_FOUND}
    return {**reply, "content": answer(client, question, results, history, model=model)}


def sidebar() -> tuple[int, str]:
    with st.sidebar:
        sidebar_brand()
        if st.button("เริ่มแชตใหม่", icon=":material/add_comment:", type="primary", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        sidebar_help()
        st.divider()
        with st.expander("ตั้งค่าขั้นสูง (สำหรับนักพัฒนา)", icon=":material/tune:"):
            top_k = st.slider("จำนวนเอกสารที่ใช้ค้นหา (top-k)", 1, 8, DEFAULT_TOP_K)
            model = st.selectbox("LLM model", AVAILABLE_MODELS)
            st.caption("RAG: PyThaiNLP chunking · multilingual-e5-small · FAISS · Groq")
        st.caption("ข้อมูลเพื่อการศึกษา ก่อนยื่นจริงควรตรวจสอบกับกรมสรรพากร (rd.go.th / 1161)")
    return top_k, model


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message)


def ask(client: Groq, store: VectorStore, question: str, top_k: int, model: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        with st.spinner("กำลังหาคำตอบจากเอกสาร ..."):
            try:
                reply = generate_reply(client, store, question, top_k, model)
            except LLMError as exc:
                reply = {"role": "assistant", "content": ERROR_MESSAGE, "error": str(exc), "sources": []}
        render_assistant(reply)
    st.session_state.messages.append(reply)


def main() -> None:
    inject_css()
    st.session_state.setdefault("messages", [])
    top_k, model = sidebar()

    client = get_client()
    if client is None:
        st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Streamlit Secrets", icon=":material/key_off:")
        st.stop()
    store = get_store()

    welcome = st.empty()
    picked = None
    if st.session_state.messages:
        compact_header()
    else:
        with welcome.container():
            hero()
            picked = topic_cards()

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("พิมพ์คำถาม เช่น ลดหย่อนประกันชีวิตได้เท่าไร") or picked
    if question:
        welcome.empty()
        ask(client, store, question, top_k, model)


main()
