"""TaxBuddy — RAG chatbot for Thai personal income tax (Streamlit app)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from groq import Groq

from rag.llm import DEFAULT_MODEL, FALLBACK_MODEL, LLMError, answer, rewrite_query
from rag.loader import build_chunks, load_documents
from rag.prompts import NOT_FOUND
from rag.retriever import SearchResult, VectorStore, load_embedder

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TOP_K = 4
# Cosine score below which the question is treated as off-topic (calibrated with eval.py).
SCORE_THRESHOLD = 0.83

EXAMPLE_QUESTIONS = [
    "ค่าลดหย่อนส่วนตัวได้เท่าไร",
    "เงินเดือน 30,000 บาท ต้องเสียภาษีเท่าไร",
    "ซื้อ RMF ลดหย่อนได้สูงสุดเท่าไร",
    "ยื่นภาษีออนไลน์ได้ถึงวันไหน",
    "What is the personal allowance in Thailand?",
]

st.set_page_config(page_title="TaxBuddy — ผู้ช่วยภาษีเงินได้", page_icon="🧾", layout="centered")


@st.cache_resource(show_spinner="กำลังโหลดเอกสารและสร้าง Vector Index ...")
def get_store() -> VectorStore:
    chunks = build_chunks(load_documents(DATA_DIR))
    return VectorStore(chunks, load_embedder())


@st.cache_resource
def get_client() -> Groq | None:
    api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else None
    return Groq(api_key=api_key) if api_key else None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 เอกสารอ้างอิง ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {src['file']} › {src['heading']}**  \n`score {src['score']:.3f}`")
            st.caption(src["text"])


def to_source_dicts(results: list[SearchResult]) -> list[dict]:
    return [
        {"file": r.chunk.file, "heading": r.chunk.heading, "text": r.chunk.text, "score": r.score}
        for r in results
    ]


def generate_reply(client: Groq, store: VectorStore, question: str, top_k: int, model: str) -> dict:
    history = st.session_state.messages[:-1]  # exclude the question just appended
    search_query = rewrite_query(client, question, history)
    results = store.search(search_query, k=top_k)
    if not results or results[0].score < SCORE_THRESHOLD:
        return {"content": NOT_FOUND, "sources": to_source_dicts(results), "query": search_query}
    reply = answer(client, question, results, history, model=model)
    return {"content": reply, "sources": to_source_dicts(results), "query": search_query}


def sidebar() -> tuple[int, str]:
    with st.sidebar:
        st.header("🧾 TaxBuddy")
        st.write(
            "แชตบอตตอบคำถาม **ภาษีเงินได้บุคคลธรรมดา (ปีภาษี 2567)** "
            "จากคลังเอกสาร 15 ไฟล์ ด้วยเทคนิค RAG"
        )
        st.caption("Embedding: multilingual-e5-small · Vector DB: FAISS · LLM: Groq")
        top_k = st.slider("จำนวนเอกสารอ้างอิง (top-k)", 1, 8, DEFAULT_TOP_K)
        model = st.selectbox("LLM model", [DEFAULT_MODEL, FALLBACK_MODEL])
        st.subheader("คำถามตัวอย่าง")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.pending = q
        if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("⚠️ ข้อมูลเพื่อการศึกษา ควรตรวจสอบกับกรมสรรพากร (rd.go.th / 1161) ก่อนยื่นภาษีจริง")
    return top_k, model


def main() -> None:
    st.title("🧾 TaxBuddy")
    st.caption("ผู้ช่วยตอบคำถามภาษีเงินได้บุคคลธรรมดา · ถามได้ทั้งภาษาไทยและภาษาอังกฤษ")

    st.session_state.setdefault("messages", [])
    top_k, model = sidebar()

    client = get_client()
    if client is None:
        st.error("ไม่พบ GROQ_API_KEY กรุณาตั้งค่าใน Streamlit Secrets (.streamlit/secrets.toml)")
        st.stop()
    store = get_store()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))

    question = st.chat_input("พิมพ์คำถามเกี่ยวกับภาษีเงินได้ ...") or st.session_state.pop("pending", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นเอกสารและสร้างคำตอบ ..."):
            try:
                reply = generate_reply(client, store, question, top_k, model)
            except LLMError as exc:
                reply = {"content": f"⚠️ {exc}", "sources": []}
        st.markdown(reply["content"])
        if reply.get("query") and reply["query"] != question:
            st.caption(f"🔎 คำค้นที่ใช้: {reply['query']}")
        render_sources(reply["sources"])

    st.session_state.messages.append({"role": "assistant", **reply})


main()
