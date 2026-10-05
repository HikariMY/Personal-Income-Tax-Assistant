import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import rag.llm
import rag.retriever
from rag.prompts import NOT_FOUND
from ui.components import TOPICS, relevance_label

FAKE_ANSWER = "ค่าลดหย่อนส่วนตัว 60,000 บาท [1]"


def fake_embed(texts):
    vectors = np.ones((len(texts), 8), dtype="float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def app(monkeypatch):
    """App with model-free embeddings and a stubbed LLM, so tests run offline."""
    st.cache_resource.clear()
    monkeypatch.setattr(rag.retriever, "load_embedder", lambda: fake_embed)
    monkeypatch.setattr(rag.llm, "answer", lambda *args, **kwargs: FAKE_ANSWER)
    monkeypatch.setattr(rag.llm, "rewrite_query", lambda client, question, history: question)
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.secrets["GROQ_API_KEY"] = "test-key"
    return at.run()


def test_welcome_screen_shows_hero_and_topic_cards(app):
    assert not app.exception
    assert any('<div class="tb-hero">' in m.value for m in app.markdown)
    assert len([b for b in app.button if b.key and b.key.startswith("ask_")]) == len(TOPICS)


def test_clicking_topic_card_asks_its_question_and_shows_sources(app):
    app.button(key="ask_0").click().run()

    messages = app.chat_message
    assert [m.name for m in messages] == ["user", "assistant"]
    assert messages[0].markdown[0].value == TOPICS[0][2]
    assert messages[1].markdown[0].value == FAKE_ANSWER
    assert any("ที่มาของคำตอบ" in e.label for e in app.expander)


def test_follow_up_question_keeps_history(app):
    app.chat_input[0].set_value("คำถามแรก").run()
    app.chat_input[0].set_value("คำถามที่สอง").run()

    assert [m.name for m in app.chat_message] == ["user", "assistant", "user", "assistant"]
    assert not any('<div class="tb-hero">' in m.value for m in app.markdown)


def test_not_found_answer_shows_hint(monkeypatch, app):
    monkeypatch.setattr(rag.llm, "answer", lambda *args, **kwargs: NOT_FOUND)

    app.chat_input[0].set_value("ภาษีมูลค่าเพิ่มเท่าไร").run()

    assert any("1161" in i.value for i in app.info)
    assert any("เอกสารที่ใกล้เคียงที่สุด" in e.label for e in app.expander)


def test_developer_settings_live_in_sidebar_expander(app):
    assert any("ตั้งค่าขั้นสูง" in e.label for e in app.sidebar.expander)
    assert app.sidebar.selectbox[0].value == rag.llm.DEFAULT_MODEL


@pytest.mark.parametrize(
    ("score", "level"),
    [(0.95, "ตรงมาก"), (0.88, "ตรงมาก"), (0.86, "เกี่ยวข้อง"), (0.80, "เกี่ยวข้องบางส่วน")],
)
def test_relevance_label_uses_plain_language(score, level):
    assert relevance_label(score)[0] == level
