import numpy as np
import pytest

from rag.loader import Chunk
from rag.prompts import NOT_FOUND, SYSTEM_PROMPT, build_messages, format_context
from rag.retriever import SearchResult, VectorStore

VOCAB = ["ภาษี", "ลดหย่อน", "บริจาค", "ประกัน"]


def fake_embed(texts):
    """Bag-of-keywords embedding: deterministic and model-free."""
    vectors = np.array(
        [[t.count(word) for word in VOCAB] for t in texts], dtype="float32"
    ) + 1e-3
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_chunk(i, text, heading="หัวข้อ"):
    return Chunk(chunk_id=i, file=f"f{i}.md", title="t", heading=heading, source="s", text=text)


@pytest.fixture
def store():
    chunks = [
        make_chunk(0, "บริจาค บริจาค เงินบริจาค"),
        make_chunk(1, "ประกัน ประกันชีวิต ประกันสุขภาพ"),
        make_chunk(2, "ลดหย่อน ลดหย่อนส่วนตัว"),
    ]
    return VectorStore(chunks, fake_embed)


def test_search_ranks_most_similar_chunk_first(store):
    results = store.search("เงินบริจาค", k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == 0
    assert results[0].score >= results[1].score


def test_search_caps_k_and_ignores_blank_query(store):
    assert len(store.search("ประกัน", k=50)) == len(store)
    assert store.search("   ") == []


def test_vector_store_rejects_empty_corpus():
    with pytest.raises(ValueError):
        VectorStore([], fake_embed)


def test_format_context_numbers_sources():
    results = [SearchResult(make_chunk(0, "ข้อความ A", "หัวข้อ A"), 0.9)]

    context = format_context(results)

    assert context.startswith("[1] แหล่งที่มา: f0.md > หัวข้อ A")
    assert "ข้อความ A" in context


def test_build_messages_has_rules_history_and_context():
    results = [SearchResult(make_chunk(0, "ข้อความ A"), 0.9)]
    history = [
        {"role": "user", "content": "ถามก่อนหน้า"},
        {"role": "assistant", "content": "ตอบก่อนหน้า", "sources": []},
    ]

    messages = build_messages("คำถามใหม่", results, history)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:3] == [
        {"role": "user", "content": "ถามก่อนหน้า"},
        {"role": "assistant", "content": "ตอบก่อนหน้า"},
    ]
    assert "ข้อความ A" in messages[-1]["content"]
    assert "คำถามใหม่" in messages[-1]["content"]


def test_system_prompt_enforces_grounding_citation_and_not_found():
    assert NOT_FOUND in SYSTEM_PROMPT
    assert "CONTEXT" in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT
