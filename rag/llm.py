"""Groq LLM calls: query rewriting and grounded answer generation."""

from __future__ import annotations

import re
from typing import Sequence

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from rag.prompts import build_messages, build_rewrite_messages
from rag.retriever import SearchResult

DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
ANSWER_TEMPERATURE = 0.1
# gpt-oss models spend tokens on hidden reasoning first, so leave headroom.
MAX_ANSWER_TOKENS = 4096
MAX_REWRITE_TOKENS = 1024


class LLMError(RuntimeError):
    """User-facing error raised when the LLM call fails."""


def _chat(client: Groq, messages: list[dict], model: str, max_tokens: int) -> str:
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=ANSWER_TEMPERATURE,
            max_tokens=max_tokens,
            **extra,
        )
    except RateLimitError as exc:
        raise LLMError("เรียกใช้ Groq API เกินโควตา กรุณารอสักครู่แล้วลองใหม่") from exc
    except APIConnectionError as exc:
        raise LLMError("เชื่อมต่อ Groq API ไม่ได้ กรุณาตรวจสอบอินเทอร์เน็ต") from exc
    except APIStatusError as exc:
        if exc.status_code == 401:
            raise LLMError("GROQ_API_KEY ไม่ถูกต้อง กรุณาตรวจสอบใน Streamlit Secrets") from exc
        raise LLMError(f"Groq API ผิดพลาด (HTTP {exc.status_code})") from exc
    return (response.choices[0].message.content or "").strip()


def rewrite_query(client: Groq, question: str, history: Sequence[dict], model: str = FALLBACK_MODEL) -> str:
    """Turn a follow-up question into a standalone one; falls back to the original."""
    if not history:
        return question
    try:
        rewritten = _chat(client, build_rewrite_messages(question, history), model, MAX_REWRITE_TOKENS)
    except LLMError:
        return question
    return rewritten or question


def answer(
    client: Groq,
    question: str,
    results: Sequence[SearchResult],
    history: Sequence[dict] = (),
    model: str = DEFAULT_MODEL,
) -> str:
    reply = _chat(client, build_messages(question, results, history), model, MAX_ANSWER_TOKENS)
    return normalize_citations(reply)


def normalize_citations(text: str) -> str:
    """gpt-oss sometimes cites with full-width 【n】; unify to [n]."""
    return re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)
