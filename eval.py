"""Run test_questions.csv through the RAG pipeline and report retrieval / not-found accuracy.

Usage:  GROQ_API_KEY=... python eval.py      (or put the key in .streamlit/secrets.toml)
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pandas as pd
from groq import Groq

from rag.llm import answer
from rag.loader import build_chunks, load_documents
from rag.prompts import NOT_FOUND
from rag.retriever import VectorStore, load_embedder

ROOT = Path(__file__).parent
TOP_K = 4
SCORE_THRESHOLD = 0.83  # keep in sync with app.py


def read_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if not key and secrets.exists():
        key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY not found in environment or .streamlit/secrets.toml")
    return key


def main() -> None:
    questions = pd.read_csv(ROOT / "test_questions.csv").fillna("")
    store = VectorStore(build_chunks(load_documents(ROOT / "data")), load_embedder())
    client = Groq(api_key=read_api_key())

    rows = []
    for q in questions.itertuples():
        results = store.search(q.question, k=TOP_K)
        top_score = results[0].score if results else 0.0
        if top_score < SCORE_THRESHOLD:
            reply = NOT_FOUND
        else:
            reply = answer(client, q.question, results)
        retrieved = [r.chunk.file for r in results]
        said_not_found = NOT_FOUND in reply
        rows.append(
            {
                "id": q.id,
                "question": q.question,
                "answerable": q.answerable,
                "top_score": round(top_score, 3),
                "retrieval_hit": (q.source_file in retrieved) if q.answerable == "yes" else "",
                "not_found_correct": said_not_found == (q.answerable == "no"),
                "expected_answer": q.expected_answer,
                "model_answer": reply,
            }
        )
        print(f"[{q.id}] score={top_score:.3f} not_found={said_not_found} :: {reply[:80]!r}")

    report = pd.DataFrame(rows)
    report.to_csv(ROOT / "eval_results.csv", index=False, encoding="utf-8-sig")
    answerable = report[report.answerable == "yes"]
    print(f"\nRetrieval hit@{TOP_K}: {answerable.retrieval_hit.mean():.0%} ({len(answerable)} answerable)")
    print(f"Answer/not-found behaviour correct: {report.not_found_correct.mean():.0%}")


if __name__ == "__main__":
    main()
