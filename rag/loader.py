"""Document loading, cleaning and chunking for the RAG pipeline."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pythainlp.tokenize import sent_tokenize
from pythainlp.util import normalize as thai_normalize

DEFAULT_MAX_CHARS = 500
DEFAULT_OVERLAP = 100
SUPPORTED_SUFFIXES = (".md", ".txt")

_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_INLINE_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class Document:
    file: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    file: str
    title: str
    heading: str
    source: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.file} > {self.heading}"


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split a simple `key: value` YAML header from the body."""
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():].lstrip("\n")


def clean_text(text: str) -> str:
    """Normalize unicode/Thai vowels, strip zero-width chars and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = thai_normalize(text)
    lines = [_INLINE_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, text) pairs on `#` / `##` headings."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in body.splitlines():
        if re.match(r"^#{1,2} ", line):
            if heading or any(buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = line.lstrip("#").strip()
            buffer = []
        else:
            buffer.append(line)
    if heading or any(buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _split_long_unit(unit: str, max_chars: int) -> list[str]:
    """Break a line longer than max_chars into sentence/phrase pieces."""
    pieces: list[str] = []
    current = ""
    for sentence in sent_tokenize(unit, engine="whitespace"):
        while len(sentence) > max_chars:  # no break point at all: hard cut
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Greedy-pack lines into chunks <= max_chars, repeating trailing lines as overlap."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        units.extend(_split_long_unit(line, max_chars) if len(line) > max_chars else [line])

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n".join([*current, unit])) > max_chars:
            chunks.append("\n".join(current))
            current = _tail_for_overlap(current, overlap, max_chars - len(unit) - 1)
        current = [*current, unit]
    if current:
        chunks.append("\n".join(current))
    return chunks


def _tail_for_overlap(units: list[str], overlap: int, room: int) -> list[str]:
    """Return the trailing units whose combined length fits both overlap and room."""
    tail: list[str] = []
    for unit in reversed(units):
        size = len("\n".join([unit, *tail]))
        if size > overlap or size > room:
            break
        tail = [unit, *tail]
    return tail


def load_documents(data_dir: Path | str) -> list[Document]:
    """Load and clean every supported file in data_dir (sorted by name)."""
    paths = sorted(p for p in Path(data_dir).iterdir() if p.suffix in SUPPORTED_SUFFIXES)
    documents = []
    for path in paths:
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        documents.append(
            Document(
                file=path.name,
                title=meta.get("title", path.stem),
                source=meta.get("source", ""),
                text=clean_text(body),
            )
        )
    return documents


def build_chunks(
    documents: list[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Section-aware chunking: split by heading first, then by size."""
    chunks: list[Chunk] = []
    for doc in documents:
        for heading, section_text in split_sections(doc.text):
            if not section_text:
                continue
            for piece in chunk_text(section_text, max_chars, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        file=doc.file,
                        title=doc.title,
                        heading=heading or doc.title,
                        source=doc.source,
                        text=piece,
                    )
                )
    return chunks
