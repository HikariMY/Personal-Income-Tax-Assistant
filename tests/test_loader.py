from pathlib import Path

from rag.loader import (
    Chunk,
    build_chunks,
    chunk_text,
    clean_text,
    load_documents,
    parse_frontmatter,
    split_sections,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_parse_frontmatter_extracts_metadata_and_body():
    raw = "---\ntitle: ทดสอบ\nsource: rd.go.th\n---\n\n# หัวเรื่อง\nเนื้อหา"

    meta, body = parse_frontmatter(raw)

    assert meta == {"title": "ทดสอบ", "source": "rd.go.th"}
    assert body.startswith("# หัวเรื่อง")


def test_parse_frontmatter_without_header_returns_empty_meta():
    meta, body = parse_frontmatter("ไม่มี frontmatter")

    assert meta == {}
    assert body == "ไม่มี frontmatter"


def test_clean_text_removes_zero_width_and_extra_spaces():
    dirty = "ภาษี​เงินได้   บุคคล\t\tธรรมดา\n\n\n\nบรรทัดใหม่  "

    cleaned = clean_text(dirty)

    assert "​" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.endswith("บรรทัดใหม่")


def test_split_sections_uses_level_two_headings():
    body = "# เรื่องหลัก\nบทนำ\n## ส่วนที่ 1\nข้อความหนึ่ง\n## ส่วนที่ 2\nข้อความสอง"

    sections = split_sections(body)

    headings = [heading for heading, _ in sections]
    assert headings == ["เรื่องหลัก", "ส่วนที่ 1", "ส่วนที่ 2"]
    assert sections[1][1] == "ข้อความหนึ่ง"


def test_chunk_text_respects_max_size_and_overlap():
    lines = [f"บรรทัดที่ {i} มีข้อความเกี่ยวกับภาษีเงินได้" for i in range(40)]
    text = "\n".join(lines)

    chunks = chunk_text(text, max_chars=200, overlap=60)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # overlap: the last line of a chunk should reappear at the start of the next
    last_line_of_first = chunks[0].split("\n")[-1]
    assert last_line_of_first in chunks[1]


def test_chunk_text_splits_single_long_line():
    long_line = " ".join(["ข้อความยาวมาก"] * 100)

    chunks = chunk_text(long_line, max_chars=150, overlap=30)

    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)


def test_chunk_text_short_text_returns_single_chunk():
    assert chunk_text("สั้น ๆ", max_chars=500, overlap=100) == ["สั้น ๆ"]


def test_build_chunks_attaches_metadata(tmp_path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "---\ntitle: ตัวอย่าง\nsource: ทดสอบ\n---\n# ตัวอย่าง\n## หัวข้อย่อย\nเนื้อหาทดสอบ",
        encoding="utf-8",
    )

    chunks = build_chunks(load_documents(tmp_path))

    assert len(chunks) >= 1
    last = chunks[-1]
    assert isinstance(last, Chunk)
    assert last.file == "sample.md"
    assert last.title == "ตัวอย่าง"
    assert last.heading == "หัวข้อย่อย"
    assert "เนื้อหาทดสอบ" in last.text


def test_real_corpus_meets_assignment_requirements():
    docs = load_documents(DATA_DIR)

    assert len(docs) >= 10
    assert sum(len(d.text) for d in docs) >= 15_000
    assert len(build_chunks(docs)) > len(docs)
