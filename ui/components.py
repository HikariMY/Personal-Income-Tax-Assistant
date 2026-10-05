"""Reusable UI pieces. Only static strings go through unsafe_allow_html;
LLM output and document text are always rendered as plain markdown."""

from __future__ import annotations

import streamlit as st

from rag.prompts import NOT_FOUND

LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<defs><linearGradient id="tbg" x1="0" y1="0" x2="24" y2="24">'
    '<stop stop-color="#7B61FF"/><stop offset="1" stop-color="#22C55E"/></linearGradient></defs>'
    '<rect width="24" height="24" rx="7" fill="url(#tbg)"/>'
    '<path d="M7.5 7.5h9M7.5 11.5h9M7.5 15.5h5" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
    "</svg>"
)
CHECK_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M5 12.5l4.5 4.5L19 7.5" stroke="#4ADE80" stroke-width="3" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

# (material icon, card title, question asked when the card is clicked)
TOPICS = (
    ("calculate", "คำนวณภาษี", "เงินเดือน 30,000 บาท ต้องเสียภาษีเท่าไร"),
    ("savings", "ลดหย่อนด้วยกองทุน", "ซื้อ RMF ลดหย่อนได้สูงสุดเท่าไร"),
    ("event", "กำหนดยื่นภาษี", "ยื่นภาษีออนไลน์ได้ถึงวันไหน"),
    ("family_restroom", "ลดหย่อนพ่อแม่", "ลดหย่อนเลี้ยงดูบิดามารดามีเงื่อนไขอะไรบ้าง"),
    ("payments", "ขอคืนภาษี", "ขอคืนภาษีต้องทำอย่างไร และได้เงินคืนทางไหน"),
    ("translate", "Ask in English", "What is the personal allowance in Thailand?"),
)

NOT_FOUND_HINT = (
    "ลองถามเรื่องค่าลดหย่อน การคำนวณภาษี การยื่นแบบ หรือการขอคืนภาษี "
    "หรือสอบถามกรมสรรพากรโดยตรงที่โทร 1161"
)
ERROR_MESSAGE = "ขออภัย ตอนนี้ระบบตอบคำถามไม่ได้ชั่วคราว กรุณาลองใหม่อีกครั้ง"


def relevance_label(score: float) -> tuple[str, str]:
    """Map cosine similarity to a plain-language level and a Streamlit colour."""
    if score >= 0.88:
        return "ตรงมาก", "green"
    if score >= 0.85:
        return "เกี่ยวข้อง", "blue"
    return "เกี่ยวข้องบางส่วน", "gray"


def hero() -> None:
    pills = "".join(
        f"<span>{CHECK_SVG}{text}</span>"
        for text in ("ข้อมูล 15 หัวข้อ", "บอกที่มาทุกคำตอบ", "ถามไทยหรืออังกฤษก็ได้")
    )
    st.markdown(
        f"""<div class="tb-hero">
<div class="tb-brand">{LOGO_SVG} TaxBuddy</div>
<div class="tb-title">ถามเรื่องภาษีเงินได้<br><span class="tb-accent">ง่ายเหมือนคุยกับเพื่อน</span></div>
<p>ตอบจากคู่มือภาษีเงินได้บุคคลธรรมดา ปีภาษี 2567 เช่น ค่าลดหย่อน การคำนวณภาษี
และการยื่นแบบ พร้อมบอกที่มาของทุกคำตอบ</p>
<div class="tb-pills">{pills}</div>
</div>""",
        unsafe_allow_html=True,
    )


def topic_cards() -> str | None:
    """Render the suggestion cards; return the question of a clicked card."""
    st.markdown('<p class="tb-section">เริ่มจากคำถามยอดฮิต</p>', unsafe_allow_html=True)
    picked = None
    for start in range(0, len(TOPICS), 3):
        columns = st.columns(3)
        for offset, column in enumerate(columns):
            index = start + offset
            icon, title, question = TOPICS[index]
            with column, st.container(key=f"topic_{index}"):
                st.markdown(f":material/{icon}: **{title}**")
                st.caption(question)
                if st.button("ถามเลย", key=f"ask_{index}", icon=":material/arrow_forward:", type="tertiary"):
                    picked = question
    st.markdown(
        '<p class="tb-footnote">ข้อมูลเพื่อการศึกษา · ก่อนยื่นจริงควรตรวจสอบกับกรมสรรพากร www.rd.go.th</p>',
        unsafe_allow_html=True,
    )
    return picked


def compact_header() -> None:
    st.markdown(
        f'<div class="tb-bar">{LOGO_SVG}<div><b>TaxBuddy</b><br>'
        "<small>ผู้ช่วยตอบคำถามภาษีเงินได้บุคคลธรรมดา</small></div></div>",
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.markdown(f'<div class="tb-side-brand">{LOGO_SVG} TaxBuddy</div>', unsafe_allow_html=True)


def sidebar_help() -> None:
    items = (
        "คำนวณภาษีจากเงินเดือนหรือรายได้",
        "อธิบายค่าลดหย่อนแต่ละประเภท",
        "บอกกำหนดและวิธียื่นแบบ ขอคืนภาษี",
    )
    rows = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(f"**TaxBuddy ช่วยอะไรได้บ้าง**<ul class='tb-side-list'>{rows}</ul>", unsafe_allow_html=True)


def render_sources(sources: list[dict], query: str | None, not_found: bool) -> None:
    if not sources:
        return
    label = "เอกสารที่ใกล้เคียงที่สุด" if not_found else "ที่มาของคำตอบ"
    with st.expander(f"{label} ({len(sources)} แหล่ง)", icon=":material/menu_book:"):
        for i, src in enumerate(sources, start=1):
            level, color = relevance_label(src["score"])
            st.markdown(f"**[{i}] {src['title']}**  \n:gray[{src['heading']}] · :{color}-badge[{level}]")
            st.caption(src["text"])
        if query:
            st.caption(f"ระบบค้นหาด้วยคำถาม: {query}")


def render_assistant(reply: dict) -> None:
    if reply.get("error"):
        st.warning(ERROR_MESSAGE, icon=":material/error:")
        st.caption(reply["error"])
        return
    content = reply["content"]
    st.markdown(content)
    not_found = content.strip().startswith(NOT_FOUND)
    if not_found:
        st.info(NOT_FOUND_HINT, icon=":material/lightbulb:")
    render_sources(reply.get("sources", []), reply.get("query"), not_found)
