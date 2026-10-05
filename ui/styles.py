"""Global look & feel.

Palette follows the Brainwave / Botly Figma kits: navy hero + sidebar,
purple primary actions, green accents.
"""

from __future__ import annotations

from string import Template

import streamlit as st

NAVY = "#14163A"
NAVY_LIGHT = "#252A6B"
PURPLE = "#5B47F0"
PURPLE_SOFT = "#EEEBFF"
GREEN = "#22C55E"
GREEN_SOFT = "#E7F8EE"
BORDER = "#E4E7F2"
SHADOW = "0 8px 24px rgba(20, 22, 58, 0.06)"
FONT = "'IBM Plex Sans Thai', 'Sarabun', sans-serif"
TOPIC_COUNT = 6

# (icon colour, tile background) per topic card, cycling purple / green / navy.
ICON_TILES = ((PURPLE, PURPLE_SOFT), ("#16A34A", GREEN_SOFT), (NAVY_LIGHT, "#E8EAF6"))

_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');

.stApp { background: #F5F7FB; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp summary { font-family: $font; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 880px; }

/* ---------- sidebar (navy, Botly style) ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, $navy 0%, $navy_light 100%); }
section[data-testid="stSidebar"] * { color: #E7E9FF; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #A9AEDB; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.14); border-radius: 14px; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
  background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.22); }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
.tb-side-brand { display: flex; align-items: center; gap: 10px; font-size: 1.25rem;
  font-weight: 700; color: #fff; margin: 0 0 .6rem; }
.tb-side-list { list-style: none; padding: 0; margin: .2rem 0 0; }
.tb-side-list li { display: flex; gap: 8px; margin-bottom: .45rem; font-size: .92rem; color: #D7DAF7; }
.tb-side-list li::before { content: ""; flex: 0 0 7px; height: 7px; margin-top: .55em;
  border-radius: 50%; background: $green; }

/* ---------- hero (navy gradient, Brainwave style) ---------- */
.tb-hero { position: relative; overflow: hidden; border-radius: 24px; padding: 34px 34px 28px;
  background: radial-gradient(circle at 92% 8%, rgba(123,97,255,.6), transparent 42%),
              radial-gradient(circle at 0% 100%, rgba(34,197,94,.22), transparent 42%),
              linear-gradient(135deg, $navy, $navy_light);
  color: #fff; box-shadow: 0 18px 40px rgba(20,22,58,.22); margin-bottom: 1.6rem; }
.tb-brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; }
.tb-title { color: #fff; font-size: 2.05rem; line-height: 1.4; margin: 18px 0 10px; font-weight: 700; }
.tb-title .tb-accent { color: #4ADE80; }
.tb-hero p { color: #C9CCF2; font-size: 1.02rem; margin: 0 0 18px; max-width: 580px; }
.tb-pills { display: flex; flex-wrap: wrap; gap: 8px; }
.tb-pills span { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
  border-radius: 999px; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
  font-size: .85rem; color: #fff; }
.tb-section { font-weight: 600; color: $navy; font-size: 1.05rem; margin: 0 0 .5rem; }
.tb-footnote { text-align: center; color: #7A7F9F; font-size: .8rem; margin-top: 1rem; }

/* ---------- compact header once chatting ---------- */
.tb-bar { display: flex; align-items: center; gap: 12px; padding: 12px 18px; border-radius: 16px;
  background: linear-gradient(135deg, $navy, $navy_light); color: #fff; margin-bottom: 1.2rem; }
.tb-bar b { font-size: 1.05rem; }
.tb-bar small { color: #C9CCF2; }

/* ---------- topic cards ---------- */
div[class*="st-key-topic_"] { background: #fff; border: 1px solid $border; border-radius: 18px;
  padding: 18px 18px 8px; box-shadow: $shadow; min-height: 172px;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
div[class*="st-key-topic_"]:hover { transform: translateY(-3px); border-color: $purple;
  box-shadow: 0 14px 30px rgba(20,22,58,.12); }
div[class*="st-key-topic_"] [data-testid="stMarkdownContainer"] [data-testid="stIconMaterial"] {
  font-size: 1.3rem; padding: 7px; border-radius: 12px; margin-right: 6px; vertical-align: middle; }
div[class*="st-key-topic_"] button { padding-left: 0; }
div[class*="st-key-topic_"] button p { color: $purple; font-weight: 600; }
$icon_tiles

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: #fff; border: 1px solid $border; border-radius: 18px;
  padding: 16px 18px; box-shadow: $shadow; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: $purple_soft; border-color: transparent; box-shadow: none; }
[data-testid="stChatMessageAvatarUser"] { background: $purple; color: #fff; }
[data-testid="stChatMessageAvatarAssistant"] { background: linear-gradient(135deg, $green, #15803D); color: #fff; }
[data-testid="stExpander"] details { border-radius: 14px; border-color: $border; background: #FAFBFF; }
[data-testid="stChatInput"] > div { border-radius: 16px; border: 1.5px solid $border;
  background: #fff; box-shadow: $shadow; }
[data-testid="stChatInput"] > div:focus-within { border-color: $purple; }
</style>
""")


def _icon_tile_rules() -> str:
    rules = []
    for i in range(TOPIC_COUNT):
        color, background = ICON_TILES[i % len(ICON_TILES)]
        rules.append(
            f'div.st-key-topic_{i} [data-testid="stMarkdownContainer"] [data-testid="stIconMaterial"]'
            f" {{ color: {color}; background: {background}; }}"
        )
    return "\n".join(rules)


def inject_css() -> None:
    css = _CSS.substitute(
        font=FONT, navy=NAVY, navy_light=NAVY_LIGHT, purple=PURPLE, purple_soft=PURPLE_SOFT,
        green=GREEN, border=BORDER, shadow=SHADOW, icon_tiles=_icon_tile_rules(),
    )
    st.markdown(css, unsafe_allow_html=True)
