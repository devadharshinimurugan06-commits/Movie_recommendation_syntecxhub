
from pathlib import Path
import zipfile
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CineAura | Movie Discovery",
    page_icon="🎬",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

# ---------- CINEMATIC CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
 --bg:#080b14; --border:rgba(255,255,255,.13);
 --muted:#a6aec6; --pink:#ff477e; --violet:#8b5cf6;
}
.stApp {
 background:
 radial-gradient(ellipse at 10% 0%,rgba(139,92,246,.22),transparent 35%),
 radial-gradient(ellipse at 95% 10%,rgba(255,71,126,.15),transparent 30%),
 var(--bg);
 color:#f5f7ff; font-family:'DM Sans',sans-serif;
}
h1,h2,h3,h4 {font-family:'Outfit',sans-serif!important}
[data-testid="stHeader"] {background:rgba(8,11,20,.7)}
.block-container {padding-top:1.5rem;max-width:1500px}
.hero {
 padding:2.2rem;border:1px solid var(--border);border-radius:26px;
 margin:.5rem 0 1.4rem;
 background:linear-gradient(115deg,rgba(139,92,246,.24),rgba(255,71,126,.12),rgba(80,216,237,.08));
 box-shadow:0 18px 70px rgba(0,0,0,.25);
 backdrop-filter:blur(18px);
}
.eyebrow {color:#ff8eaf;font-weight:700;letter-spacing:.16em;font-size:.76rem}
.hero-title {font-family:Outfit,sans-serif;font-size:clamp(2rem,4vw,3.5rem);font-weight:800;line-height:1.1;margin:.5rem 0}
.hero-sub {color:#c2c8dc}
.metric {
 background:rgba(255,255,255,.045);border:1px solid var(--border);
 border-radius:17px;padding:1rem;min-height:78px
}
.metric-label {color:var(--muted);font-size:.75rem;letter-spacing:.08em}
.metric-value {font:700 1.2rem Outfit,sans-serif;margin-top:.3rem}
.poster {width:100%;aspect-ratio:2/3;object-fit:cover;display:block}
.poster-fallback {
 width:100%;aspect-ratio:2/3;display:flex;flex-direction:column;
 align-items:center;justify-content:center;text-align:center;padding:1rem;
 box-sizing:border-box;
 background:radial-gradient(circle at 25% 20%,rgba(139,92,246,.55),transparent 40%),linear-gradient(145deg,#171d30,#29152b);
 color:#f7f3ff;font:700 1rem Outfit,sans-serif;overflow-wrap:anywhere
}
.movie-card {
 background:linear-gradient(160deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
 border:1px solid var(--border);border-radius:18px;overflow:hidden;
 transition:transform .22s,border-color .22s,box-shadow .22s;
 margin-bottom:.5rem
}
.movie-card:hover {transform:translateY(-4px);border-color:var(--pink);box-shadow:0 18px 35px #0005}
.card-body {padding:.8rem}
.movie-title {font:700 .98rem Outfit,sans-serif;line-height:1.3;min-height:2.4rem}
.movie-meta {color:var(--muted);font-size:.75rem;margin:.35rem 0 .65rem}
.score {display:inline-block;font-weight:800;padding:.3rem .5rem;border-radius:9px;font-size:.76rem}
.score-green {background:#22c55e22;color:#6ef0a0;border:1px solid #22c55e55}
.score-yellow {background:#facc1522;color:#ffe27a;border:1px solid #facc1555}
.score-red {background:#f8717122;color:#ff9999;border:1px solid #f8717155}
.stButton > button {
 border-radius:11px;
 border:1px solid rgba(255,255,255,.18);
 background:#151821;
 color:#ffffff !important;
 font-weight:600;
 min-height:44px;
 white-space:nowrap;
 font-size:14px;
 font-family:"Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
.stButton > button p {
 color:#ffffff !important;
 font-size:14px;
 font-family:"Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
.stButton > button:hover {border-color:var(--pink);background:#26202d;color:#ffffff !important}
div[data-testid="stSidebar"] {background:rgba(12,16,29,.95)}
.small-note {color:var(--muted);font-size:.78rem}
</style>
""", unsafe_allow_html=True)
