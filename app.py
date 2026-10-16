from pathlib import Path
import os
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🍿", layout="wide")
BASE = Path(__file__).resolve().parent

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
:root{--bg:#080b14;--panel:rgba(20,26,43,.78);--border:rgba(255,255,255,.12);--muted:#a6aec6;--pink:#ff477e;--violet:#8b5cf6}
.stApp{background:radial-gradient(ellipse at 10% 0%,rgba(139,92,246,.20),transparent 34%),radial-gradient(ellipse at 95% 10%,rgba(255,71,126,.13),transparent 28%),var(--bg);color:#f5f7ff;font-family:'DM Sans',sans-serif}
h1,h2,h3,h4{font-family:'Outfit',sans-serif!important;letter-spacing:-.03em}
[data-testid="stHeader"]{background:rgba(8,11,20,.75)}
.block-container{padding-top:1.5rem;padding-bottom:3rem;max-width:1500px}
.hero{padding:2rem;border:1px solid var(--border);border-radius:26px;margin:.4rem 0 1.2rem;background:linear-gradient(115deg,rgba(139,92,246,.23),rgba(255,71,126,.10) 55%,rgba(80,216,237,.08));box-shadow:0 18px 70px rgba(0,0,0,.24);backdrop-filter:blur(18px)}
.eyebrow{color:#ff8eaf;font-weight:700;letter-spacing:.16em;text-transform:uppercase;font-size:.76rem}
.hero-title{font-family:Outfit,sans-serif;font-size:clamp(2rem,4vw,3.5rem);font-weight:800;line-height:1.08;margin:.55rem 0}
.hero-sub{color:#c2c8dc;max-width:720px}
.metric{background:rgba(255,255,255,.045);border:1px solid var(--border);border-radius:17px;padding:1rem;min-height:80px}
.metric-label{color:var(--muted);font-size:.75rem;letter-spacing:.08em}.metric-value{font:700 1.25rem Outfit,sans-serif;margin-top:.3rem}
.movie-card{background:linear-gradient(160deg,rgba(255,255,255,.075),rgba(255,255,255,.025));border:1px solid var(--border);border-radius:18px;overflow:hidden;transition:transform .22s,border-color .22s,box-shadow .22s;backdrop-filter:blur(16px);margin-bottom:.3rem}
.movie-card:hover{transform:translateY(-4px);border-color:rgba(255,71,126,.65);box-shadow:0 18px 35px rgba(0,0,0,.28)}
.poster{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#171d30}
.poster-fallback{width:100%;aspect-ratio:2/3;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:1rem;box-sizing:border-box;background:radial-gradient(circle at 25% 20%,rgba(139,92,246,.55),transparent 40%),linear-gradient(145deg,#171d30,#29152b);color:#f7f3ff;font:700 1rem Outfit,sans-serif;overflow-wrap:anywhere}
.card-body{padding:.8rem}.movie-title{font:700 .98rem Outfit,sans-serif;line-height:1.25;min-height:2.4rem}.movie-meta{color:var(--muted);font-size:.75rem;margin:.35rem 0 .6rem}
.score{display:inline-block;font-weight:800;padding:.3rem .5rem;border-radius:9px;font-size:.76rem}.score-green{background:rgba(34,197,94,.15);color:#6ef0a0;border:1px solid rgba(34,197,94,.35)}.score-yellow{background:rgba(250,204,21,.14);color:#ffe27a;border:1px solid rgba(250,204,21,.3)}.score-red{background:rgba(248,113,113,.14);color:#ff9999;border:1px solid rgba(248,113,113,.3)}
.stButton>button{border-radius:11px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.06);color:#fff;font-weight:600;transition:all .18s}
.stButton>button:hover{border-color:var(--pink);background:rgba(255,71,126,.16);color:white}
div[data-testid="stSidebar"]{background:rgba(12,16,29,.92);border-right:1px solid var(--border)}
.note{color:var(--muted);font-size:.78rem}
</style>
""", unsafe_allow_html=True)

REQUIRED = ["user_movie_matrix.pkl", "user_to_row.pkl", "movie_ids_cf.pkl",
            "cf_neighbor_cache.pkl", "movies_features.pkl"]

@st.cache_resource
def load_model_files():
    missing = [name for name in REQUIRED if not (BASE / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files beside app.py: " + ", ".join(missing))
    matrix = joblib.load(BASE / "user_movie_matrix.pkl")
    user_map = joblib.load(BASE / "user_to_row.pkl")
    movie_ids = joblib.load(BASE / "movie_ids_cf.pkl")
    neighbor_cache = joblib.load(BASE / "cf_neighbor_cache.pkl")
    features = pd.read_pickle(BASE / "movies_features.pkl")
    return matrix, user_map, np.asarray(movie_ids), neighbor_cache, features

try:
    user_movie_matrix, user_to_row, movie_ids_cf, neighbor_cache, movies = load_model_files()
except Exception as e:
    st.error("The app could not load the saved recommendation files.")
    st.code(str(e))
    st.markdown("Upload all five required `.pkl` files to the **same GitHub folder as `app.py`**. Do not paste model-code fragments into the top of `app.py`.")
    st.stop()

movies = movies.copy()
if "movieId" not in movies or "title" not in movies:
    st.error("movies_features.pkl must contain at least movieId and title columns.")
    st.stop()
if "genres" not in movies:
    movies["genres"] = "Genre not listed"
movies["title"] = movies["title"].fillna("Untitled movie").astype(str)
movies["genres"] = movies["genres"].fillna("Genre not listed").astype(str)
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
movies = movies.dropna(subset=["movieId"]).copy()
movies["movieId"] = movies["movieId"].astype(int)
movies = movies.drop_duplicates("movieId")
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].fillna("—")
movie_by_id = movies.set_index("movieId", drop=False)

@st.cache_data
def read_rating_summary():
    csv_path = BASE / "ratings.csv"
    zip_path = BASE / "ml-latest-small.zip"
    try:
        if csv_path.exists():
            ratings = pd.read_csv(csv_path)
        elif zip_path.exists():
            with zipfile.ZipFile(zip_path) as z:
                member = next((x for x in z.namelist() if x.endswith("ratings.csv")), None)
                if not member:
                    return {}
                with z.open(member) as f:
                    ratings = pd.read_csv(f)
        else:
            return {}
        if not {"movieId", "rating"}.issubset(ratings.columns):
            return {}
        return ratings.groupby("movieId")["rating"].agg(["mean", "count"]).to_dict("index")
