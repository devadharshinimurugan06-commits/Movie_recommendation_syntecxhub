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
 font-family:"Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", "Twemoji Mozilla", sans-serif !important;
}
.stButton > button p,
.stButton > button span {
 color:#ffffff !important;
 font-size:14px;
 font-family:"Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", "Twemoji Mozilla", sans-serif !important;
}
.stButton > button:hover {border-color:var(--pink);background:#26202d;color:#ffffff !important}
div[data-testid="stSidebar"] {background:rgba(12,16,29,.95)}
.small-note {color:var(--muted);font-size:.78rem}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD YOUR SAVED MODEL ----------
REQUIRED = [
    "user_movie_matrix.pkl",
    "user_to_row.pkl",
    "movie_ids_cf.pkl",
    "cf_neighbor_cache.pkl",
    "movies_features.pkl",
]

@st.cache_resource
def load_model_files():
    missing = [f for f in REQUIRED if not (BASE / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

    matrix = joblib.load(BASE / "user_movie_matrix.pkl")
    user_map = joblib.load(BASE / "user_to_row.pkl")
    movie_ids = np.asarray(joblib.load(BASE / "movie_ids_cf.pkl"))
    neighbors = joblib.load(BASE / "cf_neighbor_cache.pkl")

    # This file was saved from a pandas DataFrame in Colab.
    features = pd.read_pickle(BASE / "movies_features.pkl")

    return matrix, user_map, movie_ids, neighbors, features


try:
    user_movie_matrix, user_to_row, movie_ids_cf, neighbor_cache, movies = load_model_files()
except Exception as error:
    st.error("The recommendation model could not be loaded.")
    st.code(str(error))
    st.info(
        "Upload all five required files into the same GitHub folder "
        "as app.py. Check the filenames and capitalization."
    )
    st.stop()


# ---------- CLEAN MOVIE DATA ----------
movies = movies.copy()

if "movieId" not in movies.columns or "title" not in movies.columns:
    st.error("movies_features.pkl needs movieId and title columns.")
    st.stop()

if "genres" not in movies.columns:
    movies["genres"] = "Genre not listed"

movies["title"] = movies["title"].fillna("Untitled movie").astype(str)
movies["genres"] = movies["genres"].fillna("Genre not listed").astype(str)
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
movies = movies.dropna(subset=["movieId"]).copy()
movies["movieId"] = movies["movieId"].astype(int)
movies = movies.drop_duplicates("movieId")
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].fillna("—")


# ---------- OPTIONAL MOVIE RATINGS ----------
@st.cache_data
def read_rating_summary():
    csv_path = BASE / "ratings.csv"
    zip_path = BASE / "ml-latest-small.zip"

    try:
        if csv_path.exists():
            ratings = pd.read_csv(csv_path)
        elif zip_path.exists():
            with zipfile.ZipFile(zip_path) as z:
                member = next(
                    (name for name in z.namelist()
                     if name.endswith("ratings.csv")),
                    None
                )
                if member is None:
                    return {}
                with z.open(member) as file:
                    ratings = pd.read_csv(file)
        else:
            return {}

        if not {"movieId", "rating"}.issubset(ratings.columns):
            return {}

        return ratings.groupby("movieId")["rating"].agg(
            ["mean", "count"]
        ).to_dict("index")

    except Exception:
        return {}


rating_summary = read_rating_summary()


def rating_info(movie_id):
    info = rating_summary.get(int(movie_id))
    if not info:
        return None, 0
    return float(info["mean"]), int(info["count"])


def rating_class(score):
    if score is None:
        return "score-yellow"
    if score >= 4.0:
        return "score-green"
    if score >= 2.5:
        return "score-yellow"
    return "score-red"


# ---------- OPTIONAL POSTERS ----------
# Without a TMDB API key, attractive custom poster placeholders are shown.
@st.cache_data(show_spinner=False)
def fetch_poster(title):
    api_key = st.secrets.get(
        "TMDB_API_KEY", os.getenv("TMDB_API_KEY", "")
    )
    if not api_key:
        return None

    try:
        import requests

        search_title = title.rsplit(" (", 1)[0]
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": search_title},
            timeout=4,
        )
        if response.ok:
            results = response.json().get("results", [])
            if results and results[0].get("poster_path"):
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + results[0]["poster_path"]
                )
    except Exception:
        pass

    return None


def show_poster(title, year):
    url = fetch_poster(title)
    if url:
        st.markdown(
            f'<img class="poster" src="{url}" alt="Movie poster">',
            unsafe_allow_html=True,
        )
    else:
        safe_title = (
            str(title).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;")
        )
        st.markdown(
            f'<div class="poster-fallback">'
            f'<div style="font-size:2.2rem">🍿</div>'
            f'<div>{safe_title}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------- COLLABORATIVE FILTERING ----------
def recommend_user(user_id, top_n=10):
    if user_id not in user_to_row:
        return pd.DataFrame()

    row = user_movie_matrix.getrow(user_to_row[user_id])
    scores = np.zeros(len(movie_ids_cf), dtype=float)

    for col, rating in zip(row.indices, row.data):
        if rating < 4.0:
            continue

        for neighbor_col, similarity in neighbor_cache.get(int(col), []):
            neighbor_col = int(neighbor_col)
            if 0 <= neighbor_col < len(scores) and similarity > 0:
                scores[neighbor_col] += (
                    float(rating) - 3.0
                ) * float(similarity)

    seen = set(map(int, row.indices))
    candidates = [
        i for i in range(len(movie_ids_cf))
        if i not in seen
    ]
    ranked = sorted(
        candidates, key=lambda i: scores[i], reverse=True
    )[:top_n]

    results = pd.DataFrame({
        "movieId": [int(movie_ids_cf[i]) for i in ranked],
        "Model score": [float(scores[i]) for i in ranked],
    })

    return results.merge(movies, on="movieId", how="left")


def recommend_similar(movie_id, top_n=10):
    matches = np.where(movie_ids_cf == int(movie_id))[0]
    if len(matches) == 0:
        return pd.DataFrame()

    col = int(matches[0])
    neighbors = sorted(
        neighbor_cache.get(col, []),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    results = pd.DataFrame({
        "movieId": [
            int(movie_ids_cf[int(index)])
            for index, similarity in neighbors
        ],
        "Similarity": [
            float(similarity)
            for index, similarity in neighbors
        ],
    })

    return results.merge(movies, on="movieId", how="left")


# ---------- INTERACTIVE MOVIE CARDS ----------
def render_cards(frame, prefix, max_cards=5):
    if frame is None or frame.empty:
        st.info("No recommendations found for this selection.")
        return

    frame = frame.head(max_cards).reset_index(drop=True)
    columns = st.columns(5)

    for index, movie in frame.iterrows():
        if pd.isna(movie.get("movieId")):
            continue

        movie_id = int(movie["movieId"])
        title = str(movie.get("title", "Untitled movie"))
        year = str(movie.get("year", "—"))
        genres = [
            g for g in str(movie.get("genres", "")).split("|")
            if g and g != "(no genres listed)"
        ]
        avg_rating, count = rating_info(movie_id)

        like_key = f"like_{prefix}_{movie_id}"
        dislike_key = f"dislike_{prefix}_{movie_id}"
        watch_key = f"watch_{prefix}_{movie_id}"
        hype_key = f"hype_{prefix}_{movie_id}"

        with columns[index % 5]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            show_poster(title, year)

            safe_title = (
                title.replace("&", "&amp;")
                .replace("<", "&lt;").replace(">", "&gt;")
            )
            st.markdown(
                f'<div class="card-body">'
                f'<div class="movie-title">{safe_title}</div>'
                f'<div class="movie-meta">{year} · '
                f'{", ".join(genres[:2])}</div>',
                unsafe_allow_html=True,
            )

            if avg_rating is not None:
                st.markdown(
                    f'<span class="score {rating_class(avg_rating)}">'
                    f'★ {avg_rating:.1f}/5 MovieLens</span>'
                    f'<div class="small-note">{count:,} ratings · '
                    f'not IMDb</div></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="score score-yellow">'
                    '★ Rating unavailable</span></div></div>',
                    unsafe_allow_html=True,
                )

            left, right = st.columns(2)

            with left:
                if st.button(
                    "👍 Liked" if st.session_state.get(like_key) else "👍 Like",
                    key=like_key + "_btn",
                    use_container_width=True,
                ):
                    st.session_state[like_key] = not st.session_state.get(like_key, False)
                    if st.session_state[like_key]:
                        st.session_state[dislike_key] = False
                    st.rerun()

                if st.button(
                    "❤️ Saved" if st.session_state.get(watch_key) else "❤️ Watchlist",
                    key=watch_key + "_btn",
                    use_container_width=True,
                ):
                    st.session_state[watch_key] = not st.session_state.get(watch_key, False)
                    st.rerun()

            with right:
                if st.button(
                    "👎 Disliked" if st.session_state.get(dislike_key) else "👎 Dislike",
                    key=dislike_key + "_btn",
                    use_container_width=True,
                ):
                    st.session_state[dislike_key] = not st.session_state.get(dislike_key, False)
                    if st.session_state[dislike_key]:
                        st.session_state[like_key] = False
                    st.rerun()

                if st.button(
                    "🔥 Hyped!" if st.session_state.get(hype_key) else "🔥 Hype",
                    key=hype_key + "_btn",
                    use_container_width=True,
                ):
                    st.session_state[hype_key] = not st.session_state.get(hype_key, False)
                    st.rerun()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🎬 CineAura")
    st.caption("Your next favourite is waiting.")
    page = st.radio(
        "Explore",
        ["Discover", "For You", "By Genre", "My Watchlist"]
    )

    st.markdown("---")
    st.markdown("### Your profile")

    available_users = sorted(int(uid) for uid in user_to_row.keys())
    selected_user = st.selectbox(
        "MovieLens user ID", available_users
    )

    st.caption("Recommendations use your saved collaborative-filtering model.")


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">YOUR PERSONAL CINEMA</div>
  <div class="hero-title">Find your next
  <span style="color:#ff6f9a">obsession.</span> 🍿</div>
  <div class="hero-sub">
  Discover hidden gems and explore recommendations powered by your
  trained MovieLens collaborative-filtering model.
  </div>
</div>
""", unsafe_allow_html=True)

ratings_count = (
    sum(value["count"] for value in rating_summary.values())
    if rating_summary else None
)

m1, m2, m3 = st.columns(3)
metrics = [
    (m1, "MOVIES IN CATALOG", f"{len(movies):,}"),
    (
        m2, "MOVIELENS RATINGS",
        f"{ratings_count:,}" if ratings_count is not None else "Not loaded",
    ),
    (m3, "RECOMMENDER", "Collaborative filtering"),
]

for column, label, value in metrics:
    with column:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{label}'
            f'</div><div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


# ---------- SEARCH ----------
st.markdown("### 🔎 Find a movie")

# Searchable dropdown containing every movie in the loaded dataset.
movie_options = (
    movies[["movieId", "title", "year"]]
    .drop_duplicates("movieId")
    .sort_values("title")
)
movie_labels = {
    f"{row.title} · {row.year}": int(row.movieId)
    for row in movie_options.itertuples(index=False)
}

selected_label = st.selectbox(
    "Search movies",
    options=["Choose a movie..."] + list(movie_labels.keys()),
    index=0,
    placeholder="Type a movie title to search...",
    label_visibility="collapsed",
    key="movie_search_dropdown",
)

if selected_label != "Choose a movie...":
    chosen_id = movie_labels[selected_label]
    selected_movie = movies[movies["movieId"] == chosen_id].iloc[0]

    left, right = st.columns([1, 3])

    with left:
        show_poster(selected_movie["title"], selected_movie["year"])

    with right:
        st.subheader(selected_movie["title"])
        st.write("Genres:", selected_movie["genres"])
        avg, count = rating_info(chosen_id)
        if avg is not None:
            st.markdown(
                f'<span class="score {rating_class(avg)}">'
                f'★ {avg:.1f}/5 MovieLens</span> '
                f'<span class="small-note">({count:,} ratings)</span>',
                unsafe_allow_html=True,
            )

        if st.button("✨ Recommend similar movies", key=f"similar_button_{chosen_id}"):
            st.session_state["similar_movie_id"] = chosen_id

    if st.session_state.get("similar_movie_id") == chosen_id:
        st.markdown(f"#### 🎬 More like {selected_movie['title']}")
        render_cards(recommend_similar(chosen_id, 10), "similar", 5)


# ---------- PAGE CONTENT ----------
st.markdown("---")

if page == "Discover":
    genres = sorted({
        genre
        for values in movies["genres"]
        for genre in str(values).split("|")
        if genre and genre != "(no genres listed)"
    })

    genre = st.selectbox("Choose a category", genres, key="discover_genre")
    genre_movies = movies[
        movies["genres"].str.contains(
            genre, case=False, regex=False, na=False
        )
    ].copy()

    if rating_summary:
        genre_movies["avg_rating"] = genre_movies["movieId"].map(
            lambda movie_id: rating_summary.get(
                int(movie_id), {}
            ).get("mean", 0)
        )
        genre_movies = genre_movies.sort_values(
            ["avg_rating", "title"], ascending=[False, True]
        )

    st.markdown(f"## 🔥 {genre} picks")
    render_cards(genre_movies, "discover", 5)

    st.markdown("## ✨ Curated shelf")
    render_cards(movies.sort_values("title").head(5), "curated", 5)

elif page == "For You":
    st.markdown(f"## 💜 Picks for MovieLens user {selected_user}")
    st.caption(
        "Model score ranks candidates. It is not a probability or an IMDb rating."
    )

    recommendations = recommend_user(selected_user, 10)
    render_cards(recommendations, "for_you", 5)

    st.markdown("### More recommendations")
    render_cards(recommendations.iloc[5:], "for_you_more", 5)

elif page == "By Genre":
    genres = sorted({
        genre
        for values in movies["genres"]
        for genre in str(values).split("|")
        if genre and genre != "(no genres listed)"
    })

    genre = st.selectbox("Pick a genre", genres, key="genre_page")
    genre_movies = movies[
        movies["genres"].str.contains(
            genre, case=False, regex=False, na=False
        )
    ].copy()

    if rating_summary:
        genre_movies["avg_rating"] = genre_movies["movieId"].map(
            lambda movie_id: rating_summary.get(
                int(movie_id), {}
            ).get("mean", 0)
        )
        genre_movies = genre_movies.sort_values(
            ["avg_rating", "title"], ascending=[False, True]
        )

    render_cards(genre_movies, "genre", 5)

else:
    saved_ids = []

    for key, value in list(st.session_state.items()):
        if key.startswith("watch_") and value is True:
            try:
                saved_ids.append(int(key.rsplit("_", 1)[1]))
            except ValueError:
                pass

    saved_movies = movies[
        movies["movieId"].isin(saved_ids)
    ].drop_duplicates("movieId")

    st.markdown("## ❤️ Your Watchlist")

    if saved_movies.empty:
        st.info(
            "Your watchlist is empty. Press ❤️ Watchlist on a movie card."
        )
    else:
        render_cards(saved_movies, "watchlist", 5)
        st.dataframe(
            saved_movies[["title", "year", "genres"]],
            use_container_width=True,
            hide_index=True,
        )


# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    '<div class="small-note">🎬 CineAura · Built by Devadharshini Murugan. '
    'Scores shown are MovieLens community ratings (0–5), not IMDb. '
    'Real posters are optional; configure TMDB_API_KEY in Streamlit '
    'Secrets to enable them.</div>',
    unsafe_allow_html=True,
)
