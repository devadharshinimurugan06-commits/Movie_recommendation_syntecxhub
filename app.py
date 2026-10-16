
from pathlib import Path
import os
import html
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "user_movie_matrix.pkl",
    "user_to_row.pkl",
    "movie_ids_cf.pkl",
    "cf_neighbor_cache.pkl",
    "movies_features.pkl",
]

# --------------------------------------------------
# CINEMATIC DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080b14;
    --panel: rgba(255,255,255,.055);
    --border: rgba(255,255,255,.13);
    --muted: #a6aec6;
    --pink: #ff477e;
    --violet: #8b5cf6;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(139,92,246,.22), transparent 35%),
        radial-gradient(ellipse at 95% 10%, rgba(255,71,126,.15), transparent 30%),
        var(--bg);
    color: #f5f7ff;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, h4, p, label {
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.3rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: rgba(8,11,20,.75);
}

[data-testid="stSidebar"] {
    display: none;
}

.hero {
    padding: 2.2rem;
    border: 1px solid var(--border);
    border-radius: 26px;
    margin: .5rem 0 1.5rem;
    background: linear-gradient(
        115deg,
        rgba(139,92,246,.24),
        rgba(255,71,126,.12),
        rgba(80,216,237,.08)
    );
    box-shadow: 0 18px 70px rgba(0,0,0,.25);
    backdrop-filter: blur(18px);
    text-align: center;
}

.eyebrow {
    color: #ff8eaf;
    font-weight: 700;
    letter-spacing: .16em;
    font-size: .76rem;
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.5rem);
    font-weight: 800;
    line-height: 1.12;
    margin: .65rem 0;
}

.hero-sub {
    color: #c2c8dc;
    max-width: 750px;
    margin: auto;
}

.metric {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 1rem;
    min-height: 80px;
    text-align: center;
}

.metric-label {
    color: var(--muted);
    font-size: .75rem;
    letter-spacing: .08em;
}

.metric-value {
    font-size: 1.15rem;
    font-weight: 800;
    margin-top: .35rem;
}

.movie-card {
    background: linear-gradient(
        160deg,
        rgba(255,255,255,.075),
        rgba(255,255,255,.025)
    );
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    transition: transform .22s, border-color .22s, box-shadow .22s;
    margin-bottom: .6rem;
}

.movie-card:hover {
    transform: translateY(-4px);
    border-color: var(--pink);
    box-shadow: 0 18px 35px rgba(0,0,0,.28);
}

.poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 12px;
}

.poster-fallback {
    aspect-ratio: 2 / 3;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    overflow-wrap: anywhere;
    padding: 1rem;
    border-radius: 12px;
    background:
        radial-gradient(circle at 25% 20%, rgba(139,92,246,.5), transparent 42%),
        linear-gradient(145deg, #171d30, #29152b);
    border: 1px solid var(--border);
}

.movie-title {
    font-size: .98rem;
    font-weight: 700;
    line-height: 1.35;
    min-height: 2.6rem;
    margin-top: .7rem;
}

.movie-meta {
    color: var(--muted);
    font-size: .78rem;
    margin: .35rem 0 .65rem;
}

.score {
    display: inline-block;
    font-weight: 800;
    padding: .3rem .5rem;
    border-radius: 9px;
    font-size: .76rem;
}

.score-green {
    background: rgba(34,197,94,.15);
    color: #6ef0a0;
    border: 1px solid rgba(34,197,94,.35);
}

.score-yellow {
    background: rgba(250,204,21,.14);
    color: #ffe27a;
    border: 1px solid rgba(250,204,21,.3);
}

.score-red {
    background: rgba(248,113,113,.14);
    color: #ff9999;
    border: 1px solid rgba(248,113,113,.3);
}

.stButton > button {
    width: 100%;
    border-radius: 11px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,.06);
    color: #fff;
    font-weight: 600;
    transition: all .18s;
}

.stButton > button:hover {
    border-color: var(--pink);
    background: rgba(255,71,126,.16);
    color: white;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px;
}

.small-note {
    color: var(--muted);
    font-size: .78rem;
}

hr {
    border-color: var(--border);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model_files():
    missing = [
        name for name in REQUIRED_FILES
        if not (BASE / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

    matrix = joblib.load(BASE / "user_movie_matrix.pkl")
    user_map = joblib.load(BASE / "user_to_row.pkl")
    movie_ids = np.asarray(
        joblib.load(BASE / "movie_ids_cf.pkl")
    )
    neighbor_cache = joblib.load(
        BASE / "cf_neighbor_cache.pkl"
    )
    features = pd.read_pickle(
        BASE / "movies_features.pkl"
    )

    return matrix, user_map, movie_ids, neighbor_cache, features


try:
    (
        user_movie_matrix,
        user_to_row,
        movie_ids_cf,
        neighbor_cache,
        movies,
    ) = load_model_files()

except Exception as error:
    st.error("The saved recommendation model could not be loaded.")
    st.code(str(error))
    st.info(
        "Keep all five required model files in the same GitHub "
        "folder as app.py. Check each filename carefully."
    )
    st.stop()


# --------------------------------------------------
# CLEAN MOVIE DATA
# --------------------------------------------------
movies = movies.copy()

if not {"movieId", "title"}.issubset(movies.columns):
    st.error("movies_features.pkl needs movieId and title columns.")
    st.stop()

if "genres" not in movies.columns:
    movies["genres"] = "Genre not listed"

movies["movieId"] = pd.to_numeric(
    movies["movieId"], errors="coerce"
)
movies = movies.dropna(subset=["movieId"]).copy()
movies["movieId"] = movies["movieId"].astype(int)
movies = movies.drop_duplicates("movieId")

movies["title"] = (
    movies["title"].fillna("Untitled movie").astype(str)
)
movies["genres"] = (
    movies["genres"].fillna("Genre not listed").astype(str)
)
movies["year"] = (
    movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    .fillna("Year unavailable")
)

# Only use a language filter if real language metadata exists.
LANGUAGE_COLUMN = next(
    (
        col for col in ["language", "original_language", "lang"]
        if col in movies.columns
    ),
    None,
)


# --------------------------------------------------
# LOAD MOVIELENS RATINGS
# --------------------------------------------------
@st.cache_data
def read_rating_summary():
    csv_path = BASE / "ratings.csv"
    zip_path = BASE / "ml-latest-small.zip"

    try:
        if csv_path.exists():
            ratings = pd.read_csv(csv_path)

        elif zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                member = next(
                    (
                        name for name in archive.namelist()
                        if name.endswith("ratings.csv")
                    ),
                    None,
                )

                if member is None:
                    return {}

                with archive.open(member) as file:
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


# --------------------------------------------------
# MOVIE POSTERS
# Optional: set TMDB_API_KEY in Streamlit Secrets.
# --------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(title):
    api_key = st.secrets.get(
        "TMDB_API_KEY",
        os.getenv("TMDB_API_KEY", ""),
    )

    if not api_key:
        return None

    try:
        import requests

        clean_title = title.rsplit(" (", 1)[0]

        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": clean_title,
            },
            timeout=5,
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
    poster_url = fetch_poster(title)

    if poster_url:
        st.image(poster_url, use_container_width=True)
    else:
        safe_title = html.escape(str(title))

        st.markdown(
            f"""
            <div class="poster-fallback">
                <div style="font-size:1.05rem;font-weight:800">
                    {safe_title}
                </div>
                <div style="color:#c2c8dc;margin-top:.7rem">
                    {html.escape(str(year))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# RECOMMENDATION FUNCTIONS
# --------------------------------------------------
def recommend_user(user_id, top_n=10):
    if user_id not in user_to_row:
        return pd.DataFrame()

    row = user_movie_matrix.getrow(user_to_row[user_id])
    scores = np.zeros(len(movie_ids_cf), dtype=float)

    for col, rating in zip(row.indices, row.data):
        if rating < 4.0:
            continue

        for neighbor_col, similarity in neighbor_cache.get(
            int(col), []
        ):
            neighbor_col = int(neighbor_col)

            if (
                0 <= neighbor_col < len(scores)
                and float(similarity) > 0
            ):
                scores[neighbor_col] += (
                    float(rating) - 3.0
                ) * float(similarity)

    seen = set(map(int, row.indices))

    candidates = [
        col for col in range(len(movie_ids_cf))
        if col not in seen
    ]

    ranked = sorted(
        candidates,
        key=lambda col: scores[col],
        reverse=True,
    )[:top_n]

    results = pd.DataFrame({
        "movieId": [int(movie_ids_cf[col]) for col in ranked],
        "Model score": [float(scores[col]) for col in ranked],
    })

    return results.merge(movies, on="movieId", how="left")


def recommend_similar(movie_id, top_n=10):
    positions = np.where(movie_ids_cf == int(movie_id))[0]

    if len(positions) == 0:
        return pd.DataFrame()

    col = int(positions[0])

    neighbors = sorted(
        neighbor_cache.get(col, []),
        key=lambda item: float(item[1]),
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


# --------------------------------------------------
# REACTIONS: LIKE, DISLIKE, WATCHLIST, HYPE
# --------------------------------------------------
def reaction_key(reaction, movie_id):
    return f"reaction_{reaction}_{int(movie_id)}"


def toggle_reaction(reaction, movie_id):
    key = reaction_key(reaction, movie_id)
    new_value = not st.session_state.get(key, False)

    st.session_state[key] = new_value

    # Like and dislike are mutually exclusive.
    if reaction == "like" and new_value:
        st.session_state[reaction_key("dislike", movie_id)] = False

    if reaction == "dislike" and new_value:
        st.session_state[reaction_key("like", movie_id)] = False


# --------------------------------------------------
# MOVIE CARDS
# --------------------------------------------------
def render_cards(frame, prefix, max_cards=5):
    if frame is None or frame.empty:
        st.info("No movies found for this selection.")
        return

    frame = frame.dropna(subset=["movieId"]).head(max_cards)
    columns = st.columns(5)

    for index, (_, movie) in enumerate(frame.iterrows()):
        movie_id = int(movie["movieId"])
        title = str(movie.get("title", "Untitled movie"))
        year = str(movie.get("year", "Year unavailable"))

        genres = [
            genre
            for genre in str(movie.get("genres", "")).split("|")
            if genre and genre != "(no genres listed)"
        ]

        avg_rating, rating_count = rating_info(movie_id)

        with columns[index % 5]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            show_poster(title, year)

            st.markdown(
                f"""
                <div class="movie-title">{html.escape(title)}</div>
                <div class="movie-meta">
                    {html.escape(year)} ·
                    {html.escape(", ".join(genres[:2]) or "Genre unavailable")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if avg_rating is not None:
                st.markdown(
                    f"""
                    <span class="score {rating_class(avg_rating)}">
                        ★ {avg_rating:.1f}/5 MovieLens
                    </span>
                    <div class="small-note">
                        {rating_count:,} ratings
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="small-note">MovieLens rating unavailable</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            like_key = reaction_key("like", movie_id)
            dislike_key = reaction_key("dislike", movie_id)
            watch_key = reaction_key("watch", movie_id)
            hype_key = reaction_key("hype", movie_id)

            left, right = st.columns(2)

            with left:
                st.button(
                    "👍 Liked" if st.session_state.get(like_key, False)
                    else "👍 Like",
                    key=f"{prefix}_like_{movie_id}",
                    use_container_width=True,
                    on_click=toggle_reaction,
                    args=("like", movie_id),
                )

                st.button(
                    "❤️ Saved" if st.session_state.get(watch_key, False)
                    else "❤️ Watchlist",
                    key=f"{prefix}_watch_{movie_id}",
                    use_container_width=True,
                    on_click=toggle_reaction,
                    args=("watch", movie_id),
                )

            with right:
                st.button(
                    "👎 Disliked" if st.session_state.get(dislike_key, False)
                    else "👎 Dislike",
                    key=f"{prefix}_dislike_{movie_id}",
                    use_container_width=True,
                    on_click=toggle_reaction,
                    args=("dislike", movie_id),
                )

                st.button(
                    "🔥 Hyped!" if st.session_state.get(hype_key, False)
                    else "🔥 Hype",
                    key=f"{prefix}_hype_{movie_id}",
                    use_container_width=True,
                    on_click=toggle_reaction,
                    args=("hype", movie_id),
                )


# --------------------------------------------------
# CENTERED HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL CINEMA</div>
    <div class="hero-title">
        Find your next
        <span style="color:#ff6f9a"> obsession.</span>
    </div>
    <div class="hero-sub">
        Discover your next favourite movie with recommendations
        powered by your saved MovieLens collaborative-filtering model.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CENTERED PROFILE + NAVIGATION
# --------------------------------------------------
profile_col, nav_col = st.columns([1, 2])

with profile_col:
    available_users = sorted(int(uid) for uid in user_to_row.keys())

    selected_user = st.selectbox(
        "Choose MovieLens user",
        available_users,
        help=(
            "This ID selects a user from the training ratings "
            "used by your saved recommendation model."
        ),
    )

with nav_col:
    page = st.radio(
        "Explore CineMatch",
        ["Discover", "For You", "By Genre", "My Watchlist"],
        horizontal=True,
        label_visibility="visible",
    )


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------
total_ratings = (
    sum(value["count"] for value in rating_summary.values())
    if rating_summary else None
)

m1, m2, m3 = st.columns(3)

for column, label, value in [
    (m1, "MOVIES", f"{len(movies):,}"),
    (
        m2,
        "MOVIELENS RATINGS",
        f"{total_ratings:,}" if total_ratings is not None else "Not loaded",
    ),
    (m3, "RECOMMENDATION ENGINE", "Collaborative filtering"),
]:
    with column:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# MOVIE SEARCH
# --------------------------------------------------
st.markdown("## 🔎 Find a movie")

search_text = st.text_input(
    "Search by movie title",
    placeholder="Type a title — e.g. Toy Story, Matrix, Inception",
)

if search_text.strip():
    matched = movies[
        movies["title"].str.contains(
            search_text.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ].head(30)

    if matched.empty:
        st.info("No matching movies. Try another title.")
    else:
        options = {
            f"{row['title']} · {row['year']}": int(row["movieId"])
            for _, row in matched.iterrows()
        }

        selected_label = st.selectbox(
            "Choose a matching movie",
            list(options.keys()),
            key="search_movie_choice",
        )

        chosen_id = options[selected_label]
        selected_movie = movies[
            movies["movieId"] == chosen_id
        ].iloc[0]

        poster_col, details_col = st.columns([1, 3])

        with poster_col:
            show_poster(
                selected_movie["title"],
                selected_movie["year"],
            )

        with details_col:
            st.subheader(selected_movie["title"])
            st.write("Year:", selected_movie["year"])
            st.write("Genres:", selected_movie["genres"])

            avg, count = rating_info(chosen_id)

            if avg is not None:
                st.markdown(
                    f"""
                    <span class="score {rating_class(avg)}">
                        ★ {avg:.1f}/5 MovieLens
                    </span>
                    <span class="small-note"> · {count:,} ratings</span>
                    """,
                    unsafe_allow_html=True,
                )

            st.button(
                "✨ Recommend similar movies",
                key=f"similar_button_{chosen_id}",
                on_click=lambda mid=chosen_id:
                    st.session_state.update({"similar_movie_id": mid}),
            )

        if st.session_state.get("similar_movie_id") == chosen_id:
            st.markdown("### 🎯 More movies like this")
            render_cards(
                recommend_similar(chosen_id, 10),
                "similar",
                5,
            )


# --------------------------------------------------
# DISCOVER PAGE
# --------------------------------------------------
st.markdown("---")

if page == "Discover":
    genres = sorted({
        genre
        for value in movies["genres"]
        for genre in str(value).split("|")
        if genre and genre != "(no genres listed)"
    })

    selected_genre = st.selectbox(
        "Choose a movie category",
        ["All genres"] + genres,
        key="discover_genre",
    )

    filtered = movies.copy()

    if selected_genre != "All genres":
        filtered = filtered[
            filtered["genres"].apply(
                lambda value: selected_genre in str(value).split("|")
            )
        ]

    if rating_summary and not filtered.empty:
        filtered["avg_rating"] = filtered["movieId"].map(
            lambda mid: rating_summary.get(int(mid), {}).get("mean", 0)
        )

        filtered = filtered.sort_values(
            ["avg_rating", "title"],
            ascending=[False, True],
        )

    st.markdown(
        f"## ✨ {selected_genre} movies"
    )
    render_cards(filtered, "discover", 10)

    st.markdown("## 🎬 More to explore")
    render_cards(
        movies.sort_values("title").head(5),
        "curated",
        5,
    )


# --------------------------------------------------
# PERSONAL RECOMMENDATIONS
# --------------------------------------------------
elif page == "For You":
    st.markdown(f"## 💜 Recommended for user {selected_user}")

    st.caption(
        "Recommendations come from your saved collaborative-filtering model. "
        "Model scores are ranking scores, not probabilities or movie ratings."
    )

    recommendations = recommend_user(selected_user, 10)
    render_cards(recommendations, "for_you", 10)


# --------------------------------------------------
# GENRE PAGE
# --------------------------------------------------
elif page == "By Genre":
    genres = sorted({
        genre
        for value in movies["genres"]
        for genre in str(value).split("|")
        if genre and genre != "(no genres listed)"
    })

    genre = st.selectbox(
        "Choose a genre",
        genres,
        key="genre_page",
    )

    genre_movies = movies[
        movies["genres"].apply(
            lambda value: genre in str(value).split("|")
        )
    ].copy()

    if rating_summary and not genre_movies.empty:
        genre_movies["avg_rating"] = genre_movies["movieId"].map(
            lambda mid: rating_summary.get(int(mid), {}).get("mean", 0)
        )

        genre_movies = genre_movies.sort_values(
            ["avg_rating", "title"],
            ascending=[False, True],
        )

    st.markdown(f"## 🎞️ Top {genre} movies")
    render_cards(genre_movies, "genre", 10)


# --------------------------------------------------
# WATCHLIST
# --------------------------------------------------
elif page == "My Watchlist":
    st.markdown("## ❤️ Your Watchlist")

    saved_ids = [
        int(str(key).rsplit("_", 1)[-1])
        for key, value in st.session_state.items()
        if key.startswith("reaction_watch_") and value is True
    ]

    saved_movies = movies[
        movies["movieId"].isin(saved_ids)
    ].drop_duplicates("movieId")

    if saved_movies.empty:
        st.info(
            "Your watchlist is empty. Press ❤️ Watchlist "
            "on any movie card to save a movie."
        )
    else:
        render_cards(saved_movies, "watchlist", 10)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div class="small-note" style="text-align:center">
        CineMatch · Streamlit · Saved collaborative-filtering model<br>
        Ratings shown are MovieLens community ratings on a 0–5 scale,
        not IMDb ratings.<br>
        Real posters require a TMDB API key in Streamlit Secrets.
    </div>
    """,
    unsafe_allow_html=True,
)
