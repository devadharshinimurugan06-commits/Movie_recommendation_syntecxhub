
from pathlib import Path
import os
import re
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# CINEAURA - MOVIE RECOMMENDATION SYSTEM
# Uses the saved collaborative-filtering model.
# ---------------------------------------------------------

st.set_page_config(
    page_title="CineAura | Discover Your Next Favourite",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent

# ----------------------- STYLING --------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #080b14;
  --panel: rgba(23, 29, 48, .78);
  --border: rgba(255,255,255,.12);
  --muted: #a6aec6;
  --pink: #ff477e;
  --violet: #8b5cf6;
  --cyan: #50d8ed;
}

.stApp {
  background:
    radial-gradient(ellipse at 10% 0%, rgba(139,92,246,.20), transparent 34%),
    radial-gradient(ellipse at 95% 10%, rgba(255,71,126,.13), transparent 28%),
    var(--bg);
  color: #f5f7ff;
  font-family: 'DM Sans', sans-serif;
}

h1,h2,h3,h4 {
  font-family: 'Outfit', sans-serif !important;
  letter-spacing: -.03em;
}

[data-testid="stHeader"] { background: rgba(8,11,20,.65); }

.block-container {
  max-width: 1450px;
  padding-top: 1.3rem;
  padding-bottom: 3rem;
}

section[data-testid="stSidebar"] { display: none; }

.hero {
  padding: 2.2rem;
  border: 1px solid var(--border);
  border-radius: 26px;
  margin: .5rem 0 1.4rem;
  background: linear-gradient(115deg,
    rgba(139,92,246,.25),
    rgba(255,71,126,.12) 55%,
    rgba(80,216,237,.09));
  box-shadow: 0 18px 70px rgba(0,0,0,.24);
  backdrop-filter: blur(18px);
  text-align: center;
}

.brand {
  font-family: 'Outfit', sans-serif;
  font-size: 1rem;
  color: #ff9abb;
  font-weight: 800;
  letter-spacing: .18em;
  text-transform: uppercase;
}

.hero-title {
  font-family: 'Outfit', sans-serif;
  font-size: clamp(2.2rem, 5vw, 4rem);
  font-weight: 800;
  line-height: 1.08;
  margin: .7rem 0;
}

.hero-sub {
  color: #c2c8dc;
  max-width: 720px;
  margin: auto;
}

.section-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.55rem;
  font-weight: 800;
  margin: 1.4rem 0 .3rem;
}

.section-subtitle {
  color: var(--muted);
  margin-bottom: 1rem;
  font-size: .9rem;
}

.metric {
  background: rgba(255,255,255,.045);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
  text-align: center;
}

.metric-label {
  color: var(--muted);
  font-size: .78rem;
}

.metric-value {
  font: 800 1.35rem 'Outfit', sans-serif;
  margin-top: .3rem;
}

.movie-card {
  background: linear-gradient(160deg,
    rgba(255,255,255,.075),
    rgba(255,255,255,.025));
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
  transition: transform .22s, border-color .22s, box-shadow .22s;
  backdrop-filter: blur(16px);
  margin-bottom: .6rem;
}

.movie-card:hover {
  transform: translateY(-4px);
  border-color: rgba(255,71,126,.65);
  box-shadow: 0 18px 35px rgba(0,0,0,.28);
}

.poster {
  width: 100%;
  aspect-ratio: 2 / 3;
  object-fit: cover;
  display: block;
  background: #171d30;
}

.poster-fallback {
  width: 100%;
  aspect-ratio: 2 / 3;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 1rem;
  box-sizing: border-box;
  background:
    radial-gradient(circle at 25% 20%, rgba(139,92,246,.55), transparent 40%),
    linear-gradient(145deg,#171d30,#29152b);
  color: #f7f3ff;
  font: 700 1rem 'Outfit', sans-serif;
  overflow-wrap: anywhere;
}

.card-body { padding: .8rem; }

.movie-title {
  font: 700 .98rem 'Outfit', sans-serif;
  line-height: 1.3;
  min-height: 2.4rem;
}

.movie-meta {
  color: var(--muted);
  font-size: .78rem;
  margin: .4rem 0 .65rem;
}

.score {
  display: inline-block;
  font-weight: 800;
  padding: .3rem .5rem;
  border-radius: 9px;
  font-size: .76rem;
  margin-bottom: .5rem;
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
  border-radius: 11px;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: #fff;
  font-weight: 600;
  transition: all .18s;
  width: 100%;
}

.stButton > button:hover {
  border-color: var(--pink);
  background: rgba(255,71,126,.16);
  color: white;
}

div[data-baseweb="tab-list"] {
  justify-content: center;
  gap: .45rem;
}

button[data-baseweb="tab"] {
  border-radius: 12px;
  padding: .65rem 1rem;
}

div[data-testid="stNotificationContentInfo"] {
  border-radius: 14px;
}

.footer {
  border-top: 1px solid var(--border);
  margin-top: 3rem;
  padding: 1.5rem .5rem;
  color: var(--muted);
  text-align: center;
  font-size: .85rem;
}
</style>
""", unsafe_allow_html=True)


# ----------------------- MODEL FILES ---------------------

REQUIRED = [
    "user_movie_matrix.pkl",
    "user_to_row.pkl",
    "movie_ids_cf.pkl",
    "cf_neighbor_cache.pkl",
    "movies_features.pkl",
]


@st.cache_resource
def load_model_files():
    missing = [name for name in REQUIRED if not (BASE / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing files beside app.py: " + ", ".join(missing)
        )

    matrix = joblib.load(BASE / "user_movie_matrix.pkl")
    user_map = joblib.load(BASE / "user_to_row.pkl")
    movie_ids = np.asarray(joblib.load(BASE / "movie_ids_cf.pkl"))
    neighbor_cache = joblib.load(BASE / "cf_neighbor_cache.pkl")

    # Saved with pandas.to_pickle in the notebook.
    features = pd.read_pickle(BASE / "movies_features.pkl")

    return matrix, user_map, movie_ids, neighbor_cache, features


try:
    user_movie_matrix, user_to_row, movie_ids_cf, neighbor_cache, movies = (
        load_model_files()
    )
except Exception as exc:
    st.error("CineAura could not load the saved recommendation model.")
    st.code(str(exc))
    st.info(
        "Check that all five required .pkl files are in the same GitHub "
        "folder as app.py, and that the filenames match exactly."
    )
    st.stop()


# ----------------------- MOVIE DATA ----------------------

movies = movies.copy()

if "movieId" not in movies.columns or "title" not in movies.columns:
    st.error("movies_features.pkl must contain movieId and title columns.")
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

# Language is not part of the standard MovieLens movies.csv.
if "language" not in movies.columns:
    movies["language"] = "Unknown"

movies["language"] = movies["language"].fillna("Unknown").astype(str)

# Include saved MovieLens rating summaries when the dataset is available.
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
                    (x for x in z.namelist() if x.endswith("ratings.csv")),
                    None,
                )
                if not member:
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
movies["community_rating"] = movies["movieId"].map(
    lambda mid: rating_summary.get(int(mid), {}).get("mean", np.nan)
)
movies["rating_count"] = movies["movieId"].map(
    lambda mid: rating_summary.get(int(mid), {}).get("count", 0)
)

movie_by_id = movies.set_index("movieId", drop=False)


# ----------------------- POSTERS -------------------------

# Optional secret in Streamlit Cloud:
# App settings -> Secrets:
# TMDB_API_KEY = "your_api_key"
try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")


@st.cache_data(ttl=86400, show_spinner=False)
def get_poster(title, year, api_key):
    if not api_key:
        return None

    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": clean_title,
                "year": str(year) if str(year).isdigit() else "",
            },
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return None

        path = results[0].get("poster_path")
        if path:
            return "https://image.tmdb.org/t/p/w342" + path

    except Exception:
        return None

    return None


def show_poster(title, year):
    url = get_poster(str(title), str(year), TMDB_API_KEY)
    if url:
        st.markdown(
            f'<img class="poster" src="{url}" alt="Movie poster">',
            unsafe_allow_html=True,
        )
    else:
        # No fake poster or repeated popcorn emoji.
        st.markdown(
            '<div class="poster-fallback">'
            '<span style="font-size:2rem">🎬</span>'
            '<span>Poster unavailable</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# ----------------------- SESSION STATE -------------------

if "reactions" not in st.session_state:
    st.session_state.reactions = {}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()

if "notifications" not in st.session_state:
    st.session_state.notifications = [
        "Welcome to CineAura — discover a new favourite today."
    ]

if "last_added" not in st.session_state:
    st.session_state.last_added = None


def react(movie_id, action):
    movie_id = int(movie_id)
    title = movie_by_id.loc[movie_id, "title"]

    if action == "❤️ Watchlist":
        if movie_id in st.session_state.watchlist:
            st.session_state.watchlist.remove(movie_id)
            st.session_state.notifications.insert(
                0, f"Removed from Watchlist: {title}"
            )
        else:
            st.session_state.watchlist.add(movie_id)
            st.session_state.notifications.insert(
                0, f"Added to Watchlist: {title}"
            )
            st.session_state.last_added = title

    else:
        current = st.session_state.reactions.get(movie_id)
        if current == action:
            st.session_state.reactions.pop(movie_id, None)
            st.session_state.notifications.insert(
                0, f"Reaction cleared: {title}"
            )
        else:
            st.session_state.reactions[movie_id] = action
            st.session_state.notifications.insert(
                0, f"{action} — {title}"
            )

    st.session_state.notifications = st.session_state.notifications[:8]


# ----------------------- RECOMMENDER ---------------------

def recommend_for_user(user_id, top_n=20):
    """Rank unseen movies using the saved item-neighbour cache."""
    if user_id not in user_to_row:
        return pd.DataFrame()

    row = user_movie_matrix.getrow(user_to_row[user_id])
    scores = np.zeros(len(movie_ids_cf), dtype=float)

    for col, rating in zip(row.indices, row.data):
        if float(rating) < 4.0:
            continue

        neighbors = neighbor_cache.get(int(col), [])

        for neighbor_col, similarity in neighbors:
            neighbor_col = int(neighbor_col)
            similarity = float(similarity)

            if 0 <= neighbor_col < len(scores) and similarity > 0:
                scores[neighbor_col] += (float(rating) - 3.0) * similarity

    seen = set(int(x) for x in row.indices)
    ranked = sorted(
        [
            (int(movie_ids_cf[col]), float(scores[col]))
            for col in range(len(movie_ids_cf))
            if col not in seen and scores[col] > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    if not ranked:
        return pd.DataFrame()

    result = pd.DataFrame(ranked, columns=["movieId", "cf_score"])
    return result.merge(
        movies,
        on="movieId",
        how="left",
    ).drop_duplicates("movieId")


def get_user_rated_movies(user_id, minimum_rating=4.0):
    if user_id not in user_to_row:
        return []

    row = user_movie_matrix.getrow(user_to_row[user_id])
    result = []

    for col, rating in zip(row.indices, row.data):
        if float(rating) >= minimum_rating:
            movie_id = int(movie_ids_cf[col])
            if movie_id in movie_by_id.index:
                result.append(movie_id)

    return result


# ----------------------- BRAND / HERO --------------------

st.markdown("""
<div class="hero">
  <div class="brand">CINEAURA · YOUR PERSONAL CINEMA</div>
  <div class="hero-title">Find your next <span style="color:#ff719b">favourite.</span></div>
  <div class="hero-sub">
    Discover movies matched to your taste, explore genres, and save
    the stories you want to watch next.
  </div>
</div>
""", unsafe_allow_html=True)

# No sidebar: user selection and navigation stay centered.
available_users = sorted(int(uid) for uid in user_to_row.keys())

if not available_users:
    st.error("No users were found in the saved model.")
    st.stop()

top_center = st.columns([1, 2, 1])[1]
with top_center:
    user_id = st.selectbox(
        "Choose your MovieLens profile",
        available_users,
        index=0,
        help="This selects a profile from the saved MovieLens model.",
    )

metric_cols = st.columns(4)
metric_data = [
    ("MOVIES IN CATALOG", f"{len(movies):,}"),
    ("RATINGS AVAILABLE", f"{sum(v.get('count', 0) for v in rating_summary.values()):,}" if rating_summary else "Dataset"),
    ("IN YOUR WATCHLIST", str(len(st.session_state.watchlist))),
    ("YOUR REACTIONS", str(len(st.session_state.reactions))),
]

for col, (label, value) in zip(metric_cols, metric_data):
    with col:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

tabs = st.tabs([
    "✨ Discover",
    "🎯 For You",
    "🎭 By Genre",
    "❤️ Your Watchlist",
    "🔔 Notifications",
])


# ----------------------- SHARED FILTERS -------------------

def get_filtered_catalog(frame, genre="All genres", language="All languages"):
    result = frame.copy()

    if genre != "All genres":
        result = result[
            result["genres"].fillna("").str.contains(
                re.escape(genre), case=False, regex=True
            )
        ]

    if language != "All languages" and "language" in result.columns:
        result = result[
            result["language"].fillna("Unknown").str.casefold()
            == language.casefold()
        ]

    return result


def render_movie_grid(frame, key_prefix, max_movies=20):
    if frame is None or frame.empty:
        st.info("No movies found for this selection. Try another filter.")
        return

    frame = frame.head(max_movies).copy()
    card_cols = st.columns(5)

    for index, (_, movie) in enumerate(frame.iterrows()):
        movie_id = int(movie["movieId"])
        title = str(movie.get("title", "Untitled movie"))
        year = str(movie.get("year", "—"))
        genres = str(movie.get("genres", "Genre not listed"))

        rating = movie.get("community_rating", np.nan)
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            rating = np.nan

        with card_cols[index % 5]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            show_poster(title, year)

            st.markdown('<div class="card-body">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="movie-title">{title}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="movie-meta">{year} · {genres}</div>',
                unsafe_allow_html=True,
            )

            if np.isfinite(rating):
                if rating >= 8:
                    score_class = "score-green"
                elif rating >= 5:
                    score_class = "score-yellow"
                else:
                    score_class = "score-red"

                st.markdown(
                    f'<span class="score {score_class}">'
                    f'★ {rating:.1f}/5 · MovieLens</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Community rating unavailable")

            current_reaction = st.session_state.reactions.get(movie_id)
            if current_reaction:
                st.caption(f"Your reaction: {current_reaction}")

            is_saved = movie_id in st.session_state.watchlist
            watch_label = "💗 Saved" if is_saved else "❤️ Watchlist"

            # One compact reaction box with four working actions.
            with st.expander("React to this movie", expanded=False):
                reaction_cols = st.columns(2)

                with reaction_cols[0]:
                    if st.button(
                        "👍 Like",
                        key=f"{key_prefix}_like_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, "👍 Like")
                        st.rerun()

                    if st.button(
                        "❤️ Watchlist",
                        key=f"{key_prefix}_watch_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, "❤️ Watchlist")
                        st.rerun()

                with reaction_cols[1]:
                    if st.button(
                        "👎 Dislike",
                        key=f"{key_prefix}_dislike_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, "👎 Dislike")
                        st.rerun()

                    if st.button(
                        "🔥 Hype",
                        key=f"{key_prefix}_hype_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, "🔥 Hype")
                        st.rerun()

                if is_saved:
                    st.caption("This movie is already in your Watchlist.")

            st.markdown("</div></div>", unsafe_allow_html=True)


# ----------------------- DISCOVER TAB --------------------

with tabs[0]:
    st.markdown('<div class="section-title">The movie lounge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Explore the catalog and find something worth watching.</div>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "Search movies",
        placeholder="Type a title, e.g. Toy Story, Matrix, Titanic...",
        key="discover_search",
    )

    genre_values = sorted({
        genre.strip()
        for genres in movies["genres"].dropna()
        for genre in str(genres).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    })

    language_values = sorted(
        set(movies["language"].fillna("Unknown").astype(str).tolist())
    )
    language_options = ["All languages"] + [
        value for value in language_values if value != "Unknown"
    ]

    filter_cols = st.columns(2)
    with filter_cols[0]:
        discover_genre = st.selectbox(
            "Choose a genre",
            ["All genres"] + genre_values,
            key="discover_genre",
        )
    with filter_cols[1]:
        discover_language = st.selectbox(
            "Choose a language",
            language_options,
            key="discover_language",
        )

    catalog = get_filtered_catalog(
        movies, discover_genre, discover_language
    )

    if search_query.strip():
        catalog = catalog[
            catalog["title"].str.contains(
                re.escape(search_query.strip()),
                case=False,
                na=False,
            )
        ]

    if "community_rating" in catalog.columns:
        catalog = catalog.sort_values(
            ["rating_count", "community_rating"],
            ascending=False,
            na_position="last",
        )

    st.markdown("### 🍿 Explore the collection")
    render_movie_grid(catalog, "discover", max_movies=20)


# ----------------------- FOR YOU TAB ---------------------

with tabs[1]:
    st.markdown('<div class="section-title">Picked for your taste</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Recommendations ranked using your saved collaborative-filtering model.</div>',
        unsafe_allow_html=True,
    )

    liked_seed_ids = get_user_rated_movies(user_id)

    if liked_seed_ids:
        seed_names = [
            str(movie_by_id.loc[mid, "title"])
            for mid in liked_seed_ids[:5]
        ]
        st.info("Your highly rated movies help shape recommendations.")
        st.caption("Profile favourites: " + " · ".join(seed_names))

    recs = recommend_for_user(user_id, top_n=50)

    if recs.empty:
        st.warning(
            "No recommendations could be generated for this profile. "
            "Try a different MovieLens user."
        )
    else:
        # Personalise the visible results by excluding explicit dislikes.
        disliked_ids = {
            mid for mid, reaction in st.session_state.reactions.items()
            if reaction == "👎 Dislike"
        }
        recs = recs[~recs["movieId"].isin(disliked_ids)]

        st.markdown("### ✨ Recommended for you")
        render_movie_grid(recs, "foryou", max_movies=20)

        # Different, useful secondary section.
        st.markdown("### 🎬 Popular community favourites")
        popular = movies.sort_values(
            ["rating_count", "community_rating"],
            ascending=False,
            na_position="last",
        )
        popular = popular[~popular["movieId"].isin(st.session_state.watchlist)]
        render_movie_grid(popular, "popular", max_movies=10)


# ----------------------- BY GENRE TAB --------------------

with tabs[2]:
    st.markdown('<div class="section-title">Choose your mood</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Pick a genre and discover movies that fit the vibe.</div>',
        unsafe_allow_html=True,
    )

    genre_values = sorted({
        genre.strip()
        for genres in movies["genres"].dropna()
        for genre in str(genres).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    })

    genre_cols = st.columns([1, 2, 1])
    with genre_cols[1]:
        selected_genre = st.selectbox(
            "What would you like to watch?",
            genre_values,
            key="genre_tab_choice",
        )

    genre_movies = movies[
        movies["genres"].str.contains(
            re.escape(selected_genre), case=False, na=False
        )
    ].copy()

    # Give each genre page a different feel using the genre's catalog.
    if selected_genre in {"Action", "Adventure", "Sci-Fi", "Thriller"}:
        st.info(f"⚡ High-energy picks for your {selected_genre} mood.")
    elif selected_genre in {"Comedy", "Romance"}:
        st.info(f"💫 Light-hearted picks for your {selected_genre} mood.")
    elif selected_genre in {"Drama", "Mystery", "Crime"}:
        st.info(f"🎭 Story-driven picks in {selected_genre}.")
    else:
        st.info(f"🌙 Explore the world of {selected_genre}.")

    genre_movies = genre_movies.sort_values(
        ["rating_count", "community_rating"],
        ascending=False,
        na_position="last",
    )
    render_movie_grid(genre_movies, "genre", max_movies=20)


# ----------------------- WATCHLIST TAB -------------------

with tabs[3]:
    st.markdown('<div class="section-title">Your Watchlist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Only the movies you saved. No recommendations here.</div>',
        unsafe_allow_html=True,
    )

    saved_ids = sorted(st.session_state.watchlist)

    if not saved_ids:
        st.info(
            "Your Watchlist is empty. Open Discover or For You, "
            "then use ❤️ Watchlist on a movie you want to save."
        )
    else:
        watchlist_movies = movies[movies["movieId"].isin(saved_ids)].copy()
        render_movie_grid(watchlist_movies, "watchlist", max_movies=100)


# ----------------------- NOTIFICATIONS TAB --------------

with tabs[4]:
    st.markdown('<div class="section-title">Your notifications</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Updates about your reactions and saved movies.</div>',
        unsafe_allow_html=True,
    )

    new_arrivals = movies.copy()
    new_arrivals = new_arrivals[new_arrivals["year"].astype(str).str.isdigit()]
    new_arrivals = new_arrivals[new_arrivals["year"].astype(int) >= 2015]
    new_arrivals = new_arrivals.sort_values("year", ascending=False)

    st.markdown("### 🆕 Newer titles in the catalog")
    if new_arrivals.empty:
        st.caption("No newer titles are available in the current catalog.")
    else:
        render_movie_grid(new_arrivals, "newtitles", max_movies=10)

    st.markdown("### 🔔 Recent activity")
    if st.session_state.notifications:
        for notification in st.session_state.notifications:
            st.info(notification, icon="🔔")
    else:
        st.caption("Your activity notifications will appear here.")


# ----------------------- FOOTER --------------------------

st.markdown("""
<div class="footer">
  <strong style="color:#ff8eaf;font-family:Outfit,sans-serif">
    CineAura
  </strong><br>
  Built by <strong>Devadharshini Murugan</strong><br>
  Discover · React · Save · Enjoy
</div>
""", unsafe_allow_html=True)
