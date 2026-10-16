
import os
import re
import pickle
import zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# CINEAURA — MOVIE DISCOVERY & RECOMMENDATION APP
# Uses saved MovieLens model files; does not retrain a model.
# =========================================================

st.set_page_config(
    page_title="CineAura | Discover Your Next Movie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path(".")
DATA_DIR = Path(".")

# ---------- Theme / card alignment ----------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    .stApp {
        background:
          radial-gradient(ellipse at 10% 0%, #20204a 0%, transparent 36%),
          radial-gradient(ellipse at 95% 10%, #351b3e 0%, transparent 28%),
          #0b0b13;
        color: #f4f2fa;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11,11,19,0.85);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 1.7rem;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 24px;
        background: linear-gradient(120deg,
          rgba(111,75,210,.24),
          rgba(225,79,145,.12),
          rgba(255,255,255,.025));
        margin-bottom: 1.2rem;
    }

    .brand {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 0;
    }

    .subtitle {
        color: #c4bed7;
        font-size: 1rem;
        margin-top: .7rem;
    }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.65rem;
        font-weight: 700;
        margin: .5rem 0 .25rem;
    }

    .muted {
        color: #aaa5bd;
        font-size: .9rem;
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid rgba(255,255,255,.1);
        background: rgba(255,255,255,.045);
        border-radius: 16px;
        min-height: 100px;
    }

    .movie-card {
        border: 1px solid rgba(255,255,255,.11);
        background: rgba(255,255,255,.045);
        border-radius: 16px;
        padding: 10px;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform .18s ease, border-color .18s ease;
    }

    .movie-card:hover {
        transform: translateY(-3px);
        border-color: rgba(194,151,255,.6);
    }

    .movie-title {
        font-weight: 700;
        line-height: 1.35;
        min-height: 2.75rem;
        max-height: 2.75rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-top: .65rem;
    }

    .movie-meta {
        color: #aaa5bd;
        font-size: .79rem;
        line-height: 1.45;
        min-height: 2.5rem;
        margin-top: .25rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.45rem;
        border: 1px solid rgba(255,255,255,.13);
        background: rgba(255,255,255,.065);
        color: #f4f2fa;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border-color: #bda0ff;
        color: #ffffff;
    }

    div[data-testid="stImage"] img {
        border-radius: 11px;
        object-fit: cover;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #8d879f;
        font-size: .82rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load saved model files ----------

@st.cache_resource(show_spinner="Loading your saved recommendation model...")
def load_pickle(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_data(show_spinner=False)
def load_movies():
    path = DATA_DIR / "movies_features.pkl"

    if not path.exists():
        return pd.DataFrame()

    movies = pd.read_pickle(path)

    if not isinstance(movies, pd.DataFrame):
        return pd.DataFrame()

    movies = movies.copy()

    if "movieId" not in movies.columns:
        movies = movies.reset_index()
        if "movieId" not in movies.columns:
            movies = movies.rename(columns={movies.columns[0]: "movieId"})

    movies["movieId"] = pd.to_numeric(
        movies["movieId"], errors="coerce"
    )
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)

    if "title" not in movies.columns:
        movies["title"] = movies["movieId"].astype(str)

    movies["title"] = movies["title"].fillna("Unknown title").astype(str)

    if "genres" not in movies.columns:
        movies["genres"] = ""

    movies["genres"] = movies["genres"].fillna("").astype(str)

    if "content" not in movies.columns:
        movies["content"] = ""

    if "tags_text" not in movies.columns:
        movies["tags_text"] = ""

    # Extract release year from title if available.
    # Do not assume every title has a release year.
    if "year" not in movies.columns:
        movies["year"] = movies["title"].str.extract(
            r"\((\d{4})\)\s*$", expand=False
        )

    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

    # Parse genres into a consistent list.
    movies["genre_list"] = movies["genres"].apply(
        lambda x: [
            g.strip()
            for g in re.split(r"\|", str(x))
            if g.strip() and g.strip().lower() != "(no genres listed)"
        ]
    )

    return movies.drop_duplicates("movieId").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ratings():
    """Read ratings from a CSV or the common MovieLens zip layout."""
    candidates = [
        DATA_DIR / "ratings.csv",
        DATA_DIR / "ml-latest-small" / "ratings.csv",
        DATA_DIR / "ml-latest-small.zip",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    rating_file = next(
                        (name for name in names if name.endswith("ratings.csv")),
                        None,
                    )
                    if rating_file is None:
                        continue
                    with archive.open(rating_file) as file:
                        ratings = pd.read_csv(file)
            else:
                ratings = pd.read_csv(path)

            required = {"movieId", "rating"}
            if not required.issubset(ratings.columns):
                continue

            ratings["movieId"] = pd.to_numeric(
                ratings["movieId"], errors="coerce"
            )
            ratings["rating"] = pd.to_numeric(
                ratings["rating"], errors="coerce"
            )
            ratings = ratings.dropna(subset=["movieId", "rating"])
            ratings["movieId"] = ratings["movieId"].astype(int)

            return ratings

        except Exception:
            continue

    return pd.DataFrame(columns=["movieId", "rating"])


movies = load_movies()
ratings = load_ratings()

if movies.empty:
    st.error(
        "Could not load movies_features.pkl. "
        "Upload this saved file to the same GitHub folder as app.py."
    )
    st.stop()


# ---------- Calculate real rating summaries ----------

if not ratings.empty:
    rating_summary = (
        ratings.groupby("movieId")["rating"]
        .agg(community_rating="mean", rating_count="count")
        .reset_index()
    )
    movies = movies.merge(rating_summary, on="movieId", how="left")
else:
    movies["community_rating"] = np.nan
    movies["rating_count"] = np.nan

movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)


# ---------- Optional TMDB enrichment ----------

def get_tmdb_key():
    try:
        return st.secrets.get("TMDB_API_KEY", "") or os.getenv("TMDB_API_KEY", "")
    except Exception:
        return os.getenv("TMDB_API_KEY", "")


TMDB_API_KEY = get_tmdb_key()


@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_lookup(title, year, api_key):
    """Find a movie's poster and original language using TMDB search."""
    if not api_key:
        return {}

    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()

    params = {
        "api_key": api_key,
        "query": clean_title,
        "include_adult": "false",
    }

    if pd.notna(year):
        params["year"] = int(year)

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return {}

        # Prefer exact title match when available.
        target = clean_title.casefold()
        best = next(
            (
                item for item in results
                if str(item.get("title", "")).casefold() == target
            ),
            results[0],
        )

        poster_path = best.get("poster_path")
        poster_url = (
            f"https://image.tmdb.org/t/p/w342{poster_path}"
            if poster_path else None
        )

        return {
            "poster_url": poster_url,
            "language": best.get("original_language"),
            "overview": best.get("overview", ""),
            "release_date": best.get("release_date", ""),
            "tmdb_id": best.get("id"),
            "vote_average": best.get("vote_average"),
            "vote_count": best.get("vote_count"),
        }

    except Exception:
        return {}


def movie_info(movie):
    info = {}
    if TMDB_API_KEY:
        info = tmdb_lookup(
            str(movie["title"]),
            movie.get("year", np.nan),
            TMDB_API_KEY,
        )
    return info


def poster_for(movie):
    info = movie_info(movie)
    return info.get("poster_url")


def display_rating(movie):
    value = movie.get("community_rating", np.nan)
    count = movie.get("rating_count", 0)

    if pd.notna(value) and count > 0:
        return f"⭐ {float(value):.1f} · {int(count):,} ratings"

    # Do not present a TMDB rating as if it came from MovieLens.
    return "⭐ No community rating"


def display_year(movie):
    year = movie.get("year", np.nan)
    return str(int(year)) if pd.notna(year) else "Year unavailable"


def genre_text(movie):
    genres = movie.get("genre_list", [])
    return " · ".join(genres[:3]) if genres else "Genre unavailable"


def render_poster(movie):
    url = poster_for(movie)

    if url:
        st.image(url, use_container_width=True)
    else:
        # Neutral placeholder; not a fake movie poster.
        st.markdown(
            """
            <div style="
                aspect-ratio:2/3;
                width:100%;
                border-radius:11px;
                background:linear-gradient(145deg,#29253c,#15131e);
                border:1px solid rgba(255,255,255,.10);
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
                color:#c4bed7;
                padding:12px;
                box-sizing:border-box;">
                <div>🎬<br><small>Poster unavailable</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Saved item-based collaborative-filtering model ----------

user_movie_matrix = load_pickle("user_movie_matrix.pkl")
user_to_row = load_pickle("user_to_row.pkl")
movie_ids_cf = load_pickle("movie_ids_cf.pkl")
neighbor_cache = load_pickle("cf_neighbor_cache.pkl")
movie_to_col_cf = load_pickle("movie_to_col_cf.pkl")


def normalise_movie_id(value):
    try:
        return int(value)
    except Exception:
        return None


def get_neighbors(movie_id):
    """
    Read neighbours from a saved cache.
    Supports common formats:
      {movieId: [(neighborId, similarity), ...]}
      {movieId: [neighborId, ...]}
      {movieId: {neighborId: similarity, ...}}
    """
    if neighbor_cache is None:
        return []

    key = normalise_movie_id(movie_id)
    if key is None:
        return []

    raw = None

    try:
        raw = neighbor_cache.get(key)
        if raw is None:
            raw = neighbor_cache.get(str(key))
    except Exception:
        return []

    if raw is None:
        return []

    items = []

    if isinstance(raw, dict):
        for neighbor_id, similarity in raw.items():
            try:
                items.append((int(neighbor_id), float(similarity)))
            except Exception:
                continue

    else:
        try:
            for item in raw:
                if isinstance(item, (tuple, list, np.ndarray)) and len(item) >= 2:
                    # Assume (movieId, similarity).
                    try:
                        items.append((int(item[0]), float(item[1])))
                    except Exception:
                        continue
                else:
                    try:
                        items.append((int(item), 1.0))
                    except Exception:
                        continue
        except Exception:
            return []

    # Ignore self-neighbours and non-positive similarities.
    return [
        (neighbor_id, similarity)
        for neighbor_id, similarity in items
        if neighbor_id != key and np.isfinite(similarity) and similarity > 0
    ]


def get_movie_text(movie):
    parts = [
        str(movie.get("genres", "")),
        str(movie.get("tags_text", "")),
        str(movie.get("content", "")),
        str(movie.get("title", "")),
    ]
    return " ".join(p for p in parts if p and p.lower() != "nan").lower()


def recommend_from_favourites(favourite_ids, excluded_ids=None, limit=10):
    """
    Recommend movies similar to the user's chosen favourites using
    the saved item-neighbour cache. If the cache cannot be used,
    use metadata similarity as a fallback.
    """
    excluded = set(excluded_ids or [])
    seeds = [int(x) for x in favourite_ids if pd.notna(x)]
    excluded.update(seeds)

    scores = defaultdict(float)
    evidence = defaultdict(int)

    # Saved collaborative-filtering item neighbours.
    for seed_id in seeds:
        for candidate_id, similarity in get_neighbors(seed_id):
            if candidate_id in excluded:
                continue
            scores[candidate_id] += float(similarity)
            evidence[candidate_id] += 1

    # Metadata fallback when the saved cache has no usable matches.
    if not scores and seeds:
        seed_rows = movies[movies["movieId"].isin(seeds)]
        seed_genres = Counter(
            genre
            for genres in seed_rows["genre_list"]
            for genre in genres
        )
        seed_text = " ".join(
            get_movie_text(row)
            for _, row in seed_rows.iterrows()
        )
        seed_terms = set(re.findall(r"[a-z]{3,}", seed_text))

        for _, candidate in movies.iterrows():
            candidate_id = int(candidate["movieId"])
            if candidate_id in excluded:
                continue

            overlap = len(seed_terms.intersection(
                set(re.findall(r"[a-z]{3,}", get_movie_text(candidate)))
            ))
            genre_score = sum(
                seed_genres.get(g, 0)
                for g in candidate["genre_list"]
            )

            score = genre_score + min(overlap, 30) * 0.03
            if score > 0:
                scores[candidate_id] = float(score)

    if not scores:
        # Popular, well-rated fallback rather than a blank page.
        pool = movies[~movies["movieId"].isin(excluded)].copy()
        pool = pool[pool["rating_count"] > 0]

        if not pool.empty:
            pool = pool.sort_values(
                ["rating_count", "community_rating"],
                ascending=[False, False],
            )
            return pool.head(limit)

        return movies[~movies["movieId"].isin(excluded)].head(limit)

    score_df = pd.DataFrame(
        [
            {
                "movieId": movie_id,
                "personal_score": score,
                "matched_favourites": evidence[movie_id],
            }
            for movie_id, score in scores.items()
        ]
    )

    result = movies.merge(score_df, on="movieId", how="inner")

    # Popularity is only a small tie-breaker, not the main score.
    result["rank_score"] = result["personal_score"]
    if result["rating_count"].max() > 0:
        popularity_bonus = (
            np.log1p(result["rating_count"])
            / np.log1p(result["rating_count"].max())
        )
        result["rank_score"] += popularity_bonus * 0.05

    result = result.sort_values(
        ["rank_score", "rating_count"],
        ascending=[False, False],
    )

    return result.head(limit)


# ---------- Session state ----------

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "liked_movies" not in st.session_state:
    st.session_state.liked_movies = []

if "disliked_movies" not in st.session_state:
    st.session_state.disliked_movies = []

if "hyped_movies" not in st.session_state:
    st.session_state.hyped_movies = []

if "favourite_movies" not in st.session_state:
    st.session_state.favourite_movies = []

if "notifications" not in st.session_state:
    st.session_state.notifications = []

if "last_added" not in st.session_state:
    st.session_state.last_added = None


def notify(message):
    st.session_state.notifications.insert(0, message)
    st.session_state.notifications = st.session_state.notifications[:30]


def toggle_id(state_key, movie_id, title, label):
    current = list(st.session_state[state_key])

    if movie_id in current:
        current.remove(movie_id)
        notify(f"Removed “{title}” from {label}.")
    else:
        current.append(movie_id)
        notify(f"“{title}” added to {label}.")

    st.session_state[state_key] = current


def find_movie(movie_id):
    result = movies[movies["movieId"] == int(movie_id)]
    return None if result.empty else result.iloc[0]


# ---------- Header ----------

st.markdown(
    """
    <div class="hero">
        <div class="brand">CineAura ✦</div>
        <div class="subtitle">
            Discover your next favourite. Explore by taste, genre, and mood.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# User ID is intentionally not shown on the home page.
# Personalisation is based on movies the user chooses.


# ---------- Useful catalogue data ----------

valid_years = sorted(
    movies["year"].dropna().astype(int).unique().tolist()
)
all_genres = sorted({
    genre
    for genre_list in movies["genre_list"]
    for genre in genre_list
})

# Language metadata is only added when actual TMDB data is available.
# Do not fabricate a language column for MovieLens.


# ---------- Tabs ----------

discover_tab, for_you_tab, genre_tab, watchlist_tab, notifications_tab = st.tabs(
    [
        "✨ Discover",
        "🪄 For You",
        "🎭 By Genre",
        "❤️ Your Watchlist",
        "🔔 Notifications",
    ]
)


# =========================================================
# DISCOVER TAB
# =========================================================

with discover_tab:
    st.markdown(
        '<div class="section-title">Find something worth watching</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">Filter the catalogue and explore real MovieLens titles.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns([1.4, 1.2, 1.2])

    with f1:
        discover_genre = st.selectbox(
            "Genre",
            ["All genres"] + all_genres,
            key="discover_genre",
        )

    with f2:
        year_options = ["Any year"] + [str(y) for y in reversed(valid_years)]
        discover_year = st.selectbox(
            "Release year",
            year_options,
            key="discover_year",
        )

    with f3:
        discover_sort = st.selectbox(
            "Sort by",
            [
                "Popular",
                "Highest rated",
                "Newest first",
                "Title A–Z",
            ],
            key="discover_sort",
        )

    catalogue = movies.copy()

    if discover_genre != "All genres":
        catalogue = catalogue[
            catalogue["genre_list"].apply(
                lambda genres: discover_genre in genres
            )
        ]

    if discover_year != "Any year":
        catalogue = catalogue[
            catalogue["year"] == int(discover_year)
        ]

    if discover_sort == "Popular":
        catalogue = catalogue.sort_values(
            ["rating_count", "community_rating"],
            ascending=[False, False],
        )
    elif discover_sort == "Highest rated":
        # Avoid presenting a movie with only one rating as a reliable top pick.
        catalogue = catalogue[catalogue["rating_count"] >= 10]
        catalogue = catalogue.sort_values(
            ["community_rating", "rating_count"],
            ascending=[False, False],
        )
    elif discover_sort == "Newest first":
        catalogue = catalogue.sort_values(
            ["year", "rating_count"],
            ascending=[False, False],
            na_position="last",
        )
    else:
        catalogue = catalogue.sort_values("title")

    page_size = st.slider(
        "Movies to show",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="discover_count",
    )

    st.caption(f"{len(catalogue):,} movies match your filters.")

    # Limit visible cards; keep layout stable.
    display_movies = catalogue.head(page_size)

    if display_movies.empty:
        st.info("No movies match these filters. Try another genre or year.")
    else:
        card_cols = st.columns(5)

        for index, (_, movie) in enumerate(display_movies.iterrows()):
            with card_cols[index % 5]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                render_poster(movie)

                st.markdown(
                    f'<div class="movie-title">{movie["title"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="movie-meta">{display_year(movie)}<br>'
                    f'{genre_text(movie)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(display_rating(movie))

                movie_id = int(movie["movieId"])
                title = str(movie["title"])

                with st.expander("Movie actions"):
                    if st.button(
                        "❤️ Watchlist",
                        key=f"discover_watch_{movie_id}",
                        use_container_width=True,
                    ):
                        toggle_id("watchlist", movie_id, title, "Watchlist")
                        st.rerun()

                    if st.button(
                        "👍 Like",
                        key=f"discover_like_{movie_id}",
                        use_container_width=True,
                    ):
                        toggle_id("liked_movies", movie_id, title, "Liked movies")
                        st.rerun()

                    if st.button(
                        "👎 Dislike",
                        key=f"discover_dislike_{movie_id}",
                        use_container_width=True,
                    ):
                        toggle_id("disliked_movies", movie_id, title, "Disliked movies")
                        st.rerun()

                    if st.button(
                        "🔥 Hype",
                        key=f"discover_hype_{movie_id}",
                        use_container_width=True,
                    ):
                        toggle_id("hyped_movies", movie_id, title, "Hyped movies")
                        st.rerun()

                    details = movie_info(movie)
                    if details.get("overview"):
                        st.write(details["overview"])
                    else:
                        st.caption(
                            "Movie overview is unavailable in the current metadata."
                        )

                st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOR YOU TAB
# =========================================================

with for_you_tab:
    st.markdown(
        '<div class="section-title">Your personal movie mix</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Choose a few movies you love. CineAura will use the saved item-based "
        "recommendation model to find related titles.",
    )

    # Popular titles help the user choose seed movies.
    seed_pool = movies.copy()
    seed_pool = seed_pool.sort_values(
        ["rating_count", "community_rating"],
        ascending=[False, False],
    )

    seed_titles = {
        f'{row["title"]}': int(row["movieId"])
        for _, row in seed_pool.head(1500).iterrows()
    }

    selected_titles = st.multiselect(
        "Pick movies you already like",
        options=list(seed_titles.keys()),
        default=[
            title
            for title, movie_id in seed_titles.items()
            if movie_id in st.session_state.favourite_movies
        ][:10],
        max_selections=10,
        help="Select up to 10 titles to personalise recommendations.",
        key="favourite_movie_selector",
    )

    selected_ids = [seed_titles[title] for title in selected_titles]

    slider_count = st.slider(
        "How many recommendations do you want?",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="recommendation_count",
    )

    if st.button(
        "✨ Build My Movie Mix",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.favourite_movies = selected_ids
        st.session_state.last_added = len(selected_ids)

        if selected_ids:
            notify(
                f"Updated your taste profile using {len(selected_ids)} favourite movies."
            )
        else:
            notify("Choose at least one favourite movie to personalise recommendations.")

        st.rerun()

    chosen = st.session_state.favourite_movies

    if chosen:
        st.caption(f"Your taste profile uses {len(chosen)} selected favourite movies.")

        disliked = set(st.session_state.disliked_movies)
        recommendations = recommend_from_favourites(
            chosen,
            excluded_ids=set(chosen) | disliked,
            limit=slider_count,
        )

        if recommendations.empty:
            st.info("No recommendations found. Try selecting different favourites.")
        else:
            st.markdown("### Recommended for your taste")

            rec_cols = st.columns(5)
            for index, (_, movie) in enumerate(recommendations.iterrows()):
                with rec_cols[index % 5]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    render_poster(movie)
                    st.markdown(
                        f'<div class="movie-title">{movie["title"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="movie-meta">{display_year(movie)}<br>'
                        f'{genre_text(movie)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(display_rating(movie))

                    movie_id = int(movie["movieId"])
                    title = str(movie["title"])

                    with st.expander("Actions"):
                        if st.button(
                            "❤️ Watchlist",
                            key=f"foryou_watch_{movie_id}",
                            use_container_width=True,
                        ):
                            toggle_id("watchlist", movie_id, title, "Watchlist")
                            st.rerun()

                        if st.button(
                            "👍 Like",
                            key=f"foryou_like_{movie_id}",
                            use_container_width=True,
                        ):
                            toggle_id("liked_movies", movie_id, title, "Liked movies")
                            st.rerun()

                        if st.button(
                            "🔥 Hype",
                            key=f"foryou_hype_{movie_id}",
                            use_container_width=True,
                        ):
                            toggle_id("hyped_movies", movie_id, title, "Hyped movies")
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Your taste snapshot")

        genre_counts = Counter()
        for movie_id in chosen:
            row = find_movie(movie_id)
            if row is not None:
                genre_counts.update(row["genre_list"])

        if genre_counts:
            top_genres = genre_counts.most_common(5)
            st.write(
                "Genres in your selected favourites: "
                + ", ".join(f"{genre} ({count})" for genre, count in top_genres)
            )

        if ratings.empty:
            st.caption("Rating-based personal insights need ratings.csv.")
        else:
            fav_rows = movies[movies["movieId"].isin(chosen)]
            rated = fav_rows[fav_rows["community_rating"].notna()]
            if not rated.empty:
                st.metric(
                    "Average community rating of your favourites",
                    f'{rated["community_rating"].mean():.2f} / 5',
                )
    else:
        st.info(
            "Start by selecting up to 10 favourite movies above, then press "
            "'Build My Movie Mix'. Your recommendations will appear here."
        )


# =========================================================
# BY GENRE TAB
# =========================================================

with genre_tab:
    st.markdown(
        '<div class="section-title">Explore by genre</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Pick a genre and discover well-rated, popular movies in that category."
    )

    if not all_genres:
        st.info("No genre metadata was found in movies_features.pkl.")
    else:
        selected_genre = st.selectbox(
            "Choose your genre",
            all_genres,
            key="genre_tab_selection",
        )

        genre_min_ratings = st.slider(
            "Minimum community ratings",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            key="genre_min_ratings",
        )

        genre_count = st.slider(
            "Number of genre picks",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            key="genre_result_count",
        )

        genre_movies = movies[
            movies["genre_list"].apply(
                lambda genre_list: selected_genre in genre_list
            )
        ].copy()

        genre_movies = genre_movies[
            genre_movies["rating_count"] >= genre_min_ratings
        ]

        genre_movies = genre_movies.sort_values(
            ["community_rating", "rating_count"],
            ascending=[False, False],
        )

        st.caption(
            f'{len(genre_movies):,} titles available for "{selected_genre}".'
        )

        if genre_movies.empty:
            st.info(
                "No movies match this genre and rating-count threshold. "
                "Lower the minimum rating count."
            )
        else:
            genre_cols = st.columns(5)
            for index, (_, movie) in enumerate(
                genre_movies.head(genre_count).iterrows()
            ):
                with genre_cols[index % 5]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    render_poster(movie)
                    st.markdown(
                        f'<div class="movie-title">{movie["title"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="movie-meta">{display_year(movie)}<br>'
                        f'{genre_text(movie)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(display_rating(movie))

                    movie_id = int(movie["movieId"])
                    title = str(movie["title"])

                    if st.button(
                        "❤️ Save",
                        key=f"genre_save_{movie_id}",
                        use_container_width=True,
                    ):
                        toggle_id("watchlist", movie_id, title, "Watchlist")
                        st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# WATCHLIST TAB
# =========================================================

with watchlist_tab:
    st.markdown(
        '<div class="section-title">Your Watchlist</div>',
        unsafe_allow_html=True,
    )

    saved_ids = list(dict.fromkeys(st.session_state.watchlist))
    st.metric("Saved movies", len(saved_ids))

    if not saved_ids:
        st.info(
            "Your Watchlist is empty. Go to Discover, open a movie's "
            "'Movie actions', and press '❤️ Watchlist'. It will appear here."
        )
    else:
        saved_movies = movies[movies["movieId"].isin(saved_ids)].copy()
        saved_movies["_order"] = saved_movies["movieId"].apply(
            lambda movie_id: saved_ids.index(int(movie_id))
        )
        saved_movies = saved_movies.sort_values("_order")

        watch_cols = st.columns(5)

        for index, (_, movie) in enumerate(saved_movies.iterrows()):
            with watch_cols[index % 5]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                render_poster(movie)
                st.markdown(
                    f'<div class="movie-title">{movie["title"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="movie-meta">{display_year(movie)}<br>'
                    f'{genre_text(movie)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(display_rating(movie))

                movie_id = int(movie["movieId"])
                title = str(movie["title"])

                if st.button(
                    "Remove from Watchlist",
                    key=f"remove_watch_{movie_id}",
                    use_container_width=True,
                ):
                    toggle_id("watchlist", movie_id, title, "Watchlist")
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Clear Watchlist", key="clear_watchlist"):
            st.session_state.watchlist = []
            notify("Your Watchlist was cleared.")
            st.rerun()

    st.caption(
        "Note: Watchlist data is stored in the current Streamlit session. "
        "It may reset when the session ends or the app restarts."
    )


# =========================================================
# NOTIFICATIONS TAB
# =========================================================

with notifications_tab:
    st.markdown(
        '<div class="section-title">Your CineAura activity</div>',
        unsafe_allow_html=True,
    )

    if not movies.empty:
        known_years = movies["year"].dropna()
        if not known_years.empty:
            newest_year = int(known_years.max())
            newest_movies = movies[movies["year"] == newest_year]

            st.markdown("### 🆕 Newest release year in your catalogue")
            st.write(
                f"The newest release year found in the loaded catalogue is "
                f"**{newest_year}**. This is based on the dataset, not on today's "
                f"cinema releases."
            )

            st.caption(
                f"{len(newest_movies):,} catalogue titles have this release year."
            )

            new_count = min(10, len(newest_movies))
            newest_movies = newest_movies.sort_values(
                ["rating_count", "community_rating"],
                ascending=[False, False],
            ).head(new_count)

            new_cols = st.columns(5)
            for index, (_, movie) in enumerate(newest_movies.iterrows()):
                with new_cols[index % 5]:
                    render_poster(movie)
                    st.markdown(f"**{movie['title']}**")
                    st.caption(display_rating(movie))

                    movie_id = int(movie["movieId"])
                    title = str(movie["title"])

                    if st.button(
                        "❤️ Save",
                        key=f"new_save_{movie_id}",
                        use_container_width=True,
                    ):
                        toggle_id("watchlist", movie_id, title, "Watchlist")
                        st.rerun()

    st.divider()
    st.markdown("### 🔔 Recent activity")

    if not st.session_state.notifications:
        st.info(
            "No activity yet. Like a movie, save it to your Watchlist, "
            "or build your personalised Movie Mix to see activity here."
        )
    else:
        for message in st.session_state.notifications:
            st.write(f"• {message}")

        if st.button("Clear activity notifications"):
            st.session_state.notifications = []
            st.rerun()

    st.divider()
    st.markdown("### Your activity summary")

    a, b, c, d = st.columns(4)
    a.metric("❤️ Watchlist", len(st.session_state.watchlist))
    b.metric("👍 Liked", len(st.session_state.liked_movies))
    c.metric("👎 Disliked", len(st.session_state.disliked_movies))
    d.metric("🔥 Hyped", len(st.session_state.hyped_movies))


# ---------- Footer ----------

st.markdown(
    """
    <div class="footer">
        CineAura · Built by Devadharshini Murugan
        <br>
        Movie recommendations for discovery, not movie streaming.
    </div>
    """,
    unsafe_allow_html=True,
)

if TMDB_API_KEY:
    st.markdown(
        """
        <div class="footer">
            Movie poster and metadata information provided using TMDB.
            This product uses the TMDB API but is not endorsed or certified by TMDB.
        </div>
        """,
        unsafe_allow_html=True,
    )
