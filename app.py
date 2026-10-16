
import os
import pickle
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineAura | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #100d1b, #171326, #101522);
    color: #f7f3ff;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}
h1, h2, h3, p, label {
    color: #f7f3ff !important;
}
.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(120deg, #37204d, #1b2847, #442344);
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 42px;
    margin-bottom: 4px;
}
.hero p {
    color: #e4d8f7 !important;
}
.movie-title {
    min-height: 3rem;
    max-height: 3rem;
    overflow: hidden;
    font-weight: 700;
    line-height: 1.4;
}
.movie-meta {
    min-height: 3rem;
    color: #c7bfd8;
    font-size: 13px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 38px;
    border: 1px solid #66517d;
}
.poster-placeholder {
    aspect-ratio: 2 / 3;
    border-radius: 12px;
    background: #29223a;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 12px;
    color: #d6c9e7;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- TMDB SETTINGS ----------------
try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")


# ---------------- LOAD FILES SAFELY ----------------
def load_pickle_or_joblib(filename):
    """Load a saved artifact without crashing the whole app."""
    path = BASE_DIR / filename

    if not path.exists():
        return None

    # Try joblib first, then ordinary pickle.
    for loader in (joblib.load, pickle.load):
        try:
            if loader == pickle.load:
                with open(path, "rb") as file:
                    return pickle.load(file)
            return joblib.load(path)
        except Exception:
            continue

    return None


@st.cache_data(show_spinner=False)
def load_movies():
    path = BASE_DIR / "movies_features.pkl"

    if not path.exists():
        return pd.DataFrame()

    try:
        movies = pd.read_pickle(path)
    except Exception:
        movies = load_pickle_or_joblib("movies_features.pkl")

    if not isinstance(movies, pd.DataFrame):
        return pd.DataFrame()

    movies = movies.copy()

    if "movieId" not in movies.columns:
        movies["movieId"] = range(1, len(movies) + 1)

    for col in ["title", "genres", "tags_text", "content"]:
        if col not in movies.columns:
            movies[col] = ""

        movies[col] = movies[col].fillna("").astype(str)

    movies["movieId"] = pd.to_numeric(
        movies["movieId"], errors="coerce"
    )
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)

    return movies.drop_duplicates("movieId").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ratings():
    csv_path = BASE_DIR / "ratings.csv"

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    zip_path = BASE_DIR / "ml-latest-small.zip"

    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as archive:
                with archive.open("ml-latest-small/ratings.csv") as file:
                    return pd.read_csv(file)
        except Exception:
            pass

    return pd.DataFrame(
        columns=["userId", "movieId", "rating", "timestamp"]
    )


movies = load_movies()
ratings = load_ratings()

if movies.empty:
    st.error(
        "movies_features.pkl missing or unreadable. "
        "Upload the correct movies_features.pkl file to your GitHub repository."
    )
    st.stop()


# Load only the neighbor cache needed for recommendations.
# The other model artifacts are not loaded here, avoiding the reported
# user_movie_matrix.pkl UnpicklingError.
neighbor_cache = load_pickle_or_joblib("cf_neighbor_cache.pkl")


# ---------------- DATA HELPERS ----------------
def extract_year(title):
    import re
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def clean_title(title):
    import re
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def movie_rating(movie_id):
    if ratings.empty or not {"movieId", "rating"}.issubset(ratings.columns):
        return None, 0

    subset = ratings[ratings["movieId"] == movie_id]

    if subset.empty:
        return None, 0

    return round(float(subset["rating"].mean()), 1), int(len(subset))


def get_movie_genres(movie):
    genres = str(movie.get("genres", ""))
    return [
        genre for genre in genres.split("|")
        if genre and genre.lower() != "(no genres listed)"
    ]


# ---------------- TMDB POSTERS ----------------
@st.cache_data(ttl=86400, show_spinner=False)
def find_poster(title, year, api_key):
    if not api_key:
        return None

    query = clean_title(title)

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": query,
                "year": year if year else None,
                "include_adult": "false",
            },
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return None

        # Search results are matched by title/year where possible.
        target = query.lower()
        chosen = None

        for item in results:
            if item.get("title", "").lower() == target:
                chosen = item
                break

        if chosen is None:
            chosen = results[0]

        poster_path = chosen.get("poster_path")

        if not poster_path:
            return None

        return "https://image.tmdb.org/t/p/w342" + poster_path

    except Exception:
        return None


def poster_for(movie):
    year = extract_year(movie["title"])
    return find_poster(movie["title"], year, TMDB_API_KEY)


# ---------------- SESSION STATE ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "reactions" not in st.session_state:
    st.session_state.reactions = {}

if "notifications" not in st.session_state:
    st.session_state.notifications = []


def add_notification(message):
    st.session_state.notifications.insert(0, message)
    st.session_state.notifications = st.session_state.notifications[:30]


def react(movie_id, title, action):
    movie_id = int(movie_id)

    if action == "❤️ Watchlist":
        if movie_id not in st.session_state.watchlist:
            st.session_state.watchlist.append(movie_id)
            add_notification(f"Added to Watchlist: {title}")
        else:
            st.session_state.watchlist.remove(movie_id)
            add_notification(f"Removed from Watchlist: {title}")

    else:
        st.session_state.reactions[movie_id] = action
        add_notification(f"{action} — {title}")

    st.rerun()


# ---------------- RECOMMENDATIONS ----------------
def genre_fallback(movie_id=None, count=20):
    """Fallback recommendations using genre overlap."""
    source = None

    if movie_id is not None:
        matches = movies[movies["movieId"] == int(movie_id)]
        if not matches.empty:
            source = matches.iloc[0]

    if source is not None:
        source_genres = set(get_movie_genres(source))

        def overlap(row):
            return len(source_genres & set(get_movie_genres(row)))

        candidates = movies[movies["movieId"] != int(movie_id)].copy()
        candidates["_score"] = candidates.apply(overlap, axis=1)
        candidates = candidates.sort_values(
            ["_score", "title"], ascending=[False, True]
        )
    else:
        candidates = movies.copy()
        candidates["_score"] = 0

    candidates = candidates[
        ~candidates["movieId"].isin(st.session_state.watchlist)
    ]

    return candidates.head(count).drop(
        columns=["_score"], errors="ignore"
    )


def get_recommendations(count=20):
    """Use the saved neighbor cache when its format is supported."""
    cache = neighbor_cache

    if isinstance(cache, dict) and not ratings.empty:
        try:
            # Use the most active user as a simple demonstration profile.
            if "userId" in ratings.columns and "rating" in ratings.columns:
                user_counts = ratings.groupby("userId")["rating"].count()
                user_id = int(user_counts.idxmax())

                liked = ratings[
                    (ratings["userId"] == user_id)
                    & (ratings["rating"] >= 4.0)
                ]["movieId"].astype(int).tolist()

                seen = set(
                    ratings[ratings["userId"] == user_id]["movieId"].astype(int)
                )

                scores = {}

                for liked_id in liked:
                    neighbors = cache.get(liked_id, [])

                    if isinstance(neighbors, dict):
                        neighbors = list(neighbors.items())

                    for neighbor in neighbors:
                        if isinstance(neighbor, (tuple, list)) and len(neighbor) >= 2:
                            candidate_id, similarity = neighbor[0], neighbor[1]
                        else:
                            continue

                        try:
                            candidate_id = int(candidate_id)
                            similarity = float(similarity)
                        except (ValueError, TypeError):
                            continue

                        if candidate_id not in seen:
                            scores[candidate_id] = (
                                scores.get(candidate_id, 0) + similarity
                            )

                if scores:
                    ordered_ids = [
                        movie_id for movie_id, score in
                        sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    ]
                    result = movies[
                        movies["movieId"].isin(ordered_ids)
                    ].copy()

                    rank = {movie_id: i for i, movie_id in enumerate(ordered_ids)}
                    result["_rank"] = result["movieId"].map(rank)
                    result = result.sort_values("_rank").drop(columns="_rank")
                    result = result[
                        ~result["movieId"].isin(st.session_state.watchlist)
                    ]

                    if not result.empty:
                        return result.head(count)

        except Exception:
            pass

    return genre_fallback(count=count)


# ---------------- MOVIE CARDS ----------------
def render_movie_grid(data, section_key="movies", columns_per_row=5):
    if data is None or data.empty:
        st.info("No movies found for this selection.")
        return

    data = data.drop_duplicates("movieId").reset_index(drop=True)

    for start in range(0, len(data), columns_per_row):
        row = data.iloc[start:start + columns_per_row]
        cols = st.columns(columns_per_row)

        for col, (_, movie) in zip(cols, row.iterrows()):
            movie_id = int(movie["movieId"])
            title = str(movie["title"])
            year = extract_year(title)
            genres = ", ".join(get_movie_genres(movie)) or "Genre unavailable"
            rating, votes = movie_rating(movie_id)

            with col:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                poster_url = poster_for(movie)

                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        f'<div class="poster-placeholder">{clean_title(title)}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="movie-title">{clean_title(title)}</div>',
                    unsafe_allow_html=True,
                )

                year_text = str(year) if year else "Year unavailable"
                rating_text = (
                    f"⭐ {rating}/5 · {votes:,} ratings"
                    if rating is not None
                    else "⭐ Rating unavailable"
                )

                st.markdown(
                    f'<div class="movie-meta">{year_text}<br>{genres}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(rating_text)

                current = st.session_state.reactions.get(movie_id)
                if current:
                    st.caption(f"Your reaction: {current}")

                c1, c2 = st.columns(2)

                with c1:
                    if st.button(
                        "👍 Like",
                        key=f"{section_key}_like_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, title, "👍 Like")

                    if st.button(
                        "❤️ Watchlist",
                        key=f"{section_key}_watch_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, title, "❤️ Watchlist")

                with c2:
                    if st.button(
                        "👎 Dislike",
                        key=f"{section_key}_dislike_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, title, "👎 Dislike")

                    if st.button(
                        "🔥 Hype",
                        key=f"{section_key}_hype_{movie_id}",
                        use_container_width=True,
                    ):
                        react(movie_id, title, "🔥 Hype")

                st.markdown("</div>", unsafe_allow_html=True)


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 CineAura</h1>
    <p>Find your next favourite movie. Discover stories that match your vibe.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SPOTLIGHT STRIP ----------------
st.subheader("✨ Spotlight Movies")
st.caption("A handpicked-looking discovery strip powered by your movie dataset.")

spotlight = movies.copy()
spotlight["_year"] = spotlight["title"].apply(extract_year)
spotlight = spotlight.sort_values(
    "_year", ascending=False, na_position="last"
).head(5)

if spotlight.empty:
    spotlight = movies.head(5)

render_movie_grid(spotlight, section_key="spotlight", columns_per_row=5)

if not TMDB_API_KEY:
    st.caption(
        "Posters are loaded automatically when TMDB_API_KEY is configured. "
        "Without the key, poster images may be unavailable."
    )


# ---------------- NAVIGATION ----------------
tabs = st.tabs([
    "🔎 Discover",
    "✨ For You",
    "🎭 By Genre",
    "❤️ Your Watchlist",
    "🔔 Notifications",
])


# ---------------- DISCOVER ----------------
with tabs[0]:
    st.subheader("Discover Movies")

    genres_available = sorted({
        genre
        for values in movies["genres"].fillna("").astype(str)
        for genre in values.split("|")
        if genre and genre.lower() != "(no genres listed)"
    })

    c1, c2 = st.columns([2, 1])

    with c1:
        search_text = st.text_input(
            "Search by movie title",
            placeholder="e.g. Toy Story, Jumanji...",
            key="discover_search",
        )

    with c2:
        selected_genre = st.selectbox(
            "Filter by genre",
            ["All genres"] + genres_available,
            key="discover_genre",
        )

    discover = movies.copy()

    if search_text.strip():
        discover = discover[
            discover["title"].str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        ]

    if selected_genre != "All genres":
        discover = discover[
            discover["genres"].str.contains(
                selected_genre, case=False, na=False, regex=False
            )
        ]

    st.caption(f"{len(discover):,} matching movies")
    render_movie_grid(discover.head(50), section_key="discover")


# ---------------- FOR YOU ----------------
with tabs[1]:
    st.subheader("✨ Recommended for You")
    st.write(
        "Recommendations use the saved neighbor cache when compatible. "
        "Otherwise, similar-genre movies are shown."
    )

    recommendations = get_recommendations(30)
    render_movie_grid(recommendations, section_key="for_you")


# ---------------- BY GENRE ----------------
with tabs[2]:
    st.subheader("Explore by Genre")

    chosen = st.selectbox(
        "Choose a genre",
        ["Select a genre"] + genres_available,
        key="genre_tab_select",
    )

    if chosen != "Select a genre":
        genre_movies = movies[
            movies["genres"].str.contains(
                chosen, case=False, na=False, regex=False
            )
        ].head(30)

        render_movie_grid(genre_movies, section_key="genre")


# ---------------- WATCHLIST ----------------
with tabs[3]:
    st.subheader("❤️ Your Watchlist")

    # Only movies explicitly saved by the user appear here.
    watchlist_ids = st.session_state.watchlist

    if not watchlist_ids:
        st.info("Your Watchlist is empty. Press ❤️ Watchlist on a movie to save it.")
    else:
        saved = movies[movies["movieId"].isin(watchlist_ids)]
        render_movie_grid(saved, section_key="watchlist")


# ---------------- NOTIFICATIONS ----------------
with tabs[4]:
    st.subheader("🔔 Notifications")

    if not st.session_state.notifications:
        st.info("No notifications yet. Your movie reactions will appear here.")
    else:
        for message in st.session_state.notifications:
            st.markdown(f"- {message}")


# ---------------- FOOTER ----------------
st.divider()
st.caption("Built by Devadharshini Murugan")
st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
