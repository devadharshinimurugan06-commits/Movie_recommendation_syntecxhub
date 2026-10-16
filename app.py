
import os
import re
import pickle
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st


# =========================================================
# CINEAURA | MOVIE RECOMMENDATION SYSTEM
# =========================================================

st.set_page_config(
    page_title="CineAura | Discover Your Next Movie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent

# ------------------------- THEME -------------------------

st.markdown("""
<style>
:root {
    --bg: #100e19;
    --panel: #1b1729;
    --panel2: #241d35;
    --accent: #d9a7ff;
    --accent2: #f0c4ff;
    --text: #f8f4ff;
    --muted: #bdb2ce;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #302044 0%, transparent 38%),
        linear-gradient(145deg, #100e19, #151324 55%, #101722);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, p, label {
    color: var(--text);
}

.hero {
    padding: 32px 34px;
    border: 1px solid rgba(217,167,255,.25);
    border-radius: 24px;
    background:
        linear-gradient(110deg, rgba(64,38,85,.96),
        rgba(29,36,67,.94), rgba(58,31,63,.92));
    box-shadow: 0 15px 45px rgba(0,0,0,.22);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: clamp(34px, 4vw, 49px);
    letter-spacing: -1.3px;
    margin: 0;
}

.hero p {
    color: #e3d5f2 !important;
    margin-top: 8px;
    font-size: 16px;
}

.eyebrow {
    color: #e4baff;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 9px;
}

.section-heading {
    font-size: 25px;
    font-weight: 750;
    margin: 8px 0 3px;
}

.section-subtitle {
    color: var(--muted);
    margin-bottom: 16px;
}

div[data-testid="stTabs"] button {
    border-radius: 12px 12px 0 0;
    font-weight: 650;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #edc8ff;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(217,167,255,.2);
}

.movie-title {
    height: 3.15rem;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    line-height: 1.45;
    font-size: 15px;
    font-weight: 750;
    color: #fff;
    margin-top: 10px;
}

.movie-meta {
    height: 3.15rem;
    overflow: hidden;
    color: #c5bad5;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 3px;
}

.movie-rating {
    height: 1.6rem;
    color: #f3d59c;
    font-size: 12px;
    margin-top: 5px;
}

.poster-fallback {
    aspect-ratio: 2 / 3;
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(145deg, #302440, #191b2e);
    border: 1px solid #51405f;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 14px;
    color: #d8c5e8;
    font-weight: 700;
    overflow: hidden;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 36px;
    padding: 5px 3px;
    font-size: 12px;
    white-space: nowrap;
    border: 1px solid rgba(217,167,255,.22);
    background: #211b30;
    color: #f8f4ff;
}

div.stButton > button:hover {
    border-color: #d9a7ff;
    color: #f0c4ff;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    object-fit: cover;
}

div[data-testid="stSelectbox"],
div[data-testid="stTextInput"],
div[data-testid="stMultiSelect"] {
    border-radius: 12px;
}

div[data-testid="stMetric"] {
    background: rgba(36,29,53,.72);
    padding: 13px;
    border-radius: 14px;
    border: 1px solid rgba(217,167,255,.12);
}

hr {
    border-color: rgba(217,167,255,.17);
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ------------------------- TMDB -------------------------

try:
    TMDB_KEY = st.secrets.get("TMDB_API_KEY", "")
except Exception:
    TMDB_KEY = os.getenv("TMDB_API_KEY", "")


@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_movie_info(title, year, api_key):
    """Return a poster URL and original language from TMDB."""
    if not api_key:
        return None, None

    search_title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()

    try:
        params = {
            "api_key": api_key,
            "query": search_title,
            "include_adult": "false",
        }
        if year:
            params["year"] = int(year)

        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=7,
        )
        response.raise_for_status()

        results = response.json().get("results", [])
        if not results:
            return None, None

        target = search_title.casefold()
        chosen = next(
            (
                item for item in results
                if str(item.get("title", "")).casefold() == target
            ),
            results[0],
        )

        poster_path = chosen.get("poster_path")
        poster_url = (
            "https://image.tmdb.org/t/p/w342" + poster_path
            if poster_path else None
        )

        return poster_url, chosen.get("original_language")

    except Exception:
        return None, None


# ------------------------- DATA LOADING -------------------------

@st.cache_data(show_spinner=False)
def load_movies():
    path = ROOT / "movies_features.pkl"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_pickle(path)
    except Exception:
        try:
            df = joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as file:
                    df = pickle.load(file)
            except Exception:
                return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    df = df.copy()

    if "movieId" not in df.columns:
        return pd.DataFrame()

    for column in ["title", "genres", "content", "tags_text"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
    df = df.dropna(subset=["movieId"])
    df["movieId"] = df["movieId"].astype(int)

    return df.drop_duplicates("movieId").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ratings():
    path = ROOT / "ratings.csv"

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    zip_path = ROOT / "ml-latest-small.zip"

    if zip_path.exists():
        try:
            import zipfile
            with zipfile.ZipFile(zip_path) as archive:
                candidate = next(
                    name for name in archive.namelist()
                    if name.endswith("ratings.csv")
                )
                with archive.open(candidate) as file:
                    return pd.read_csv(file)
        except Exception:
            pass

    return pd.DataFrame(columns=["movieId", "rating", "userId"])


def load_artifact(filename):
    """Try supported loaders; a broken artifact won't crash the app."""
    path = ROOT / filename

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        pass

    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception:
        return None


movies = load_movies()
ratings = load_ratings()
neighbor_cache = load_artifact("cf_neighbor_cache.pkl")

if movies.empty:
    st.error(
        "Could not load movies_features.pkl. "
        "Please check that the correct file is in your GitHub repository."
    )
    st.stop()


# ------------------------- SESSION STATE -------------------------

defaults = {
    "watchlist": [],
    "reactions": {},
    "favourites": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------- MOVIE HELPERS -------------------------

def get_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def display_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def genres_of(movie):
    genres = str(movie.get("genres", ""))
    return [
        g for g in genres.split("|")
        if g and g.lower() != "(no genres listed)"
    ]


def rating_summary(movie_id):
    if ratings.empty:
        return None, 0

    if not {"movieId", "rating"}.issubset(ratings.columns):
        return None, 0

    rows = ratings[ratings["movieId"] == int(movie_id)]

    if rows.empty:
        return None, 0

    return round(float(rows["rating"].mean()), 1), len(rows)


def get_poster(movie):
    return tmdb_movie_info(
        movie["title"],
        get_year(movie["title"]),
        TMDB_KEY,
    )[0]


def notify_action(title, action):
    st.session_state["last_notice"] = f"{action}: {title}"


def handle_reaction(movie_id, title, action):
    movie_id = int(movie_id)

    if action == "❤️":
        if movie_id in st.session_state.watchlist:
            st.session_state.watchlist.remove(movie_id)
            notify_action(title, "Removed from Watchlist")
        else:
            st.session_state.watchlist.append(movie_id)
            notify_action(title, "Added to Watchlist")

    else:
        st.session_state.reactions[movie_id] = action
        notify_action(title, f"Reaction saved ({action})")

    st.rerun()


# ------------------------- RECOMMENDER -------------------------

def genre_recommendations(seed_ids=None, count=20):
    """Fallback based on favourite movie genres."""
    seed_ids = seed_ids or []
    seeds = movies[movies["movieId"].isin(seed_ids)]

    preferred_genres = set()
    for _, movie in seeds.iterrows():
        preferred_genres.update(genres_of(movie))

    pool = movies[
        ~movies["movieId"].isin(seed_ids)
        & ~movies["movieId"].isin(st.session_state.watchlist)
    ].copy()

    if preferred_genres:
        pool["_score"] = pool["genres"].apply(
            lambda value: len(
                preferred_genres
                & set(
                    g for g in str(value).split("|")
                    if g and g.lower() != "(no genres listed)"
                )
            )
        )
        pool = pool.sort_values(
            ["_score", "title"], ascending=[False, True]
        )
    else:
        pool["_score"] = 0
        pool = pool.sort_values("title")

    return pool.drop(columns="_score", errors="ignore").head(count)


def saved_model_recommendations(seed_ids, count=20):
    """
    Expected cache format:
    {movieId: [(similar_movieId, similarity_score), ...]}
    """
    if not isinstance(neighbor_cache, dict) or not seed_ids:
        return pd.DataFrame()

    scores = {}

    try:
        for seed_id in seed_ids:
            neighbors = neighbor_cache.get(int(seed_id), [])

            if isinstance(neighbors, dict):
                neighbors = list(neighbors.items())

            for item in neighbors:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue

                candidate_id, score = item[0], item[1]

                try:
                    candidate_id = int(candidate_id)
                    score = float(score)
                except (TypeError, ValueError):
                    continue

                if candidate_id in seed_ids:
                    continue

                if candidate_id in st.session_state.watchlist:
                    continue

                scores[candidate_id] = scores.get(candidate_id, 0.0) + score

        if not scores:
            return pd.DataFrame()

        ordered_ids = [
            movie_id for movie_id, _ in
            sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        ]

        result = movies[movies["movieId"].isin(ordered_ids)].copy()
        order_map = {movie_id: i for i, movie_id in enumerate(ordered_ids)}
        result["_rank"] = result["movieId"].map(order_map)

        return result.sort_values("_rank").drop(
            columns="_rank"
        ).head(count)

    except Exception:
        return pd.DataFrame()


# ------------------------- MOVIE CARD GRID -------------------------

def render_grid(data, prefix="movies", columns_per_row=5):
    if data is None or data.empty:
        st.info("No movies found for this selection.")
        return

    data = data.drop_duplicates("movieId").reset_index(drop=True)

    for start in range(0, len(data), columns_per_row):
        chunk = data.iloc[start:start + columns_per_row]
        columns = st.columns(columns_per_row, gap="medium")

        for col, (_, movie) in zip(columns, chunk.iterrows()):
            movie_id = int(movie["movieId"])
            title = str(movie["title"])
            year = get_year(title)
            movie_genres = ", ".join(genres_of(movie)) or "Genre unavailable"
            average, vote_count = rating_summary(movie_id)

            with col:
                with st.container(border=True):
                    poster = get_poster(movie)

                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.markdown(
                            '<div class="poster-fallback">'
                            '🎬<br>Poster unavailable'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="movie-title">{display_title(title)}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="movie-meta">'
                        f'{year if year else "Year unavailable"}'
                        f'<br>{movie_genres}</div>',
                        unsafe_allow_html=True,
                    )

                    if average is not None:
                        rating_label = f"⭐ {average}/5 · {vote_count:,} ratings"
                    else:
                        rating_label = "⭐ Rating unavailable"

                    st.markdown(
                        f'<div class="movie-rating">{rating_label}</div>',
                        unsafe_allow_html=True,
                    )

                    if movie_id in st.session_state.reactions:
                        st.caption(
                            f"Your reaction: "
                            f"{st.session_state.reactions[movie_id]}"
                        )

                    # ONE compact reaction row per movie.
                    r1, r2, r3, r4 = st.columns(4, gap="small")

                    with r1:
                        if st.button(
                            "👍",
                            key=f"{prefix}_like_{movie_id}",
                            help="Like",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, "👍")

                    with r2:
                        if st.button(
                            "👎",
                            key=f"{prefix}_dislike_{movie_id}",
                            help="Dislike",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, "👎")

                    with r3:
                        saved = movie_id in st.session_state.watchlist
                        if st.button(
                            "❤️" if not saved else "💜",
                            key=f"{prefix}_watch_{movie_id}",
                            help="Add to / remove from Watchlist",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, "❤️")

                    with r4:
                        if st.button(
                            "🔥",
                            key=f"{prefix}_hype_{movie_id}",
                            help="Hype",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, "🔥")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL MOVIE UNIVERSE</div>
    <h1>🎬 CineAura</h1>
    <p>Discover stories. Find your vibe. Fall in love with your next movie.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------- SPOTLIGHT -------------------------

st.markdown(
    '<div class="section-heading">✨ Spotlight</div>'
    '<div class="section-subtitle">'
    'A quick look at movies from your catalogue.'
    '</div>',
    unsafe_allow_html=True,
)

spotlight = movies.copy()
spotlight["_year"] = spotlight["title"].apply(get_year)
spotlight = spotlight.sort_values(
    "_year", ascending=False, na_position="last"
).head(5)

render_grid(spotlight, prefix="spotlight", columns_per_row=5)

st.divider()


# =========================================================
# FOUR MAIN TABS
# =========================================================

discover_tab, foryou_tab, genre_tab, watchlist_tab = st.tabs([
    "🔎 Discover",
    "✨ For You",
    "🎭 By Genre",
    "❤️ Your Watchlist",
])


# ------------------------- DISCOVER -------------------------

with discover_tab:
    st.markdown(
        '<div class="section-heading">Discover Movies</div>'
        '<div class="section-subtitle">'
        'Search the catalogue and find something worth watching.'
        '</div>',
        unsafe_allow_html=True,
    )

    all_genres = sorted({
        genre
        for value in movies["genres"].fillna("").astype(str)
        for genre in value.split("|")
        if genre and genre.lower() != "(no genres listed)"
    })

    search_col, genre_col = st.columns([1.5, 1])

    with search_col:
        search_query = st.text_input(
            "Search movies",
            placeholder="Type a movie title...",
            key="discover_query",
        )

    with genre_col:
        discover_genre = st.selectbox(
            "Genre",
            ["All genres"] + all_genres,
            key="discover_genre",
        )

    discover_data = movies.copy()

    if search_query.strip():
        discover_data = discover_data[
            discover_data["title"].str.contains(
                search_query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if discover_genre != "All genres":
        discover_data = discover_data[
            discover_data["genres"].str.contains(
                discover_genre,
                case=False,
                na=False,
                regex=False,
            )
        ]

    st.caption(f"{len(discover_data):,} movies found")
    render_grid(discover_data.head(40), prefix="discover")


# ------------------------- FOR YOU -------------------------

with foryou_tab:
    st.markdown(
        '<div class="section-heading">Made for Your Taste ✨</div>'
        '<div class="section-subtitle">'
        'Choose favourite movies to personalise your recommendations.'
        '</div>',
        unsafe_allow_html=True,
    )

    movie_options = dict(
        zip(movies["title"], movies["movieId"].astype(int))
    )

    selected_titles = st.multiselect(
        "Your favourite movies (up to 10)",
        options=list(movie_options.keys()),
        max_selections=10,
        key="favourite_movies_select",
        placeholder="Choose movies you enjoy...",
    )

    selected_ids = [
        int(movie_options[title]) for title in selected_titles
    ]

    st.session_state.favourites = selected_ids

    number_to_show = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="recommendation_count",
    )

    if selected_ids:
        result = saved_model_recommendations(
            selected_ids, number_to_show
        )

        if result.empty:
            result = genre_recommendations(
                selected_ids, number_to_show
            )
            st.caption(
                "Showing genre-based matches because the saved neighbour "
                "cache did not return compatible recommendations."
            )
        else:
            st.caption("Recommendations generated from the saved neighbour cache.")

    else:
        result = genre_recommendations(
            count=number_to_show
        )
        st.caption(
            "Choose favourite movies above for more personalised recommendations."
        )

    render_grid(result, prefix="foryou")


# ------------------------- BY GENRE -------------------------

with genre_tab:
    st.markdown(
        '<div class="section-heading">Explore by Genre 🎭</div>'
        '<div class="section-subtitle">'
        'Pick a mood and explore movies from that category.'
        '</div>',
        unsafe_allow_html=True,
    )

    genre_choice = st.selectbox(
        "Choose your movie vibe",
        all_genres,
        key="genre_explorer",
        placeholder="Select a genre...",
    )

    genre_results = movies[
        movies["genres"].str.contains(
            genre_choice,
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    st.caption(f"{len(genre_results):,} movies in this genre")
    render_grid(genre_results.head(30), prefix="genre")


# ------------------------- WATCHLIST -------------------------

with watchlist_tab:
    st.markdown(
        '<div class="section-heading">Your Watchlist ❤️</div>'
        '<div class="section-subtitle">'
        'Only the movies you saved appear here.'
        '</div>',
        unsafe_allow_html=True,
    )

    saved_ids = st.session_state.watchlist

    if not saved_ids:
        st.info(
            "Your Watchlist is empty. Tap ❤️ on a movie card to save it here."
        )
    else:
        saved_movies = movies[
            movies["movieId"].isin(saved_ids)
        ].copy()

        st.caption(f"{len(saved_movies)} saved movies")
        render_grid(saved_movies, prefix="watchlist")


# ------------------------- FOOTER -------------------------

st.divider()

last_notice = st.session_state.get("last_notice")
if last_notice:
    st.caption(f"Recent activity · {last_notice}")

st.markdown(
    "<div style='text-align:center;color:#bdb2ce;padding:10px'>"
    "CineAura · Built by Devadharshini Murugan"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
