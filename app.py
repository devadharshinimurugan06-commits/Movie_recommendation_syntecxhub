# -*- coding: utf-8 -*-
import re
import html
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# CINEAURA | MOVIE RECOMMENDATION SYSTEM
# Local dataset and saved artifacts only. No external APIs.
# =========================================================

# ---------------------------------------------------------
# EMOJI / SYMBOL CONSTANTS (Unicode escapes = always render
# correctly, regardless of how the file was saved)
# ---------------------------------------------------------
CLAPPER = "\U0001F3AC"          # 🎬
SEARCH = "\U0001F50E"           # 🔎
SPARKLES = "\u2728"             # ✨
MASKS = "\U0001F3AD"            # 🎭
HEART = "\u2764\ufe0f"          # ❤️
PURPLE_HEART = "\U0001F49C"     # 💜
THUMBS_UP = "\U0001F44D"        # 👍
THUMBS_DOWN = "\U0001F44E"      # 👎
FIRE = "\U0001F525"             # 🔥
STAR = "\u2605"                 # ★
DOT = "\u00b7"                  # ·

st.set_page_config(
    page_title="CineAura | Movie Discovery",
    page_icon=CLAPPER,
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent


# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #090909, #141414 55%, #090909);
    color: #f5f5f5;
}
header[data-testid="stHeader"] {
    background: transparent;
}
section[data-testid="stSidebar"] {
    display: none;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}
h1, h2, h3, h4, p, label {
    color: #f5f5f5;
}
.hero {
    padding: 32px;
    border: 1px solid #393939;
    border-radius: 22px;
    background: linear-gradient(120deg, #222222, #101010, #292929);
    margin-bottom: 25px;
}
.hero h1 {
    font-size: clamp(34px, 4vw, 48px);
    letter-spacing: -1px;
    margin: 0;
    color: #ffffff;
    font-family: "Source Sans Pro", "Segoe UI", "Segoe UI Emoji",
                 "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
.hero p {
    color: #c8c8c8 !important;
    font-size: 16px;
    margin-top: 8px;
}
.eyebrow {
    color: #c4c4c4;
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 800;
    margin-bottom: 9px;
}
.section-heading {
    font-size: 25px;
    font-weight: 750;
    color: #ffffff;
    margin: 8px 0 4px;
}
.section-subtitle {
    color: #aaaaaa;
    margin-bottom: 18px;
}
div[data-testid="stTabs"] button {
    border-radius: 10px 10px 0 0;
    font-weight: 650;
    font-family: "Source Sans Pro", "Segoe UI", "Segoe UI Emoji",
                 "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
div[data-testid="stTabs"] button p {
    font-family: "Source Sans Pro", "Segoe UI", "Segoe UI Emoji",
                 "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #363636;
    background: #111111;
    border-radius: 15px;
}
.movie-poster {
    height: 255px;
    width: 100%;
    box-sizing: border-box;
    padding: 17px 13px;
    border-radius: 10px;
    border: 1px solid #494949;
    background: linear-gradient(150deg, #303030, #080808 75%);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
}
.poster-label {
    color: #aaaaaa;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 700;
}
.poster-title {
    color: #ffffff;
    font-size: 21px;
    line-height: 1.25;
    font-weight: 800;
    overflow-wrap: anywhere;
    display: -webkit-box;
    -webkit-line-clamp: 5;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.poster-bottom {
    color: #bdbdbd;
    font-size: 11px;
    line-height: 1.5;
    overflow: hidden;
}
.movie-title {
    height: 43px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    font-size: 14px;
    line-height: 1.45;
    font-weight: 750;
    color: #ffffff;
    margin-top: 10px;
}
.movie-meta {
    height: 58px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    font-size: 12px;
    line-height: 1.5;
    color: #b8b8b8;
}
.movie-rating {
    height: 22px;
    font-size: 13px;
    font-weight: 700;
    color: #f0f0f0;
}
div.stButton > button {
    min-height: 38px;
    padding: 4px 2px;
    font-size: 16px;
    line-height: 1.3;
    border-radius: 9px;
    border: 1px solid #404040;
    background: #1c1c1c;
    color: #ffffff;
    white-space: nowrap;
    overflow: visible;
    font-family: "Source Sans Pro", "Segoe UI", "Segoe UI Emoji",
                 "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
div.stButton > button p {
    font-size: 16px;
    line-height: 1.3;
    margin: 0;
    overflow: visible;
    font-family: "Source Sans Pro", "Segoe UI", "Segoe UI Emoji",
                 "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
div.stButton > button:hover {
    border-color: #bdbdbd;
    color: #ffffff;
    background: #303030;
}
div[data-testid="stImage"] img {
    border-radius: 10px;
}
hr {
    border-color: #333333;
}
div[data-testid="stMetric"] {
    background: #171717;
    border: 1px solid #333333;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD LOCAL FILES
# =========================================================

@st.cache_data(show_spinner=False)
def load_movies():
    path = ROOT / "movies_features.pkl"

    if not path.exists():
        return pd.DataFrame()

    df = None

    try:
        df = pd.read_pickle(path)
    except Exception:
        pass

    if not isinstance(df, pd.DataFrame):
        try:
            df = joblib.load(path)
        except Exception:
            pass

    if not isinstance(df, pd.DataFrame):
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
    csv_path = ROOT / "ratings.csv"

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if {"movieId", "rating"}.issubset(df.columns):
                return df
        except Exception:
            pass

    zip_path = ROOT / "ml-latest-small.zip"

    if zip_path.exists():
        try:
            import zipfile

            with zipfile.ZipFile(zip_path) as archive:
                matches = [
                    name for name in archive.namelist()
                    if name.endswith("ratings.csv")
                ]
                if matches:
                    with archive.open(matches[0]) as file:
                        df = pd.read_csv(file)
                    if {"movieId", "rating"}.issubset(df.columns):
                        return df
        except Exception:
            pass

    return pd.DataFrame(columns=["movieId", "rating", "userId"])


def load_artifact(filename):
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

# The app only attempts to load the saved neighbor cache.
# It deliberately does not load user_movie_matrix.pkl.
neighbor_cache = load_artifact("cf_neighbor_cache.pkl")

if movies.empty:
    st.error(
        "Could not load movies_features.pkl. "
        "Check that the correct file exists in your GitHub repository."
    )
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if not isinstance(st.session_state.get("watchlist"), list):
    st.session_state["watchlist"] = []

if not isinstance(st.session_state.get("reactions"), dict):
    st.session_state["reactions"] = {}

if not isinstance(st.session_state.get("favourites"), list):
    st.session_state["favourites"] = []

if "last_notice" not in st.session_state:
    st.session_state["last_notice"] = ""


# =========================================================
# MOVIE HELPERS
# =========================================================

def get_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def display_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def safe_text(value):
    return html.escape(str(value), quote=True)


def genres_of(movie):
    value = str(movie.get("genres", ""))

    return [
        genre for genre in value.split("|")
        if genre and genre.lower() != "(no genres listed)"
    ]


def rating_summary(movie_id):
    if ratings.empty:
        return None, 0

    if not {"movieId", "rating"}.issubset(ratings.columns):
        return None, 0

    rows = ratings[ratings["movieId"] == int(movie_id)]

    if rows.empty:
        return None, 0

    average = float(rows["rating"].mean())
    return round(average, 1), len(rows)


# =========================================================
# LOCAL BLACK-AND-WHITE POSTER DESIGN
# =========================================================

def render_movie_poster(movie, average):
    title = safe_text(display_title(movie["title"]))
    genres = safe_text(f" {DOT} ".join(genres_of(movie)) or "Genre unavailable")
    year = get_year(movie["title"])
    year_text = str(year) if year else "Movie collection"

    if average is not None:
        rating_text = f"{STAR} {average:.1f} / 5"
    else:
        rating_text = "Rating unavailable"

    st.markdown(
        f"""
        <div class="movie-poster">
            <div class="poster-label">CINEAURA {DOT} FILM COLLECTION</div>
            <div>
                <div class="poster-title">{title}</div>
                <div class="poster-bottom" style="margin-top:12px">
                    {genres}
                </div>
            </div>
            <div class="poster-bottom">
                <div style="
                    color:#ffffff;
                    font-size:16px;
                    font-weight:800;
                    margin-bottom:5px;
                ">{rating_text}</div>
                {year_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# REACTIONS — SAFE SESSION STATE
# =========================================================

def handle_reaction(movie_id, title, action):
    movie_id = int(movie_id)

    # Repair old/invalid session values before any list operation.
    current_watchlist = st.session_state.get("watchlist")

    if not isinstance(current_watchlist, list):
        current_watchlist = []

    current_watchlist = [
        int(item) for item in current_watchlist
        if str(item).isdigit()
    ]

    if not isinstance(st.session_state.get("reactions"), dict):
        st.session_state["reactions"] = {}

    if action == HEART:
        if movie_id in current_watchlist:
            current_watchlist = [
                item for item in current_watchlist
                if item != movie_id
            ]
            message = f"Removed from Watchlist: {title}"
        else:
            current_watchlist.append(movie_id)
            message = f"Added to Watchlist: {title}"

        st.session_state["watchlist"] = current_watchlist

    else:
        st.session_state["reactions"][movie_id] = action
        message = f"{action} saved: {title}"

    st.session_state["last_notice"] = message
    st.rerun()


# =========================================================
# RECOMMENDATION FUNCTIONS
# =========================================================

def genre_recommendations(seed_ids=None, count=20):
    seed_ids = [int(value) for value in (seed_ids or [])]
    seeds = movies[movies["movieId"].isin(seed_ids)]

    preferred_genres = set()

    for _, movie in seeds.iterrows():
        preferred_genres.update(genres_of(movie))

    excluded = set(seed_ids) | set(st.session_state["watchlist"])

    pool = movies[~movies["movieId"].isin(excluded)].copy()

    if preferred_genres:
        def score_genres(value):
            candidate = {
                genre for genre in str(value).split("|")
                if genre and genre.lower() != "(no genres listed)"
            }
            return len(preferred_genres & candidate)

        pool["_score"] = pool["genres"].apply(score_genres)
        pool = pool.sort_values(
            ["_score", "title"],
            ascending=[False, True],
        )
    else:
        pool["_score"] = 0
        pool = pool.sort_values("title")

    return pool.drop(columns="_score", errors="ignore").head(count)


def saved_model_recommendations(seed_ids, count=20):
    """
    Expected neighbor cache:
    {movieId: [(similar_movieId, similarity_score), ...]}
    """
    if not isinstance(neighbor_cache, dict) or not seed_ids:
        return pd.DataFrame()

    seed_ids = [int(value) for value in seed_ids]
    excluded = set(seed_ids) | set(st.session_state["watchlist"])
    scores = {}

    try:
        for seed_id in seed_ids:
            neighbors = neighbor_cache.get(seed_id, [])

            if isinstance(neighbors, dict):
                neighbors = list(neighbors.items())

            for item in neighbors:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue

                try:
                    candidate_id = int(item[0])
                    similarity = float(item[1])
                except (ValueError, TypeError):
                    continue

                if candidate_id in excluded:
                    continue

                scores[candidate_id] = (
                    scores.get(candidate_id, 0.0) + similarity
                )

        if not scores:
            return pd.DataFrame()

        ranked_ids = [
            movie_id
            for movie_id, _ in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        result = movies[movies["movieId"].isin(ranked_ids)].copy()
        rank_order = {
            movie_id: index for index, movie_id in enumerate(ranked_ids)
        }

        result["_rank"] = result["movieId"].map(rank_order)

        return (
            result.sort_values("_rank")
            .drop(columns="_rank")
            .head(count)
        )

    except Exception:
        return pd.DataFrame()


# =========================================================
# ALIGNED MOVIE CARDS + ONE REACTION ROW
# =========================================================

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
            average, vote_count = rating_summary(movie_id)

            with col:
                with st.container(border=True):
                    render_movie_poster(movie, average)

                    st.markdown(
                        f'<div class="movie-title">'
                        f'{safe_text(display_title(title))}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    year = get_year(title)
                    genre_text = safe_text(
                        ", ".join(genres_of(movie)) or "Genre unavailable"
                    )

                    st.markdown(
                        f'<div class="movie-meta">'
                        f'{year if year else "Year unavailable"}'
                        f'<br>{genre_text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if average is not None:
                        rating_label = (
                            f"{STAR} {average:.1f}/5 {DOT} "
                            f"{vote_count:,} ratings"
                        )
                    else:
                        rating_label = "Rating unavailable"

                    st.markdown(
                        f'<div class="movie-rating">'
                        f'{safe_text(rating_label)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    previous_reaction = st.session_state["reactions"].get(
                        movie_id
                    )

                    if previous_reaction:
                        st.caption(f"Your reaction: {previous_reaction}")

                    # Exactly one reaction row for each movie.
                    b1, b2, b3, b4 = st.columns(4, gap="small")

                    with b1:
                        if st.button(
                            THUMBS_UP,
                            key=f"{prefix}_like_{movie_id}",
                            help="Like this movie",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, THUMBS_UP)

                    with b2:
                        if st.button(
                            THUMBS_DOWN,
                            key=f"{prefix}_dislike_{movie_id}",
                            help="Dislike this movie",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, THUMBS_DOWN)

                    with b3:
                        is_saved = movie_id in st.session_state["watchlist"]

                        if st.button(
                            PURPLE_HEART if is_saved else HEART,
                            key=f"{prefix}_watch_{movie_id}",
                            help="Remove from Watchlist" if is_saved
                            else "Add to Watchlist",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, HEART)

                    with b4:
                        if st.button(
                            FIRE,
                            key=f"{prefix}_hype_{movie_id}",
                            help="Hype this movie",
                            use_container_width=True,
                        ):
                            handle_reaction(movie_id, title, FIRE)


# =========================================================
# HEADER
# =========================================================

st.markdown(f"""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL MOVIE UNIVERSE</div>
    <h1>{CLAPPER} CineAura</h1>
    <p>Discover stories. Find your vibe. Choose your next favourite.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SPOTLIGHT — FIVE MOVIES FROM YOUR OWN DATASET
# =========================================================

st.markdown(
    '<div class="section-heading">Spotlight</div>'
    '<div class="section-subtitle">'
    'A selection from your movie collection.'
    '</div>',
    unsafe_allow_html=True,
)

spotlight = movies.copy()
spotlight["_year"] = spotlight["title"].apply(get_year)
spotlight = (
    spotlight.sort_values("_year", ascending=False, na_position="last")
    .head(5)
    .drop(columns="_year")
)

render_grid(spotlight, prefix="spotlight", columns_per_row=5)

st.divider()


# =========================================================
# FOUR MAIN TABS — KEEPING YOUR ORIGINAL NAVIGATION
# =========================================================

discover_tab, foryou_tab, genre_tab, watchlist_tab = st.tabs([
    f"{SEARCH} Discover",
    f"{SPARKLES} For You",
    f"{MASKS} By Genre",
    f"{HEART} Your Watchlist",
])


# =========================================================
# DISCOVER
# =========================================================

all_genres = sorted({
    genre
    for value in movies["genres"].fillna("").astype(str)
    for genre in value.split("|")
    if genre and genre.lower() != "(no genres listed)"
})

with discover_tab:
    st.markdown(
        '<div class="section-heading">Discover Movies</div>'
        '<div class="section-subtitle">'
        'Search and filter movies in your dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    search_col, genre_col = st.columns([1.5, 1])

    with search_col:
        search_query = st.text_input(
            "Search by movie title",
            placeholder="Type a movie title...",
            key="discover_query",
        )

    with genre_col:
        discover_genre = st.selectbox(
            "Filter by genre",
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

    # ---- NEW: dropdown box. Click it, pick a movie, its poster shows ----
    dropdown_data = discover_data.sort_values("title").head(500)
    dropdown_titles = {
        int(row.movieId): str(row.title)
        for row in dropdown_data[["movieId", "title"]].itertuples(index=False)
    }

    picked_movie_id = st.selectbox(
        "Select a movie to view its poster",
        options=list(dropdown_titles.keys()),
        index=None,
        format_func=lambda movie_id: dropdown_titles.get(movie_id, str(movie_id)),
        placeholder="Click here and choose a movie...",
        key="discover_poster_pick",
    )

    if picked_movie_id is not None:
        picked_movie = movies[movies["movieId"] == int(picked_movie_id)]
        render_grid(picked_movie, prefix="discover_pick")

    st.caption(f"{len(discover_data):,} matching movies")
    render_grid(discover_data.head(40), prefix="discover")


# =========================================================
# FOR YOU
# =========================================================

with foryou_tab:
    st.markdown(
        '<div class="section-heading">Made for Your Taste</div>'
        '<div class="section-subtitle">'
        'Choose favourite movies to personalise recommendations.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Prefix IDs in labels to avoid duplicate titles mapping to the wrong ID.
    movie_options = {
        f"{row.title} {DOT} {int(row.movieId)}": int(row.movieId)
        for row in movies[["movieId", "title"]].itertuples(index=False)
    }

    selected_labels = st.multiselect(
        "Choose favourite movies (up to 10)",
        options=list(movie_options.keys()),
        max_selections=10,
        key="favourite_movies_select",
        placeholder="Choose movies you enjoy...",
    )

    selected_ids = [movie_options[label] for label in selected_labels]
    st.session_state["favourites"] = selected_ids

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
            selected_ids,
            number_to_show,
        )

        if result.empty:
            result = genre_recommendations(
                selected_ids,
                number_to_show,
            )
            st.caption(
                "The saved neighbour cache was unavailable or incompatible. "
                "Showing genre-based recommendations instead."
            )
        else:
            st.caption("Recommendations generated from the saved neighbour cache.")
    else:
        result = genre_recommendations(count=number_to_show)
        st.caption(
            "Select your favourite movies above to personalise the results."
        )

    render_grid(result, prefix="foryou")


# =========================================================
# BY GENRE
# =========================================================

with genre_tab:
    st.markdown(
        '<div class="section-heading">Explore by Genre</div>'
        '<div class="section-subtitle">'
        'Choose a genre to discover movies from your dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not all_genres:
        st.info("No genre information was found in the movie dataset.")
    else:
        genre_choice = st.selectbox(
            "Choose a genre",
            all_genres,
            key="genre_explorer",
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


# =========================================================
# WATCHLIST — ONLY SAVED MOVIES
# =========================================================

with watchlist_tab:
    st.markdown(
        '<div class="section-heading">Your Watchlist</div>'
        '<div class="section-subtitle">'
        'Your saved movies, all in one place.'
        '</div>',
        unsafe_allow_html=True,
    )

    saved_ids = st.session_state["watchlist"]

    if not saved_ids:
        st.info(f"Your Watchlist is empty. Press {HEART} on a movie to save it.")
    else:
        saved_movies = movies[
            movies["movieId"].isin(saved_ids)
        ].copy()

        st.caption(f"{len(saved_movies)} saved movies")
        render_grid(saved_movies, prefix="watchlist")


# =========================================================
# FOOTER — NO EXTERNAL SERVICE ATTRIBUTION
# =========================================================

st.divider()

if st.session_state["last_notice"]:
    st.caption(f"Recent activity {DOT} {st.session_state['last_notice']}")

st.markdown(
    f"""
    <div style="
        text-align:center;
        color:#aaaaaa;
        padding:12px;
        letter-spacing:.4px;
    ">
        CineAura {DOT} Built by Devadharshini Murugan
    </div>
    """,
    unsafe_allow_html=True,
)
