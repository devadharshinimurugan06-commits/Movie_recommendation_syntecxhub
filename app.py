    for col, rating in zip(row.indices, row.data):
        if rating < 4: continue
        for neighbor, similarity in neighbor_cache.get(int(col), []):
            if similarity > 0:
                scores[int(neighbor)] += (float(rating)-3.0)*float(similarity)
    seen = set(map(int, row.indices))
    candidates = [i for i in range(len(movie_ids_cf)) if i not in seen]
    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:n]
    recs = pd.DataFrame({"movieId":[int(movie_ids_cf[i]) for i in ranked],
                         "Model score":[float(scores[i]) for i in ranked]})
    return recs.merge(movies, on="movieId", how="left")

def recommend_similar(mid, n=10):
    locations = {int(movie_id): i for i, movie_id in enumerate(movie_ids_cf)}
    if int(mid) not in locations: return pd.DataFrame()
    col = locations[int(mid)]
    neighbors = sorted(neighbor_cache.get(col, []), key=lambda x: x[1], reverse=True)[:n]
    recs = pd.DataFrame({"movieId":[int(movie_ids_cf[int(i)]) for i, _ in neighbors],
                         "Similarity":[float(s) for _, s in neighbors]})
    return recs.merge(movies, on="movieId", how="left")

def render_cards(df, context, limit=5):
    if df is None or df.empty:
        st.info("No movies to show here yet."); return
    df = df.head(limit).reset_index(drop=True)
    cols = st.columns(min(5, len(df)))
    for i, (_, m) in enumerate(df.iterrows()):
        title = str(m.get("title", "Untitled movie"))
        mid = int(m["movieId"])
        year = str(m.get("year", "—"))
        genres = [g for g in str(m.get("genres","")).split("|") if g and g != "(no genres listed)"]
        avg, count = rating_info(mid)
        with cols[i]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.markdown(poster_markup(title, year, get_poster(title)), unsafe_allow_html=True)
            safe_title = title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(f'<div class="card-body"><div class="movie-title">{safe_title}</div><div class="movie-meta">{year} · {", ".join(genres[:2])}</div>', unsafe_allow_html=True)
            if avg is not None:
                st.markdown(f'<span class="score {rating_css(avg)}">★ {avg:.1f}/5 MovieLens</span><div class="small-note">{count:,} community ratings · not IMDb</div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="score score-yellow">★ Rating unavailable</span></div></div>', unsafe_allow_html=True)
            lk=f"like_{context}_{mid}"; dk=f"dislike_{context}_{mid}"; wk=f"watch_{context}_{mid}"; hk=f"hype_{context}_{mid}"
            a,b=st.columns(2)
            with a:
                if st.button("👍 Liked" if st.session_state.get(lk) else "👍 Like", key=lk+"_btn", use_container_width=True):
                    st.session_state[lk]=not st.session_state.get(lk,False)
                    if st.session_state[lk]: st.session_state[dk]=False
                    st.rerun()
                if st.button("❤️ Saved" if st.session_state.get(wk) else "❤️ Watchlist", key=wk+"_btn", use_container_width=True):
                    st.session_state[wk]=not st.session_state.get(wk,False); st.rerun()
            with b:
                if st.button("👎 Disliked" if st.session_state.get(dk) else "👎 Dislike", key=dk+"_btn", use_container_width=True):
                    st.session_state[dk]=not st.session_state.get(dk,False)
                    if st.session_state[dk]: st.session_state[lk]=False
                    st.rerun()
                if st.button("🔥 Hyped!" if st.session_state.get(hk) else "🔥 Hype", key=hk+"_btn", use_container_width=True):
                    st.session_state[hk]=not st.session_state.get(hk,False); st.rerun()

with st.sidebar:
    st.markdown("## 🍿 CineMatch")
    st.caption("Your next favourite is waiting.")
    page=st.radio("Explore",["Discover","For You","By Genre","My Watchlist"])
    st.markdown("---")
    st.markdown("### Your profile")
    selected_user=st.selectbox("MovieLens user ID", sorted(user_to_row.keys()))
    st.caption("Recommendations use your trained collaborative-filtering model.")

st.markdown("""
<div class="cine-hero"><div class="eyebrow">YOUR PERSONAL CINEMA</div>
<div class="hero-title">Find your next <span style="color:#ff6f9a">obsession.</span> 🍿</div>
<div class="hero-sub">Explore films, discover hidden gems, and get recommendations powered by your trained MovieLens model.</div></div>
""", unsafe_allow_html=True)

a,b,c=st.columns(3)
for col,label,value in [(a,"MOVIES IN CATALOG",f"{len(movies):,}"),
                        (b,"RATINGS AVAILABLE",f"{sum(x['count'] for x in rating_summary.values()):,}" if rating_summary else "Not loaded"),
                        (c,"RECOMMENDER","Collaborative filtering")]:
    with col:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)

st.markdown("### 🔎 Search your next movie")
query=st.text_input("Search movie titles",placeholder="Type a title, e.g. Inception, Toy Story, The Matrix...",label_visibility="collapsed")
if query.strip():
    found=movies[movies["title"].str.contains(query.strip(),case=False,regex=False,na=False)].head(15)
    if found.empty: st.info("No matching title. Try another search.")
    else:
        choices={f"{r['title']} · {r['year']}":int(r["movieId"]) for _,r in found.iterrows()}
        label=st.selectbox("Choose a movie",list(choices.keys()))
        mid=choices[label]; selected=movies[movies.movieId==mid].iloc[0]
        l,r=st.columns([1,3])
        with l: st.markdown(poster_markup(selected["title"],selected["year"],get_poster(selected["title"])),unsafe_allow_html=True)
        with r:
            st.subheader(selected["title"]); st.write("Genres:",selected["genres"])
            avg,count=rating_info(mid)
            if avg is not None: st.markdown(f'<span class="score {rating_css(avg)}">★ {avg:.1f}/5 MovieLens</span> <span class="small-note">({count:,} ratings)</span>',unsafe_allow_html=True)
            if st.button("✨ Find similar movies",key="find_similar"): st.session_state["similar_mid"]=mid
        if st.session_state.get("similar_mid")==mid:
            st.markdown(f"#### 🎯 Because you searched for {selected['title']}")
            render_cards(recommend_similar(mid,10),"similar",5)

st.markdown("---")
if page=="Discover":
    genres=sorted({g for s in movies.genres for g in str(s).split("|") if g and g!="(no genres listed)"})
    genre=st.selectbox("Explore a category",genres,key="discover_genre")
    df=movies[movies.genres.str.contains(genre,case=False,regex=False,na=False)].copy()
    if rating_summary:
        df["avg_rating"]=df.movieId.map(lambda x:rating_summary.get(int(x),{}).get("mean",0))
        df=df.sort_values("avg_rating",ascending=False)
    st.markdown(f"## 🔥 {genre} picks"); render_cards(df,"discover",5)
    st.markdown("## ✨ Curated movie shelf"); render_cards(movies.sample(min(5,len(movies)),random_state=42),"curated",5)
elif page=="For You":
    st.markdown(f"## 💜 Made for User {selected_user}")
    st.caption("Model score is a ranking score, not a probability or IMDb rating.")
    recs=recommend_user(selected_user,10)
    render_cards(recs,"for_you",5); st.markdown("### More picks for you"); render_cards(recs.iloc[5:],"for_you_more",5)
elif page=="By Genre":
    genres=sorted({g for s in movies.genres for g in str(s).split("|") if g and g!="(no genres listed)"})
    genre=st.selectbox("Pick a genre",genres,key="browse_genre")
    df=movies[movies.genres.str.contains(genre,case=False,regex=False,na=False)].copy()
    if rating_summary:
        df["avg_rating"]=df.movieId.map(lambda x:rating_summary.get(int(x),{}).get("mean",0))
        df=df.sort_values("avg_rating",ascending=False)
    render_cards(df,"by_genre",5)
else:
    saved_ids=[]
    for key,val in st.session_state.items():
        if key.startswith("watch_") and val is True:
            try: saved_ids.append(int(key.rsplit("_",1)[1]))
            except ValueError: pass
    saved=movies[movies.movieId.isin(saved_ids)].drop_duplicates("movieId")
    st.markdown("## ❤️ Your Watchlist")
    if saved.empty: st.info("Your watchlist is empty. Tap ❤️ Watchlist on a movie card to save it.")
    else:
        render_cards(saved,"watchlist",5)
        st.dataframe(saved[["title","year","genres"]],use_container_width=True,hide_index=True)

st.markdown("---")
st.markdown('<div class="small-note">🍿 CineMatch · Streamlit + your saved model. MovieLens ratings are on a 0–5 scale, not IMDb ratings. Real posters are optional via a TMDB API key in Streamlit Secrets.</div>',unsafe_allow_html=True)

