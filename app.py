import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.movie-card {
    background-color: #1b1f2a;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    min-height: 300px;
}

.movie-title {
    font-size: 22px;
    font-weight: bold;
}

.rating {
    color: #ffd700;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies you may enjoy based on genres and story similarity."
)

# -----------------------------
# Load Dataset
# -----------------------------
movies = pd.read_csv("movies.csv")

# -----------------------------
# Create Features
# -----------------------------
movies["features"] = (
    movies["genre"].fillna("") + " " +
    movies["overview"].fillna("")
)

# -----------------------------
# TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    movies["features"]
)

# -----------------------------
# Cosine Similarity
# -----------------------------
similarity = cosine_similarity(tfidf_matrix)

# -----------------------------
# Movie Selection
# -----------------------------
st.subheader("🎥 Choose a Movie")

movie_name = st.selectbox(
    "Select a movie",
    movies["title"].tolist()
)

# -----------------------------
# Movie Details
# -----------------------------
selected_movie = movies[
    movies["title"] == movie_name
].iloc[0]

st.markdown("### 📖 Selected Movie")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(
        "<div style='font-size:100px;text-align:center;'>🎞️</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"## {selected_movie['title']}"
    )

    st.write(
        f"⭐ **Rating:** {selected_movie['rating']}/10"
    )

    st.write(
        f"🎭 **Genre:** {selected_movie['genre']}"
    )

    st.write(
        f"📅 **Year:** {selected_movie['year']}"
    )

    st.write(
        selected_movie["overview"]
    )

# -----------------------------
# Recommendation Button
# -----------------------------
if st.button(
    "🎯 Recommend Movies",
    use_container_width=True
):

    movie_index = movies[
        movies["title"] == movie_name
    ].index[0]

    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = similarity_scores[1:7]

    st.subheader("🍿 You May Also Like")

    # Create 3 columns
    cols = st.columns(3)

    for i, (index, score) in enumerate(recommendations):

        movie = movies.iloc[index]
        with cols[i % 3]:

            poster_file = (
               f"posters/"
               f"{movie['title'].lower().replace(' ', '_')}.jpg"
            )

            st.image(
             poster_file,
            use_container_width=True
            )

            st.markdown(
            f"### {movie['title']}"
            )

            st.write(f"⭐ Rating: {movie['rating']}/10")
            st.write(f"🎭 Genre: {movie['genre']}")
            st.write(f"📅 Year: {movie['year']}")
            st.write(f"🔎 Similarity: {score:.1%}")

            st.write(movie["overview"])

            st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "🎬 Movie Recommendation System | "
    "Built with Python, Streamlit & Machine Learning"
)