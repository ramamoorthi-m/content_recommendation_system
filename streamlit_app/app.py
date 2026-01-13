import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Hybrid Movie Recommender", layout="centered")

st.title("🎬 Content Recommendation System (Hybrid Movie Recommendation System)")

user_id = st.number_input("User ID", min_value=0, step=1)
k = st.slider("Number of recommendations", 5, 20, 10)
alpha = st.slider("Alpha (ALS vs Content)", 0.0, 1.0, 0.7)

if st.button("Get Recommendations"):
    payload = {
        "user_id": int(user_id),
        "k": k,
        "alpha": alpha
    }

    with st.spinner("Fetching recommendations..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        st.success("Recommendations loaded")

        for i, rec in enumerate(data["recommendations"], 1):
            st.markdown(f"**{i}. {rec['title']}**")
            st.caption(", ".join(rec["genres"]))
    else:
        st.error("API error")
        st.json(response.text)
