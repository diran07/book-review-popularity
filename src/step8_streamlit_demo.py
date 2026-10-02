# src/step8_streamlit_demo.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import shap
import joblib
from sentence_transformers import SentenceTransformer

st.title("Goodreads Review Popularity Predictor")

@st.cache_resource
def load_artifacts():
    xgb = joblib.load("results/xgb_model.pkl")
    sc = joblib.load("results/scaler.pkl")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    explainer = shap.TreeExplainer(xgb)
    return xgb, sc, st_model, explainer

xgb, sc, st_model, explainer = load_artifacts()

NON_TEXT = ["user_reviews", "days_since_review", "user_rating", "rating_diff"]
TEXT_STATS = ["num_words", "avg_sent_len", "avg_word_len", "quote",
              "n_excl", "n_quest", "sentiment"]
B = NON_TEXT + TEXT_STATS

def compute_text_stats(text):
    words = text.split()
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    num_words = len(words)
    avg_sent_len = num_words / max(len(sentences), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    quote = int('"' in text or "'" in text)
    n_excl = text.count('!')
    n_quest = text.count('?')
    try:
        from textblob import TextBlob
        sentiment = TextBlob(text).sentiment.polarity
    except ImportError:
        sentiment = 0.0  # placeholder if textblob isn't installed — replace with your actual sentiment function
    return [num_words, avg_sent_len, avg_word_len, quote, n_excl, n_quest, sentiment]

st.subheader("Review text")
review_text = st.text_area("Paste a review:", height=150)

st.subheader("Reviewer / book context")
col1, col2 = st.columns(2)
with col1:
    user_reviews = st.number_input("Reviewer's total review count", min_value=0, value=10)
    user_rating = st.slider("This review's star rating", 1, 5, 4)
with col2:
    days_since_review = st.number_input("Days since the review was posted", min_value=0, value=30)
    rating_diff = st.number_input("Rating minus book's average rating", value=0.0, step=0.1)

if st.button("Predict") and review_text.strip():
    text_feats = compute_text_stats(review_text)
    non_text_feats = [user_reviews, days_since_review, user_rating, rating_diff]
    engineered = np.array([non_text_feats + text_feats])
    engineered_scaled = sc.transform(engineered)

    emb = st_model.encode([review_text], normalize_embeddings=True)
    X = np.hstack([engineered_scaled, emb])

    prob = xgb.predict_proba(X)[0, 1]
    st.metric("Predicted probability of being popular", f"{prob:.1%}")

    shap_vals = explainer.shap_values(X)[0]
    abs_shap = np.abs(shap_vals)
    n_eng = len(B)
    eng_importance = dict(zip(B, abs_shap[:n_eng]))
    emb_importance = abs_shap[n_eng:].sum()

    grouped = pd.Series({**eng_importance, "MiniLM embeddings (sum)": emb_importance})
    grouped = grouped.sort_values(ascending=True)

    st.subheader("What drove this prediction")
    st.bar_chart(grouped)