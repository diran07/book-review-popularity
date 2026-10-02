import re
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

IN_PATH = r"C:\data\goodreads\reviews_processed.pkl"
OUT_PATH = r"C:\data\goodreads\features.pkl"
SAMPLE_N = 200_000

tqdm.pandas()
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

df = pd.read_pickle(IN_PATH)
print("rating values:\n", df["rating"].value_counts().sort_index())

# ---- non-text features (computed on the full filtered set, before sampling)
df["days_since_review"] = (df["date_added"].max() - df["date_added"]).dt.days

# rating 0 means "no rating given", so exclude it from the book average
rated = df["rating"] > 0
book_mean = df[rated].groupby("book_id")["rating"].mean()
df["book_avg_rating"] = df["book_id"].map(book_mean)
df["rating_diff"] = (df["rating"] - df["book_avg_rating"]).where(rated, 0).fillna(0)
df["user_rating"] = df["rating"]

# ---- sample for speed
if SAMPLE_N and len(df) > SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
print("working rows:", len(df))

# ---- text features
def text_feats(t):
    words = t.split()
    n_words = len(words)
    sents = [s for s in re.split(r"[.!?]+\s+|\n+", t) if s.strip()]
    n_sents = max(len(sents), 1)
    avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
    quote = int('"' in t or "\u201c" in t)
    return n_words, n_words / n_sents, avg_word_len, quote, t.count("!"), t.count("?")

cols = ["num_words", "avg_sent_len", "avg_word_len", "quote", "n_excl", "n_quest"]
feats = df["review_text"].progress_apply(text_feats)
df[cols] = pd.DataFrame(feats.tolist(), index=df.index)

df["sentiment"] = df["review_text"].progress_apply(
    lambda t: sia.polarity_scores(t)["compound"])

NON_TEXT = ["user_reviews", "days_since_review", "user_rating", "rating_diff"]
TEXT_STATS = cols + ["sentiment"]
print(df[NON_TEXT + TEXT_STATS].describe().T)
print("popular rate:", round(df["popular"].mean(), 3))
df.to_pickle(OUT_PATH)
print("saved to", OUT_PATH)