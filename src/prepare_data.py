import gzip, json, random, re
from collections import Counter
import pandas as pd

DATA_PATH = r"C:\data\goodreads\goodreads_reviews_dedup.json.gz"
OUT_PATH = r"C:\data\goodreads\reviews_processed.pkl"
MAX_LINES = None      # quick test; set to None for the full run
KEEP_PROB = 0.25         # fraction of engaged reviews whose text we keep
THRESHOLD = 0.02         # likes-share cutoff from the paper
random.seed(42)

user_reviews = Counter()   # all reviews written per user
book_eng_n = Counter()     # engaged reviews per book
book_eng = Counter()       # total likes+comments per book
kept = []

with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if MAX_LINES and i >= MAX_LINES:
            break
        d = json.loads(line)
        u, b = d["user_id"], d["book_id"]
        eng = (d.get("n_votes") or 0) + (d.get("n_comments") or 0)
        user_reviews[u] += 1
        if eng > 0:
            book_eng_n[b] += 1
            book_eng[b] += eng
            if random.random() < KEEP_PROB:
                kept.append((u, b, d.get("rating"), d.get("review_text"),
                             d.get("date_added"), eng))
        if i % 1_000_000 == 0:
            print("lines read:", i, flush=True)

df = pd.DataFrame(kept, columns=["user_id", "book_id", "rating",
                                 "review_text", "date_added", "engagement"])
print("kept before filters:", df.shape)

df["user_reviews"] = df["user_id"].map(user_reviews)
df["book_total"] = df["book_id"].map(book_eng)
df["book_n_reviews"] = df["book_id"].map(book_eng_n)

# paper's filters: books with >=10 reviews and >=60 likes+comments
df = df[(df["book_n_reviews"] >= 10) & (df["book_total"] >= 60)]

df["review_text"] = df["review_text"].fillna("").str.strip()
df = df[df["review_text"].str.len() > 0]

STOP = re.compile(r"\b(the|and|is|was|this|that|it|of|to|a)\b", re.I)
def looks_english(t):
    if sum(c.isascii() for c in t) / len(t) < 0.95:
        return False
    return len(STOP.findall(t)) >= max(1, len(t.split()) * 0.15)
df = df[df["review_text"].map(looks_english)]

df["likes_share"] = df["engagement"] / df["book_total"]
df["popular"] = (df["likes_share"] > THRESHOLD).astype(int)

df["date_added"] = pd.to_datetime(
    df["date_added"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
df = df.dropna(subset=["date_added", "rating"])

print("final:", df.shape)
print("books:", df["book_id"].nunique())
print("popular rate:", round(df["popular"].mean(), 3))
df.reset_index(drop=True).to_pickle(OUT_PATH)
print("saved to", OUT_PATH)