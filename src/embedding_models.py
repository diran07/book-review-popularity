import os
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LIMIT = None        # 5000 for a quick test, None for the full run
DEVICE = "cpu"      # cuda crashed your laptop, so stay on cpu
BATCH = 32
CHUNK = 1000

df = pd.read_pickle(r"C:\data\goodreads\features.pkl")
if LIMIT:
    df = df.iloc[:LIMIT].reset_index(drop=True)
EMB_PATH = rf"C:\data\goodreads\embeddings_{len(df)}.npy"

# ---- embeddings (chunked + resumable)
CK_DIR = rf"C:\data\goodreads\emb_chunks_{DEVICE}_{len(df)}"
os.makedirs(CK_DIR, exist_ok=True)

if os.path.exists(EMB_PATH):
    emb = np.load(EMB_PATH)
else:
    print("embedding on", DEVICE, flush=True)
    st = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    st.max_seq_length = 128
    if DEVICE == "cuda":
        st.half()
    texts = df["review_text"].tolist()
    parts = []
    for s in range(0, len(texts), CHUNK):
        f = os.path.join(CK_DIR, f"{s}.npy")
        if os.path.exists(f):
            parts.append(np.load(f))
            continue
        e = st.encode(texts[s:s + CHUNK], batch_size=BATCH,
                      normalize_embeddings=True)
        np.save(f, e)
        parts.append(e)
        print("done", s + len(e), "/", len(texts), flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    emb = np.vstack(parts).astype(np.float32)
    np.save(EMB_PATH, emb)
print("embeddings:", emb.shape)

NON_TEXT = ["user_reviews", "days_since_review", "user_rating", "rating_diff"]
TEXT_STATS = ["num_words", "avg_sent_len", "avg_word_len", "quote",
              "n_excl", "n_quest", "sentiment"]
B = NON_TEXT + TEXT_STATS

# ---- same split recipe as baselines.py (grouped by book, 70/15/15)
g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
trval_i, test_i = next(g1.split(df, df["popular"], groups=df["book_id"]))
trval = df.iloc[trval_i]
g2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
tr_rel, val_rel = next(g2.split(trval, trval["popular"], groups=trval["book_id"]))
tr_i, val_i = trval_i[tr_rel], trval_i[val_rel]

# undersample the training indices only
tr_df = df.iloc[tr_i]
pos_i = tr_df.index[tr_df["popular"] == 1]
neg_i = tr_df[tr_df["popular"] == 0].sample(len(pos_i), random_state=42).index
tr_i = np.random.RandomState(42).permutation(np.concatenate([pos_i, neg_i]))

y_tr, y_val, y_te = (df["popular"].values[i] for i in (tr_i, val_i, test_i))
print("train/val/test:", len(tr_i), len(val_i), len(test_i))

def evaluate(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob)}

sc = StandardScaler().fit(df[B].values[tr_i])
Xb = sc.transform(df[B].values)
tfidf = TfidfVectorizer(max_features=5000, min_df=5, stop_words="english",
                        sublinear_tf=True).fit(df["review_text"].values[tr_i])
Xtf = tfidf.transform(df["review_text"])

feature_sets = {
    "Text-only: TF-IDF": (Xtf.tocsr(), ["LogReg"]),
    "Text-only: MiniLM embeddings": (emb, ["LogReg", "XGBoost"]),
    "B + MiniLM embeddings": (np.hstack([Xb, emb]), ["LogReg", "XGBoost"]),
}

results = []
for name, (X, models) in feature_sets.items():
    Xtr, Xval, Xte = X[tr_i], X[val_i], X[test_i]
    if "LogReg" in models:
        best = None
        for C in [0.01, 0.1, 1, 10]:
            lr = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_tr)
            auc = roc_auc_score(y_val, lr.predict_proba(Xval)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, C, lr)
        _, C, lr = best
        results.append({"features": name, "model": "LogReg", "params": f"C={C}",
                        **evaluate(y_te, lr.predict_proba(Xte)[:, 1])})
        print(results[-1], flush=True)
    if "XGBoost" in models:
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", eval_metric="auc",
                            early_stopping_rounds=30, random_state=42)
        xgb.fit(Xtr, y_tr, eval_set=[(Xval, y_val)], verbose=False)
        results.append({"features": name, "model": "XGBoost",
                        "params": f"trees={xgb.best_iteration + 1}, depth=6",
                        **evaluate(y_te, xgb.predict_proba(Xte)[:, 1])})
        print(results[-1], flush=True)

res = pd.DataFrame(results).round(3)
print("\n", res.to_string(index=False))
os.makedirs("results", exist_ok=True)
if not LIMIT:
    res.to_csv("results/embedding_results.csv", index=False)
    print("saved results/embedding_results.csv")
import joblib
joblib.dump(xgb, "results/xgb_model.pkl")
joblib.dump(sc, "results/scaler.pkl")
print("saved results/xgb_model.pkl and results/scaler.pkl")