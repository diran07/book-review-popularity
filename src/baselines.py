import os
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_pickle(r"C:\data\goodreads\features.pkl")

NON_TEXT = ["user_reviews", "days_since_review", "user_rating", "rating_diff"]
TEXT_STATS = ["num_words", "avg_sent_len", "avg_word_len", "quote",
              "n_excl", "n_quest", "sentiment"]

# ---- split by book: 70 / 15 / 15
g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
trval_i, test_i = next(g1.split(df, df["popular"], groups=df["book_id"]))
trval, test = df.iloc[trval_i], df.iloc[test_i]
g2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
tr_i, val_i = next(g2.split(trval, trval["popular"], groups=trval["book_id"]))
train, val = trval.iloc[tr_i], trval.iloc[val_i]

# ---- undersample the training set only
pos = train[train["popular"] == 1]
neg = train[train["popular"] == 0].sample(len(pos), random_state=42)
train_bal = pd.concat([pos, neg]).sample(frac=1, random_state=42)

print("train/val/test:", len(train_bal), len(val), len(test))
print("popular rate  train(bal)/val/test:",
      round(train_bal.popular.mean(), 3), round(val.popular.mean(), 3),
      round(test.popular.mean(), 3))
majority_acc = 1 - test["popular"].mean()
print("majority-class accuracy on test:", round(majority_acc, 3))

def evaluate(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob)}

tfidf = TfidfVectorizer(max_features=5000, min_df=5, stop_words="english",
                        sublinear_tf=True).fit(train_bal["review_text"])

def build(cols, use_tfidf):
    sc = StandardScaler().fit(train_bal[cols])
    def mk(d):
        X = csr_matrix(sc.transform(d[cols]))
        if use_tfidf:
            X = hstack([X, tfidf.transform(d["review_text"])]).tocsr()
        return X
    return mk(train_bal), mk(val), mk(test)

sets = {"A (non-text)": (NON_TEXT, False),
        "B (+text stats)": (NON_TEXT + TEXT_STATS, False),
        "D (+TF-IDF)": (NON_TEXT + TEXT_STATS, True)}

y_tr, y_val, y_te = train_bal["popular"], val["popular"], test["popular"]
results = []

for name, (cols, use_tf) in sets.items():
    Xtr, Xval, Xte = build(cols, use_tf)

    # logistic regression, C chosen on validation AUC
    best = None
    for C in [0.01, 0.1, 1, 10]:
        lr = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_tr)
        auc = roc_auc_score(y_val, lr.predict_proba(Xval)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, lr)
    _, C, lr = best
    r = evaluate(y_te, lr.predict_proba(Xte)[:, 1])
    results.append({"features": name, "model": "LogReg", "params": f"C={C}", **r})
    print(results[-1], flush=True)

    # XGBoost with early stopping on validation
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", eval_metric="auc",
                        early_stopping_rounds=30, random_state=42)
    xgb.fit(Xtr, y_tr, eval_set=[(Xval, y_val)], verbose=False)
    r = evaluate(y_te, xgb.predict_proba(Xte)[:, 1])
    results.append({"features": name, "model": "XGBoost",
                    "params": f"trees={xgb.best_iteration + 1}, depth=6", **r})
    print(results[-1], flush=True)

res = pd.DataFrame(results).round(3)
print("\n", res.to_string(index=False))
os.makedirs("results", exist_ok=True)
res.to_csv("results/baseline_results.csv", index=False)
print("\nsaved results/baseline_results.csv")