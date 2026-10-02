import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

LIMIT = None

df = pd.read_pickle(r"C:\data\goodreads\features.pkl")
if LIMIT:
    df = df.iloc[:LIMIT].reset_index(drop=True)
EMB_PATH = rf"C:\data\goodreads\embeddings_{len(df)}.npy"
emb = np.load(EMB_PATH)

NON_TEXT = ["user_reviews", "days_since_review", "user_rating", "rating_diff"]
TEXT_STATS = ["num_words", "avg_sent_len", "avg_word_len", "quote",
              "n_excl", "n_quest", "sentiment"]
B = NON_TEXT + TEXT_STATS

g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
trval_i, test_i = next(g1.split(df, df["popular"], groups=df["book_id"]))

xgb = joblib.load("results/xgb_model.pkl")
sc = joblib.load("results/scaler.pkl")

Xb = sc.transform(df[B].values)
X = np.hstack([Xb, emb])
Xte = X[test_i]

n_engineered = len(B)
feature_names = B + [f"emb_{i}" for i in range(emb.shape[1])]

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(Xte)

abs_shap = np.abs(shap_values)
per_feature_importance = abs_shap.mean(axis=0)

engineered_importance = dict(zip(B, per_feature_importance[:n_engineered]))
embedding_block_importance = per_feature_importance[n_engineered:].sum()

grouped = pd.Series({**engineered_importance, "MiniLM embeddings (sum)": embedding_block_importance})
grouped = grouped.sort_values(ascending=True)

os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))
grouped.plot.barh(ax=ax)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Feature importance: engineered features vs. embedding block")
plt.tight_layout()
plt.savefig("results/shap_grouped_importance.png", dpi=150)
print(grouped)

plt.figure()
shap.summary_plot(shap_values[:, :n_engineered], Xte[:, :n_engineered],
                   feature_names=B, show=False)
plt.tight_layout()
plt.savefig("results/shap_summary_engineered.png", dpi=150)

np.save("results/shap_values_test.npy", shap_values)
pd.Series(feature_names).to_csv("results/feature_names.csv", index=False, header=False)
print("saved results/shap_grouped_importance.png, shap_summary_engineered.png, shap_values_test.npy")