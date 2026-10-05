# Goodreads Review Popularity

Predicts whether a Goodreads book review will become "popular" (high reader
engagement), using the review text and the reviewer's activity history.
Built as an AI/ML capstone project.

## Problem

Some reviews get far more engagement than others. This project tries to
predict which reviews will be popular, and explain why, using:

- the review's text (via TF-IDF and MiniLM sentence embeddings)
- engineered metadata (reviewer activity, rating behavior, text statistics)

The question driving the comparison: do semantic embeddings plus simple
engineered features beat either one alone?

## Dataset

- ~200,000 sampled Goodreads reviews
- Popular rate in the data: ~21% (imbalanced)
- Training set undersampled to 50/50 class balance; validation/test sets
  left at the natural (imbalanced) rate

## Approach

Three feature representations, two model families, to isolate what text
understanding adds over simple metadata:

| Feature set                   | Models tried        | Role                        |
|--------------------------------|----------------------|------------------------------|
| TF-IDF (5,000 terms)           | Logistic Regression  | Classical text baseline     |
| MiniLM embeddings (384-d)      | LogReg, XGBoost       | Semantic text baseline      |
| Engineered features + MiniLM   | LogReg, XGBoost       | Does metadata add signal?   |

Engineered features: `user_reviews`, `days_since_review`, `user_rating`,
`rating_diff`, plus text statistics (`num_words`, `avg_sent_len`,
`avg_word_len`, `quote`, `n_excl`, `n_quest`, `sentiment`). Sentiment is
computed with NLTK VADER (`SentimentIntensityAnalyzer`).

Best model is explained with SHAP (TreeExplainer); embedding dimensions are
grouped into a single block for interpretability.

## Results

| Features                     | Model   | ROC-AUC | PR-AUC | F1    |
|-------------------------------|---------|---------|--------|-------|
| TF-IDF                        | LogReg  | 0.694   | 0.370  | 0.430 |
| MiniLM embeddings             | LogReg  | 0.651   | 0.311  | 0.396 |
| MiniLM embeddings             | XGBoost | 0.655   | 0.315  | 0.403 |
| Engineered + MiniLM           | LogReg  | 0.687   | 0.351  | 0.426 |
| **Engineered + MiniLM**       | **XGBoost** | **0.733** | **0.417** | **0.457** |

Combining embeddings with engineered metadata, scored with XGBoost, beats
every single-feature-set baseline. ROC-AUC of 0.73 is real predictive
power, but popularity clearly depends on more than text and basic metadata.

Note: the model is trained on class-balanced data but evaluated/deployed
against the natural (imbalanced) rate, so predicted probabilities run
higher on average than the true ~21% popularity rate. This is expected and
doesn't affect the ROC-AUC/PR-AUC metrics above, which are
threshold-independent; it does mean a raw 0.5 cutoff isn't perfectly
calibrated against the real-world base rate.

## Explainability

Mean |SHAP value| on the test set, best model (XGBoost, Engineered + MiniLM):

| Feature                     | Mean \|SHAP\| |
|------------------------------|---------------|
| MiniLM embeddings (sum)      | 2.91          |
| num_words                    | 0.33          |
| user_reviews                 | 0.31          |
| rating_diff                  | 0.24          |
| other 7 features combined    | 0.27          |

Embeddings dominate overall, but review length and reviewer activity are
the strongest single engineered signals. Lower star ratings show a mild
positive link to popularity — critical reviews may draw more engagement.

## Project structure

```
src/
  build_features.py        # feature engineering: engineered metadata + text stats
  embedding_models.py       # trains/evaluates all model x feature-set combinations
  step8_streamlit_demo.py   # interactive demo: paste a review, get a prediction + SHAP breakdown
results/
  embedding_results.csv     # full results table
  xgb_model.pkl              # best model (XGBoost, Engineered + MiniLM)
  scaler.pkl                 # StandardScaler fit on engineered features
```

## Running it

```
pip install -r requirements.txt
python src/build_features.py        # build engineered features from raw reviews
python src/embedding_models.py       # train and evaluate all models
streamlit run src/step8_streamlit_demo.py   # launch the interactive demo
```

## Demo

The Streamlit demo takes a pasted review plus reviewer/book context
(reviewer's total review count, star rating, days since posted, rating
vs. book average) and returns a predicted probability of popularity along
with a SHAP breakdown of what drove the prediction.

## Conclusions

- Combining embeddings and engineered metadata, scored with XGBoost, beats
  every single-feature-set baseline.
- The signal is real but moderate (ROC-AUC 0.73) — popularity depends on
  more than text and basic metadata.
- The model is explainable via SHAP and demonstrated through a live,
  interactive demo.
