# Movie Recommender - Matrix Factorization (MDS Project)

A small collaborative-filtering demo that uses matrix factorization to analyze
user-item rating data and recommend movies, built with NumPy and Streamlit.

## What it does

- Analyzes the MovieLens 100K rating data (100,000 ratings, 943 users, 1,682 movies)
- Trains a regularized matrix factorization model (alternating least squares,
  rank 10, with user and item bias terms) on observed ratings only
- Shows the top-N movie recommendations for any dataset user, with predicted star ratings
- Falls back to item-popularity picks for unknown user IDs (cold start)

## Evaluation

Per-user 80/10/10 train/validation/test split (seed 42). Rank and checkpoint
were selected on validation only; the held-out test set was evaluated once.

| Model | Test RMSE | Test MAE |
|---|---|---|
| Global mean | 1.1385 | - |
| User/item bias baseline | 0.9478 | - |
| Matrix factorization (rank 10) | 0.9135 | 0.7139 |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run downloads MovieLens 100K from the official GroupLens source and
trains the model (a few seconds). Ratings are never bundled with this repo:
GroupLens terms do not allow redistribution. Research/education use only.

Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4 (2015). https://grouplens.org/datasets/movielens/
