"""Movie recommendation demo: matrix factorization (ALS) on MovieLens 100K.

Educational MDS project demo. Ratings are downloaded from the official
GroupLens source at first run and are never redistributed with this app.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import streamlit as st
from src.data import fetch_data
from src.model import MatrixFactorization, errors, split_by_user

st.set_page_config(page_title='Movie Recommender - Matrix Factorization', page_icon='🎬', layout='centered')


@st.cache_resource(show_spinner='Downloading MovieLens 100K from GroupLens and training the ALS model (first run only)...')
def load_model():
    ratings, titles = fetch_data()
    train, validation, test = split_by_user(ratings)  # seed 42, same split as the project report
    n_users, n_items = int(ratings[:, 0].max() + 1), int(ratings[:, 1].max() + 1)
    baseline = MatrixFactorization(n_users, n_items, rank=0).fit(train, validation)
    model = MatrixFactorization(n_users, n_items, rank=10).fit(train, validation)
    metrics = {
        'global_mean': errors(test[:, 2], np.full(len(test), train[:, 2].mean())),
        'bias_baseline': errors(test[:, 2], baseline.predict(test[:, 0], test[:, 1])),
        'matrix_factorization': errors(test[:, 2], model.predict(test[:, 0], test[:, 1])),
    }
    return ratings, titles, model, metrics, (len(train), len(validation), len(test))


ratings, titles, model, metrics, split_sizes = load_model()

st.title('🎬 Movie Recommender')
st.caption('Matrix factorization (regularized ALS) trained on MovieLens 100K - MDS academic project demo')

user = st.number_input('MovieLens user ID (1-943 are dataset users)', min_value=1, max_value=999999, value=1, step=1)
k = st.selectbox('Number of recommendations', [5, 10, 20], index=1)

internal = int(user) - 1
seen = ratings[ratings[:, 0] == internal, 1].astype(int)
cold = not (0 <= internal < model.n_users and model.known_users[internal])
ids, scores = model.recommend(max(internal, 0), seen, k)

if cold:
    st.info('New user: no rating history, so these are item-popularity fallback picks, not personalized.')
else:
    st.success(f'Personalized from the training ratings of MovieLens user {user}. '
               f'All {len(seen)} historically rated movies are hidden.')

table = pd.DataFrame({
    'Rank': range(1, len(ids) + 1),
    'Movie': [titles[int(i)] for i in ids],
    'Movie ID': [int(i) + 1 for i in ids],
    'Predicted stars': [round(float(s), 2) for s in scores],
})
st.dataframe(table, hide_index=True, use_container_width=True)

with st.expander('How it works and how it was evaluated'):
    st.markdown(
        '- Model: regularized alternating least squares (ALS) with user and item biases, rank 10, '
        'regularization 10, seed 42. Only observed ratings are used; no missing rating is treated as zero.\n'
        '- Evaluation: per-user 80/10/10 train/validation/test split '
        f'({split_sizes[0]} / {split_sizes[1]} / {split_sizes[2]} ratings). Rank and checkpoint were '
        'selected on validation only; the test set was touched once.\n'
        f"- Test RMSE: **{metrics['matrix_factorization']['rmse']:.4f}** stars "
        f"(MAE {metrics['matrix_factorization']['mae']:.4f}), vs bias-only baseline "
        f"{metrics['bias_baseline']['rmse']:.4f} and global mean {metrics['global_mean']['rmse']:.4f}.\n"
        '- Predicted scores are clipped to 1-5 stars and are estimates, not guarantees.\n'
        '- User IDs above 943 show the cold-start fallback. These are historical dataset users, '
        'not personal accounts.'
    )
    chart = pd.DataFrame(
        {'Model': ['Global mean', 'User/item bias', 'Matrix factorization (rank 10)'],
         'Test RMSE': [metrics['global_mean']['rmse'], metrics['bias_baseline']['rmse'],
                       metrics['matrix_factorization']['rmse']]}
    ).set_index('Model')
    st.bar_chart(chart)

st.divider()
st.caption('Dataset: MovieLens 100K, GroupLens Research, University of Minnesota '
           '(downloaded from the official source; research/education use only, no endorsement). '
           'Latent factors are learned rating patterns, not named genres.')
