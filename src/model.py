"""Regularized alternating least squares for explicit ratings.
No missing rating is ever treated as zero.
"""
import numpy as np


def errors(actual, predicted):
    difference = np.asarray(actual) - np.asarray(predicted)
    return {"rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mae": float(np.mean(np.abs(difference)))}


def split_by_user(ratings, seed=42):
    """Random per-user 80/10/10 split, rounding holdouts down."""
    rng = np.random.default_rng(seed)
    groups = [[], [], []]
    for user in np.unique(ratings[:, 0]):
        indices = np.flatnonzero(ratings[:, 0] == user)
        rng.shuffle(indices)
        n = max(1, len(indices) // 10)
        for target, part in zip(groups, [indices[2*n:], indices[:n], indices[n:2*n]]):
            target.extend(part.tolist())
    return tuple(ratings[np.array(x)] for x in groups)


class MatrixFactorization:
    """Joint bias + latent-factor ALS; rank=0 gives a bias-only baseline.

    Objective: sum_observed (r-mu-bu-bi-pu.qi)^2
               + reg * (sum(bu^2)+sum(bi^2)+sum(P^2)+sum(Q^2)).
    Mean is fixed from train. Predictions are clipped only at inference.
    """
    def __init__(self, n_users, n_items, rank=20, reg=10.0, seed=42):
        if rank < 0 or reg <= 0:
            raise ValueError("rank must be nonnegative and reg positive")
        self.n_users, self.n_items = n_users, n_items
        self.rank, self.reg, self.seed = rank, reg, seed

    def predict(self, users, items, clip=True):
        users, items = np.broadcast_arrays(np.asarray(users, dtype=int), np.asarray(items, dtype=int))
        shape = users.shape
        u, i = users.ravel(), items.ravel()
        uk = (u >= 0) & (u < self.n_users)
        ik = (i >= 0) & (i < self.n_items)
        out = np.full(len(u), self.mean, dtype=float)
        out[uk] += self.bu[u[uk]]
        out[ik] += self.bi[i[ik]]
        both = uk & ik
        out[both] += np.sum(self.P[u[both]] * self.Q[i[both]], axis=1)
        return (np.clip(out, 1, 5) if clip else out).reshape(shape)

    def fit(self, train, validation=None, epochs=20):
        rng = np.random.default_rng(self.seed)
        self.mean = float(train[:, 2].mean())
        self.bu, self.bi = np.zeros(self.n_users), np.zeros(self.n_items)
        self.P = rng.normal(0, .05, (self.n_users, self.rank))
        self.Q = rng.normal(0, .05, (self.n_items, self.rank))
        ug = [train[train[:, 0] == u] for u in range(self.n_users)]
        ig = [train[train[:, 1] == i] for i in range(self.n_items)]
        self.known_users = np.array([len(x) > 0 for x in ug])
        self.known_items = np.array([len(x) > 0 for x in ig])
        self.P[~self.known_users] = 0
        self.Q[~self.known_items] = 0
        penalty = self.reg * np.eye(self.rank + 1)
        self.history, best, snapshot = [], float('inf'), None
        for epoch in range(1, epochs + 1):
            for u, rows in enumerate(ug):
                if not len(rows):
                    continue
                items = rows[:, 1].astype(int)
                X = np.column_stack((np.ones(len(rows)), self.Q[items]))
                y = rows[:, 2] - self.mean - self.bi[items]
                w = np.linalg.solve(X.T @ X + penalty, X.T @ y)
                self.bu[u], self.P[u] = w[0], w[1:]
            for i, rows in enumerate(ig):
                if not len(rows):
                    continue
                users = rows[:, 0].astype(int)
                X = np.column_stack((np.ones(len(rows)), self.P[users]))
                y = rows[:, 2] - self.mean - self.bu[users]
                w = np.linalg.solve(X.T @ X + penalty, X.T @ y)
                self.bi[i], self.Q[i] = w[0], w[1:]
            row = {"epoch": epoch, "train_rmse": errors(train[:, 2], self.predict(train[:, 0], train[:, 1]))['rmse']}
            if validation is not None:
                row['validation_rmse'] = errors(validation[:, 2], self.predict(validation[:, 0], validation[:, 1]))['rmse']
                if row['validation_rmse'] < best:
                    best, self.best_epoch = row['validation_rmse'], epoch
                    snapshot = [x.copy() for x in (self.bu, self.bi, self.P, self.Q)]
            self.history.append(row)
        if snapshot is not None:
            self.bu, self.bi, self.P, self.Q = snapshot
        else:
            self.best_epoch = epochs
        return self

    def recommend(self, user, seen_items=(), k=10):
        if k < 1:
            raise ValueError('k must be positive')
        candidates = np.flatnonzero(self.known_items)
        candidates = candidates[~np.isin(candidates, list(seen_items))]
        scores = self.predict(np.full(len(candidates), user), candidates)
        # Break equal scores by item ID for repeatable output.
        order = np.lexsort((candidates, -scores))[:k]
        return candidates[order], scores[order]

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, bu=self.bu, bi=self.bi, P=self.P,
                            Q=self.Q, known_users=self.known_users, known_items=self.known_items,
                            reg=self.reg, seed=self.seed)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(len(z['bu']), len(z['bi']), z['P'].shape[1], float(z['reg']), int(z['seed']))
            for field in ('mean', 'bu', 'bi', 'P', 'Q', 'known_users', 'known_items'):
                setattr(model, field, z[field].copy())
        return model


def ranking_metrics(model, train, validation, test, k=10):
    """Full warm-item candidates, observed train/validation items excluded.
    Relevant: warm test items rated >=4. Macro averages over users with >=1.
    Unobserved items are not known negatives: this is a limited offline proxy.
    """
    recalls, ndcgs = [], []
    for u in range(model.n_users):
        rows = test[test[:, 0] == u]
        relevant = {int(i) for _, i, r, *_ in rows if r >= 4 and model.known_items[int(i)]}
        if not relevant:
            continue
        seen = set(train[train[:, 0] == u, 1].astype(int)) | set(validation[validation[:, 0] == u, 1].astype(int))
        ids, _ = model.recommend(u, seen, k)
        hits = np.array([i in relevant for i in ids], dtype=float)
        recalls.append(float(hits.sum() / len(relevant)))
        dcg = np.sum(hits / np.log2(np.arange(len(hits)) + 2))
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
        ndcgs.append(float(dcg / ideal))
    return {f'recall_at_{k}': float(np.mean(recalls)), f'ndcg_at_{k}': float(np.mean(ndcgs)), 'eligible_users': len(recalls)}
