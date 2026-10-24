"""Tanimoto helpers shared by the ACE regression notebooks.

Kept in a real module (not the notebook's __main__) so joblib pickles of
TanimotoKNN / SVR(kernel=tanimoto_kernel) can be loaded from any notebook.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def tanimoto_kernel(A, B):
    """Tanimoto (Jaccard) similarity between rows of binary matrices A and B."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    dot = A @ B.T
    denom = A.sum(axis=1)[:, None] + B.sum(axis=1)[None, :] - dot
    return np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)


def max_tanimoto_similarity(X, X_ref, chunk=2000):
    """Max Tanimoto similarity of each row of X to any row of X_ref."""
    X = np.asarray(X)
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), chunk):
        out[start:start + chunk] = tanimoto_kernel(X[start:start + chunk], X_ref).max(axis=1)
    return out


class TanimotoKNN(BaseEstimator, RegressorMixin):
    """Similarity-weighted k-NN regressor using Tanimoto similarity."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X, chunk=2000):
        X = np.asarray(X, dtype=np.float32)
        k = min(self.n_neighbors, len(self.y_))
        pred = np.empty(len(X))
        for start in range(0, len(X), chunk):
            sim = tanimoto_kernel(X[start:start + chunk], self.X_)
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            w = np.take_along_axis(sim, idx, axis=1)
            y_nb = self.y_[idx]
            wsum = w.sum(axis=1)
            pred[start:start + chunk] = np.where(
                wsum > 0, (w * y_nb).sum(axis=1) / np.where(wsum > 0, wsum, 1), self.y_.mean())
        return pred


def predict_in_chunks(model, X, chunk=5000):
    """Predict large libraries without building one giant kernel matrix."""
    X = np.asarray(X)
    return np.concatenate([model.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])
