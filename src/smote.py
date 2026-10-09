"""SMOTE oversampling. Uses imbalanced-learn when installed, else a minimal fallback."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_resample(X, y, k=5, random_state=42):
    """Minimal SMOTE: interpolate between a minority sample and one of its k nearest minority neighbours."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[counts.argmin()]
    n_new = counts.max() - counts.min()
    X_min = X[y == minority]
    if n_new <= 0 or len(X_min) < 2:
        return X, y
    k = min(k, len(X_min) - 1)
    neigh = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
    base = rng.integers(0, len(X_min), n_new)
    nb = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    X_new = X_min[base] + gap * (X_min[nb] - X_min[base])
    return np.vstack([X, X_new]), np.concatenate([y, np.full(n_new, minority)])


def oversample(X, y, random_state=42):
    """Return (X_res, y_res, backend_name)."""
    try:
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)
        return X_res, y_res, "imbalanced-learn"
    except ImportError:
        X_res, y_res = smote_resample(X, y, random_state=random_state)
        return X_res, y_res, "fallback (built-in)"
