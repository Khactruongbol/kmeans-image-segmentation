from __future__ import annotations

import numpy as np


class KMeansFromScratch:
    """Small NumPy implementation of standard K-Means clustering."""

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf")
        return X

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = X.shape[0]
        replace = self.n_clusters > n_samples
        indices = rng.choice(n_samples, size=self.n_clusters, replace=replace)
        return X[indices].copy()

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        X = self._validate_X(X)
        rng = np.random.default_rng(self.random_state)
        centers = self._init_centroids(X, rng)

        labels = np.zeros(X.shape[0], dtype=np.int64)
        for iteration in range(1, self.max_iter + 1):
            distances = self._squared_distances(X, centers)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()

            for cluster_idx in range(self.n_clusters):
                mask = labels == cluster_idx
                if mask.any():
                    new_centers[cluster_idx] = X[mask].mean(axis=0)
                else:
                    farthest_idx = np.argmax(distances.min(axis=1))
                    new_centers[cluster_idx] = X[farthest_idx]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:
                break

        final_distances = self._squared_distances(X, centers)
        labels = final_distances.argmin(axis=1)
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = float(final_distances[np.arange(X.shape[0]), labels].sum())
        self.n_iter_ = iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeansFromScratch must be fitted before predict")
        X = self._validate_X(X)
        return self._squared_distances(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
