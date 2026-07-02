from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_io import load_image_rgb, resize_max_side
from src.kmeans import KMeansFromScratch
from src.pipeline import compute_cluster_quality_metrics
from src.segment import build_pixel_features, reconstruct_segmented_image


def test_load_image_rgb_returns_uint8_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8), mode="RGB").save(image_path)
    image = load_image_rgb(image_path)
    assert image.shape == (12, 20, 3)
    assert image.dtype == np.uint8


def test_resize_max_side_preserves_aspect_ratio() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_max_side(image, max_side=50)
    assert resized.shape[:2] == (25, 50)


def test_build_pixel_features_shape() -> None:
    image = np.zeros((10, 8, 3), dtype=np.uint8)
    features, shape = build_pixel_features(image, color_space="rgb")
    assert features.shape == (80, 3)
    assert shape == (10, 8)


def test_kmeans_output_shapes() -> None:
    X = np.array([[0, 0, 0], [1, 1, 1], [250, 250, 250], [255, 255, 255]], dtype=float)
    model = KMeansFromScratch(n_clusters=2, random_state=0).fit(X)
    assert model.labels_.shape == (4,)
    assert model.cluster_centers_.shape == (2, 3)
    assert model.inertia_ >= 0


def test_kmeans_converges_on_two_color_image() -> None:
    X = np.vstack([np.zeros((20, 3)), np.ones((20, 3)) * 255])
    model = KMeansFromScratch(n_clusters=2, random_state=1).fit(X)
    assert model.n_iter_ <= model.max_iter
    assert len(np.unique(model.labels_)) == 2


def test_kmeans_handles_empty_cluster_without_nan() -> None:
    X = np.zeros((5, 3), dtype=float)
    model = KMeansFromScratch(n_clusters=4, random_state=2).fit(X)
    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(model.inertia_)


def test_reconstruct_segmented_image_shape_and_dtype() -> None:
    labels = np.array([0, 1, 0, 1])
    centers = np.array([[0, 0, 0], [255, 255, 255]], dtype=float)
    image = reconstruct_segmented_image(labels, centers, (2, 2), color_space="rgb")
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8


def test_compute_cluster_quality_metrics_has_ranking_score() -> None:
    X = np.vstack([np.zeros((10, 3)), np.ones((10, 3)) * 255])
    labels = np.array([0] * 10 + [1] * 10)
    centers = np.array([[0, 0, 0], [255, 255, 255]], dtype=float)
    metrics = compute_cluster_quality_metrics(X, labels, centers, inertia_per_pixel=0.0, random_state=42)
    assert metrics["silhouette_sample"] > 0.9
    assert metrics["cluster_balance"] == 1.0
    assert np.isfinite(metrics["ranking_score"])
