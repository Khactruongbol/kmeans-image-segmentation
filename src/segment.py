from __future__ import annotations

from pathlib import Path

import numpy as np

from .image_io import convert_color_space, convert_feature_image_to_rgb, load_image_rgb, resize_max_side
from .kmeans import KMeansFromScratch


def build_pixel_features(
    image_rgb: np.ndarray,
    color_space: str = "lab",
    use_xy: bool = False,
    xy_weight: float = 0.1,
) -> tuple[np.ndarray, tuple[int, int]]:
    feature_image = convert_color_space(image_rgb, color_space)
    height, width = feature_image.shape[:2]
    color_features = feature_image.reshape(-1, 3).astype(np.float64)
    if not use_xy:
        return color_features, (height, width)

    ys, xs = np.mgrid[0:height, 0:width]
    xy = np.column_stack(
        [
            xs.reshape(-1) / max(width - 1, 1),
            ys.reshape(-1) / max(height - 1, 1),
        ]
    )
    xy = xy.astype(np.float64) * 255.0 * float(xy_weight)
    return np.column_stack([color_features, xy]), (height, width)


def reconstruct_segmented_image(
    labels: np.ndarray,
    centers: np.ndarray,
    image_shape: tuple[int, int],
    color_space: str = "lab",
) -> np.ndarray:
    height, width = image_shape
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)
    color_centers = centers[:, :3]
    feature_image = color_centers[labels].reshape(height, width, 3)
    return convert_feature_image_to_rgb(feature_image, color_space)


def segment_image_array(
    image_rgb: np.ndarray,
    k: int,
    color_space: str = "lab",
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    use_xy: bool = False,
    xy_weight: float = 0.1,
) -> tuple[np.ndarray, KMeansFromScratch, np.ndarray]:
    features, image_shape = build_pixel_features(image_rgb, color_space, use_xy, xy_weight)
    model = KMeansFromScratch(k, max_iter=max_iter, tol=tol, random_state=random_state)
    labels = model.fit_predict(features)
    assert model.cluster_centers_ is not None
    segmented = reconstruct_segmented_image(labels, model.cluster_centers_, image_shape, color_space)
    return segmented, model, labels


def segment_image_file(
    image_path: str | Path,
    k: int,
    color_space: str = "lab",
    max_side: int = 512,
    random_state: int = 42,
    use_xy: bool = False,
    xy_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, KMeansFromScratch, np.ndarray]:
    image_rgb = load_image_rgb(image_path)
    resized_rgb = resize_max_side(image_rgb, max_side)
    segmented, model, labels = segment_image_array(
        resized_rgb,
        k=k,
        color_space=color_space,
        random_state=random_state,
        use_xy=use_xy,
        xy_weight=xy_weight,
    )
    return resized_rgb, segmented, model, labels
