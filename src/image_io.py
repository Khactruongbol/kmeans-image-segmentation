from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save_rgb_image(image_rgb: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(path)


def resize_max_side(image_rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image_rgb.copy()
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray(image_rgb, mode="RGB")
    return np.asarray(image.resize(new_size, Image.Resampling.LANCZOS), dtype=np.uint8)


def convert_color_space(image_rgb: np.ndarray, color_space: str) -> np.ndarray:
    color_space = color_space.lower()
    if color_space == "rgb":
        return image_rgb.copy()
    mode = {"hsv": "HSV", "lab": "LAB"}.get(color_space)
    if mode is None:
        raise ValueError("color_space must be one of: rgb, hsv, lab")
    image = Image.fromarray(image_rgb, mode="RGB").convert(mode)
    return np.asarray(image, dtype=np.uint8)


def convert_feature_image_to_rgb(feature_image: np.ndarray, color_space: str) -> np.ndarray:
    color_space = color_space.lower()
    feature_image = np.clip(feature_image, 0, 255).astype(np.uint8)
    if color_space == "rgb":
        return feature_image
    mode = {"hsv": "HSV", "lab": "LAB"}.get(color_space)
    if mode is None:
        raise ValueError("color_space must be one of: rgb, hsv, lab")
    return np.asarray(Image.fromarray(feature_image, mode=mode).convert("RGB"), dtype=np.uint8)
