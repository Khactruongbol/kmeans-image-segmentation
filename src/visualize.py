from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .image_io import save_rgb_image


def save_image(image_rgb: np.ndarray, output_path: str | Path) -> None:
    save_rgb_image(image_rgb, output_path)


def save_side_by_side(
    original_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Landscape")
    axes[0].axis("off")
    axes[1].imshow(segmented_rgb)
    axes[1].set_title("K-Means Segmentation")
    axes[1].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_k_grid(
    original_rgb: np.ndarray,
    segmented_by_k: dict[int, np.ndarray],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(segmented_by_k.items())
    cols = min(4, len(items) + 1)
    rows = int(np.ceil((len(items) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    axes_array[0].imshow(original_rgb)
    axes_array[0].set_title("Original")
    axes_array[0].axis("off")
    for ax, (k, image) in zip(axes_array[1:], items):
        ax.imshow(image)
        ax.set_title(f"K={k}")
        ax.axis("off")
    for ax in axes_array[len(items) + 1 :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
