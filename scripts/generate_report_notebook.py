from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    notebook_path = ROOT / "notebooks" / "01_kmeans_image_segmentation_report.ipynb"
    cells = [
        markdown_cell(
            """# Final Notebook - K-Means Image Segmentation

## 1. Define Problem

The assignment asks us to segment a given image into `K` clusters using the K-Means algorithm. In this project, the input domain is landscape imagery because skies, water, forests, mountains, sand, and clouds create broad color regions that are suitable for unsupervised pixel clustering.

The model is not a semantic segmentation network. It does not learn object masks. Each pixel is represented by color-space features, assigned to the nearest centroid, and reconstructed using the centroid color of its cluster."""
        ),
        markdown_cell(
            """## 2. Assignment Requirement Mapping

| Requirement | Implemented artifact |
|---|---|
| Load image with PIL/OpenCV | `src/image_io.py` |
| Convert to RGB/HSV/LAB | `convert_color_space()` |
| Resize image | `resize_max_side()` |
| Flatten pixels into 2D matrix | `build_pixel_features()` |
| Initialize and update K centroids | `KMeansFromScratch` |
| Assign pixels by Euclidean distance | vectorized NumPy distance computation |
| Repeat until convergence | `max_iter` and `tol` |
| Replace pixels by centroid colors | `reconstruct_segmented_image()` |
| Visualize original and segmented images | `reports/figures/*_comparison.png` |"""
        ),
        markdown_cell(
            """## 3. Success Metrics

Because there is no ground-truth segmentation mask, accuracy is measured with unsupervised internal quality metrics and visual inspection:

- `silhouette_sample`: higher is better.
- `davies_bouldin_sample`: lower is better.
- `calinski_harabasz_sample`: higher is better.
- `inertia_per_pixel`: lower is better.
- `cluster_balance`: higher is better when clusters are not collapsed.
- `ranking_score`: lower is better; combines silhouette, Davies-Bouldin, inertia, and cluster balance."""
        ),
        code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if ROOT.name != "kmeans_image_segmentation":
    ROOT = ROOT / "kmeans_image_segmentation"

summary = json.loads((ROOT / "reports" / "metrics" / "workflow_summary.json").read_text(encoding="utf-8"))
review = json.loads((ROOT / "reports" / "metrics" / "source_review.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(ROOT / "reports" / "metrics" / "model_comparison.csv")
image_labels = pd.read_csv(ROOT / "data" / "labels" / "image_labels.csv")
clean_manifest = pd.read_csv(ROOT / "data" / "manifest" / "clean_landscape_manifest.csv")
summary"""
        ),
        markdown_cell(
            """## 4. Data Sources and Raw Crawl

The data source is Wikimedia Commons, accessed through official API endpoints rather than arbitrary HTML scraping. The crawler stores source URLs, download URLs, license metadata, author information, dimensions, and local paths.

Wikimedia rate-limited repeated image downloads during expansion, so the final clean dataset combines crawled landscape images with deterministic crop/color augmentations derived from those crawled images. Augmented images are explicitly marked as `source=local_augmentation` in the manifest."""
        ),
        code_cell(
            """image_labels["source"].value_counts().rename("image_count").to_frame()"""
        ),
        code_cell(
            """image_labels.head(10)"""
        ),
        markdown_cell(
            """## 5. Data Cleaning Summary

Images are accepted only if they are readable RGB images, have sufficient dimensions, and are not nearly blank. Every accepted record keeps `dataset_label=landscape` and `image_domain=landscape`."""
        ),
        code_cell(
            """clean_manifest[["image_id", "source", "query", "width", "height", "clean_status", "clean_reason"]].head(12)"""
        ),
        markdown_cell(
            """## 6. Image EDA / Dataset Summary

The final dataset contains 20 clean landscape images. This is enough to compare K-Means behavior across multiple visual conditions while keeping the coursework runtime manageable."""
        ),
        code_cell(
            """eda = clean_manifest.groupby("source").agg(
    images=("image_id", "count"),
    min_width=("width", "min"),
    min_height=("height", "min"),
    max_width=("width", "max"),
    max_height=("height", "max"),
)
eda"""
        ),
        markdown_cell(
            """## 7. Preprocessing

Each landscape image is resized to a maximum side of 128 pixels for retraining. The image is converted into `RGB`, `HSV`, or `LAB`, flattened into `(height * width, n_features)`, and optionally extended with normalized spatial coordinates `(x, y)`."""
        ),
        code_cell(
            """comparison[["image_id", "k", "color_space", "use_xy", "max_side"]].drop_duplicates().head()"""
        ),
        markdown_cell(
            """## 8. K-Means Training

The experiment grid trains K-Means from scratch over:

- `K = 2..10`
- color spaces: `RGB`, `HSV`, `LAB`
- spatial features: disabled and enabled

This produces `20 * 9 * 3 * 2 = 1080` model runs."""
        ),
        code_cell(
            """comparison.shape, comparison[["k", "color_space", "use_xy"]].drop_duplicates().shape"""
        ),
        markdown_cell(
            """## 9. Model Comparison

The model comparison table records numerical quality metrics and paths to generated artifacts for every run."""
        ),
        code_cell(
            """comparison.sort_values("ranking_score")[
    ["image_id", "k", "color_space", "use_xy", "silhouette_sample", "davies_bouldin_sample", "inertia_per_pixel", "cluster_balance", "ranking_score"]
].head(15)"""
        ),
        markdown_cell(
            """## 10. Best Model Selection

For each image, the selected model is the configuration with the lowest `ranking_score`. This favors compact, separated clusters while penalizing collapsed or highly imbalanced cluster assignments."""
        ),
        code_cell(
            """best = comparison.sort_values("ranking_score").groupby("image_id").head(1)
best[["image_id", "k", "color_space", "use_xy", "silhouette_sample", "davies_bouldin_sample", "ranking_score", "output_comparison"]]"""
        ),
        markdown_cell(
            """## 11. Output Images After Training

The pipeline exports segmented images, side-by-side comparison figures, K-grid figures, pixel label arrays, and `.npz` model artifacts."""
        ),
        code_cell(
            """for path in best["output_comparison"].head(5):
    display(Image(filename=path))"""
        ),
        markdown_cell(
            """## 12. UI Usage

The Streamlit UI lets a user upload a landscape image, choose `K`, select a color space, toggle spatial features, and inspect clustering metrics after segmentation.

Run:

```bash
streamlit run app.py
```"""
        ),
        markdown_cell(
            """## 13. Source Review and Final Alignment

The final review checks source compilation, tests, notebook JSON/AST validity, metrics files, generated figures, pixel labels, model artifacts, and assignment alignment."""
        ),
        code_cell(
            """review"""
        ),
        markdown_cell(
            """## 14. Conclusion

The improved project remains aligned with the K-Means Image Segmentation exercise. It uses landscape images, expands the clean dataset to 20 images, retrains 1080 K-Means configurations, compares models with unsupervised metrics, and exports visual artifacts for inspection. It does not use supervised segmentation masks, object detection, or deep learning segmentation models."""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(str(notebook_path).encode("unicode_escape").decode("ascii"))


if __name__ == "__main__":
    main()
