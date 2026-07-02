from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_exists(path: str | Path, label: str) -> None:
    path = Path(path)
    if not path.exists():
        raise AssertionError(f"Missing {label}: {path}")


def validate_notebook(notebook_path: Path) -> None:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4:
        raise AssertionError("Notebook is not nbformat v4")
    for idx, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            ast.parse(source)


def main() -> None:
    metrics_dir = ROOT / "reports" / "metrics"
    manifest_dir = ROOT / "data" / "manifest"
    labels_dir = ROOT / "data" / "labels"
    figures_dir = ROOT / "reports" / "figures"
    models_dir = ROOT / "models"
    notebook_path = ROOT / "notebooks" / "01_kmeans_image_segmentation_report.ipynb"

    required = [
        (manifest_dir / "raw_landscape_manifest.csv", "raw manifest"),
        (manifest_dir / "clean_landscape_manifest.csv", "clean manifest"),
        (labels_dir / "image_labels.csv", "image labels"),
        (metrics_dir / "model_comparison.csv", "model comparison"),
        (metrics_dir / "best_model_by_image.json", "best-model report"),
        (metrics_dir / "workflow_summary.json", "workflow summary"),
        (notebook_path, "Jupyter notebook"),
    ]
    for path, label in required:
        assert_exists(path, label)

    with (metrics_dir / "workflow_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("problem") != "K-Means Image Segmentation on landscape images":
        raise AssertionError("Workflow summary is not aligned with the assignment")
    if summary.get("clean_images", 0) <= 0:
        raise AssertionError("No clean landscape images found")
    if summary.get("clean_images", 0) < 20:
        raise AssertionError("Expected at least 20 clean landscape images")

    with (metrics_dir / "model_comparison.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AssertionError("No trained K-Means runs recorded")
    expected_runs = (
        int(summary["clean_images"])
        * len(summary["k_values"])
        * len(summary["color_spaces"])
        * len(summary["use_xy_modes"])
    )
    if len(rows) != expected_runs:
        raise AssertionError(f"Expected {expected_runs} model runs, found {len(rows)}")
    for row in rows:
        assert row["dataset_label"] == "landscape"
        assert row["image_domain"] == "landscape"
        for metric in [
            "inertia_per_pixel",
            "silhouette_sample",
            "davies_bouldin_sample",
            "calinski_harabasz_sample",
            "cluster_balance",
            "ranking_score",
        ]:
            if row.get(metric, "") == "":
                raise AssertionError(f"Missing metric {metric}")
        assert_exists(row["output_segmented"], "segmented image")
        assert_exists(row["output_comparison"], "comparison image")
        assert_exists(row["output_labels"], "pixel label array")
        assert_exists(row["output_model"], "model artifact")

    if not any(figures_dir.glob("*_k_grid_*.png")):
        raise AssertionError("No K-grid comparison images found")
    if not any(models_dir.glob("*_kmeans_model.npz")):
        raise AssertionError("No K-Means model artifacts found")
    validate_notebook(notebook_path)

    review = {
        "status": "passed",
        "clean_images": summary.get("clean_images"),
        "model_runs": len(rows),
        "expected_model_runs": expected_runs,
        "alignment": "The source implements unsupervised K-Means Image Segmentation with landscape images.",
        "checked_outputs": [
            "raw manifest",
            "clean manifest",
            "image labels",
            "pixel cluster labels",
            "model artifacts",
            "segmented images",
            "side-by-side comparisons",
            "K-grid comparisons",
            "Jupyter notebook JSON and code-cell AST",
        ],
    }
    output_path = metrics_dir / "source_review.json"
    output_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
