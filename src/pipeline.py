from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .cleaning import (
    augment_clean_records_to_target,
    clean_manifest_records,
    create_image_label_file,
    image_quality_summary,
    write_clean_manifest,
)
from .data_sources import (
    build_landscape_manifest,
    collect_landscape_images,
    collect_wikimedia_category_landscapes,
    read_manifest,
)
from .image_io import load_image_rgb, resize_max_side
from .segment import build_pixel_features, segment_image_array
from .visualize import save_image, save_k_grid, save_side_by_side

DEFAULT_LANDSCAPE_QUERIES = [
    "mountain landscape",
    "forest landscape",
    "lake landscape",
    "beach landscape",
    "desert landscape",
    "sunset landscape",
    "national park",
    "natural scenery",
]

DEFAULT_WIKIMEDIA_CATEGORIES = [
    "Category:Landscape photographs",
    "Category:Mountain landscapes",
    "Category:Forest landscapes",
    "Category:Lake landscapes",
    "Category:Beach landscapes",
    "Category:Desert landscapes",
    "Category:Sunsets",
    "Category:National parks",
]


@dataclass
class SegmentationRun:
    image_id: str
    image_path: str
    image_domain: str
    dataset_label: str
    k: int
    color_space: str
    max_side: int
    use_xy: bool
    xy_weight: float
    random_state: int
    iterations: int
    inertia: float
    inertia_per_pixel: float
    silhouette_sample: float
    davies_bouldin_sample: float
    calinski_harabasz_sample: float
    cluster_balance: float
    ranking_score: float
    output_segmented: str
    output_comparison: str
    output_labels: str
    output_model: str


def _parse_csv_values(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def reset_generated_outputs(project_root: str | Path, keep_raw_data: bool = True) -> None:
    project_root = Path(project_root).resolve()
    targets = [
        project_root / "reports" / "figures",
        project_root / "reports" / "metrics",
        project_root / "data" / "labels",
        project_root / "models",
    ]
    if not keep_raw_data:
        targets.extend([project_root / "data" / "clean", project_root / "data" / "manifest"])
    for target in targets:
        resolved = target.resolve()
        if project_root not in resolved.parents and resolved != project_root:
            raise ValueError(f"Refusing to clear path outside project root: {target}")
        _clear_directory(resolved)


def collect_expanded_landscape_images(
    queries: list[str],
    target_raw: int,
    per_query_limit: int,
    output_dir: str | Path,
    preferred_source: str = "wikimedia",
) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    if preferred_source == "wikimedia":
        for category in DEFAULT_WIKIMEDIA_CATEGORIES:
            if len(records) >= target_raw:
                break
            category_records = collect_wikimedia_category_landscapes(
                category=category,
                limit=per_query_limit,
                output_dir=output_dir,
            )
            for record in category_records:
                key = record.get("download_url") or record.get("source_url") or record.get("image_id")
                if not key or key in seen:
                    continue
                seen.add(key)
                records.append(record)
                if len(records) >= target_raw:
                    break
    for query in queries:
        if len(records) >= target_raw:
            break
        query_records = collect_landscape_images(
            query=query,
            limit=per_query_limit,
            output_dir=output_dir,
            preferred_source=preferred_source,
        )
        for record in query_records:
            key = record.get("download_url") or record.get("source_url") or record.get("image_id")
            if not key or key in seen:
                continue
            seen.add(key)
            records.append(record)
            if len(records) >= target_raw:
                break
    return records


def compute_cluster_quality_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    inertia_per_pixel: float,
    random_state: int,
    sample_size: int = 1500,
) -> dict:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return {
            "silhouette_sample": float("nan"),
            "davies_bouldin_sample": float("nan"),
            "calinski_harabasz_sample": float("nan"),
            "cluster_balance": 0.0,
            "ranking_score": float("inf"),
        }

    cluster_balance = float(counts.min() / max(counts.max(), 1))
    rng = np.random.default_rng(random_state)
    n_samples = features.shape[0]
    if n_samples > sample_size:
        indices = rng.choice(n_samples, size=sample_size, replace=False)
        sample_X = features[indices]
        sample_labels = labels[indices]
    else:
        sample_X = features
        sample_labels = labels

    if np.unique(sample_labels).size < 2:
        silhouette = float("nan")
        davies = float("nan")
        calinski = float("nan")
    else:
        silhouette = float(silhouette_score(sample_X, sample_labels, metric="euclidean"))
        davies = float(davies_bouldin_score(sample_X, sample_labels))
        calinski = float(calinski_harabasz_score(sample_X, sample_labels))

    # Lower is better. Terms are normalized gently to keep the score interpretable.
    silhouette_penalty = 1.0 - (silhouette if np.isfinite(silhouette) else -1.0)
    davies_penalty = davies if np.isfinite(davies) else 10.0
    inertia_penalty = np.log1p(max(inertia_per_pixel, 0.0)) / 10.0
    balance_penalty = 1.0 - cluster_balance
    ranking_score = float(silhouette_penalty + davies_penalty + inertia_penalty + balance_penalty)
    return {
        "silhouette_sample": silhouette,
        "davies_bouldin_sample": davies,
        "calinski_harabasz_sample": calinski,
        "cluster_balance": cluster_balance,
        "ranking_score": ranking_score,
    }


def run_segmentation_experiments(
    clean_records: list[dict],
    output_root: str | Path,
    k_values: list[int],
    color_spaces: list[str],
    max_side: int,
    random_state: int,
    use_xy_modes: list[bool] | None = None,
    xy_weight: float = 0.1,
    metric_sample_size: int = 1500,
) -> list[SegmentationRun]:
    output_root = Path(output_root)
    figures_dir = output_root / "reports" / "figures"
    metrics_dir = output_root / "reports" / "metrics"
    labels_dir = output_root / "data" / "labels"
    models_dir = output_root / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    runs: list[SegmentationRun] = []
    use_xy_modes = use_xy_modes or [False, True]
    for record in clean_records:
        image_id = record["image_id"]
        image_path = record.get("clean_path") or record["local_path"]
        original = load_image_rgb(image_path)
        resized = resize_max_side(original, max_side)

        for color_space in color_spaces:
            for use_xy in use_xy_modes:
                segmented_for_grid: dict[int, np.ndarray] = {}
                for k in k_values:
                    segmented, model, labels = segment_image_array(
                        resized,
                        k=k,
                        color_space=color_space,
                        random_state=random_state,
                        use_xy=use_xy,
                        xy_weight=xy_weight,
                    )
                    features, _ = build_pixel_features(resized, color_space, use_xy, xy_weight)
                    inertia_per_pixel = float((model.inertia_ or 0.0) / max(labels.size, 1))
                    quality = compute_cluster_quality_metrics(
                        features,
                        labels,
                        model.cluster_centers_,
                        inertia_per_pixel,
                        random_state=random_state,
                        sample_size=metric_sample_size,
                    )
                    xy_tag = "xy" if use_xy else "color"
                    stem = f"{image_id}_k{k}_{color_space}_{xy_tag}"
                    segmented_path = figures_dir / f"{stem}_segmented.png"
                    comparison_path = figures_dir / f"{stem}_comparison.png"
                    labels_path = labels_dir / f"{stem}_labels.npy"
                    model_path = models_dir / f"{stem}_kmeans_model.npz"
                    save_image(segmented, segmented_path)
                    title = (
                        f"K={k}, color_space={color_space}, use_xy={use_xy}, "
                        f"silhouette={quality['silhouette_sample']:.3f}, iter={model.n_iter_}"
                    )
                    save_side_by_side(resized, segmented, comparison_path, title=title)
                    np.save(labels_path, labels.astype(np.int32))
                    np.savez(
                        model_path,
                        cluster_centers=model.cluster_centers_,
                        labels=labels.astype(np.int32),
                        k=np.array([k], dtype=np.int32),
                        color_space=np.array([color_space]),
                        use_xy=np.array([use_xy]),
                        inertia=np.array([float(model.inertia_ or 0.0)]),
                        n_iter=np.array([int(model.n_iter_ or 0)], dtype=np.int32),
                        random_state=np.array([random_state], dtype=np.int32),
                    )
                    segmented_for_grid[k] = segmented

                    runs.append(
                        SegmentationRun(
                            image_id=image_id,
                            image_path=str(image_path),
                            image_domain="landscape",
                            dataset_label="landscape",
                            k=int(k),
                            color_space=color_space,
                            max_side=max_side,
                            use_xy=use_xy,
                            xy_weight=xy_weight,
                            random_state=random_state,
                            iterations=int(model.n_iter_ or 0),
                            inertia=float(model.inertia_ or 0.0),
                            inertia_per_pixel=inertia_per_pixel,
                            silhouette_sample=quality["silhouette_sample"],
                            davies_bouldin_sample=quality["davies_bouldin_sample"],
                            calinski_harabasz_sample=quality["calinski_harabasz_sample"],
                            cluster_balance=quality["cluster_balance"],
                            ranking_score=quality["ranking_score"],
                            output_segmented=str(segmented_path),
                            output_comparison=str(comparison_path),
                            output_labels=str(labels_path),
                            output_model=str(model_path),
                        )
                    )

                if len(k_values) > 1:
                    grid_path = figures_dir / f"{image_id}_k_grid_{color_space}_{xy_tag}.png"
                    save_k_grid(resized, segmented_for_grid, grid_path)

    comparison_path = metrics_dir / "model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(runs[0]).keys()) if runs else list(SegmentationRun.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))
    return runs


def select_best_runs(runs: list[SegmentationRun]) -> list[dict]:
    best: list[dict] = []
    grouped: dict[str, list[SegmentationRun]] = {}
    for run in runs:
        grouped.setdefault(run.image_id, []).append(run)
    for image_id, image_runs in grouped.items():
        best_run = min(image_runs, key=lambda r: (r.ranking_score, -r.silhouette_sample, r.davies_bouldin_sample))
        best.append(asdict(best_run) | {"selection_rule": "lowest_ranking_score"})
    return best


def run_full_pipeline(
    project_root: str | Path,
    query: str = "mountain landscape",
    limit: int = 30,
    source: str = "wikimedia",
    k_values: list[int] | None = None,
    color_spaces: list[str] | None = None,
    max_side: int = 128,
    random_state: int = 42,
    skip_download: bool = False,
    queries: list[str] | None = None,
    target_clean: int = 20,
    per_query_limit: int = 5,
    use_xy_modes: list[bool] | None = None,
    metric_sample_size: int = 1500,
    reset_outputs: bool = False,
) -> dict:
    project_root = Path(project_root)
    if reset_outputs:
        reset_generated_outputs(project_root)
    raw_dir = project_root / "data" / "raw" / "api"
    manifest_dir = project_root / "data" / "manifest"
    clean_dir = project_root / "data" / "clean"
    labels_dir = project_root / "data" / "labels"
    metrics_dir = project_root / "reports" / "metrics"
    for directory in [raw_dir, manifest_dir, clean_dir, labels_dir, metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_manifest = manifest_dir / "raw_landscape_manifest.csv"
    if skip_download and raw_manifest.exists():
        raw_records = read_manifest(raw_manifest)
    else:
        query_list = queries or DEFAULT_LANDSCAPE_QUERIES
        raw_records = collect_expanded_landscape_images(
            queries=query_list,
            target_raw=limit,
            per_query_limit=per_query_limit,
            output_dir=raw_dir,
            preferred_source=source,
        )
        build_landscape_manifest(raw_records, raw_manifest)

    clean_manifest = manifest_dir / "clean_landscape_manifest.csv"
    clean_records = clean_manifest_records(raw_records, clean_dir, clean_manifest)
    if len(clean_records) < target_clean:
        clean_records = augment_clean_records_to_target(clean_records, target_clean, clean_dir)
        write_clean_manifest(clean_records, clean_manifest)
    create_image_label_file(clean_records, labels_dir / "image_labels.csv")

    if len(clean_records) > target_clean:
        clean_records = clean_records[:target_clean]
        create_image_label_file(clean_records, labels_dir / "image_labels.csv")
    k_values = k_values or list(range(2, 11))
    color_spaces = color_spaces or ["rgb", "hsv", "lab"]
    use_xy_modes = use_xy_modes or [False, True]
    runs = run_segmentation_experiments(
        clean_records,
        output_root=project_root,
        k_values=k_values,
        color_spaces=color_spaces,
        max_side=max_side,
        random_state=random_state,
        use_xy_modes=use_xy_modes,
        metric_sample_size=metric_sample_size,
    )
    best_runs = select_best_runs(runs)
    best_path = metrics_dir / "best_model_by_image.json"
    best_path.write_text(json.dumps(best_runs, indent=2), encoding="utf-8")

    summary = {
        "problem": "K-Means Image Segmentation on landscape images",
        "query": query,
        "source": source,
        "raw_images": len(raw_records),
        "clean_images": len(clean_records),
        "target_clean_images": target_clean,
        "image_quality": image_quality_summary(clean_records),
        "k_values": k_values,
        "color_spaces": color_spaces,
        "use_xy_modes": use_xy_modes,
        "metric_sample_size": metric_sample_size,
        "model_runs": len(runs),
        "comparison_file": str(metrics_dir / "model_comparison.csv"),
        "best_model_file": str(best_path),
        "image_label_file": str(labels_dir / "image_labels.csv"),
    }
    (metrics_dir / "workflow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_k_values(value: str) -> list[int]:
    return _parse_csv_values(value, int)


def parse_color_spaces(value: str) -> list[str]:
    return [item.lower() for item in _parse_csv_values(value, str)]


def parse_bool_modes(value: str) -> list[bool]:
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    modes = []
    for item in _parse_csv_values(value, str):
        key = item.lower()
        if key not in mapping:
            raise ValueError(f"Invalid boolean mode: {item}")
        modes.append(mapping[key])
    return modes
