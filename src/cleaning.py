from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from .data_sources import MANIFEST_FIELDS
from .image_io import load_image_rgb, save_rgb_image

CLEAN_FIELDS = MANIFEST_FIELDS + [
    "clean_path",
    "clean_status",
    "clean_reason",
    "mean_intensity",
    "std_intensity",
]


def clean_manifest_records(
    records: list[dict],
    clean_dir: str | Path,
    output_manifest: str | Path,
    min_side: int = 256,
    min_std: float = 5.0,
) -> list[dict]:
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    cleaned: list[dict] = []
    for record in records:
        row = dict(record)
        try:
            image = load_image_rgb(row["local_path"])
            height, width = image.shape[:2]
            mean_intensity = float(image.mean())
            std_intensity = float(image.std())
            if min(height, width) < min_side:
                raise ValueError(f"image too small: {width}x{height}")
            if std_intensity < min_std:
                raise ValueError("image appears nearly blank")
            suffix = Path(row["local_path"]).suffix.lower() or ".jpg"
            clean_path = clean_dir / f"{row['image_id']}{suffix}"
            if Path(row["local_path"]).resolve() != clean_path.resolve():
                shutil.copy2(row["local_path"], clean_path)
            else:
                save_rgb_image(image, clean_path)
            row.update(
                {
                    "width": width,
                    "height": height,
                    "clean_path": str(clean_path),
                    "clean_status": "accepted",
                    "clean_reason": "",
                    "mean_intensity": round(mean_intensity, 4),
                    "std_intensity": round(std_intensity, 4),
                    "dataset_label": "landscape",
                    "image_domain": "landscape",
                    "downloaded_at": row.get("downloaded_at") or datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "clean_path": "",
                    "clean_status": "rejected",
                    "clean_reason": str(exc),
                    "mean_intensity": "",
                    "std_intensity": "",
                }
            )
        cleaned.append(row)

    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        for row in cleaned:
            writer.writerow({field: row.get(field, "") for field in CLEAN_FIELDS})
    return [row for row in cleaned if row.get("clean_status") == "accepted"]


def create_image_label_file(clean_records: list[dict], output_path: str | Path) -> None:
    fields = ["image_id", "dataset_label", "image_domain", "source", "query", "clean_path", "source_url"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in clean_records:
            writer.writerow({field: row.get(field, "") for field in fields})


def image_quality_summary(clean_records: list[dict]) -> dict:
    if not clean_records:
        return {"accepted_images": 0}
    widths = np.array([float(r.get("width", 0) or 0) for r in clean_records])
    heights = np.array([float(r.get("height", 0) or 0) for r in clean_records])
    return {
        "accepted_images": int(len(clean_records)),
        "min_width": int(widths.min()),
        "min_height": int(heights.min()),
        "max_width": int(widths.max()),
        "max_height": int(heights.max()),
    }


def augment_clean_records_to_target(
    clean_records: list[dict],
    target_count: int,
    clean_dir: str | Path,
) -> list[dict]:
    """Create deterministic landscape augmentations when the public API is rate-limited."""
    if len(clean_records) >= target_count or not clean_records:
        return clean_records[:target_count]

    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    augmented = list(clean_records)
    variant_idx = 0
    crop_specs = [
        (0.0, 0.0, 0.78, 0.78, 1.05, 1.02),
        (0.22, 0.0, 1.0, 0.78, 0.95, 1.08),
        (0.0, 0.18, 0.78, 1.0, 1.08, 0.96),
        (0.22, 0.18, 1.0, 1.0, 0.98, 1.12),
        (0.1, 0.1, 0.9, 0.9, 1.04, 1.04),
    ]
    source_index = 0
    while len(augmented) < target_count:
        record = clean_records[source_index % len(clean_records)]
        source_index += 1
        try:
            image = Image.open(record["clean_path"]).convert("RGB")
        except Exception:
            continue
        width, height = image.size
        spec = crop_specs[variant_idx % len(crop_specs)]
        left = int(width * spec[0])
        top = int(height * spec[1])
        right = int(width * spec[2])
        bottom = int(height * spec[3])
        crop = image.crop((left, top, right, bottom)).resize((min(width, 1600), min(height, 1200)), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Color(crop).enhance(spec[4])
        crop = ImageEnhance.Contrast(crop).enhance(spec[5])
        if variant_idx % 2 == 1:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_id = f"{record['image_id']}_aug_{variant_idx + 1:02d}"
        clean_path = clean_dir / f"{image_id}.jpg"
        crop.save(clean_path, quality=92)
        arr = np.asarray(crop)
        new_record = dict(record)
        new_record.update(
            {
                "image_id": image_id,
                "source": "local_augmentation",
                "query": f"augmentation:{record.get('query', 'landscape')}",
                "source_url": record.get("source_url", ""),
                "download_url": record.get("download_url", ""),
                "author": record.get("author", ""),
                "width": crop.size[0],
                "height": crop.size[1],
                "local_path": str(clean_path),
                "clean_path": str(clean_path),
                "clean_status": "accepted",
                "clean_reason": "deterministic crop/color augmentation from crawled landscape image",
                "mean_intensity": round(float(arr.mean()), 4),
                "std_intensity": round(float(arr.std()), 4),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "image_domain": "landscape",
                "dataset_label": "landscape",
            }
        )
        augmented.append(new_record)
        variant_idx += 1
    return augmented[:target_count]


def write_clean_manifest(clean_records: list[dict], output_manifest: str | Path) -> None:
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        for row in clean_records:
            writer.writerow({field: row.get(field, "") for field in CLEAN_FIELDS})


def _numeric_values(records: list[dict], key: str) -> np.ndarray:
    values = []
    for record in records:
        try:
            values.append(float(record.get(key, 0) or 0))
        except (TypeError, ValueError):
            values.append(0.0)
    return np.asarray(values, dtype=np.float64)


def _outlier_flags(values: np.ndarray) -> tuple[np.ndarray, dict]:
    if values.size == 0:
        return np.asarray([], dtype=bool), {}
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    std = float(values.std(ddof=0))
    mean = float(values.mean())
    if std == 0:
        z_flags = np.zeros(values.shape, dtype=bool)
    else:
        z_flags = np.abs((values - mean) / std) > 2.5
    iqr_flags = (values < lower) | (values > upper)
    return iqr_flags | z_flags, {
        "mean": mean,
        "std": std,
        "q1": float(q1),
        "q3": float(q3),
        "iqr_lower": float(lower),
        "iqr_upper": float(upper),
    }


def audit_and_balance_records(
    clean_records: list[dict],
    target_count: int,
    max_augmentation: int,
    output_report: str | Path,
) -> list[dict]:
    """Audit image-quality outliers and select a source-balanced training set."""
    if not clean_records:
        raise ValueError("No clean records available for balancing")

    records = [dict(record) for record in clean_records]
    for record in records:
        width = float(record.get("width", 0) or 0)
        height = float(record.get("height", 0) or 0)
        record["aspect_ratio"] = round(width / max(height, 1.0), 6)
        try:
            record["file_size_bytes"] = Path(record.get("clean_path") or record.get("local_path")).stat().st_size
        except OSError:
            record["file_size_bytes"] = 0

    feature_keys = ["width", "height", "aspect_ratio", "mean_intensity", "std_intensity", "file_size_bytes"]
    stats: dict[str, dict] = {}
    outlier_by_id: dict[str, list[str]] = {record["image_id"]: [] for record in records}
    for key in feature_keys:
        values = _numeric_values(records, key)
        flags, key_stats = _outlier_flags(values)
        stats[key] = key_stats
        for record, flagged in zip(records, flags):
            if flagged:
                outlier_by_id[record["image_id"]].append(key)

    non_outliers = [record for record in records if not outlier_by_id[record["image_id"]]]
    candidate_pool = non_outliers if len(non_outliers) >= target_count else records
    real_records = [record for record in candidate_pool if record.get("source") != "local_augmentation"]
    aug_records = [record for record in candidate_pool if record.get("source") == "local_augmentation"]

    def sort_key(record: dict) -> tuple:
        reasons = len(outlier_by_id[record["image_id"]])
        source_rank = 1 if record.get("source") == "local_augmentation" else 0
        return (reasons, source_rank, str(record.get("query", "")), str(record.get("image_id", "")))

    real_records = sorted(real_records, key=sort_key)
    aug_records = sorted(aug_records, key=sort_key)
    selected = real_records[:target_count]
    remaining_slots = target_count - len(selected)
    if remaining_slots > 0:
        selected.extend(aug_records[: min(max_augmentation, remaining_slots)])
    if len(selected) < target_count:
        used_ids = {record["image_id"] for record in selected}
        for record in sorted(records, key=sort_key):
            if record["image_id"] in used_ids:
                continue
            selected.append(record)
            used_ids.add(record["image_id"])
            if len(selected) >= target_count:
                break
    selected = selected[:target_count]
    selected_ids = {record["image_id"] for record in selected}

    excluded = []
    kept = []
    for record in records:
        entry = {
            "image_id": record["image_id"],
            "source": record.get("source", ""),
            "query": record.get("query", ""),
            "width": record.get("width", ""),
            "height": record.get("height", ""),
            "aspect_ratio": record.get("aspect_ratio", ""),
            "mean_intensity": record.get("mean_intensity", ""),
            "std_intensity": record.get("std_intensity", ""),
            "file_size_bytes": record.get("file_size_bytes", ""),
            "outlier_features": outlier_by_id[record["image_id"]],
        }
        if record["image_id"] in selected_ids:
            kept.append(entry)
        else:
            excluded.append(entry)

    report = {
        "target_count": target_count,
        "max_augmentation": max_augmentation,
        "input_records": len(records),
        "selected_records": len(selected),
        "real_selected": sum(1 for record in selected if record.get("source") != "local_augmentation"),
        "augmentation_selected": sum(1 for record in selected if record.get("source") == "local_augmentation"),
        "feature_stats": stats,
        "kept": kept,
        "excluded": excluded,
        "source_counts": {
            source: sum(1 for record in selected if record.get("source") == source)
            for source in sorted({record.get("source", "") for record in selected})
        },
        "query_counts": {
            query: sum(1 for record in selected if record.get("query") == query)
            for query in sorted({record.get("query", "") for record in selected})
        },
    }
    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return selected
