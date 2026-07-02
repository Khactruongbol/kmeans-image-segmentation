from __future__ import annotations

import csv
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
