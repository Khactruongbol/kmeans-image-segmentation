from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

MANIFEST_FIELDS = [
    "image_id",
    "source",
    "query",
    "source_url",
    "download_url",
    "license",
    "author",
    "width",
    "height",
    "local_path",
    "downloaded_at",
    "image_domain",
    "dataset_label",
]


def _safe_stem(value: str, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return stem[:80] or fallback


def build_landscape_manifest(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in MANIFEST_FIELDS}
            writer.writerow(row)


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_image(url: str, output_path: str | Path, timeout: int = 30) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "kmeans-image-segmentation-coursework/1.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "image" not in content_type and output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"URL did not return an image: {url}")
    output_path.write_bytes(response.content)
    return str(output_path)


def collect_wikimedia_landscapes(
    query: str,
    limit: int,
    output_dir: str | Path,
    min_side: int = 512,
) -> list[dict]:
    """Collect landscape images from Wikimedia Commons using the public API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_url = "https://commons.wikimedia.org/w/api.php"
    headers = {"User-Agent": "kmeans-image-segmentation-coursework/1.0"}
    search_params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": max(limit * 8, 20),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|user|extmetadata",
        "iiurlwidth": 1600,
    }
    response = requests.get(api_url, params=search_params, headers=headers, timeout=30)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    records: list[dict] = []

    for page in pages.values():
        if len(records) >= limit:
            break
        title = page.get("title", "")
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        original_url = info.get("url", "")
        url = info.get("thumburl") or original_url
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        mime = info.get("mime", "")
        if min(width, height) < min_side:
            continue
        if not any(url.lower().split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            continue
        if mime and not any(token in mime.lower() for token in ("jpeg", "png")):
            continue

        ext = Path(urlparse(url).path).suffix.lower()
        image_id = _safe_stem(title.replace("File:", ""), f"wikimedia_{len(records) + 1}")
        local_path = output_dir / f"{image_id}{ext}"
        try:
            download_image(url, local_path)
        except Exception:
            continue

        metadata = info.get("extmetadata", {}) or {}
        license_name = metadata.get("LicenseShortName", {}).get("value", "")
        author = metadata.get("Artist", {}).get("value", "") or info.get("user", "")
        records.append(
            {
                "image_id": image_id,
                "source": "wikimedia_commons",
                "query": query,
                "source_url": info.get("descriptionurl", ""),
                "download_url": url,
                "original_url": original_url,
                "license": re.sub("<[^<]+?>", "", license_name),
                "author": re.sub("<[^<]+?>", "", author),
                "width": width,
                "height": height,
                "local_path": str(local_path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "image_domain": "landscape",
                "dataset_label": "landscape",
            }
        )
        time.sleep(0.2)
    return records


def collect_wikimedia_category_landscapes(
    category: str,
    limit: int,
    output_dir: str | Path,
    min_side: int = 512,
) -> list[dict]:
    """Collect landscape images from a Wikimedia Commons category."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_url = "https://commons.wikimedia.org/w/api.php"
    headers = {"User-Agent": "kmeans-image-segmentation-coursework/1.0"}
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmnamespace": 6,
        "gcmlimit": max(limit * 8, 20),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|user|extmetadata",
        "iiurlwidth": 1600,
    }
    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    records: list[dict] = []
    for page in pages.values():
        if len(records) >= limit:
            break
        title = page.get("title", "")
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        original_url = info.get("url", "")
        url = info.get("thumburl") or original_url
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        mime = info.get("mime", "")
        if min(width, height) < min_side:
            continue
        if not any(token in mime.lower() for token in ("jpeg", "png")):
            continue
        ext = ".jpg" if "jpeg" in mime.lower() else ".png"
        image_id = _safe_stem(title.replace("File:", ""), f"wikimedia_category_{len(records) + 1}")
        local_path = output_dir / f"{image_id}{ext}"
        try:
            download_image(url, local_path)
        except Exception:
            continue
        metadata = info.get("extmetadata", {}) or {}
        license_name = metadata.get("LicenseShortName", {}).get("value", "")
        author = metadata.get("Artist", {}).get("value", "") or info.get("user", "")
        records.append(
            {
                "image_id": image_id,
                "source": "wikimedia_commons_category",
                "query": category.replace("Category:", ""),
                "source_url": info.get("descriptionurl", ""),
                "download_url": url,
                "license": re.sub("<[^<]+?>", "", license_name),
                "author": re.sub("<[^<]+?>", "", author),
                "width": width,
                "height": height,
                "local_path": str(local_path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "image_domain": "landscape",
                "dataset_label": "landscape",
            }
        )
        time.sleep(0.1)
    return records


def collect_pexels_landscapes(query: str, limit: int, output_dir: str | Path) -> list[dict]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []
    output_dir = Path(output_dir)
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": min(limit, 80), "orientation": "landscape"}
    response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=30)
    response.raise_for_status()
    records: list[dict] = []
    for item in response.json().get("photos", [])[:limit]:
        url = item.get("src", {}).get("large2x") or item.get("src", {}).get("large")
        if not url:
            continue
        image_id = f"pexels_{item.get('id')}"
        local_path = output_dir / f"{image_id}.jpg"
        try:
            download_image(url, local_path)
        except Exception:
            continue
        records.append(
            {
                "image_id": image_id,
                "source": "pexels",
                "query": query,
                "source_url": item.get("url", ""),
                "download_url": url,
                "license": "Pexels License",
                "author": item.get("photographer", ""),
                "width": item.get("width", ""),
                "height": item.get("height", ""),
                "local_path": str(local_path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "image_domain": "landscape",
                "dataset_label": "landscape",
            }
        )
    return records


def collect_unsplash_landscapes(query: str, limit: int, output_dir: str | Path) -> list[dict]:
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        return []
    output_dir = Path(output_dir)
    params = {
        "query": query,
        "per_page": min(limit, 30),
        "orientation": "landscape",
        "client_id": access_key,
    }
    response = requests.get("https://api.unsplash.com/search/photos", params=params, timeout=30)
    response.raise_for_status()
    records: list[dict] = []
    for item in response.json().get("results", [])[:limit]:
        url = item.get("urls", {}).get("regular")
        if not url:
            continue
        image_id = f"unsplash_{item.get('id')}"
        local_path = output_dir / f"{image_id}.jpg"
        try:
            download_image(url, local_path)
        except Exception:
            continue
        user = item.get("user", {}) or {}
        records.append(
            {
                "image_id": image_id,
                "source": "unsplash",
                "query": query,
                "source_url": item.get("links", {}).get("html", ""),
                "download_url": url,
                "license": "Unsplash License",
                "author": user.get("name", ""),
                "width": item.get("width", ""),
                "height": item.get("height", ""),
                "local_path": str(local_path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "image_domain": "landscape",
                "dataset_label": "landscape",
            }
        )
    return records


def collect_landscape_images(
    query: str,
    limit: int,
    output_dir: str | Path,
    preferred_source: str = "wikimedia",
) -> list[dict]:
    if preferred_source == "pexels":
        records = collect_pexels_landscapes(query, limit, output_dir)
        if records:
            return records
    if preferred_source == "unsplash":
        records = collect_unsplash_landscapes(query, limit, output_dir)
        if records:
            return records
    return collect_wikimedia_landscapes(query, limit, output_dir)
