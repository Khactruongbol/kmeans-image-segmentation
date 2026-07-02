# K-Means Image Segmentation - Professional Implementation Plan

## 1. Objective

Build a reproducible Computer Vision pipeline for **K-Means-based image segmentation** on landscape and natural-scene images.

The implementation must solve the assignment directly:

> Given an image, segment it into `K` clusters using the K-Means algorithm and visualize the original and segmented images side by side.

This is an unsupervised color-clustering workflow. The purpose is not to detect semantic objects, but to partition pixels into visually coherent color groups.

## 2. Recommended Landscape Data Strategy

Landscape images are preferred because K-Means can clearly separate large color regions such as sky, vegetation, water, snow, rock, sand, and clouds.

Use this priority order:

1. **Local curated image**: keep at least one landscape image in `data/raw/manual/`.
2. **Wikimedia Commons API**: best for license-transparent academic examples.
3. **Pexels or Unsplash API**: best for high-quality natural-scene photos.
4. **COCO/Open Images subset**: best for dataset-backed reproducibility.
5. **Flickr API**: useful when license filtering is required.

Recommended queries:

```text
landscape
mountain landscape
forest landscape
lake landscape
beach landscape
desert landscape
sunset landscape
natural scenery
national park
```

## 3. Standard Crawl and Download Links

Use these official links in the implementation documentation and code comments:

| Source | Official link | Practical endpoint or method |
|---|---|---|
| COCO Dataset | `https://cocodataset.org/` | Download validation images from the official dataset portal |
| Open Images V7 | `https://storage.googleapis.com/openimages/web/index.html` | Dataset overview and metadata |
| Open Images V7 Download | `https://storage.googleapis.com/openimages/web/download_v7.html` | Official download instructions |
| Wikimedia Commons API | `https://commons.wikimedia.org/wiki/Commons:API` | Commons API reference |
| Wikimedia MediaWiki API | `https://commons.wikimedia.org/wiki/Commons:API/MediaWiki` | Query `imageinfo`, metadata, URL, dimensions, license |
| Unsplash API | `https://unsplash.com/documentation` | `GET /search/photos?query=landscape` |
| Pexels API | `https://www.pexels.com/api/documentation/` | `GET https://api.pexels.com/v1/search?query=landscape` |
| Flickr Photo Search API | `https://www.flickr.com/services/api/flickr.photos.search.html` | `flickr.photos.search` with `text`, `tags`, `license`, `media=photos` |

Do not implement browser scraping for image-search pages. Treat "crawl data" as API-based retrieval or official dataset download.

## 4. End-to-End Workflow

```text
Landscape image source
        |
        v
API/dataset download or local image selection
        |
        v
Manifest creation and image validation
        |
        v
Image loading with PIL/OpenCV
        |
        v
RGB normalization and optional color-space conversion
        |
        v
Resize and pixel-feature flattening
        |
        v
K-Means clustering
        |
        v
Pixel-to-centroid replacement
        |
        v
Segmented image reconstruction
        |
        v
Side-by-side visualization and run summary
```

## 5. Project Structure

Generate the following structure:

```text
kmeans_image_segmentation/
├── data/
│   ├── raw/
│   │   ├── manual/
│   │   └── api/
│   └── manifest/
├── reports/
│   ├── figures/
│   └── metrics/
├── scripts/
│   └── run_segmentation.py
├── src/
│   ├── data_sources.py
│   ├── image_io.py
│   ├── kmeans.py
│   ├── segment.py
│   └── visualize.py
└── tests/
```

## 6. Module Implementation Contract

### 6.1. `src/data_sources.py`

Responsibilities:

- Retrieve landscape image URLs through official APIs.
- Download selected images into `data/raw/api/`.
- Write a manifest under `data/manifest/`.
- Fall back to `data/raw/manual/` when API keys or network access are unavailable.

Required public functions:

```python
def build_landscape_manifest(records: list[dict], output_path: str) -> None:
    """Persist image metadata for reproducibility."""

def download_image(url: str, output_path: str, timeout: int = 30) -> str:
    """Download one image file and return the local path."""

def collect_wikimedia_landscapes(query: str, limit: int, output_dir: str) -> list[dict]:
    """Collect landscape image metadata and files from Wikimedia Commons."""

def collect_pexels_landscapes(query: str, limit: int, output_dir: str) -> list[dict]:
    """Collect landscape image metadata and files from Pexels API."""

def collect_unsplash_landscapes(query: str, limit: int, output_dir: str) -> list[dict]:
    """Collect landscape image metadata and files from Unsplash API."""
```

Environment variables:

```text
PEXELS_API_KEY
UNSPLASH_ACCESS_KEY
FLICKR_API_KEY
```

Minimum manifest fields:

```text
image_id, source, query, source_url, download_url, license, author,
width, height, local_path, downloaded_at
```

### 6.2. `src/image_io.py`

Responsibilities:

- Load image files.
- Convert all inputs to RGB.
- Convert RGB images to clustering color spaces.
- Resize images while preserving aspect ratio.

Required public functions:

```python
def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as RGB with shape (H, W, 3) and dtype uint8."""

def resize_max_side(image_rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Resize image so the longest side is at most max_side."""

def convert_color_space(image_rgb: np.ndarray, color_space: str) -> np.ndarray:
    """Convert RGB image to rgb, hsv, or lab feature image."""
```

Rules:

- If using OpenCV, convert `BGR -> RGB` after reading.
- If using PIL, call `.convert("RGB")`.
- Preserve a clean RGB copy for visualization.

### 6.3. `src/kmeans.py`

Responsibilities:

- Implement K-Means from scratch with NumPy.
- Support deterministic runs through `random_state`.
- Store standard estimator attributes.

Required class:

```python
class KMeansFromScratch:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        ...

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        ...
```

Required attributes after `fit`:

```python
labels_: np.ndarray
cluster_centers_: np.ndarray
inertia_: float
n_iter_: int
```

Algorithm pseudocode:

```text
validate X as a 2D numeric matrix
sample K rows from X as initial centroids
for iteration in range(max_iter):
    compute pairwise squared Euclidean distances
    assign each sample to nearest centroid
    recompute each centroid as the mean of assigned samples
    if a cluster is empty, reinitialize its centroid from a valid sample
    compute centroid shift
    if centroid shift < tol, stop
store labels, centers, inertia, and iteration count
```

### 6.4. `src/segment.py`

Responsibilities:

- Build pixel features.
- Reconstruct the segmented RGB image.
- Handle inverse conversion from `HSV` or `LAB` back to RGB.

Required public functions:

```python
def build_pixel_features(
    image_rgb: np.ndarray,
    color_space: str = "lab",
    use_xy: bool = False,
    xy_weight: float = 0.1,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return pixel features with shape (H*W, n_features) and image shape."""

def reconstruct_segmented_image(
    labels: np.ndarray,
    centers: np.ndarray,
    image_shape: tuple[int, int],
    color_space: str = "lab",
) -> np.ndarray:
    """Return segmented RGB image with shape (H, W, 3), dtype uint8."""
```

Default recommendation:

- Use `LAB` as the default color space because it often produces more perceptually meaningful color clusters than raw RGB.
- Keep `RGB` available because it is easiest for students to understand.

### 6.5. `src/visualize.py`

Responsibilities:

- Save segmented image.
- Save original-vs-segmented comparison.
- Save optional multi-`K` comparison grid.

Required public functions:

```python
def save_image(image_rgb: np.ndarray, output_path: str) -> None:
    """Save an RGB image as PNG or JPG."""

def save_side_by_side(
    original_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
    output_path: str,
    title: str | None = None,
) -> None:
    """Save a side-by-side comparison plot."""

def save_k_grid(
    original_rgb: np.ndarray,
    segmented_by_k: dict[int, np.ndarray],
    output_path: str,
) -> None:
    """Save a grid comparing segmentation results for multiple K values."""
```

## 7. Command-Line Interface

Generate `scripts/run_segmentation.py` with this interface:

```bash
python scripts/run_segmentation.py \
  --image data/raw/manual/landscape.jpg \
  --k 4 \
  --color-space lab \
  --max-side 512 \
  --output-dir reports
```

Required arguments:

```text
--image
--k
--color-space {rgb,hsv,lab}
--max-side
--random-state
--output-dir
```

Optional arguments:

```text
--compare-k 2,3,4,5,6,7,8
--use-xy
--xy-weight 0.1
```

Expected outputs:

```text
reports/figures/{image_stem}_k4_lab_segmented.png
reports/figures/{image_stem}_k4_lab_comparison.png
reports/metrics/{image_stem}_k4_lab_summary.json
```

## 8. Run Summary Schema

Write a JSON summary for every run:

```json
{
  "image_path": "data/raw/manual/landscape.jpg",
  "image_domain": "landscape",
  "k": 4,
  "color_space": "lab",
  "max_side": 512,
  "use_xy": false,
  "xy_weight": 0.1,
  "random_state": 42,
  "iterations": 14,
  "inertia": 123456.78,
  "output_segmented": "reports/figures/landscape_k4_lab_segmented.png",
  "output_comparison": "reports/figures/landscape_k4_lab_comparison.png"
}
```

## 9. AI Code Generation Prompt

Use this prompt to generate implementation code:

```text
Create a Python project for K-Means Image Segmentation on landscape images.

Requirements:
1. Keep the scope limited to unsupervised pixel clustering with K-Means.
2. Prefer landscape or natural-scene images as input examples.
3. Support local image loading and optional API-based image retrieval.
4. Use official data sources only: COCO, Open Images, Wikimedia Commons API,
   Unsplash API, Pexels API, or Flickr API.
5. Do not scrape Google Images or arbitrary HTML image-search pages.
6. Implement K-Means from scratch with NumPy.
7. Support RGB, HSV, and LAB color spaces.
8. Resize the image, flatten pixel features, cluster pixels, replace each pixel
   with its assigned centroid color, and reconstruct the segmented image.
9. Expose labels_, cluster_centers_, inertia_, and n_iter_.
10. Handle empty clusters explicitly.
11. Save the segmented image, side-by-side comparison, optional multi-K grid,
    and a JSON run summary.
12. Provide a CLI script and tests for shapes, convergence, reconstruction,
    empty clusters, and smoke execution.
13. Read API keys from environment variables only.
```

## 10. Testing Plan

### Unit Tests

- `test_load_image_rgb_returns_uint8_rgb`
- `test_resize_max_side_preserves_aspect_ratio`
- `test_build_pixel_features_shape`
- `test_kmeans_output_shapes`
- `test_kmeans_converges_on_two_color_image`
- `test_kmeans_handles_empty_cluster_without_nan`
- `test_reconstruct_segmented_image_shape_and_dtype`

### Smoke Test

```bash
python scripts/run_segmentation.py \
  --image data/raw/manual/sample_landscape.jpg \
  --k 4 \
  --color-space lab \
  --max-side 256
```

Smoke test success criteria:

- Process exits with code `0`.
- Segmented image exists.
- Side-by-side comparison exists.
- JSON summary exists.
- Output image shape matches the resized input shape.

## 11. Alignment Check Against the Assignment

| Assignment requirement | Implementation coverage |
|---|---|
| Segment an image into K clusters | `KMeansFromScratch` and pixel-to-centroid reconstruction |
| Use any image of choice | Landscape/natural-scene image domain with local and API options |
| Read image using OpenCV or PIL | `load_image_rgb()` |
| Convert to RGB, HSV, or LAB | `convert_color_space()` |
| Resize for faster processing | `resize_max_side()` |
| Flatten into a 2D pixel array | `build_pixel_features()` |
| Initialize K centroids | `KMeansFromScratch.fit()` |
| Assign pixels by Euclidean distance | Vectorized distance computation |
| Update centroids by mean | K-Means update step |
| Repeat until convergence | `max_iter` and `tol` |
| Replace pixels with centroid colors | `reconstruct_segmented_image()` |
| Reshape back to image dimensions | Reconstruction to `(H, W, 3)` |
| Visualize original and segmented images side by side | `save_side_by_side()` |

Conclusion: the workflow, rules, data strategy, and implementation contract are aligned with the K-Means Image Segmentation exercise. The data layer supplies landscape images only; the core learning task remains unsupervised pixel clustering.

## 12. Risk Controls

- **No API key**: use local image fallback.
- **Large images**: enforce `max_side`.
- **Poor RGB segmentation**: compare `LAB` and `HSV`.
- **Empty clusters**: reinitialize centroid safely.
- **Slow convergence**: enforce `max_iter`.
- **Scope drift**: reject supervised labels and deep learning models in the core pipeline.
