# K-Means Image Segmentation - Working Rules

## 1. Problem Scope

This project implements **unsupervised image segmentation with K-Means clustering**.

The target exercise is:

> Segment a given image into `K` clusters using the K-Means algorithm.

The system must stay focused on pixel-level color clustering:

- Load a natural landscape image.
- Convert the image into a suitable color space: `RGB`, `HSV`, or `LAB`.
- Resize the image for computational efficiency.
- Flatten the image into a 2D pixel-feature matrix.
- Apply K-Means clustering using Euclidean distance.
- Replace each pixel with the centroid color of its assigned cluster.
- Reconstruct and visualize the segmented image.

The project must not drift into supervised semantic segmentation, object detection, instance segmentation, U-Net, Mask R-CNN, SAM, or any deep learning model. Dataset annotations may be used only for filtering or image selection, not for training.

## 2. Preferred Image Domain

Landscape and natural-scene images are the default input domain because they usually contain broad, visually separable color regions such as sky, mountain, water, grass, forest, sand, and clouds. This makes K-Means color quantization easier to inspect and explain.

Recommended search queries:

- `landscape`
- `mountain landscape`
- `forest landscape`
- `beach landscape`
- `lake landscape`
- `desert landscape`
- `sunset landscape`
- `national park`
- `countryside`
- `natural scenery`

Avoid primary examples dominated by text overlays, logos, screenshots, cartoons, low-light noise, heavy watermarks, or extreme close-ups.

## 3. Standard Data Sources and Crawl Links

Use official APIs, dataset portals, or documented download mechanisms. Do not scrape arbitrary image-search result pages.

| Priority | Source | Standard access link | Recommended use |
|---|---|---|---|
| 1 | Local curated landscape image | `data/raw/manual/` | Reliable offline baseline for development and grading |
| 2 | COCO Dataset | `https://cocodataset.org/` | Download a small validation subset and select natural-scene images |
| 3 | Open Images V7 | `https://storage.googleapis.com/openimages/web/index.html` and `https://storage.googleapis.com/openimages/web/download_v7.html` | Download a small subset by class/query metadata |
| 4 | Wikimedia Commons API | `https://commons.wikimedia.org/wiki/Commons:API` and `https://commons.wikimedia.org/wiki/Commons:API/MediaWiki` | Search landscape files with explicit metadata and license information |
| 5 | Unsplash API | `https://unsplash.com/documentation` | Search high-quality landscape photos through the official JSON API |
| 6 | Pexels API | `https://www.pexels.com/api/documentation/` | Search landscape photos through a REST JSON API |
| 7 | Flickr API | `https://www.flickr.com/services/api/flickr.photos.search.html` | Search public photos with license filters and metadata |

## 4. Data Acquisition Rules

- The default image category must be landscape or natural scenery.
- The implementation must support a local-image fallback so the workflow can run without network access.
- API credentials must be read from environment variables, never hard-coded.
- Each downloaded image must be recorded in a manifest file.
- Each image must have a traceable source URL and license or terms reference.
- Prefer images with width and height of at least 512 pixels.
- Accepted file formats are `jpg`, `jpeg`, and `png`.
- Exclude corrupted files, very small files, blank images, screenshots, memes, text-heavy images, and heavily watermarked images.
- Do not bypass API rate limits, robots rules, authentication requirements, or provider terms.
- Do not scrape Google Images, Facebook, Instagram, Pinterest, or login-gated pages.

Required manifest schema:

```csv
image_id,source,query,source_url,download_url,license,author,width,height,local_path,downloaded_at
```

## 5. Algorithm Rules

- `K` must be a positive integer. Recommended experiments use `K = 2..8`.
- The primary feature vector is a color vector: `[R, G, B]`, `[H, S, V]`, or `[L, A, B]`.
- Euclidean distance is the default clustering distance.
- Initialization must be reproducible through `random_state`.
- The required baseline initialization is random pixel-centroid initialization.
- `k-means++` may be implemented only as an optional enhancement.
- The iterative loop must follow the standard K-Means procedure:
  1. Initialize `K` centroids.
  2. Assign each pixel to the nearest centroid.
  3. Update each centroid as the mean of assigned pixels.
  4. Repeat until convergence or `max_iter`.
- Convergence must be defined by centroid shift tolerance, unchanged labels, or reaching `max_iter`.
- Empty clusters must be handled explicitly by reinitializing the centroid from a valid pixel or a high-error sample.
- The implementation must expose `labels_`, `cluster_centers_`, `inertia_`, and `n_iter_`.

## 6. Preprocessing Rules

- If OpenCV is used, convert `BGR` to `RGB` before display or RGB processing.
- Images with alpha channels must be converted to RGB.
- Resize before clustering; default `max_side` should be `512` pixels.
- Preserve the resized image shape `(height, width)` for reconstruction.
- Flatten the image into shape `(height * width, n_features)`.
- Use `float32` or `float64` for numerical clustering.
- Convert the final image back to `uint8`.
- Optional spatial features `[x, y]` are allowed only as an advanced mode and must be clearly documented.

## 7. Output Rules

The program must produce:

- A segmented image.
- A side-by-side comparison of original and segmented images.
- A run summary file with parameters and metrics.

Recommended output paths:

```text
reports/figures/{image_stem}_k{K}_{color_space}_segmented.png
reports/figures/{image_stem}_k{K}_{color_space}_comparison.png
reports/metrics/{image_stem}_k{K}_{color_space}_summary.json
```

For multiple `K` values, save a comparison grid such as:

```text
reports/figures/{image_stem}_k_grid_{color_space}.png
```

## 8. Project Structure Rules

Recommended implementation structure:

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

Core clustering logic should live in `src/`, not only inside a notebook. A notebook may be added for reporting, but it must call the same reusable functions.

## 9. Validation Rules

The final system must verify:

- The input image exists and is readable.
- The image is converted to RGB correctly.
- The resized image has valid dimensions.
- The flattened feature matrix has shape `(height * width, n_features)`.
- K-Means returns one label per pixel.
- The centroid matrix has shape `(K, n_features)`.
- The segmented image has shape `(height, width, 3)`.
- Output arrays contain no `NaN` or `Inf`.
- The side-by-side visualization is generated.
- The workflow remains centered on K-Means image segmentation.

## 10. Acceptance Checklist

- [ ] Landscape/natural-scene images are used as the default data domain.
- [ ] Standard crawl/download links are documented.
- [ ] Data collection uses official APIs or dataset download mechanisms.
- [ ] A local-image fallback exists.
- [ ] K-Means is implemented as the core segmentation algorithm.
- [ ] Color spaces `RGB`, `HSV`, and `LAB` are supported or clearly planned.
- [ ] Output includes segmented image, comparison image, and run summary.
- [ ] Tests cover shapes, convergence, empty clusters, reconstruction, and smoke execution.
- [ ] The workflow matches the original K-Means Image Segmentation exercise.
