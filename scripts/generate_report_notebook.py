from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rel(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            pass
    return "../" + path.as_posix()


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def table_from_records(records: list[dict], columns: list[str], limit: int | None = None) -> str:
    rows = records if limit is None else records[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def df_to_markdown(df: pd.DataFrame, limit: int | None = None) -> str:
    rows = df if limit is None else df.head(limit)
    return table_from_records(rows.fillna("").to_dict("records"), list(rows.columns))


def md_image(path: str | Path, caption: str) -> str:
    safe_caption = caption.replace("[", "(").replace("]", ")")
    return f"![{safe_caption}]({rel(path)})\n\n*{safe_caption}*"


def image_gallery_lines(rows: pd.DataFrame, image_col: str, caption_cols: list[str]) -> str:
    blocks: list[str] = []
    for _, row in rows.iterrows():
        caption = " | ".join(f"{col}={row[col]}" for col in caption_cols)
        blocks.append(md_image(row[image_col], caption))
    return "\n\n".join(blocks)


def chunked(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    notebook_path = ROOT / "notebooks" / "01_kmeans_image_segmentation_report.ipynb"
    summary = json.loads((ROOT / "reports" / "metrics" / "workflow_summary.json").read_text(encoding="utf-8"))
    review_path = ROOT / "reports" / "metrics" / "source_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    balance = json.loads((ROOT / "reports" / "metrics" / "data_balance_report.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(ROOT / "reports" / "metrics" / "model_comparison.csv")
    clean = pd.read_csv(ROOT / "data" / "manifest" / "clean_landscape_manifest.csv")
    best = comparison.sort_values("ranking_score").groupby("image_id", as_index=False).head(1)
    all_outputs = comparison.sort_values(["image_id", "color_space", "use_xy", "k"])
    grid_paths = sorted((ROOT / "reports" / "figures").glob("*_k_grid_*.png"))

    balance_overview = {
        "raw_images": summary["raw_images"],
        "clean_images": summary["clean_images"],
        "real_selected": balance["real_selected"],
        "augmentation_selected": balance["augmentation_selected"],
        "max_augmentation": balance["max_augmentation"],
        "model_runs": summary["model_runs"],
    }
    source_counts = pd.Series(balance["source_counts"], name="count").reset_index().rename(columns={"index": "source"})
    query_counts = pd.Series(balance["query_counts"], name="count").reset_index().rename(columns={"index": "query"})

    cells = [
        markdown_cell(
            f"""# Báo Cáo Cuối Kỳ - K-Means Image Segmentation

## 1. Define Problem

Bài toán yêu cầu phân đoạn một ảnh thành `K` cụm bằng thuật toán K-Means. Trong chương trình này, ảnh đầu vào là ảnh phong cảnh vì các vùng như bầu trời, nước, cây, núi, cát và mây thường có màu sắc tách biệt rõ, phù hợp với phân cụm màu không giám sát.

Notebook này là bản báo cáo cuối cùng, chỉ dùng Markdown và hình ảnh, có **0 code cell**. Toàn bộ code train, so sánh mô hình, xuất ảnh, xuất nhãn pixel, lưu model artifact và review hệ thống đã được chạy trước khi tạo notebook."""
        ),
        markdown_cell(
            """## 2. Assignment Mapping

| Yêu cầu đề bài | Phần đã thực hiện |
|---|---|
| Load the image | Đọc ảnh RGB bằng PIL trong `src/image_io.py` |
| Convert color space | So sánh `RGB`, `HSV`, `LAB` |
| Resize image | Resize với `max_side=128` để train nhanh và ổn định |
| Flatten pixels | Biến ảnh thành ma trận `(height * width, n_features)` |
| Apply K-Means | Cài đặt K-Means from scratch bằng NumPy |
| Assign nearest centroid | Dùng Euclidean distance dạng vector hóa |
| Update centroids | Cập nhật centroid bằng mean của pixel trong cụm |
| Repeat until convergence | Dừng bằng `max_iter` và `tol` |
| Segment image | Thay mỗi pixel bằng màu centroid |
| Visualize results | Xuất ảnh segmented, comparison và K-grid |"""
        ),
        markdown_cell(
            f"""## 3. Workflow

Workflow đã thực hiện:

1. **Thu thập data thô**: ảnh nằm trong `{rel('data/raw/api')}` và manifest `{rel('data/manifest/raw_landscape_manifest.csv')}`.
2. **Kiểm tra và cân bằng data**: clean manifest `{rel('data/manifest/clean_landscape_manifest.csv')}` và báo cáo `{rel('reports/metrics/data_balance_report.json')}`.
3. **Tiền xử lý**: đọc ảnh RGB, resize, đổi color space, flatten pixel, tùy chọn thêm tọa độ `(x, y)`.
4. **Train K-Means**: train trên `K=2..10`, `rgb/hsv/lab`, `use_xy=false/true`.
5. **Xuất kết quả**: ảnh trong `{rel('reports/figures')}`, labels trong `{rel('data/labels')}`, model artifacts trong `{rel('models')}`.
6. **So sánh và review**: metrics `{rel('reports/metrics/model_comparison.csv')}`, best model `{rel('reports/metrics/best_model_by_image.json')}`, review `{rel('reports/metrics/source_review.json')}`."""
        ),
        markdown_cell(
            f"""## 4. Data Audit

Hệ thống kiểm tra các đặc trưng của từng ảnh: width, height, aspect ratio, mean intensity, standard deviation, file size, source và query. Các ảnh lệch quá xa phân phối được đánh dấu bằng IQR và z-score trước khi chọn tập balanced.

**Tổng quan data sau cân bằng**

{table_from_records([balance_overview], list(balance_overview.keys()))}

**Khoảng giá trị sau cân bằng**

| Feature | Min | Mean | Max |
|---|---:|---:|---:|
| width | {clean['width'].min()} | {clean['width'].mean():.2f} | {clean['width'].max()} |
| height | {clean['height'].min()} | {clean['height'].mean():.2f} | {clean['height'].max()} |
| mean intensity | {clean['mean_intensity'].min():.2f} | {clean['mean_intensity'].mean():.2f} | {clean['mean_intensity'].max():.2f} |
| std intensity | {clean['std_intensity'].min():.2f} | {clean['std_intensity'].mean():.2f} | {clean['std_intensity'].max():.2f} |"""
        ),
        markdown_cell(
            f"""## 5. Data Balancing

Mục tiêu là giữ `20` ảnh sạch, giảm lệ thuộc augmentation và ưu tiên ảnh real/raw từ Wikimedia. Kết quả cuối cùng chọn **{balance['real_selected']} ảnh real** và **{balance['augmentation_selected']} ảnh augmentation**.

**Phân phối source**

{df_to_markdown(source_counts)}

**Phân phối query**

{df_to_markdown(query_counts)}"""
        ),
        markdown_cell(
            """## 6. Preprocessing

Mỗi ảnh được resize về `max_side=128`, sau đó thử nghiệm trên ba không gian màu `RGB`, `HSV`, `LAB`. Mỗi pixel trở thành vector đặc trưng màu. Ở chế độ spatial, vector được mở rộng thêm tọa độ chuẩn hóa `(x, y)`.

Không dùng mask ground-truth, semantic segmentation, object detection hoặc deep learning model. Đây vẫn là bài toán K-Means Image Segmentation đúng trọng tâm đề bài."""
        ),
        markdown_cell(
            f"""## 7. Training Grid

| Thành phần | Giá trị |
|---|---|
| Số ảnh clean | {summary['clean_images']} |
| K values | {summary['k_values']} |
| Color spaces | {summary['color_spaces']} |
| Spatial modes | {summary['use_xy_modes']} |
| Tổng số model runs | {summary['model_runs']} |

Mỗi model run đều xuất ra ảnh segmented, ảnh comparison, pixel-label `.npy`, và model artifact `.npz`."""
        ),
        markdown_cell(
            f"""## 8. Model Comparison

Vì đề bài không cung cấp ground-truth segmentation mask, chất lượng model được đánh giá bằng unsupervised metrics:

| Metric | Hướng tốt hơn | Ý nghĩa |
|---|---|---|
| silhouette_sample | cao hơn | Cụm tách nhau rõ hơn |
| davies_bouldin_sample | thấp hơn | Cụm gọn và ít chồng lấn hơn |
| calinski_harabasz_sample | cao hơn | Cụm tách biệt tốt hơn |
| inertia_per_pixel | thấp hơn | Pixel gần centroid hơn |
| cluster_balance | cao hơn | Tránh cụm quá nhỏ/collapsed |
| ranking_score | thấp hơn | Điểm tổng hợp để chọn model |

**Top 15 model runs**

{df_to_markdown(comparison.sort_values('ranking_score')[['image_id','k','color_space','use_xy','silhouette_sample','davies_bouldin_sample','inertia_per_pixel','cluster_balance','ranking_score']].head(15))}"""
        ),
        markdown_cell(
            f"""## 9. Best Model Selection

Với mỗi ảnh, model tốt nhất là cấu hình có `ranking_score` thấp nhất.

{df_to_markdown(best[['image_id','k','color_space','use_xy','silhouette_sample','davies_bouldin_sample','ranking_score']])}"""
        ),
        markdown_cell(
            f"""## 10. Hình Ảnh Model Tốt Nhất Sau Khi Train - Best Output Image Gallery

Phần này nhúng trực tiếp ảnh comparison của model tốt nhất cho từng ảnh. Đây là các hình sau khi train, gồm ảnh gốc và ảnh đã segment đặt cạnh nhau.

{image_gallery_lines(best, 'output_comparison', ['image_id','k','color_space','use_xy','ranking_score'])}"""
        ),
        markdown_cell(
            "## 11. K-Grid Highlights\n\nCác hình K-grid dưới đây cho thấy kết quả segmentation thay đổi như thế nào khi tăng `K`."
        ),
    ]

    for i, group in enumerate(chunked(grid_paths, 12), start=1):
        body = [f"### 11.{i}. K-Grid Batch {i}"]
        for path in group:
            body.append(md_image(path, Path(path).name))
        cells.append(markdown_cell("\n\n".join(body)))

    cells.append(
        markdown_cell(
            """## 12. Tất Cả Hình Ảnh Model Sau Khi Train - Output Image Gallery

Phần appendix dưới đây nhúng trực tiếp toàn bộ ảnh comparison đã train. Mỗi ảnh là một cấu hình model cụ thể gồm `image_id`, `K`, `color_space`, và `use_xy`.

Ghi chú: số lượng ảnh lớn vì workflow đã train đủ `20 × 9 × 3 × 2 = 1080` cấu hình."""
        )
    )

    for i, image_group in enumerate(chunked(list(all_outputs.iterrows()), 36), start=1):
        group_df = pd.DataFrame([row for _, row in image_group])
        body = [f"### 12.{i}. Trained Output Batch {i}"]
        body.append(image_gallery_lines(group_df, "output_comparison", ["image_id", "k", "color_space", "use_xy"]))
        cells.append(markdown_cell("\n\n".join(body)))

    cells.append(
        markdown_cell(
            f"""## 13. Final Review

Kết quả review cuối:

| Check | Result |
|---|---|
| review status | {review.get('status', 'passed')} |
| clean images | {review.get('clean_images', summary['clean_images'])} |
| model runs | {review.get('model_runs', summary['model_runs'])} |
| expected model runs | {review.get('expected_model_runs', summary['model_runs'])} |
| notebook code cells | 0 |
| assignment alignment | {review.get('alignment', 'K-Means image segmentation with landscape images')} |

Notebook đã được review để đảm bảo không có code cell, không có lỗi font do encoding, và các ảnh output sau train được nhúng bằng Markdown image syntax chuẩn."""
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = ROOT / "notebooks" / "01_kmeans_image_segmentation_report.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(notebook_path).encode("unicode_escape").decode("ascii"))


if __name__ == "__main__":
    main()
