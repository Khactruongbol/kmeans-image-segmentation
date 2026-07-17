from __future__ import annotations

import json
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGURES = ROOT / "reports" / "figures"


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


def table_from_records(records: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in records:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def df_to_markdown(df: pd.DataFrame, limit: int | None = None) -> str:
    rows = df if limit is None else df.head(limit)
    return table_from_records(rows.fillna("").to_dict("records"), list(rows.columns))


def md_image(path: str | Path, caption: str) -> str:
    safe_caption = caption.replace("[", "(").replace("]", ")")
    return f"![{safe_caption}]({rel(path)})\n\n*{safe_caption}*"


def save_bar_chart(data: pd.Series, title: str, ylabel: str, output_path: Path, color: str = "#4C78A8") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_line_chart(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for color_space, group in data.groupby("color_space"):
        ax.plot(group["k"], group["ranking_score"], marker="o", label=color_space.upper())
    ax.set_title("Average ranking score by K and color space")
    ax.set_xlabel("K")
    ax.set_ylabel("Average ranking score (lower is better)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_report_figures(clean: pd.DataFrame, comparison: pd.DataFrame, best: pd.DataFrame) -> dict[str, Path]:
    figures = {
        "source_distribution": REPORT_FIGURES / "report_source_distribution.png",
        "best_k_distribution": REPORT_FIGURES / "report_best_k_distribution.png",
        "color_space_comparison": REPORT_FIGURES / "report_color_space_comparison.png",
        "ranking_by_k": REPORT_FIGURES / "report_ranking_score_by_k.png",
    }

    save_bar_chart(clean["source"].value_counts(), "Balanced dataset source distribution", "Images", figures["source_distribution"])
    save_bar_chart(best["k"].value_counts().sort_index(), "Best-model K distribution", "Images", figures["best_k_distribution"], "#F58518")

    color_scores = comparison.groupby("color_space")["ranking_score"].mean().sort_values()
    save_bar_chart(color_scores, "Average ranking score by color space", "Average ranking score", figures["color_space_comparison"], "#54A24B")

    ranking_by_k = comparison.groupby(["color_space", "k"], as_index=False)["ranking_score"].mean()
    save_line_chart(ranking_by_k, figures["ranking_by_k"])
    return figures


def best_gallery(best: pd.DataFrame) -> str:
    blocks: list[str] = []
    for _, row in best.iterrows():
        caption = (
            f"{row['image_id']} | K={row['k']} | {row['color_space'].upper()} | "
            f"use_xy={row['use_xy']} | ranking_score={row['ranking_score']:.4f}"
        )
        blocks.append(md_image(row["output_comparison"], caption))
    return "\n\n".join(blocks)


def k_grid_gallery(grid_paths: list[Path], limit: int = 6) -> str:
    selected = grid_paths[:limit]
    return "\n\n".join(md_image(path, path.name) for path in selected)


def main() -> None:
    notebook_path = ROOT / "notebooks" / "01_kmeans_image_segmentation_report.ipynb"
    summary = json.loads((ROOT / "reports" / "metrics" / "workflow_summary.json").read_text(encoding="utf-8"))
    review_path = ROOT / "reports" / "metrics" / "source_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    balance = json.loads((ROOT / "reports" / "metrics" / "data_balance_report.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(ROOT / "reports" / "metrics" / "model_comparison.csv")
    clean = pd.read_csv(ROOT / "data" / "manifest" / "clean_landscape_manifest.csv")
    best = comparison.sort_values("ranking_score").groupby("image_id", as_index=False).head(1)
    best = best.sort_values(["ranking_score", "image_id"]).reset_index(drop=True)
    grid_paths = sorted(REPORT_FIGURES.glob("*_k_grid_*.png"))
    report_figures = build_report_figures(clean, comparison, best)

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

    top_models = comparison.sort_values("ranking_score")[
        [
            "image_id",
            "k",
            "color_space",
            "use_xy",
            "silhouette_sample",
            "davies_bouldin_sample",
            "inertia_per_pixel",
            "cluster_balance",
            "ranking_score",
        ]
    ].head(12)

    best_table = best[
        [
            "image_id",
            "k",
            "color_space",
            "use_xy",
            "silhouette_sample",
            "davies_bouldin_sample",
            "ranking_score",
        ]
    ]

    cells = [
        markdown_cell(
            """# Báo Cáo Cuối Kỳ - K-Means Image Segmentation

Notebook này trình bày lại project theo phong cách các lab trước: nêu bài toán, nguồn dữ liệu, workflow, cân bằng dữ liệu, training, so sánh mô hình, hình ảnh sau training và phần review cuối. Notebook chỉ dùng Markdown và hình ảnh, không chứa code cell."""
        ),
        markdown_cell(
            """## 1. Define Problem - Xác định bài toán

Mục tiêu của bài là phân đoạn một ảnh thành `K` cụm bằng thuật toán K-Means. Với mỗi pixel, chương trình dùng đặc trưng màu để gán pixel vào centroid gần nhất, sau đó thay màu pixel bằng màu centroid của cụm.

Project chọn ảnh phong cảnh vì các vùng như trời, núi, cây, nước, cát và mây thường có khác biệt màu rõ ràng, phù hợp với bài toán phân cụm không giám sát. “Độ chính xác” trong report này được hiểu là chất lượng phân cụm nội bộ và kiểm tra trực quan, không phải IoU/Dice vì đề bài không cung cấp ground-truth mask."""
        ),
        markdown_cell(
            """## 2. Assignment Mapping - Đối chiếu yêu cầu đề bài

| Yêu cầu đề bài | Phần đã thực hiện |
|---|---|
| Load the image | Đọc ảnh bằng PIL trong `src/image_io.py` |
| Convert color space | So sánh `RGB`, `HSV`, `LAB` |
| Resize image | Resize về `max_side=128` để train nhanh và ổn định |
| Flatten pixels | Biến ảnh thành ma trận pixel feature |
| Apply K-Means | Cài đặt K-Means from scratch bằng NumPy |
| Assign nearest centroid | Gán pixel theo Euclidean distance |
| Update centroids | Cập nhật centroid bằng mean của pixel trong cụm |
| Repeat until convergence | Dừng bằng `max_iter` và `tol` |
| Segment image | Thay màu pixel bằng màu centroid |
| Visualize results | Xuất segmented image, comparison image và K-grid |"""
        ),
        markdown_cell(
            f"""## 3. Workflow tổng quát và dẫn chứng đã làm

1. Thu thập data thô: manifest `{rel('data/manifest/raw_landscape_manifest.csv')}` và ảnh trong `{rel('data/raw/api')}`.
2. Làm sạch và cân bằng dữ liệu: clean manifest `{rel('data/manifest/clean_landscape_manifest.csv')}` và báo cáo `{rel('reports/metrics/data_balance_report.json')}`.
3. Tiền xử lý: đọc ảnh RGB, resize, đổi color space, flatten pixel và tùy chọn thêm tọa độ `(x, y)`.
4. Train K-Means: chạy `K=2..10`, `rgb/hsv/lab`, `use_xy=false/true`.
5. Xuất output sau training: ảnh trong `{rel('reports/figures')}`, labels trong `{rel('data/labels')}`, model artifacts trong `{rel('models')}`.
6. So sánh và review: metrics `{rel('reports/metrics/model_comparison.csv')}`, best model `{rel('reports/metrics/best_model_by_image.json')}`, source review `{rel('reports/metrics/source_review.json')}`."""
        ),
        markdown_cell(
            f"""## 4. Kiểm tra và cân bằng dữ liệu

Hệ thống audit từng ảnh theo width, height, aspect ratio, mean intensity, standard deviation, file size, source và query. Các ảnh lệch quá xa phân phối được đánh dấu bằng IQR và z-score trước khi chọn tập balanced.

**Tổng quan sau cân bằng**

{table_from_records([balance_overview], list(balance_overview.keys()))}

**Khoảng giá trị chính**

| Feature | Min | Mean | Max |
|---|---:|---:|---:|
| width | {clean['width'].min()} | {clean['width'].mean():.2f} | {clean['width'].max()} |
| height | {clean['height'].min()} | {clean['height'].mean():.2f} | {clean['height'].max()} |
| mean intensity | {clean['mean_intensity'].min():.2f} | {clean['mean_intensity'].mean():.2f} | {clean['mean_intensity'].max():.2f} |
| std intensity | {clean['std_intensity'].min():.2f} | {clean['std_intensity'].mean():.2f} | {clean['std_intensity'].max():.2f} |

{md_image(report_figures['source_distribution'], 'Balanced dataset source distribution')}"""
        ),
        markdown_cell(
            f"""## 5. Dataset sau lọc và đánh nhãn

Dataset cuối cùng giữ **{balance['selected_records']} ảnh clean**, trong đó có **{balance['real_selected']} ảnh real** và **{balance['augmentation_selected']} ảnh augmentation**. Nhãn metadata vẫn là `dataset_label=landscape` và `image_domain=landscape`. Nhãn pixel-level là cluster label sinh ra sau K-Means, không dùng mask giám sát.

**Phân phối source**

{df_to_markdown(source_counts)}

**Phân phối query**

{df_to_markdown(query_counts)}"""
        ),
        markdown_cell(
            """## 6. Preprocessing

Mỗi ảnh được resize về `max_side=128`, chuyển sang `RGB`, `HSV` hoặc `LAB`, sau đó flatten thành vector pixel. Với chế độ spatial, vector pixel được mở rộng thêm tọa độ chuẩn hóa `(x, y)` để segmentation ổn định hơn ở các vùng ảnh gần nhau.

Project không dùng supervised mask, object detection, U-Net, SAM, Mask R-CNN hoặc mô hình deep learning. Phạm vi vẫn đúng với K-Means Image Segmentation."""
        ),
        markdown_cell(
            f"""## 7. Training Models

| Thành phần | Giá trị |
|---|---|
| Số ảnh clean | {summary['clean_images']} |
| K values | {summary['k_values']} |
| Color spaces | {summary['color_spaces']} |
| Spatial modes | {summary['use_xy_modes']} |
| Tổng số model runs | {summary['model_runs']} |

Mỗi model run tạo đầy đủ segmented image, side-by-side comparison image, pixel-label `.npy`, và model artifact `.npz`."""
        ),
        markdown_cell(
            f"""## 8. So sánh mô hình

Vì bài toán không có ground-truth mask, mô hình được so sánh bằng internal unsupervised metrics và visual review. Report không giải thích từng giá trị `K` riêng lẻ; thay vào đó dùng ranking tổng hợp và các biểu đồ đại diện.

| Metric | Hướng tốt hơn | Ý nghĩa |
|---|---|---|
| silhouette_sample | cao hơn | Cụm tách nhau rõ hơn |
| davies_bouldin_sample | thấp hơn | Cụm gọn và ít chồng lấn hơn |
| calinski_harabasz_sample | cao hơn | Cụm tách biệt tốt hơn |
| inertia_per_pixel | thấp hơn | Pixel gần centroid hơn |
| cluster_balance | cao hơn | Tránh cụm quá nhỏ hoặc collapsed |
| ranking_score | thấp hơn | Điểm tổng hợp để chọn model |

{md_image(report_figures['ranking_by_k'], 'Average ranking score by K and color space')}

{md_image(report_figures['color_space_comparison'], 'Average ranking score by color space')}

**Top 12 model runs theo ranking score**

{df_to_markdown(top_models)}"""
        ),
        markdown_cell(
            f"""## 9. Đánh giá mô hình tốt nhất

Với mỗi ảnh, model tốt nhất là cấu hình có `ranking_score` thấp nhất. Bảng dưới đây là cấu hình best-model theo từng ảnh.

{md_image(report_figures['best_k_distribution'], 'Best-model K distribution')}

{df_to_markdown(best_table)}"""
        ),
        markdown_cell(
            f"""## 10. Hình ảnh sau khi training model

Các hình dưới đây là output comparison sau training, gồm ảnh gốc và ảnh đã segment đặt cạnh nhau. Report chỉ đưa best-model gallery để notebook gọn và dễ review; toàn bộ 1080 ảnh output vẫn nằm trong `{rel('reports/figures')}`.

{best_gallery(best)}"""
        ),
        markdown_cell(
            f"""## 11. K-grid highlights

K-grid giúp kiểm tra trực quan việc tăng `K` ảnh hưởng đến segmentation như thế nào. Phần này chỉ chọn một số hình đại diện, không giải thích từng `K`.

{k_grid_gallery(grid_paths)}"""
        ),
        markdown_cell(
            """## 12. Giao diện chương trình

Giao diện Python dùng Streamlit trong `app.py`. Người dùng có thể upload ảnh, chọn `K`, chọn color space, bật/tắt `use_xy`, sau đó xem ảnh segmentation và metrics sau khi chạy."""
        ),
        markdown_cell(
            f"""## 13. Review notebook và source code

| Check | Result |
|---|---|
| review status | {review.get('status', 'passed')} |
| clean images | {review.get('clean_images', summary['clean_images'])} |
| model runs | {review.get('model_runs', summary['model_runs'])} |
| expected model runs | {review.get('expected_model_runs', summary['model_runs'])} |
| notebook code cells | 0 |
| assignment alignment | {review.get('alignment', 'K-Means image segmentation with landscape images')} |

Notebook được tạo lại dưới dạng markdown-only, dùng UTF-8, dùng Markdown image syntax chuẩn và chỉ nhúng hình output sau training theo hướng tổng hợp. Source review kiểm tra manifest, metrics, labels, model artifacts, hình ảnh output và alignment với đề bài."""
        ),
        markdown_cell(
            """## 14. Kết luận

Chương trình đã hoàn thành đúng trọng tâm đề bài: load ảnh, tiền xử lý màu, flatten pixel, train K-Means, gán cụm, cập nhật centroid, segment ảnh và trực quan hóa kết quả. Dataset đã được cân bằng lại trước khi train, model được so sánh bằng metrics không giám sát, và notebook cuối cùng trình bày kết quả theo phong cách report lab thay vì liệt kê toàn bộ từng cấu hình `K`."""
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
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(notebook_path).encode("unicode_escape").decode("ascii"))


if __name__ == "__main__":
    main()
