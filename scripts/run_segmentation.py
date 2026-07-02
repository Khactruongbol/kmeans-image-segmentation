from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.segment import segment_image_file
from src.visualize import save_image, save_k_grid, save_side_by_side


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-Means image segmentation on one landscape image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--color-space", choices=["rgb", "hsv", "lab"], default="lab")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--compare-k", default="")
    parser.add_argument("--use-xy", action="store_true")
    parser.add_argument("--xy-weight", type=float, default=0.1)
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    original, segmented, model, labels = segment_image_file(
        image_path,
        k=args.k,
        color_space=args.color_space,
        max_side=args.max_side,
        random_state=args.random_state,
        use_xy=args.use_xy,
        xy_weight=args.xy_weight,
    )
    stem = f"{image_path.stem}_k{args.k}_{args.color_space}"
    segmented_path = figures_dir / f"{stem}_segmented.png"
    comparison_path = figures_dir / f"{stem}_comparison.png"
    save_image(segmented, segmented_path)
    save_side_by_side(
        original,
        segmented,
        comparison_path,
        title=f"K={args.k}, color_space={args.color_space}, inertia={model.inertia_:.2f}, iter={model.n_iter_}",
    )

    if args.compare_k:
        segmented_by_k = {}
        for k_text in args.compare_k.split(","):
            k = int(k_text.strip())
            _, k_segmented, _, _ = segment_image_file(
                image_path,
                k=k,
                color_space=args.color_space,
                max_side=args.max_side,
                random_state=args.random_state,
                use_xy=args.use_xy,
                xy_weight=args.xy_weight,
            )
            segmented_by_k[k] = k_segmented
        save_k_grid(original, segmented_by_k, figures_dir / f"{image_path.stem}_k_grid_{args.color_space}.png")

    summary = {
        "image_path": str(image_path),
        "image_domain": "landscape",
        "k": args.k,
        "color_space": args.color_space,
        "max_side": args.max_side,
        "use_xy": args.use_xy,
        "xy_weight": args.xy_weight,
        "random_state": args.random_state,
        "iterations": model.n_iter_,
        "inertia": model.inertia_,
        "labels_count": int(labels.size),
        "output_segmented": str(segmented_path),
        "output_comparison": str(comparison_path),
    }
    (metrics_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
