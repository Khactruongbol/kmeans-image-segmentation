from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import parse_bool_modes, parse_color_spaces, parse_k_values, run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full K-Means landscape segmentation workflow.")
    parser.add_argument("--query", default="mountain landscape")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--source", choices=["wikimedia", "pexels", "unsplash"], default="wikimedia")
    parser.add_argument("--k-values", default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--color-spaces", default="rgb,hsv,lab")
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--queries", default="mountain landscape,forest landscape,lake landscape,beach landscape,desert landscape,sunset landscape,national park,natural scenery")
    parser.add_argument("--target-clean", type=int, default=20)
    parser.add_argument("--max-augmentation", type=int, default=8)
    parser.add_argument("--per-query-limit", type=int, default=5)
    parser.add_argument("--use-xy-modes", default="false,true")
    parser.add_argument("--metric-sample-size", type=int, default=1500)
    parser.add_argument("--reset-outputs", action="store_true")
    args = parser.parse_args()

    summary = run_full_pipeline(
        project_root=ROOT,
        query=args.query,
        limit=args.limit,
        source=args.source,
        k_values=parse_k_values(args.k_values),
        color_spaces=parse_color_spaces(args.color_spaces),
        max_side=args.max_side,
        random_state=args.random_state,
        skip_download=args.skip_download,
        queries=parse_color_spaces(args.queries),
        target_clean=args.target_clean,
        max_augmentation=args.max_augmentation,
        per_query_limit=args.per_query_limit,
        use_xy_modes=parse_bool_modes(args.use_xy_modes),
        metric_sample_size=args.metric_sample_size,
        reset_outputs=args.reset_outputs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
