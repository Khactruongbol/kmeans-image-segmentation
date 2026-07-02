from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.image_io import load_image_rgb, resize_max_side
from src.pipeline import compute_cluster_quality_metrics
from src.segment import build_pixel_features, segment_image_array

st.set_page_config(page_title="K-Means Image Segmentation", layout="wide")
st.title("K-Means Image Segmentation")

uploaded = st.file_uploader("Upload a landscape image", type=["jpg", "jpeg", "png"])
k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=4)
color_space = st.selectbox("Color space", ["lab", "rgb", "hsv"])
max_side = st.slider("Max side for processing", min_value=128, max_value=768, value=384, step=64)
use_xy = st.checkbox("Use spatial x/y features", value=False)
xy_weight = st.slider("Spatial feature weight", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    image_rgb = resize_max_side(load_image_rgb(tmp_path), max_side=max_side)
    segmented, model, labels = segment_image_array(
        image_rgb,
        k=k,
        color_space=color_space,
        random_state=42,
        use_xy=use_xy,
        xy_weight=xy_weight,
    )
    features, _ = build_pixel_features(image_rgb, color_space=color_space, use_xy=use_xy, xy_weight=xy_weight)
    inertia_per_pixel = (model.inertia_ or 0.0) / max(labels.size, 1)
    metrics = compute_cluster_quality_metrics(
        features,
        labels,
        model.cluster_centers_,
        inertia_per_pixel=inertia_per_pixel,
        random_state=42,
        sample_size=1500,
    )

    left, right = st.columns(2)
    with left:
        st.image(Image.fromarray(image_rgb), caption="Original landscape", use_container_width=True)
    with right:
        st.image(Image.fromarray(segmented), caption="K-Means segmented image", use_container_width=True)

    st.subheader("Run Summary")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "K": k,
                    "color_space": color_space,
                    "iterations": model.n_iter_,
                    "inertia": model.inertia_,
                    "pixels": labels.size,
                    "inertia_per_pixel": inertia_per_pixel,
                    "silhouette_sample": metrics["silhouette_sample"],
                    "davies_bouldin_sample": metrics["davies_bouldin_sample"],
                    "calinski_harabasz_sample": metrics["calinski_harabasz_sample"],
                    "cluster_balance": metrics["cluster_balance"],
                    "ranking_score": metrics["ranking_score"],
                }
            ]
        ),
        use_container_width=True,
    )
else:
    st.info("Upload a landscape image to run unsupervised K-Means segmentation.")
