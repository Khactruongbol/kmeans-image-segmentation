# K-Means Image Segmentation

This project implements an end-to-end K-Means image segmentation workflow for landscape images.

## Main Commands

```bash
python scripts/run_full_pipeline.py --reset-outputs --skip-download --limit 30 --target-clean 20 --max-augmentation 8 --k-values 2,3,4,5,6,7,8,9,10 --color-spaces rgb,hsv,lab --use-xy-modes false,true --max-side 128
python scripts/generate_report_notebook.py
python -m pytest -q
python scripts/review_system.py
streamlit run app.py
```

## Outputs

- Raw images: `data/raw/api/`
- Clean images: `data/clean/`
- Image metadata labels: `data/labels/image_labels.csv`
- Pixel cluster labels: `data/labels/*_labels.npy`
- Segmentation figures: `reports/figures/`
- Model comparison: `reports/metrics/model_comparison.csv`
- Best model report: `reports/metrics/best_model_by_image.json`
- Data balance report: `reports/metrics/data_balance_report.json`
- Workflow summary: `reports/metrics/workflow_summary.json`
- Notebook report: `notebooks/01_kmeans_image_segmentation_report.ipynb` (lab-style, markdown-only, 0 code cells)
- Report summary figures: `reports/figures/report_*.png`

## GitHub

Target public repository: `kmeans-image-segmentation`
Working branch: `codex/report-trained-images-review`
