# Drone Inspection

Drone inspection pipeline for part segmentation and defect detection experiments.

This repository contains code for:

- Drone part segmentation with YOLO segmentation models
- ROI crop generation for `propeller`, `arm`, and `body`
- Defect classification with transfer learning
- Normal-only anomaly detection with PatchCore and PaDiM
- Few-shot defect detection with ProtoNet
- Defect instance segmentation comparison between YOLO-seg and Mask R-CNN
- Report-ready metric and confusion-matrix visualization

## Project Layout

```text
src/
  ROI/              ROI crop utilities
  anomaly/          PatchCore, PaDiM, synthetic defect generation
  classifier/       transfer learning, ProtoNet few-shot, evaluation, plotting
  eval/             scenario dataset setup scripts
  pipeline/         end-to-end inspection runner
  seg/              segmentation training, prediction, evaluation, comparison

data/
  drone.yaml        YOLO drone part segmentation config

runs/
  report_figures_20260603/   report-ready figures, generated locally
```

Large datasets, trained weights, and experiment outputs are intentionally excluded
from GitHub by `.gitignore`.

## Research Scope

The pipeline structure is fixed across experiments:

```text
input drone image -> part segmentation -> ROI crop -> defect decision -> visualization/evaluation
```

The study compares model choices under four data scenarios:

- Few-shot defects: ProtoNet
- Normal-only data: PatchCore and PaDiM
- Defect classification with transfer learning: MVTec AD pretraining + drone ROI fine-tuning
- Fully labeled defect location data: YOLO-seg and Mask R-CNN with BBox/Mask labels

## Environment

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-core.txt
```

For full reproduction on a GPU machine, install the CUDA-compatible PyTorch build
for that machine before installing the remaining packages.

## Segmentation Training

Part segmentation example:

```bash
python src/seg/train.py \
  --model yolo11n-seg.pt \
  --data data/drone.yaml \
  --epochs 30 \
  --imgsz 640 \
  --batch 8 \
  --run-name drone_seg_v1 \
  --project runs/segment
```

Defect segmentation example:

```bash
python src/seg/train.py \
  --model yolo11n-seg.pt \
  --data data/defect_detseg_v2/data.yaml \
  --epochs 30 \
  --imgsz 640 \
  --batch 8 \
  --run-name defectseg_yolo_v2_e30 \
  --project runs/segment
```

## Evaluation And Comparison

YOLO defect segmentation evaluation:

```bash
python src/seg/eval_yolo_defectseg.py \
  --weights runs/segment/defectseg_yolo_v2_e30/weights/best.pt \
  --data data/defect_detseg_v2/data.yaml \
  --out-dir runs/seg/yolo_defect_v2_e30_eval \
  --conf 0.10
```

YOLO-seg vs Mask R-CNN comparison:

```bash
python src/seg/compare_defectseg_models.py \
  --yolo-summary runs/seg/yolo_defect_v2_e30_eval/evaluation_summary.json \
  --maskrcnn-summary runs/seg/maskrcnn_defect_v2_e30/evaluation_summary.json \
  --out-dir runs/seg/compare_yolo_e30_vs_mask_e30
```

## Report Figures

Report figures are generated into:

```text
runs/report_figures_20260603/
```

That folder includes:

- `comparison_graphs/`
- `confusion_matrices/`
- `manifest.json`

The manifest links each figure back to the original result file.

## Notes

- Do not commit private datasets or large trained weights.
- Use `rsync` or cloud storage to move `data/`, `runs/`, and `.pt` files between machines.
- The GitHub repo should mainly contain source code, configs, and documentation.
