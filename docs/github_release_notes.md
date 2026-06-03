# GitHub Release Notes

## What To Include

- `src/` source code
- `data/drone.yaml`
- `requirements-core.txt`
- `requirements.txt` if exact local environment reproduction is needed
- `README.md`
- `.gitignore`

## What To Exclude

- Raw datasets under `data/images`, `data/labels`, `data/roi`, `data/mvtec_ad`
- Generated synthetic datasets under `data/defect_detseg*`
- Trained weights such as `*.pt`
- Experiment outputs under `runs/`
- Local virtual environments such as `.venv`

## Suggested GitHub Description

Drone part segmentation and defect detection experiments using YOLO-seg,
PatchCore, PaDiM, ProtoNet, transfer learning, and Mask R-CNN comparison.

## Suggested README Figure Bundle

For reports, use the generated local bundle:

```text
runs/report_figures_20260603/
```

This folder is intentionally kept outside GitHub by default. Upload selected
figures manually if the report needs public visual examples.
