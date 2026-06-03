# Data

This repository includes small sample datasets so the code structure and commands
can be tested without private or large training data.

## Included Samples

```text
data/sample_part_seg/
  images/train, images/val
  labels/train, labels/val
  data.yaml
```

Small YOLO segmentation sample for drone part segmentation:

- `propeller`
- `body`
- `arm`

```text
data/sample_defect_detseg/
  images/train, images/val
  labels/train, labels/val
  data.yaml
```

Small YOLO segmentation sample for defect instance segmentation:

- `defect`

## Excluded Data

The following are intentionally excluded from GitHub:

- Full drone datasets
- ROI datasets
- MVTec AD data
- Generated synthetic datasets
- Trained weights and experiment outputs

Place full datasets locally under `data/` using the same folder structure as the
sample datasets, then update the `data.yaml` path if needed.
