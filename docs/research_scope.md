# Research Scope

This project keeps the overall drone inspection pipeline fixed and changes only
the model or learning scenario used inside the pipeline.

## Fixed Pipeline

```text
drone image input
-> drone part segmentation
-> ROI crop for propeller / arm / body
-> defect detection
-> visualization and metric evaluation
```

## Compared Scenarios

| Scenario | Data condition | Main method |
| --- | --- | --- |
| Few-shot defect data | Very small number of defect samples | ProtoNet |
| Normal-only data | Normal ROI data only | PatchCore, PaDiM |
| Transfer learning | MVTec AD + synthetic/ROI defect data | ResNet / RegNet classifier |
| Full BBox/Mask labels | Defect location labels available | YOLO-seg, Mask R-CNN |

## Excluded From The Public Release

The following experiments were useful during exploration but are not part of the
final public research scope:

- pseudo-labeling experiments
- noisy-label robustness experiments
- RT-DETR trial
- body-specific augmentation utilities
- ad hoc benchmark scripts
