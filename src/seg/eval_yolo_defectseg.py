import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_YAML = ROOT_DIR / "data" / "defect_detseg_v1" / "data.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "seg" / "yolo_defect_eval"


def _parse_yaml(data_yaml: Path) -> tuple[Path, Path, Path]:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"]).expanduser()
    val_rel = Path(cfg["val"])
    val_dir = val_rel if val_rel.is_absolute() else root / val_rel
    label_root = root / "labels" / val_dir.name
    return root, val_dir, label_root


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _calc_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "fpr": _safe_div(fp, fp + tn),
    }


def _read_gt_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return m
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        vals = ln.split()
        if len(vals) < 7:
            continue
        coords = np.array([float(v) for v in vals[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= max(1, w - 1)
        coords[:, 1] *= max(1, h - 1)
        pts = np.round(coords).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
    return m


def evaluate(weights: Path, data_yaml: Path, out_dir: Path, conf: float = 0.25) -> None:
    _, val_dir, label_root = _parse_yaml(data_yaml)
    images = sorted([p for p in val_dir.iterdir() if p.is_file()])
    model = YOLO(str(weights))

    out_dir.mkdir(parents=True, exist_ok=True)
    tp = tn = fp = fn = 0
    infer_sec = 0.0
    infer_n = 0
    rows = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt_mask = _read_gt_mask(label_root / f"{img_path.stem}.txt", h=h, w=w)

        t0 = time.perf_counter()
        r = model(str(img_path), conf=conf, verbose=False)[0]
        infer_sec += time.perf_counter() - t0
        infer_n += 1

        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if r.masks is not None and len(r.masks.xy) > 0:
            for poly in r.masks.xy:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(pred_mask, [pts], 1)

        gt_def = int(gt_mask.any())
        pred_def = int(pred_mask.any())
        if pred_def == 1 and gt_def == 1:
            tp += 1
        elif pred_def == 0 and gt_def == 0:
            tn += 1
        elif pred_def == 1 and gt_def == 0:
            fp += 1
        else:
            fn += 1

        pix_tp = int(((pred_mask == 1) & (gt_mask == 1)).sum())
        pix_fp = int(((pred_mask == 1) & (gt_mask == 0)).sum())
        pix_fn = int(((pred_mask == 0) & (gt_mask == 1)).sum())
        pix_iou = _safe_div(pix_tp, pix_tp + pix_fp + pix_fn)
        pix_dice = _safe_div(2 * pix_tp, 2 * pix_tp + pix_fp + pix_fn)
        rows.append(
            {
                "image": str(img_path),
                "gt_defect": gt_def,
                "pred_defect": pred_def,
                "pixel_iou": pix_iou,
                "pixel_dice": pix_dice,
            }
        )

    metrics = _calc_binary_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    metrics["latency_ms_per_image"] = (infer_sec / max(1, infer_n)) * 1000.0
    metrics["mean_pixel_iou"] = float(np.mean([r["pixel_iou"] for r in rows])) if rows else 0.0
    metrics["mean_pixel_dice"] = float(np.mean([r["pixel_dice"] for r in rows])) if rows else 0.0
    summary = {
        "method": "yolo_seg",
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "conf": conf,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": metrics,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "evaluation_predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {out_dir / 'evaluation_summary.json'}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO defect segmentation 평가(픽셀/검출 지표)")
    p.add_argument("--weights", required=True, help="YOLO-seg weights")
    p.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--conf", type=float, default=0.25)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate(
        weights=Path(args.weights),
        data_yaml=Path(args.data),
        out_dir=Path(args.out_dir),
        conf=args.conf,
    )

