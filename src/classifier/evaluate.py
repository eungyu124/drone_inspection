import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_DIR = ROOT_DIR / "data" / "roi"
DEFAULT_CKPT_DIR = ROOT_DIR / "runs" / "classifier"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "classifier" / "eval"
PARTS = ("propeller", "arm", "body")
NORMAL_ALIASES = {"good", "normal", "ok"}
DEFECT_ALIASES = {"defect", "bad", "ng", "abnormal"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label_idx: int  # 0 normal, 1 defect


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_model(num_classes: int = 2, backbone: str = "resnet18") -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if backbone == "regnety_400mf":
        model = models.regnet_y_400mf(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"지원하지 않는 backbone: {backbone}")


def _collect_samples(part_dir: Path) -> list[Sample]:
    if not part_dir.exists():
        return []
    samples: list[Sample] = []
    for class_dir in sorted(path for path in part_dir.iterdir() if path.is_dir()):
        class_name = class_dir.name.lower()
        if class_name in NORMAL_ALIASES:
            label_idx = 0
        elif class_name in DEFECT_ALIASES:
            label_idx = 1
        else:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                samples.append(Sample(path=image_path, label_idx=label_idx))
    return samples


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _binary_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / pos
        fpr = fp / neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5
        prev_fpr, prev_tpr = fpr, tpr
    return float(auc)


def _binary_pr_auc(y_true: list[int], y_score: list[float]) -> float | None:
    pos = sum(y_true)
    if pos == 0:
        return None

    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    precisions = [1.0]
    recalls = [0.0]
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(_safe_div(tp, tp + fp))
        recalls.append(tp / pos)

    ap = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        ap += dr * precisions[i]
    return float(ap)


def _calc_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced_acc = (recall + specificity) / 2.0
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "specificity": specificity,
        "fpr": _safe_div(fp, fp + tn),
        "balanced_accuracy": balanced_acc,
    }


def _sweep_threshold_metrics(y_true: list[int], y_score: list[float]) -> dict:
    if not y_true:
        return {"num_thresholds": 0, "best_by_f1": None}
    thresholds = sorted(set(float(s) for s in y_score))
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    thresholds = sorted(set(thresholds))

    best = None
    for th in thresholds:
        tp = tn = fp = fn = 0
        for yt, ys in zip(y_true, y_score):
            pred = 1 if ys >= th else 0
            if pred == 1 and yt == 1:
                tp += 1
            elif pred == 0 and yt == 0:
                tn += 1
            elif pred == 1 and yt == 0:
                fp += 1
            else:
                fn += 1
        m = _calc_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
        cand = {
            "threshold": th,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": m,
        }
        if best is None:
            best = cand
            continue
        # F1 최대, 동률이면 balanced_accuracy 높은 쪽
        if (cand["metrics"]["f1"], cand["metrics"]["balanced_accuracy"]) > (
            best["metrics"]["f1"],
            best["metrics"]["balanced_accuracy"],
        ):
            best = cand

    return {
        "num_thresholds": len(thresholds),
        "best_by_f1": best,
    }


def evaluate_part(
    part: str,
    roi_dir: Path,
    ckpt_dir: Path,
    device: str,
    threshold: float | None,
    backbone: str = "resnet18",
) -> dict:
    ckpt_path = ckpt_dir / f"{part}_{backbone}_binary.pt"
    if not ckpt_path.exists() and backbone == "resnet18":
        # backward compatibility
        legacy = ckpt_dir / f"{part}_resnet18_binary.pt"
        if legacy.exists():
            ckpt_path = legacy
    if not ckpt_path.exists():
        return {"part": part, "status": "missing_checkpoint", "checkpoint": str(ckpt_path)}

    samples = _collect_samples(roi_dir / part)
    if not samples:
        return {"part": part, "status": "no_samples"}

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    class_names = checkpoint.get("class_names", ["normal", "defect"])
    if "defect" not in class_names:
        return {"part": part, "status": "missing_defect_class", "class_names": class_names}
    defect_idx = class_names.index("defect")

    model = _build_model(num_classes=len(class_names), backbone=backbone)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    tp = tn = fp = fn = 0
    defect_probs: list[float] = []
    gt_labels: list[int] = []
    rows: list[dict] = []
    infer_total_sec = 0.0
    infer_count = 0
    with torch.no_grad():
        for s in samples:
            image = Image.open(s.path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            logits = model(x)
            infer_total_sec += time.perf_counter() - t0
            infer_count += 1
            probs = torch.softmax(logits, dim=1)[0]
            defect_prob = float(probs[defect_idx].item())
            defect_probs.append(defect_prob)
            gt_labels.append(s.label_idx)

            if threshold is None:
                pred_idx = int(torch.argmax(probs).item())
                pred_label = class_names[pred_idx]
                pred_is_defect = 1 if pred_label == "defect" else 0
            else:
                pred_is_defect = 1 if defect_prob >= threshold else 0
                pred_label = "defect" if pred_is_defect else "normal"

            gt_is_defect = s.label_idx
            if pred_is_defect == 1 and gt_is_defect == 1:
                tp += 1
            elif pred_is_defect == 0 and gt_is_defect == 0:
                tn += 1
            elif pred_is_defect == 1 and gt_is_defect == 0:
                fp += 1
            else:
                fn += 1

            rows.append(
                {
                    "part": part,
                    "image": str(s.path),
                    "gt_label": "defect" if gt_is_defect else "normal",
                    "pred_label": pred_label,
                    "defect_prob": defect_prob,
                }
            )

    metrics = _calc_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    metrics["auroc"] = _binary_roc_auc(gt_labels, defect_probs)
    metrics["auprc"] = _binary_pr_auc(gt_labels, defect_probs)
    threshold_sweep = _sweep_threshold_metrics(gt_labels, defect_probs)
    return {
        "part": part,
        "status": "ok",
        "checkpoint": str(ckpt_path),
        "threshold": threshold,
        "num_samples": len(samples),
        "num_normal": sum(1 for s in samples if s.label_idx == 0),
        "num_defect": sum(1 for s in samples if s.label_idx == 1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": metrics,
        "threshold_sweep": threshold_sweep,
        "defect_prob_stats": {
            "min": min(defect_probs) if defect_probs else None,
            "max": max(defect_probs) if defect_probs else None,
            "mean": sum(defect_probs) / len(defect_probs) if defect_probs else None,
        },
        "latency_ms_per_image": (infer_total_sec * 1000.0 / infer_count) if infer_count else None,
        "rows": rows,
    }


def evaluate_all(
    roi_dir: Path,
    ckpt_dir: Path,
    out_dir: Path,
    threshold: float | None,
    backbone: str = "resnet18",
) -> None:
    device = _resolve_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_rows: list[dict] = []
    for part in PARTS:
        part_result = evaluate_part(
            part=part,
            roi_dir=roi_dir,
            ckpt_dir=ckpt_dir,
            device=device,
            threshold=threshold,
            backbone=backbone,
        )
        results.append(part_result)
        all_rows.extend(part_result.get("rows", []))

    valid = [r for r in results if r.get("status") == "ok"]
    if valid:
        total_tp = sum(r["confusion"]["tp"] for r in valid)
        total_tn = sum(r["confusion"]["tn"] for r in valid)
        total_fp = sum(r["confusion"]["fp"] for r in valid)
        total_fn = sum(r["confusion"]["fn"] for r in valid)
        overall = {
            "confusion": {"tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn},
            "metrics": _calc_metrics(tp=total_tp, tn=total_tn, fp=total_fp, fn=total_fn),
        }
        gt_all = [1 if r["gt_label"] == "defect" else 0 for r in all_rows]
        score_all = [float(r["defect_prob"]) for r in all_rows]
        overall["metrics"]["auroc"] = _binary_roc_auc(gt_all, score_all)
        overall["metrics"]["auprc"] = _binary_pr_auc(gt_all, score_all)
        overall["threshold_sweep"] = _sweep_threshold_metrics(gt_all, score_all)
        part_latencies = [r.get("latency_ms_per_image") for r in valid if r.get("latency_ms_per_image") is not None]
        overall["latency_ms_per_image"] = (sum(part_latencies) / len(part_latencies)) if part_latencies else None
    else:
        overall = {"confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, "metrics": _calc_metrics(0, 0, 0, 0)}
        overall["metrics"]["auroc"] = None
        overall["metrics"]["auprc"] = None
        overall["threshold_sweep"] = {"num_thresholds": 0, "best_by_f1": None}
        overall["latency_ms_per_image"] = None

    payload = {
        "roi_dir": str(roi_dir),
        "ckpt_dir": str(ckpt_dir),
        "device": device,
        "threshold_mode": "argmax" if threshold is None else "fixed_threshold",
        "threshold": threshold,
        "backbone": backbone,
        "overall": overall,
        "parts": results,
    }

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "evaluation_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["part", "image", "gt_label", "pred_label", "defect_prob"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"평가 완료: {summary_path}")
    print(f"예측 상세: {csv_path}")
    auroc = overall["metrics"]["auroc"]
    auprc = overall["metrics"]["auprc"]
    latency = overall.get("latency_ms_per_image")
    print(
        f"overall F1={overall['metrics']['f1']:.4f}, Recall={overall['metrics']['recall']:.4f}, "
        f"Precision={overall['metrics']['precision']:.4f}, FPR={overall['metrics']['fpr']:.4f}, "
        f"AUROC={f'{auroc:.4f}' if auroc is not None else 'N/A'}, "
        f"AUPRC={f'{auprc:.4f}' if auprc is not None else 'N/A'}, "
        f"Latency(ms/img)={f'{latency:.3f}' if latency is not None else 'N/A'}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="부품별 normal/defect 분류기 평가")
    parser.add_argument("--roi-dir", default=str(DEFAULT_ROI_DIR), help="평가 데이터셋 루트 디렉터리")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR), help="체크포인트 디렉터리")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="평가 결과 저장 디렉터리")
    parser.add_argument(
        "--threshold",
        type=float,
        help="고정 defect 확률 임계값 (미지정 시 argmax 판정)",
    )
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "regnety_400mf"], help="평가 backbone")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    evaluate_all(
        roi_dir=Path(args.roi_dir),
        ckpt_dir=Path(args.ckpt_dir),
        out_dir=Path(args.out_dir),
        threshold=args.threshold,
        backbone=args.backbone,
    )
