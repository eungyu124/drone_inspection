import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

import torch

from patchcore import PatchCoreRunner, _iter_images, _load_bgr_image, _resolve_device


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_DIR = ROOT_DIR / "data" / "roi"
DEFAULT_MODEL_DIR = ROOT_DIR / "runs" / "patchcore" / "models"
DEFAULT_THRESHOLD_DIR = ROOT_DIR / "runs" / "patchcore" / "calibration"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "patchcore" / "eval_patchcore"
PARTS = ("propeller", "arm", "body")
NORMAL_ALIASES = {"good", "normal", "ok"}
DEFECT_ALIASES = {"defect", "bad", "ng", "abnormal"}


@dataclass
class Sample:
    path: Path
    label_idx: int  # 0 normal, 1 defect


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
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return float(ap)


def _calc_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
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
        "balanced_accuracy": (recall + specificity) / 2.0,
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
        cand = {
            "threshold": th,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": _calc_metrics(tp=tp, tn=tn, fp=fp, fn=fn),
        }
        if best is None:
            best = cand
            continue
        if (cand["metrics"]["f1"], cand["metrics"]["balanced_accuracy"]) > (
            best["metrics"]["f1"],
            best["metrics"]["balanced_accuracy"],
        ):
            best = cand

    return {
        "num_thresholds": len(thresholds),
        "best_by_f1": best,
    }


def _collect_samples(part_dir: Path) -> list[Sample]:
    if not part_dir.exists():
        return []
    samples: list[Sample] = []
    for class_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
        name = class_dir.name.lower()
        if name in NORMAL_ALIASES:
            label = 0
        elif name in DEFECT_ALIASES:
            label = 1
        else:
            continue
        for img in _iter_images(class_dir):
            samples.append(Sample(path=img, label_idx=label))
    return samples


def _load_threshold(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["threshold"])


def evaluate_part(
    part: str,
    roi_dir: Path,
    model_dir: Path,
    threshold_dir: Path | None,
    runner: PatchCoreRunner,
) -> dict:
    ckpt_path = model_dir / f"{part}_patchcore.pt"
    if not ckpt_path.exists():
        return {"part": part, "status": "missing_checkpoint", "checkpoint": str(ckpt_path)}

    samples = _collect_samples(roi_dir / part)
    if not samples:
        return {"part": part, "status": "no_samples"}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    memory_bank: torch.Tensor = ckpt["memory_bank"]

    threshold_file = threshold_dir / f"{part}_threshold_p99.json" if threshold_dir else None
    threshold = _load_threshold(threshold_file)

    tp = tn = fp = fn = 0
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []
    infer_total_sec = 0.0
    infer_count = 0

    for sample in samples:
        bgr = _load_bgr_image(sample.path)
        t0 = time.perf_counter()
        score = runner.score_image(bgr, memory_bank=memory_bank)
        infer_total_sec += time.perf_counter() - t0
        infer_count += 1
        y_true.append(sample.label_idx)
        y_score.append(float(score))
        gt = sample.label_idx
        pred = None if threshold is None else (1 if score > threshold else 0)
        if pred is not None:
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 0 and gt == 0:
                tn += 1
            elif pred == 1 and gt == 0:
                fp += 1
            else:
                fn += 1
        rows.append(
            {
                "part": part,
                "image": str(sample.path),
                "gt_label": "defect" if gt else "normal",
                "pred_label": ("defect" if pred == 1 else "normal") if pred is not None else "",
                "anomaly_score": float(score),
                "threshold": threshold,
            }
        )

    if threshold is None:
        # 임계값 없는 경우는 점수 기반 지표만 기록
        metrics = _calc_metrics(0, 0, 0, 0)
        confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    else:
        metrics = _calc_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
        confusion = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    metrics["auroc"] = _binary_roc_auc(y_true, y_score)
    metrics["auprc"] = _binary_pr_auc(y_true, y_score)
    return {
        "part": part,
        "status": "ok",
        "checkpoint": str(ckpt_path),
        "threshold_file": str(threshold_file) if threshold_file else None,
        "threshold": threshold,
        "num_samples": len(samples),
        "num_normal": sum(1 for s in samples if s.label_idx == 0),
        "num_defect": sum(1 for s in samples if s.label_idx == 1),
        "confusion": confusion,
        "metrics": metrics,
        "threshold_sweep": _sweep_threshold_metrics(y_true, y_score),
        "score_stats": {
            "min": min(y_score) if y_score else None,
            "max": max(y_score) if y_score else None,
            "mean": (sum(y_score) / len(y_score)) if y_score else None,
        },
        "latency_ms_per_image": (infer_total_sec * 1000.0 / infer_count) if infer_count else None,
        "rows": rows,
    }


def evaluate_all(roi_dir: Path, model_dir: Path, threshold_dir: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = PatchCoreRunner(device=_resolve_device())

    parts = []
    all_rows: list[dict] = []
    for part in PARTS:
        r = evaluate_part(part=part, roi_dir=roi_dir, model_dir=model_dir, threshold_dir=threshold_dir, runner=runner)
        parts.append(r)
        all_rows.extend(r.get("rows", []))

    valid = [p for p in parts if p.get("status") == "ok"]
    total_tp = sum(p["confusion"]["tp"] for p in valid)
    total_tn = sum(p["confusion"]["tn"] for p in valid)
    total_fp = sum(p["confusion"]["fp"] for p in valid)
    total_fn = sum(p["confusion"]["fn"] for p in valid)
    overall = {
        "confusion": {"tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn},
        "metrics": _calc_metrics(tp=total_tp, tn=total_tn, fp=total_fp, fn=total_fn),
    }
    if all_rows:
        gt_all = [1 if r["gt_label"] == "defect" else 0 for r in all_rows]
        score_all = [float(r["anomaly_score"]) for r in all_rows]
        overall["metrics"]["auroc"] = _binary_roc_auc(gt_all, score_all)
        overall["metrics"]["auprc"] = _binary_pr_auc(gt_all, score_all)
        overall["threshold_sweep"] = _sweep_threshold_metrics(gt_all, score_all)
        part_latencies = [p.get("latency_ms_per_image") for p in valid if p.get("latency_ms_per_image") is not None]
        overall["latency_ms_per_image"] = (sum(part_latencies) / len(part_latencies)) if part_latencies else None
    else:
        overall["metrics"]["auroc"] = None
        overall["metrics"]["auprc"] = None
        overall["threshold_sweep"] = {"num_thresholds": 0, "best_by_f1": None}
        overall["latency_ms_per_image"] = None

    summary = {
        "roi_dir": str(roi_dir),
        "model_dir": str(model_dir),
        "threshold_dir": str(threshold_dir) if threshold_dir else None,
        "device": _resolve_device(),
        "overall": overall,
        "parts": parts,
    }
    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "evaluation_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["part", "image", "gt_label", "pred_label", "anomaly_score", "threshold"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    auroc = overall["metrics"]["auroc"]
    auprc = overall["metrics"]["auprc"]
    latency = overall.get("latency_ms_per_image")
    print(f"평가 완료: {summary_path}")
    print(f"예측 상세: {csv_path}")
    print(
        f"overall F1={overall['metrics']['f1']:.4f}, Recall={overall['metrics']['recall']:.4f}, "
        f"Precision={overall['metrics']['precision']:.4f}, FPR={overall['metrics']['fpr']:.4f}, "
        f"AUROC={f'{auroc:.4f}' if auroc is not None else 'N/A'}, "
        f"AUPRC={f'{auprc:.4f}' if auprc is not None else 'N/A'}, "
        f"Latency(ms/img)={f'{latency:.3f}' if latency is not None else 'N/A'}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchCore 전용 평가")
    parser.add_argument("--roi-dir", default=str(DEFAULT_ROI_DIR), help="평가 데이터셋 루트 디렉터리")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="PatchCore 모델 디렉터리")
    parser.add_argument(
        "--threshold-dir",
        default=str(DEFAULT_THRESHOLD_DIR),
        help="threshold json 디렉터리 (part_threshold_p99.json 기대)",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="평가 결과 저장 디렉터리")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    threshold_dir = Path(args.threshold_dir) if args.threshold_dir else None
    evaluate_all(
        roi_dir=Path(args.roi_dir),
        model_dir=Path(args.model_dir),
        threshold_dir=threshold_dir,
        out_dir=Path(args.out_dir),
    )
