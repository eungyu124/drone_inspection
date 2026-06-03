import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_ROOT = ROOT_DIR / "data" / "roi"
DEFAULT_MODEL_DIR = ROOT_DIR / "runs" / "padim" / "models"
DEFAULT_CALIB_DIR = ROOT_DIR / "runs" / "padim" / "calibration"
DEFAULT_PRED_DIR = ROOT_DIR / "runs" / "padim" / "predict"
DEFAULT_EVAL_DIR = ROOT_DIR / "runs" / "padim" / "eval"
PARTS = ("propeller", "arm", "body")
NORMAL_ALIASES = {"good", "normal", "ok"}
DEFECT_ALIASES = {"defect", "bad", "ng", "abnormal"}


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


def _iter_images(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


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


class PaDiMRunner:
    """Lightweight PaDiM-style global Mahalanobis scorer."""

    def __init__(self, device: str):
        self.device = device
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature = torch.nn.Sequential(*(list(backbone.children())[:-1])).to(device).eval()
        self.pre = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def embedding(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.pre(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.feature(x).reshape(-1).detach().cpu().numpy()  # 512
        return f.astype(np.float32)

    def score(self, emb: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        d = emb - mean
        return float(np.sqrt(max(0.0, d @ inv_cov @ d)))


def _collect_samples(part_dir: Path) -> list[Sample]:
    if not part_dir.exists():
        return []
    out: list[Sample] = []
    for class_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
        name = class_dir.name.lower()
        if name in NORMAL_ALIASES:
            label = 0
        elif name in DEFECT_ALIASES:
            label = 1
        else:
            continue
        for p in _iter_images(class_dir):
            out.append(Sample(p, label))
    return out


def fit_part(part: str, roi_root: Path, model_dir: Path, reg_eps: float) -> Path:
    normal_dir = roi_root / part / "normal"
    if not normal_dir.exists():
        raise FileNotFoundError(f"정상 폴더 없음: {normal_dir}")
    imgs = list(_iter_images(normal_dir))
    if not imgs:
        raise FileNotFoundError(f"정상 이미지 없음: {normal_dir}")
    runner = PaDiMRunner(_resolve_device())
    embs = np.stack([runner.embedding(cv2.imread(str(p))) for p in imgs], axis=0)
    mean = embs.mean(axis=0)
    cov = np.cov(embs, rowvar=False)
    cov = cov + np.eye(cov.shape[0], dtype=np.float32) * reg_eps
    inv_cov = np.linalg.pinv(cov).astype(np.float32)
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / f"{part}_padim.npz"
    np.savez_compressed(out, mean=mean.astype(np.float32), inv_cov=inv_cov, part=part, n=len(imgs))
    print(f"[{part}] 저장 완료: {out} (normal={len(imgs)})")
    return out


def calibrate(part: str, ckpt: Path, normal_dir: Path, out_dir: Path, percentile: float) -> Path:
    if not ckpt.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt}")
    data = np.load(ckpt)
    mean = data["mean"]
    inv_cov = data["inv_cov"]
    imgs = list(_iter_images(normal_dir))
    if not imgs:
        raise FileNotFoundError(f"정상 이미지 없음: {normal_dir}")
    runner = PaDiMRunner(_resolve_device())
    scores = [runner.score(runner.embedding(cv2.imread(str(p))), mean, inv_cov) for p in imgs]
    threshold = float(np.percentile(scores, percentile))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{part}_threshold_p{int(percentile)}.json"
    out.write_text(
        json.dumps(
            {
                "part": part,
                "threshold": threshold,
                "percentile": percentile,
                "min_score": float(min(scores)),
                "max_score": float(max(scores)),
                "mean_score": float(sum(scores) / len(scores)),
                "num_samples": len(scores),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{part}] threshold={threshold:.6f} -> {out}")
    return out


def predict(
    part: str,
    ckpt: Path,
    input_dir: Path,
    out_dir: Path,
    threshold: float | None,
    threshold_file: Path | None,
) -> None:
    if not ckpt.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt}")
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더 없음: {input_dir}")
    if threshold_file:
        threshold = float(json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"])
    data = np.load(ckpt)
    mean = data["mean"]
    inv_cov = data["inv_cov"]
    runner = PaDiMRunner(_resolve_device())
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in _iter_images(input_dir):
        img = cv2.imread(str(p))
        score = runner.score(runner.embedding(img), mean, inv_cov)
        verdict = None if threshold is None else ("defect" if score > threshold else "normal")
        rows.append({"image": str(p), "part": part, "score": score, "threshold": threshold, "verdict": verdict})
        print(f"{p.name}: score={score:.4f}" + (f" ({verdict})" if verdict else ""))
    (out_dir / f"{part}_padim_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(roi_dir: Path, model_dir: Path, threshold_dir: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = PaDiMRunner(_resolve_device())
    part_results = []
    all_rows = []

    for part in PARTS:
        ckpt = model_dir / f"{part}_padim.npz"
        if not ckpt.exists():
            part_results.append({"part": part, "status": "missing_checkpoint", "checkpoint": str(ckpt)})
            continue
        samples = _collect_samples(roi_dir / part)
        if not samples:
            part_results.append({"part": part, "status": "no_samples"})
            continue

        pack = np.load(ckpt)
        mean = pack["mean"]
        inv_cov = pack["inv_cov"]
        th = None
        th_file = threshold_dir / f"{part}_threshold_p99.json" if threshold_dir else None
        if th_file and th_file.exists():
            th = float(json.loads(th_file.read_text(encoding="utf-8"))["threshold"])

        tp = tn = fp = fn = 0
        y_true, y_score = [], []
        t_sum = 0.0
        n_inf = 0
        rows = []
        for s in samples:
            img = cv2.imread(str(s.path))
            t0 = time.perf_counter()
            score = runner.score(runner.embedding(img), mean, inv_cov)
            t_sum += time.perf_counter() - t0
            n_inf += 1
            y_true.append(s.label_idx)
            y_score.append(score)
            pred = 1 if (th is not None and score > th) else 0
            gt = s.label_idx
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
                    "image": str(s.path),
                    "gt_label": "defect" if gt else "normal",
                    "pred_label": "defect" if pred else "normal",
                    "anomaly_score": float(score),
                    "threshold": th,
                }
            )
        m = _calc_metrics(tp, tn, fp, fn)
        m["auroc"] = _binary_roc_auc(y_true, y_score)
        m["auprc"] = _binary_pr_auc(y_true, y_score)
        pr = {
            "part": part,
            "status": "ok",
            "threshold": th,
            "num_samples": len(samples),
            "num_normal": sum(1 for s in samples if s.label_idx == 0),
            "num_defect": sum(1 for s in samples if s.label_idx == 1),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": m,
            "latency_ms_per_image": (t_sum * 1000.0 / n_inf) if n_inf else None,
            "rows": rows,
        }
        part_results.append(pr)
        all_rows.extend(rows)

    valid = [p for p in part_results if p.get("status") == "ok"]
    if valid:
        ttp = sum(p["confusion"]["tp"] for p in valid)
        ttn = sum(p["confusion"]["tn"] for p in valid)
        tfp = sum(p["confusion"]["fp"] for p in valid)
        tfn = sum(p["confusion"]["fn"] for p in valid)
        overall = {"confusion": {"tp": ttp, "tn": ttn, "fp": tfp, "fn": tfn}, "metrics": _calc_metrics(ttp, ttn, tfp, tfn)}
        gt_all = [1 if r["gt_label"] == "defect" else 0 for r in all_rows]
        sc_all = [float(r["anomaly_score"]) for r in all_rows]
        overall["metrics"]["auroc"] = _binary_roc_auc(gt_all, sc_all)
        overall["metrics"]["auprc"] = _binary_pr_auc(gt_all, sc_all)
        lats = [p.get("latency_ms_per_image") for p in valid if p.get("latency_ms_per_image") is not None]
        overall["latency_ms_per_image"] = (sum(lats) / len(lats)) if lats else None
    else:
        overall = {"confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, "metrics": _calc_metrics(0, 0, 0, 0), "latency_ms_per_image": None}
        overall["metrics"]["auroc"] = None
        overall["metrics"]["auprc"] = None

    payload = {
        "roi_dir": str(roi_dir),
        "model_dir": str(model_dir),
        "threshold_dir": str(threshold_dir) if threshold_dir else None,
        "device": _resolve_device(),
        "overall": overall,
        "parts": part_results,
    }
    summary = out_dir / "evaluation_summary.json"
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_csv = out_dir / "evaluation_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "image", "gt_label", "pred_label", "anomaly_score", "threshold"])
        w.writeheader()
        w.writerows(all_rows)
    auroc = overall["metrics"]["auroc"]
    auprc = overall["metrics"]["auprc"]
    lat = overall.get("latency_ms_per_image")
    print(f"평가 완료: {summary}")
    print(f"예측 상세: {pred_csv}")
    print(
        f"overall F1={overall['metrics']['f1']:.4f}, Recall={overall['metrics']['recall']:.4f}, "
        f"Precision={overall['metrics']['precision']:.4f}, FPR={overall['metrics']['fpr']:.4f}, "
        f"AUROC={f'{auroc:.4f}' if auroc is not None else 'N/A'}, "
        f"AUPRC={f'{auprc:.4f}' if auprc is not None else 'N/A'}, "
        f"Latency(ms/img)={f'{lat:.3f}' if lat is not None else 'N/A'}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PaDiM-style anomaly detection")
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--part", required=True, choices=list(PARTS))
    fit.add_argument("--roi-root", default=str(DEFAULT_ROI_ROOT))
    fit.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    fit.add_argument("--reg-eps", type=float, default=1e-2)

    cal = sub.add_parser("calibrate")
    cal.add_argument("--part", required=True, choices=list(PARTS))
    cal.add_argument("--ckpt", required=True)
    cal.add_argument("--normal-dir", required=True)
    cal.add_argument("--output-dir", default=str(DEFAULT_CALIB_DIR))
    cal.add_argument("--percentile", type=float, default=99.0)

    pred = sub.add_parser("predict")
    pred.add_argument("--part", required=True, choices=list(PARTS))
    pred.add_argument("--ckpt", required=True)
    pred.add_argument("--input-dir", required=True)
    pred.add_argument("--output-dir", default=str(DEFAULT_PRED_DIR))
    pred.add_argument("--threshold", type=float)
    pred.add_argument("--threshold-file")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--roi-dir", default=str(DEFAULT_ROI_ROOT))
    ev.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ev.add_argument("--threshold-dir", default=str(DEFAULT_CALIB_DIR))
    ev.add_argument("--out-dir", default=str(DEFAULT_EVAL_DIR))
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.cmd == "fit":
        fit_part(args.part, Path(args.roi_root), Path(args.model_dir), args.reg_eps)
    elif args.cmd == "calibrate":
        calibrate(args.part, Path(args.ckpt), Path(args.normal_dir), Path(args.output_dir), args.percentile)
    elif args.cmd == "predict":
        predict(
            args.part,
            Path(args.ckpt),
            Path(args.input_dir),
            Path(args.output_dir),
            args.threshold,
            Path(args.threshold_file) if args.threshold_file else None,
        )
    elif args.cmd == "evaluate":
        evaluate(Path(args.roi_dir), Path(args.model_dir), Path(args.threshold_dir), Path(args.out_dir))
