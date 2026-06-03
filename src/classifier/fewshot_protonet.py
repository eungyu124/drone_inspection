import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_DIR = ROOT_DIR / "data" / "validation_scenarios_v1" / "synthetic_holdout"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "classifier" / "fewshot_protonet"
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


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


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


def _collect_samples(part_dir: Path) -> list[Sample]:
    if not part_dir.exists():
        return []
    samples: list[Sample] = []
    for class_dir in sorted(path for path in part_dir.iterdir() if path.is_dir()):
        name = class_dir.name.lower()
        if name in NORMAL_ALIASES:
            label_idx = 0
        elif name in DEFECT_ALIASES:
            label_idx = 1
        else:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                samples.append(Sample(path=image_path, label_idx=label_idx))
    return samples


def _build_encoder(backbone: str = "resnet18") -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(model.children())[:-1])
    if backbone == "regnety_400mf":
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        return nn.Sequential(*list(model.children())[:-1])
    raise ValueError(f"지원하지 않는 backbone: {backbone}")


def _split_support_query(samples: list[Sample], shots_normal: int, shots_defect: int, seed: int) -> tuple[list[Sample], list[Sample]]:
    by_class = {0: [], 1: []}
    for s in samples:
        by_class[s.label_idx].append(s)
    rnd = random.Random(seed)
    for k in by_class:
        rnd.shuffle(by_class[k])

    support = by_class[0][:shots_normal] + by_class[1][:shots_defect]
    support_paths = {str(s.path) for s in support}
    query = [s for s in samples if str(s.path) not in support_paths]
    return support, query


def _embed_batch(encoder: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    z = encoder(batch).flatten(1)
    return torch.nn.functional.normalize(z, p=2, dim=1)


def _compute_prototypes(
    encoder: nn.Module,
    samples: list[Sample],
    transform: transforms.Compose,
    device: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for s in samples:
        xs.append(transform(Image.open(s.path).convert("RGB")))
        ys.append(s.label_idx)
    if not xs:
        raise ValueError("support 샘플이 비어 있습니다.")
    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            b = torch.stack(xs[i : i + batch_size]).to(device)
            feats.append(_embed_batch(encoder, b).cpu())
    feat = torch.cat(feats, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    prototypes: dict[int, torch.Tensor] = {}
    for cls in [0, 1]:
        idx = (y == cls).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            raise ValueError(f"class={cls} support 샘플이 없습니다.")
        p = feat[idx].mean(dim=0, keepdim=True)
        prototypes[cls] = torch.nn.functional.normalize(p, p=2, dim=1).squeeze(0)
    return prototypes


def evaluate_part(
    part: str,
    train_root: Path,
    test_root: Path,
    encoder: nn.Module,
    device: str,
    shots_normal: int,
    shots_defect: int,
    threshold: float,
    seed: int,
    batch_size: int,
) -> dict:
    train_samples = _collect_samples(train_root / part)
    test_samples = _collect_samples(test_root / part)
    if not train_samples or not test_samples:
        return {"part": part, "status": "missing_samples"}

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    support, _ = _split_support_query(train_samples, shots_normal=shots_normal, shots_defect=shots_defect, seed=seed)
    proto = _compute_prototypes(encoder, support, transform=transform, device=device, batch_size=batch_size)
    p_normal = proto[0].to(device)
    p_defect = proto[1].to(device)

    rows = []
    y_true: list[int] = []
    y_score: list[float] = []
    tp = tn = fp = fn = 0
    infer_total_sec = 0.0
    infer_count = 0

    with torch.no_grad():
        for s in test_samples:
            x = transform(Image.open(s.path).convert("RGB")).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            z = _embed_batch(encoder, x).squeeze(0)
            sim_n = torch.dot(z, p_normal)
            sim_d = torch.dot(z, p_defect)
            logits = torch.stack([sim_n, sim_d], dim=0)
            probs = torch.softmax(logits, dim=0)
            infer_total_sec += time.perf_counter() - t0
            infer_count += 1

            defect_prob = float(probs[1].item())
            pred = 1 if defect_prob >= threshold else 0
            gt = s.label_idx

            y_true.append(gt)
            y_score.append(defect_prob)

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
                    "image_path": str(s.path),
                    "gt_label": "defect" if gt == 1 else "normal",
                    "pred_label": "defect" if pred == 1 else "normal",
                    "defect_prob": defect_prob,
                }
            )

    metrics = _calc_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    metrics["auroc"] = _binary_roc_auc(y_true, y_score) or 0.0
    metrics["auprc"] = _binary_pr_auc(y_true, y_score) or 0.0
    latency_ms = (infer_total_sec / infer_count) * 1000.0 if infer_count else 0.0

    return {
        "part": part,
        "status": "ok",
        "support_counts": {
            "normal": sum(1 for s in support if s.label_idx == 0),
            "defect": sum(1 for s in support if s.label_idx == 1),
        },
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "num_test": len(test_samples)},
        "metrics": metrics,
        "latency_ms_per_image": latency_ms,
        "rows": rows,
    }


def run_all(
    train_dir: Path,
    test_dir: Path,
    out_dir: Path,
    shots_normal: int,
    shots_defect: int,
    threshold: float,
    seed: int,
    backbone: str,
    batch_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    encoder = _build_encoder(backbone=backbone).to(device)
    encoder.eval()

    results = []
    all_rows = []
    latencies = []
    for part in PARTS:
        r = evaluate_part(
            part=part,
            train_root=train_dir,
            test_root=test_dir,
            encoder=encoder,
            device=device,
            shots_normal=shots_normal,
            shots_defect=shots_defect,
            threshold=threshold,
            seed=seed,
            batch_size=batch_size,
        )
        results.append(r)
        if r.get("status") == "ok":
            all_rows.extend(r["rows"])
            latencies.append(float(r.get("latency_ms_per_image", 0.0)))

    tp = sum(r.get("counts", {}).get("tp", 0) for r in results if r.get("status") == "ok")
    tn = sum(r.get("counts", {}).get("tn", 0) for r in results if r.get("status") == "ok")
    fp = sum(r.get("counts", {}).get("fp", 0) for r in results if r.get("status") == "ok")
    fn = sum(r.get("counts", {}).get("fn", 0) for r in results if r.get("status") == "ok")

    y_true = [1 if row["gt_label"] == "defect" else 0 for row in all_rows]
    y_score = [float(row["defect_prob"]) for row in all_rows]
    overall_metrics = _calc_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    overall_metrics["auroc"] = _binary_roc_auc(y_true, y_score) or 0.0
    overall_metrics["auprc"] = _binary_pr_auc(y_true, y_score) or 0.0

    summary = {
        "method": "fewshot_protonet",
        "backbone": backbone,
        "device": device,
        "shots_normal": shots_normal,
        "shots_defect": shots_defect,
        "threshold": threshold,
        "seed": seed,
        "overall": {
            "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": overall_metrics,
            "latency_ms_per_image": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        },
        "parts": results,
    }

    summary_path = out_dir / "evaluation_summary.json"
    pred_path = out_dir / "evaluation_predictions.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["part", "image_path", "gt_label", "pred_label", "defect_prob"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"평가 완료: {summary_path}")
    print(f"예측 상세: {pred_path}")
    print(
        "overall "
        f"F1={overall_metrics['f1']:.4f}, "
        f"Recall={overall_metrics['recall']:.4f}, "
        f"Precision={overall_metrics['precision']:.4f}, "
        f"FPR={overall_metrics['fpr']:.4f}, "
        f"AUROC={overall_metrics['auroc']:.4f}, "
        f"AUPRC={overall_metrics['auprc']:.4f}, "
        f"Latency(ms/img)={summary['overall']['latency_ms_per_image']:.3f}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Few-shot ProtoNet(embedding prototype) 평가")
    p.add_argument("--train-dir", default=str(DEFAULT_ROI_DIR / "train"), help="support 샘플 루트")
    p.add_argument("--test-dir", default=str(DEFAULT_ROI_DIR / "test"), help="query/test 샘플 루트")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="결과 저장 디렉터리")
    p.add_argument("--shots-normal", type=int, default=20, help="normal support 샷 수")
    p.add_argument("--shots-defect", type=int, default=5, help="defect support 샷 수")
    p.add_argument("--threshold", type=float, default=0.5, help="defect 확률 임계값")
    p.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "regnety_400mf"], help="임베딩 backbone")
    p.add_argument("--batch-size", type=int, default=32, help="임베딩 배치 크기")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_all(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        out_dir=Path(args.out_dir),
        shots_normal=args.shots_normal,
        shots_defect=args.shots_defect,
        threshold=args.threshold,
        seed=args.seed,
        backbone=args.backbone,
        batch_size=args.batch_size,
    )

