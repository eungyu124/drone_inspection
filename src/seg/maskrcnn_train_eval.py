import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_YAML = ROOT_DIR / "data" / "defect_detseg_v1" / "data.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "seg" / "maskrcnn_defect"


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_yaml(data_yaml: Path) -> tuple[Path, Path, Path]:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"]).expanduser()
    train_rel = Path(cfg["train"])
    val_rel = Path(cfg["val"])
    train_dir = train_rel if train_rel.is_absolute() else root / train_rel
    val_dir = val_rel if val_rel.is_absolute() else root / val_rel
    label_root = root / "labels"
    return train_dir, val_dir, label_root


def _label_path_from_image(img_path: Path, label_root: Path, split_name: str) -> Path:
    return label_root / split_name / f"{img_path.stem}.txt"


def _polygon_to_mask(points_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_xy) >= 3:
        pts = np.round(points_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _read_yolo_seg_instances(label_path: Path, h: int, w: int) -> list[np.ndarray]:
    if not label_path.exists():
        return []
    masks = []
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
        m = _polygon_to_mask(coords, h=h, w=w)
        if m.sum() > 0:
            masks.append(m)
    return masks


class YoloSegMaskDataset(Dataset):
    def __init__(self, images_dir: Path, label_root: Path):
        self.images_dir = images_dir
        self.label_root = label_root
        self.split_name = images_dir.name
        self.image_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        label_path = _label_path_from_image(img_path, self.label_root, self.split_name)
        masks = _read_yolo_seg_instances(label_path, h=h, w=w)

        boxes = []
        masks_out = []
        labels = []
        areas = []
        for m in masks:
            ys, xs = np.where(m > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            boxes.append([x1, y1, x2, y2])
            masks_out.append(m)
            labels.append(1)  # defect class
            areas.append(float((m > 0).sum()))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            areas_t = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            masks_t = torch.tensor(np.stack(masks_out, axis=0), dtype=torch.uint8)
            areas_t = torch.tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64),
        }
        return F.to_tensor(img), target, str(img_path)


def _collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return images, targets, paths


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


def _merge_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    if masks.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    m = (masks > 0).any(axis=0).astype(np.uint8)
    return m


def _evaluate_model(
    model: nn.Module,
    dl_val: DataLoader,
    score_thr: float,
) -> tuple[dict, list[dict], dict]:
    device = next(model.parameters()).device
    model.eval()
    tp = tn = fp = fn = 0
    infer_sec = 0.0
    infer_n = 0
    rows = []
    with torch.no_grad():
        for images, targets, paths in dl_val:
            img = images[0].to(device)
            gt = targets[0]
            h, w = img.shape[1], img.shape[2]

            t0 = time.perf_counter()
            pred = model([img])[0]
            infer_sec += time.perf_counter() - t0
            infer_n += 1

            gt_mask = _merge_masks(gt["masks"].cpu().numpy(), h=h, w=w)
            keep = pred["scores"].detach().cpu().numpy() >= score_thr
            if keep.any():
                pred_masks = pred["masks"].detach().cpu().numpy()[keep, 0]
                pred_mask = _merge_masks((pred_masks >= 0.5).astype(np.uint8), h=h, w=w)
            else:
                pred_mask = np.zeros((h, w), dtype=np.uint8)

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
                    "image": paths[0],
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
    confusion = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return metrics, rows, confusion


def train_and_evaluate(
    data_yaml: Path,
    out_dir: Path,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    score_thr: float = 0.5,
) -> None:
    train_dir, val_dir, label_root = _parse_yaml(data_yaml)
    ds_train = YoloSegMaskDataset(train_dir, label_root=label_root)
    ds_val = YoloSegMaskDataset(val_dir, label_root=label_root)
    if len(ds_train) == 0 or len(ds_val) == 0:
        raise FileNotFoundError("train/val 이미지가 없습니다.")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=0)

    device = _resolve_device()
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for images, targets, _ in dl_train:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n += 1
        ep_loss = loss_sum / max(1, n)
        history.append({"epoch": ep + 1, "train_loss": ep_loss})
        print(f"[maskrcnn] epoch {ep+1}/{epochs} loss={ep_loss:.4f}")

    ckpt = out_dir / "maskrcnn_defect.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)

    metrics, rows, confusion = _evaluate_model(model=model, dl_val=dl_val, score_thr=score_thr)

    summary = {
        "method": "maskrcnn_resnet50_fpn",
        "data_yaml": str(data_yaml),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "score_thr": score_thr,
        "device": device,
        "confusion": confusion,
        "metrics": metrics,
        "history": history,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "evaluation_predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {out_dir / 'evaluation_summary.json'}")


def evaluate_only(
    data_yaml: Path,
    out_dir: Path,
    weights: Path,
    score_thr: float = 0.5,
) -> None:
    _, val_dir, label_root = _parse_yaml(data_yaml)
    ds_val = YoloSegMaskDataset(val_dir, label_root=label_root)
    if len(ds_val) == 0:
        raise FileNotFoundError("val 이미지가 없습니다.")

    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=0)
    device = _resolve_device()
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2).to(device)
    ckpt = torch.load(weights, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, rows, confusion = _evaluate_model(model=model, dl_val=dl_val, score_thr=score_thr)
    summary = {
        "method": "maskrcnn_resnet50_fpn",
        "mode": "eval_only",
        "data_yaml": str(data_yaml),
        "weights": str(weights),
        "score_thr": score_thr,
        "device": device,
        "confusion": confusion,
        "metrics": metrics,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "evaluation_predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {out_dir / 'evaluation_summary.json'}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mask R-CNN 결함 인스턴스 분할 학습/평가")
    p.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="YOLO-seg data.yaml")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="출력 디렉터리")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--eval-only", action="store_true", help="학습 없이 checkpoint 로드 후 평가만 수행")
    p.add_argument("--weights", default="", help="--eval-only 일 때 사용할 checkpoint(.pt)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.eval_only:
        if not args.weights:
            raise ValueError("--eval-only 사용 시 --weights 경로가 필요합니다.")
        evaluate_only(
            data_yaml=Path(args.data),
            out_dir=Path(args.out_dir),
            weights=Path(args.weights),
            score_thr=args.score_thr,
        )
    else:
        train_and_evaluate(
            data_yaml=Path(args.data),
            out_dir=Path(args.out_dir),
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            score_thr=args.score_thr,
        )
