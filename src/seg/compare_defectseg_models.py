import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "seg" / "compare_defectseg"
METRICS = ["f1", "recall", "precision", "fpr", "iou", "dice", "mean_pixel_iou", "mean_pixel_dice", "latency_ms_per_image"]


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _m(d: dict, key: str) -> float:
    v = d.get("metrics", {}).get(key)
    if v is None:
        return 0.0
    return float(v)


def compare(yolo_summary: Path, maskrcnn_summary: Path, out_dir: Path) -> None:
    y = _load(yolo_summary)
    m = _load(maskrcnn_summary)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["YOLO-seg", "Mask R-CNN"]
    rows = [
        {"model": labels[0], **{k: _m(y, k) for k in METRICS}},
        {"model": labels[1], **{k: _m(m, k) for k in METRICS}},
    ]
    (out_dir / "compare_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # core classification-style metrics
    core = ["f1", "recall", "precision", "fpr"]
    x = np.arange(len(core))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    yvals = [_m(y, k) for k in core]
    mvals = [_m(m, k) for k in core]
    b1 = ax.bar(x - width / 2, yvals, width, label="YOLO-seg", color="#1f77b4")
    b2 = ax.bar(x + width / 2, mvals, width, label="Mask R-CNN", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(core)
    ax.set_ylim(0, 1.02)
    ax.set_title("Defect Segmentation Core Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for b in bars:
            h = float(b.get_height())
            ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "core_metrics.png", dpi=220)
    plt.close(fig)

    # mask quality + latency
    ext = ["iou", "dice", "mean_pixel_iou", "mean_pixel_dice", "latency_ms_per_image"]
    x = np.arange(len(ext))
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    yvals = [_m(y, k) for k in ext]
    mvals = [_m(m, k) for k in ext]
    b1 = ax.bar(x - width / 2, yvals, width, label="YOLO-seg", color="#1f77b4")
    b2 = ax.bar(x + width / 2, mvals, width, label="Mask R-CNN", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(ext, rotation=15, ha="right")
    ax.set_title("Mask Quality & Latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars, is_lat in ((b1, False), (b2, False)):
        for i, b in enumerate(bars):
            h = float(b.get_height())
            fmt = "{:.2f}" if ext[i] == "latency_ms_per_image" else "{:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, h + (0.01 if ext[i] != "latency_ms_per_image" else max(yvals + mvals) * 0.01), fmt.format(h), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "mask_latency_metrics.png", dpi=220)
    plt.close(fig)

    print(f"저장 완료: {out_dir / 'core_metrics.png'}")
    print(f"저장 완료: {out_dir / 'mask_latency_metrics.png'}")
    print(f"저장 완료: {out_dir / 'compare_metrics.json'}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO-seg vs Mask R-CNN 비교 그래프")
    p.add_argument("--yolo-summary", required=True)
    p.add_argument("--maskrcnn-summary", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    compare(
        yolo_summary=Path(args.yolo_summary),
        maskrcnn_summary=Path(args.maskrcnn_summary),
        out_dir=Path(args.out_dir),
    )

