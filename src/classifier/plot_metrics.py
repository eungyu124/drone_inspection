import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT_DIR / "runs" / "classifier" / "eval" / "evaluation_summary.json"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "classifier" / "eval"

METRICS = ["accuracy", "precision", "recall", "f1", "iou", "dice", "auroc", "auprc", "fpr"]


def _get_metric(d: dict, key: str) -> float:
    v = d.get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _get_latency(payload: dict) -> float:
    v = payload.get("overall", {}).get("latency_ms_per_image")
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def plot_metrics(summary_path: Path, out_dir: Path) -> None:
    if not summary_path.exists():
        raise FileNotFoundError(f"evaluation_summary.json을 찾을 수 없습니다: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = payload.get("overall", {}).get("metrics", {})
    parts = [p for p in payload.get("parts", []) if p.get("status") == "ok"]

    # 1) Overall bar chart
    overall_vals = [_get_metric(overall, m) for m in METRICS]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(METRICS))
    bars = ax.bar(x, overall_vals, color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Overall Classification Metrics")
    for b, v in zip(bars, overall_vals):
        ax.text(b.get_x() + b.get_width() / 2, min(1.0, v) + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    overall_path = out_dir / "metrics_overall.png"
    fig.savefig(overall_path, dpi=160)
    plt.close(fig)

    grouped_metrics = ["f1", "iou", "dice", "auroc", "auprc"]

    # 2) Per-part grouped chart
    if parts:
        part_names = [p["part"] for p in parts]
        width = 0.14
        x = np.arange(len(part_names))
        fig, ax = plt.subplots(figsize=(10, 5.2))
        colors = ["#1B9E77", "#D95F02", "#7570B3", "#66A61E", "#E7298A"]
        for i, m in enumerate(grouped_metrics):
            vals = [_get_metric(p.get("metrics", {}), m) for p in parts]
            ax.bar(x + (i - 2) * width, vals, width=width, label=m, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(part_names)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.set_title("Per-Part Metrics")
        ax.legend(loc="lower right", ncol=2, fontsize=9)
        fig.tight_layout()
        part_path = out_dir / "metrics_by_part.png"
        fig.savefig(part_path, dpi=160)
        plt.close(fig)
    else:
        part_path = None

    # 3) Operational metrics chart (FPR + Latency)
    fpr_val = _get_metric(overall, "fpr")
    latency_val = _get_latency(payload)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    axes[0].bar(["overall"], [fpr_val], color="#E15759")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Overall FPR")
    axes[0].set_ylabel("Rate")
    axes[0].text(0, min(1.0, fpr_val) + 0.02, f"{fpr_val:.4f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(["overall"], [latency_val], color="#4E79A7")
    axes[1].set_title("Latency (ms/image)")
    axes[1].set_ylabel("ms")
    axes[1].text(0, latency_val + max(0.1, latency_val * 0.02), f"{latency_val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    ops_path = out_dir / "metrics_operational.png"
    fig.savefig(ops_path, dpi=160)
    plt.close(fig)

    # 4) Single dashboard image
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.4], hspace=0.35, wspace=0.25)
    ax_overall = fig.add_subplot(gs[0, :])
    ax_fpr = fig.add_subplot(gs[1, 0])
    ax_latency = fig.add_subplot(gs[1, 1])

    x = np.arange(len(METRICS))
    bars = ax_overall.bar(x, overall_vals, color="#2E86AB")
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(METRICS, rotation=20, ha="right")
    ax_overall.set_ylim(0, 1.02)
    ax_overall.set_ylabel("Score")
    ax_overall.set_title("Overall Metrics (Core + FPR)")
    for b, v in zip(bars, overall_vals):
        ax_overall.text(b.get_x() + b.get_width() / 2, min(1.0, v) + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax_fpr.bar(["overall"], [fpr_val], color="#E15759")
    ax_fpr.set_ylim(0, 1.02)
    ax_fpr.set_title("FPR")
    ax_fpr.set_ylabel("Rate")
    ax_fpr.text(0, min(1.0, fpr_val) + 0.02, f"{fpr_val:.4f}", ha="center", va="bottom", fontsize=9)

    ax_latency.bar(["overall"], [latency_val], color="#4E79A7")
    ax_latency.set_title("Latency (ms/image)")
    ax_latency.set_ylabel("ms")
    ax_latency.text(0, latency_val + max(0.1, latency_val * 0.02), f"{latency_val:.3f}", ha="center", va="bottom", fontsize=9)

    dashboard_path = out_dir / "metrics_dashboard.png"
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=170)
    plt.close(fig)

    print(f"시각화 저장 완료:")
    print(f" - {overall_path}")
    if part_path:
        print(f" - {part_path}")
    print(f" - {ops_path}")
    print(f" - {dashboard_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="evaluation_summary.json 지표 시각화")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="evaluation_summary.json 경로")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="시각화 이미지 저장 디렉터리")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    plot_metrics(summary_path=Path(args.summary), out_dir=Path(args.out_dir))
