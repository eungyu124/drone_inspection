import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRANSFER_SUMMARY = ROOT_DIR / "runs" / "classifier" / "eval_transfer" / "evaluation_summary.json"
DEFAULT_PATCHCORE_SUMMARY = ROOT_DIR / "runs" / "patchcore" / "eval_patchcore" / "evaluation_summary.json"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "comparison"
METRICS = ["accuracy", "precision", "recall", "f1", "iou", "dice", "auroc", "auprc"]
PARTS = ("propeller", "arm", "body")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"summary 파일을 찾을 수 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(d: dict, key: str) -> float:
    v = d.get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _part_metric_map(payload: dict) -> dict[str, dict]:
    out = {}
    for p in payload.get("parts", []):
        if p.get("status") == "ok":
            out[str(p.get("part"))] = p.get("metrics", {})
    return out


def plot_compare(transfer_summary: Path, patchcore_summary: Path, out_dir: Path) -> None:
    transfer = _load_json(transfer_summary)
    patchcore = _load_json(patchcore_summary)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Overall compare
    t_over = transfer.get("overall", {}).get("metrics", {})
    p_over = patchcore.get("overall", {}).get("metrics", {})
    t_vals = [_metric(t_over, m) for m in METRICS]
    p_vals = [_metric(p_over, m) for m in METRICS]

    x = np.arange(len(METRICS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - width / 2, t_vals, width=width, label="Transfer", color="#1f77b4")
    b2 = ax.bar(x + width / 2, p_vals, width=width, label="PatchCore", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics: Transfer vs PatchCore")
    ax.legend(loc="lower right")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, min(1.0, h) + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    overall_path = out_dir / "compare_overall.png"
    fig.savefig(overall_path, dpi=170)
    plt.close(fig)

    # 2) Per-part F1 compare
    t_part = _part_metric_map(transfer)
    p_part = _part_metric_map(patchcore)
    part_names = [p for p in PARTS if p in t_part or p in p_part]
    if part_names:
        t_f1 = [_metric(t_part.get(p, {}), "f1") for p in part_names]
        p_f1 = [_metric(p_part.get(p, {}), "f1") for p in part_names]
        x = np.arange(len(part_names))
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(x - width / 2, t_f1, width=width, label="Transfer", color="#1f77b4")
        ax.bar(x + width / 2, p_f1, width=width, label="PatchCore", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(part_names)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("F1")
        ax.set_title("Per-Part F1 Comparison")
        ax.legend(loc="lower right")
        fig.tight_layout()
        part_f1_path = out_dir / "compare_part_f1.png"
        fig.savefig(part_f1_path, dpi=170)
        plt.close(fig)
    else:
        part_f1_path = None

    # 3) Save compact table json for quick read
    summary = {
        "transfer_summary": str(transfer_summary),
        "patchcore_summary": str(patchcore_summary),
        "overall": {
            "transfer": {m: _metric(t_over, m) for m in METRICS},
            "patchcore": {m: _metric(p_over, m) for m in METRICS},
        },
        "parts_f1": {
            p: {
                "transfer_f1": _metric(t_part.get(p, {}), "f1"),
                "patchcore_f1": _metric(p_part.get(p, {}), "f1"),
            }
            for p in part_names
        },
    }
    summary_path = out_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("비교 시각화 저장 완료:")
    print(f" - {overall_path}")
    if part_f1_path:
        print(f" - {part_f1_path}")
    print(f" - {summary_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="전이학습 vs PatchCore 평가 비교 시각화")
    parser.add_argument("--transfer-summary", default=str(DEFAULT_TRANSFER_SUMMARY), help="전이학습 evaluation_summary.json")
    parser.add_argument("--patchcore-summary", default=str(DEFAULT_PATCHCORE_SUMMARY), help="PatchCore evaluation_summary.json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="비교 시각화 저장 디렉터리")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    plot_compare(
        transfer_summary=Path(args.transfer_summary),
        patchcore_summary=Path(args.patchcore_summary),
        out_dir=Path(args.out_dir),
    )
