import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRANSFER_SUMMARY = ROOT_DIR / "runs" / "classifier" / "eval_transfer" / "evaluation_summary.json"
DEFAULT_TRANSFER_PRED = ROOT_DIR / "runs" / "classifier" / "eval_transfer" / "evaluation_predictions.csv"
DEFAULT_PATCHCORE_SUMMARY = ROOT_DIR / "runs" / "patchcore" / "eval_patchcore" / "evaluation_summary.json"
DEFAULT_PATCHCORE_PRED = ROOT_DIR / "runs" / "patchcore" / "eval_patchcore" / "evaluation_predictions.csv"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "report_artifacts"
METRICS = ("accuracy", "precision", "recall", "f1", "iou", "dice", "auroc", "auprc")


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _calc_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scores(pred_csv: Path, score_col: str) -> tuple[list[int], list[float]]:
    if not pred_csv.exists():
        raise FileNotFoundError(f"예측 CSV를 찾을 수 없습니다: {pred_csv}")
    y_true: list[int] = []
    y_score: list[float] = []
    with pred_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row.get("gt_label", "").strip().lower()
            if gt not in {"normal", "defect"}:
                continue
            sc = row.get(score_col, "")
            try:
                score = float(sc)
            except Exception:
                continue
            y_true.append(1 if gt == "defect" else 0)
            y_score.append(score)
    return y_true, y_score


def _sweep(y_true: list[int], y_score: list[float]) -> list[dict]:
    if not y_true:
        return []
    thresholds = sorted(set(y_score))
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    thresholds = sorted(set(thresholds))
    out = []
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
        out.append({"threshold": th, **m})
    return out


def _write_core_table(transfer_summary: dict, patchcore_summary: dict, out_dir: Path) -> Path:
    t = transfer_summary.get("overall", {}).get("metrics", {})
    p = patchcore_summary.get("overall", {}).get("metrics", {})
    out_csv = out_dir / "core_metrics_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "transfer", "patchcore"])
        for m in METRICS:
            writer.writerow([m, t.get(m), p.get(m)])
    return out_csv


def _plot_threshold_curves(
    transfer_curve: list[dict],
    patchcore_curve: list[dict],
    out_dir: Path,
) -> Path:
    out_png = out_dir / "threshold_curves_overall.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    items = [
        ("f1", "F1 vs Threshold"),
        ("precision", "Precision vs Threshold"),
        ("recall", "Recall vs Threshold"),
    ]
    for ax, (k, title) in zip(axes, items):
        if transfer_curve:
            ax.plot([r["threshold"] for r in transfer_curve], [r[k] for r in transfer_curve], label="Transfer", color="#1f77b4")
        if patchcore_curve:
            ax.plot([r["threshold"] for r in patchcore_curve], [r[k] for r in patchcore_curve], label="PatchCore", color="#ff7f0e")
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(k.capitalize())
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


def build_report_artifacts(
    transfer_summary_path: Path,
    transfer_pred_path: Path,
    patchcore_summary_path: Path,
    patchcore_pred_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer_summary = _load_json(transfer_summary_path)
    patchcore_summary = _load_json(patchcore_summary_path)

    y_t_true, y_t_score = _read_scores(transfer_pred_path, score_col="defect_prob")
    y_p_true, y_p_score = _read_scores(patchcore_pred_path, score_col="anomaly_score")
    curve_t = _sweep(y_t_true, y_t_score)
    curve_p = _sweep(y_p_true, y_p_score)

    table_csv = _write_core_table(transfer_summary, patchcore_summary, out_dir=out_dir)
    curve_png = _plot_threshold_curves(curve_t, curve_p, out_dir=out_dir)

    (out_dir / "threshold_curve_transfer.json").write_text(
        json.dumps(curve_t, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "threshold_curve_patchcore.json").write_text(
        json.dumps(curve_p, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "transfer_summary": str(transfer_summary_path),
        "patchcore_summary": str(patchcore_summary_path),
        "transfer_predictions": str(transfer_pred_path),
        "patchcore_predictions": str(patchcore_pred_path),
        "core_metrics_table_csv": str(table_csv),
        "threshold_curves_png": str(curve_png),
    }
    out_summary = out_dir / "report_artifacts_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"보고서 산출물 생성 완료: {out_dir}")
    print(f" - {table_csv}")
    print(f" - {curve_png}")
    print(f" - {out_summary}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="보고서용 핵심 지표 표/임계값 그래프 생성")
    parser.add_argument("--transfer-summary", default=str(DEFAULT_TRANSFER_SUMMARY))
    parser.add_argument("--transfer-pred", default=str(DEFAULT_TRANSFER_PRED))
    parser.add_argument("--patchcore-summary", default=str(DEFAULT_PATCHCORE_SUMMARY))
    parser.add_argument("--patchcore-pred", default=str(DEFAULT_PATCHCORE_PRED))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    build_report_artifacts(
        transfer_summary_path=Path(args.transfer_summary),
        transfer_pred_path=Path(args.transfer_pred),
        patchcore_summary_path=Path(args.patchcore_summary),
        patchcore_pred_path=Path(args.patchcore_pred),
        out_dir=Path(args.out_dir),
    )
