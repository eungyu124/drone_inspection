import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "anomaly_compare"


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"evaluation_summary.json을 찾을 수 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _m(payload: dict, key: str) -> float | None:
    v = payload.get("overall", {}).get("metrics", {}).get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _lat(payload: dict) -> float | None:
    v = payload.get("overall", {}).get("latency_ms_per_image")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe(v: float | None) -> float:
    return v if v is not None else 0.0


def _write_table(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "model", "f1", "recall", "precision", "fpr", "auroc", "auprc", "latency_ms_per_image"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot(rows: list[dict], out_dir: Path) -> None:
    scenarios = sorted(set(r["scenario"] for r in rows))
    metrics = ["f1", "recall", "precision", "fpr", "latency_ms_per_image"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    axes = axes.flatten()
    colors = {"PatchCore": "#1f77b4", "PaDiM": "#ff7f0e"}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(scenarios))
        width = 0.35
        p_vals = []
        d_vals = []
        for sc in scenarios:
            p_row = next((r for r in rows if r["scenario"] == sc and r["model"] == "PatchCore"), None)
            d_row = next((r for r in rows if r["scenario"] == sc and r["model"] == "PaDiM"), None)
            p_vals.append(_safe(p_row.get(metric) if p_row else None))
            d_vals.append(_safe(d_row.get(metric) if d_row else None))
        ax.bar(x - width / 2, p_vals, width=width, color=colors["PatchCore"], label="PatchCore")
        ax.bar(x + width / 2, d_vals, width=width, color=colors["PaDiM"], label="PaDiM")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        if metric != "latency_ms_per_image":
            ax.set_ylim(0, 1.02)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    axes[-1].axis("off")
    fig.tight_layout()
    out = out_dir / "patchcore_vs_padim_dashboard.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)


def run(
    patchcore_normal: Path,
    patchcore_defect: Path,
    patchcore_mixed: Path,
    padim_normal: Path,
    padim_defect: Path,
    padim_mixed: Path,
    out_dir: Path,
) -> None:
    sources = [
        ("normal_only", "PatchCore", patchcore_normal),
        ("defect_only", "PatchCore", patchcore_defect),
        ("mixed_holdout", "PatchCore", patchcore_mixed),
        ("normal_only", "PaDiM", padim_normal),
        ("defect_only", "PaDiM", padim_defect),
        ("mixed_holdout", "PaDiM", padim_mixed),
    ]
    rows = []
    for scenario, model, path in sources:
        payload = _load_summary(path)
        rows.append(
            {
                "scenario": scenario,
                "model": model,
                "f1": _m(payload, "f1"),
                "recall": _m(payload, "recall"),
                "precision": _m(payload, "precision"),
                "fpr": _m(payload, "fpr"),
                "auroc": _m(payload, "auroc"),
                "auprc": _m(payload, "auprc"),
                "latency_ms_per_image": _lat(payload),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    table_csv = out_dir / "patchcore_vs_padim_metrics.csv"
    _write_table(rows, table_csv)
    _plot(rows, out_dir)
    summary = {
        "table_csv": str(table_csv),
        "dashboard_png": str(out_dir / "patchcore_vs_padim_dashboard.png"),
        "rows": rows,
    }
    (out_dir / "patchcore_vs_padim_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {table_csv}")
    print(f"완료: {out_dir / 'patchcore_vs_padim_dashboard.png'}")
    print(f"완료: {out_dir / 'patchcore_vs_padim_summary.json'}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PatchCore vs PaDiM 시나리오 비교 리포트")
    p.add_argument("--patchcore-normal", required=True)
    p.add_argument("--patchcore-defect", required=True)
    p.add_argument("--patchcore-mixed", required=True)
    p.add_argument("--padim-normal", required=True)
    p.add_argument("--padim-defect", required=True)
    p.add_argument("--padim-mixed", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        patchcore_normal=Path(args.patchcore_normal),
        patchcore_defect=Path(args.patchcore_defect),
        patchcore_mixed=Path(args.patchcore_mixed),
        padim_normal=Path(args.padim_normal),
        padim_defect=Path(args.padim_defect),
        padim_mixed=Path(args.padim_mixed),
        out_dir=Path(args.out_dir),
    )
