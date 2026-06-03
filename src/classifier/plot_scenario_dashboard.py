import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "scenario_dashboard"
CORE_METRICS = ["auroc", "auprc", "f1", "recall", "precision"]


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"evaluation_summary.json을 찾을 수 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_summary_path"] = str(path)
    return payload


def _metric(payload: dict, key: str) -> float:
    v = payload.get("overall", {}).get("metrics", {}).get(key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _latency(payload: dict) -> float:
    v = payload.get("overall", {}).get("latency_ms_per_image")
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _fpr(payload: dict) -> float:
    return _metric(payload, "fpr")


def _best_threshold(payload: dict) -> tuple[list[float], list[float], list[float], list[float]]:
    sweep = payload.get("overall", {}).get("threshold_sweep", {})
    best = sweep.get("best_by_f1")
    if not best:
        return [], [], [], []
    # summary에 전체 curve가 없으므로, best point만 찍어서라도 비교 가능하게 반환
    th = float(best.get("threshold", 0.0))
    m = best.get("metrics", {})
    return [th], [float(m.get("f1", 0.0))], [float(m.get("recall", 0.0))], [float(m.get("precision", 0.0))]


def _parse_named_summaries(named: list[str]) -> tuple[list[str], list[dict]]:
    labels: list[str] = []
    payloads: list[dict] = []
    for item in named:
        if "=" not in item:
            raise ValueError(f"--summary는 'label=/abs/path/evaluation_summary.json' 형식이어야 합니다: {item}")
        label, path_str = item.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        labels.append(label)
        payloads.append(_load_summary(path))
    return labels, payloads


def _annotate_bars(ax, bars, fmt: str = "{:.3f}", rotation: int = 0) -> None:
    for bar in bars:
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.01,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=rotation,
        )


def plot_dashboard(labels: list[str], payloads: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 핵심지표 비교 (scenario x metric)
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    for i, m in enumerate(CORE_METRICS):
        vals = [_metric(p, m) for p in payloads]
        bars = ax.bar(x + (i - 2) * width, vals, width=width, label=m.upper(), color=colors[i])
        _annotate_bars(ax, bars, fmt="{:.3f}", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Core Metrics by Scenario")
    ax.legend(loc="lower right", ncol=3, fontsize=10)
    fig.tight_layout()
    core_path = out_dir / "scenario_core_metrics.png"
    fig.savefig(core_path, dpi=240)
    plt.close(fig)

    # 2) 임계값 그래프 (best threshold point 기준)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    titles = [("f1", "F1 @ Best Threshold"), ("recall", "Recall @ Best Threshold"), ("precision", "Precision @ Best Threshold")]
    for ax, (k, title) in zip(axes, titles):
        vals = []
        for p in payloads:
            _, f1s, recs, pres = _best_threshold(p)
            if k == "f1":
                vals.append(f1s[0] if f1s else 0.0)
            elif k == "recall":
                vals.append(recs[0] if recs else 0.0)
            else:
                vals.append(pres[0] if pres else 0.0)
        bars = ax.bar(labels, vals, color="#17becf")
        _annotate_bars(ax, bars, fmt="{:.3f}")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    thr_path = out_dir / "scenario_threshold_metrics.png"
    fig.savefig(thr_path, dpi=240)
    plt.close(fig)

    # 3) 운영지표 (FPR + Latency)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fpr_vals = [_fpr(p) for p in payloads]
    lat_vals = [_latency(p) for p in payloads]
    bars_fpr = axes[0].bar(labels, fpr_vals, color="#e45756")
    _annotate_bars(axes[0], bars_fpr, fmt="{:.3f}")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("FPR by Scenario")
    axes[0].tick_params(axis="x", rotation=15)
    bars_lat = axes[1].bar(labels, lat_vals, color="#4c78a8")
    _annotate_bars(axes[1], bars_lat, fmt="{:.2f}", rotation=0)
    axes[1].set_title("Latency (ms/image) by Scenario")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    ops_path = out_dir / "scenario_operational_metrics.png"
    fig.savefig(ops_path, dpi=240)
    plt.close(fig)

    # save compact table
    rows = []
    for label, payload in zip(labels, payloads):
        row = {"scenario": label}
        for m in CORE_METRICS:
            row[m] = _metric(payload, m)
        row["fpr"] = _fpr(payload)
        row["latency_ms_per_image"] = _latency(payload)
        rows.append(row)
    (out_dir / "scenario_metrics_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("시나리오 대시보드 생성 완료:")
    print(f" - {core_path}")
    print(f" - {thr_path}")
    print(f" - {ops_path}")
    print(f" - {out_dir / 'scenario_metrics_table.json'}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="시나리오별 평가 대시보드 생성")
    p.add_argument(
        "--summary",
        action="append",
        required=True,
        help="label=/abs/path/evaluation_summary.json 형식. 여러 번 사용 가능",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="출력 디렉터리")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    labels_, payloads_ = _parse_named_summaries(args.summary)
    plot_dashboard(labels_, payloads_, out_dir=Path(args.out_dir))
