import argparse
import json
import random
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PARTS = ("propeller", "arm", "body")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _copy_files(files: list[Path], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in files:
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n


def setup(
    roi_root: Path,
    out_root: Path,
    holdout_ratio: float,
    seed: int,
    max_real_normal_per_part: int,
) -> Path:
    random.seed(seed)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    synth_train = out_root / "synthetic_holdout" / "train"
    synth_test = out_root / "synthetic_holdout" / "test"
    real_normal = out_root / "real_normal_eval"
    meta = {"roi_root": str(roi_root), "parts": {}, "holdout_ratio": holdout_ratio, "seed": seed}

    for part in PARTS:
        normal_dir = roi_root / part / "normal"
        defect_dir = roi_root / part / "defect_synth_realistic"

        normal_files = sorted([p for p in normal_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if normal_dir.exists() else []
        defect_files = sorted([p for p in defect_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if defect_dir.exists() else []

        random.shuffle(normal_files)
        random.shuffle(defect_files)

        n_norm_test = max(1, int(len(normal_files) * holdout_ratio)) if normal_files else 0
        n_def_test = max(1, int(len(defect_files) * holdout_ratio)) if defect_files else 0

        norm_test, norm_train = normal_files[:n_norm_test], normal_files[n_norm_test:]
        def_test, def_train = defect_files[:n_def_test], defect_files[n_def_test:]

        # synthetic holdout split
        n1 = _copy_files(norm_train, synth_train / part / "normal")
        n2 = _copy_files(def_train, synth_train / part / "defect")
        n3 = _copy_files(norm_test, synth_test / part / "normal")
        n4 = _copy_files(def_test, synth_test / part / "defect")

        # real normal eval set (normal only)
        real_pick = normal_files[: max_real_normal_per_part or len(normal_files)]
        n5 = _copy_files(real_pick, real_normal / part / "normal")

        meta["parts"][part] = {
            "synthetic_train_normal": n1,
            "synthetic_train_defect": n2,
            "synthetic_test_normal": n3,
            "synthetic_test_defect": n4,
            "real_normal_eval": n5,
        }

    out_meta = out_root / "scenario_setup_summary.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="합성 홀드아웃 + 실제 정상 오탐 평가셋 세팅")
    p.add_argument("--roi-root", default=str(ROOT_DIR / "data" / "roi"))
    p.add_argument("--out-root", default=str(ROOT_DIR / "data" / "validation_scenarios_v1"))
    p.add_argument("--holdout-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-real-normal-per-part", type=int, default=300)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    summary = setup(
        roi_root=Path(args.roi_root),
        out_root=Path(args.out_root),
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        max_real_normal_per_part=args.max_real_normal_per_part,
    )
    print(f"완료: {summary}")
