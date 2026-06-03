import argparse
import json
import random
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PARTS = ("propeller", "arm", "body")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _copy(files: list[Path], dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in files:
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def make_scenario(
    src_train_root: Path,
    out_root: Path,
    defect_per_part: int,
    normal_multiplier: float,
    seed: int,
) -> Path:
    random.seed(seed)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "src_train_root": str(src_train_root),
        "out_root": str(out_root),
        "defect_per_part": defect_per_part,
        "normal_multiplier": normal_multiplier,
        "seed": seed,
        "parts": {},
    }

    for part in PARTS:
        normals = _collect_images(src_train_root / part / "normal")
        defects = _collect_images(src_train_root / part / "defect")
        random.shuffle(normals)
        random.shuffle(defects)

        d_take = min(defect_per_part, len(defects))
        selected_defects = defects[:d_take]

        # keep normal many-shot while defect is few-shot
        n_take = min(len(normals), max(1, int(round(d_take * normal_multiplier))))
        selected_normals = normals[:n_take]

        n_norm = _copy(selected_normals, out_root / part / "normal")
        n_def = _copy(selected_defects, out_root / part / "defect")
        summary["parts"][part] = {
            "available_normal": len(normals),
            "available_defect": len(defects),
            "selected_normal": n_norm,
            "selected_defect": n_def,
        }

    summary_path = out_root.parent / f"{out_root.name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Few-shot defect 시나리오 데이터셋 생성")
    p.add_argument(
        "--src-train-root",
        default=str(ROOT_DIR / "data" / "validation_scenarios_v1" / "synthetic_holdout" / "train"),
        help="원본 train 루트(part/normal, part/defect)",
    )
    p.add_argument(
        "--out-prefix",
        default=str(ROOT_DIR / "data" / "validation_scenarios_v1" / "fewshot_defect"),
        help="출력 prefix. 예) .../fewshot_defect -> .../fewshot_defect_k5 생성",
    )
    p.add_argument(
        "--defect-shots",
        default="5,10,20",
        help="부품당 defect 샘플 수 목록, 콤마 구분",
    )
    p.add_argument(
        "--normal-multiplier",
        type=float,
        default=5.0,
        help="normal 샘플 비율(= defect_shot * multiplier)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    src = Path(args.src_train_root)
    out_prefix = Path(args.out_prefix)
    ks = [int(x.strip()) for x in args.defect_shots.split(",") if x.strip()]

    for k in ks:
        out_root = Path(f"{out_prefix}_k{k}")
        summary = make_scenario(
            src_train_root=src,
            out_root=out_root,
            defect_per_part=k,
            normal_multiplier=args.normal_multiplier,
            seed=args.seed,
        )
        print(f"k={k} 완료: {summary}")
