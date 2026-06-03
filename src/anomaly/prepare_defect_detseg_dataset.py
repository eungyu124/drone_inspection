import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_ROOT = ROOT_DIR / "data" / "roi"
DEFAULT_OUT_ROOT = ROOT_DIR / "data" / "defect_detseg_v1"
PARTS = ("propeller", "arm", "body")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _find_image_by_stem(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _build_defect_mask(normal_bgr: np.ndarray, defect_bgr: np.ndarray, diff_thresh: int, blur_ksize: int) -> np.ndarray:
    if normal_bgr.shape[:2] != defect_bgr.shape[:2]:
        defect_bgr = cv2.resize(defect_bgr, (normal_bgr.shape[1], normal_bgr.shape[0]))
    diff = cv2.absdiff(defect_bgr, normal_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _mask_to_yolo_seg_line(mask: np.ndarray, class_id: int = 0, min_area: int = 12) -> str | None:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: list[tuple[float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        eps = 0.003 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        for p in approx[:, 0, :]:
            x = float(np.clip(p[0] / max(1, w - 1), 0.0, 1.0))
            y = float(np.clip(p[1] / max(1, h - 1), 0.0, 1.0))
            points.append((x, y))
    if len(points) < 3:
        return None
    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {flat}"


def _write_yaml(path: Path, out_root: Path) -> None:
    content = "\n".join(
        [
            f"path: {out_root}",
            "train: images/train",
            "val: images/test",
            "names:",
            "  0: defect",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def build_dataset(
    roi_root: Path,
    out_root: Path,
    test_ratio: float,
    seed: int,
    diff_thresh: int,
    blur_ksize: int,
    parts: tuple[str, ...],
) -> None:
    random.seed(seed)
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, str | None]] = []  # (image_path, yolo_seg_line or None)
    skipped_no_match = 0
    skipped_no_mask = 0

    for part in parts:
        normal_dir = roi_root / part / "normal"
        defect_dir = roi_root / part / "defect_maskonly_20260505_161835"
        if not normal_dir.exists() or not defect_dir.exists():
            continue

        # Add normal images with empty label
        for p in sorted(defect_dir.parent.joinpath("normal").iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                items.append((p, None))

        # Build labels for defect images by diff with matching normal stem
        for defect_img in sorted(defect_dir.iterdir()):
            if not defect_img.is_file() or defect_img.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = defect_img.stem
            if "_synthdefect_" not in stem:
                continue
            normal_stem = stem.split("_synthdefect_")[0]
            normal_img = _find_image_by_stem(normal_dir, normal_stem)
            if normal_img is None:
                skipped_no_match += 1
                continue
            n = cv2.imread(str(normal_img))
            d = cv2.imread(str(defect_img))
            if n is None or d is None:
                skipped_no_match += 1
                continue
            mask = _build_defect_mask(n, d, diff_thresh=diff_thresh, blur_ksize=blur_ksize)
            line = _mask_to_yolo_seg_line(mask, class_id=0)
            if line is None:
                skipped_no_mask += 1
                continue
            items.append((defect_img, line))

    random.shuffle(items)
    n_test = int(len(items) * test_ratio)
    test_items = items[:n_test]
    train_items = items[n_test:]

    def _copy(items_split: list[tuple[Path, str | None]], split: str) -> None:
        for src, line in items_split:
            # include part prefix to avoid name collisions
            part = src.parts[-3] if len(src.parts) >= 3 else "x"
            dst_name = f"{part}__{src.name}"
            img_dst = out_root / "images" / split / dst_name
            lbl_dst = out_root / "labels" / split / f"{Path(dst_name).stem}.txt"
            shutil.copy2(src, img_dst)
            if line is None:
                lbl_dst.write_text("", encoding="utf-8")
            else:
                lbl_dst.write_text(line + "\n", encoding="utf-8")

    _copy(train_items, "train")
    _copy(test_items, "test")
    _write_yaml(out_root / "data.yaml", out_root)

    num_defect = sum(1 for _, l in items if l is not None)
    num_normal = sum(1 for _, l in items if l is None)
    print(f"dataset: {out_root}")
    print(f"total={len(items)} train={len(train_items)} test={len(test_items)}")
    print(f"normal={num_normal} defect={num_defect}")
    print(f"skipped_no_match={skipped_no_match} skipped_no_mask={skipped_no_mask}")
    print(f"yaml: {out_root / 'data.yaml'}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROI normal/defect에서 BBox+Mask(YOLO-seg) 데이터셋 생성 + 80/20 split")
    parser.add_argument("--roi-root", default=str(DEFAULT_ROI_ROOT), help="ROI 루트")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="출력 데이터셋 루트")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="test 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--diff-thresh", type=int, default=18, help="차영상 임계값")
    parser.add_argument("--blur-ksize", type=int, default=3, help="가우시안 블러 커널(홀수)")
    parser.add_argument("--part", choices=list(PARTS), help="특정 부품만 생성 (미지정 시 전체)")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    target_parts = (args.part,) if args.part else PARTS
    build_dataset(
        roi_root=Path(args.roi_root),
        out_root=Path(args.out_root),
        test_ratio=args.test_ratio,
        seed=args.seed,
        diff_thresh=args.diff_thresh,
        blur_ksize=args.blur_ksize,
        parts=target_parts,
    )
