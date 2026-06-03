import argparse
import random
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "roi" / "propeller" / "normal"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "roi" / "propeller" / "defect_synth"
DEFAULT_MASK_DIR = ROOT_DIR / "data" / "roi" / "propeller" / "normal_mask"


class RealisticCrackGenerator:
    """물리 기반에 가까운 크랙 합성기."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _branch(
        self,
        start_xy: tuple[int, int],
        angle: float,
        length: int,
        shape_hw: tuple[int, int],
        allowed_mask: np.ndarray | None,
    ) -> list[tuple[int, int]]:
        h, w = shape_hw
        path = [start_xy]
        for _ in range(max(8, length)):
            angle += self.rng.gauss(0.0, 0.22)
            step = self.rng.uniform(0.6, 1.8)
            dx = int(np.cos(angle) * step)
            dy = int(np.sin(angle) * step)
            nx = int(np.clip(path[-1][0] + dx, 0, w - 1))
            ny = int(np.clip(path[-1][1] + dy, 0, h - 1))
            if allowed_mask is not None and allowed_mask[ny, nx] == 0:
                continue
            path.append((nx, ny))
        return path

    def generate_crack_path(
        self,
        start_xy: tuple[int, int],
        shape_hw: tuple[int, int],
        allowed_mask: np.ndarray | None,
    ) -> list[tuple[int, int]]:
        h, w = shape_hw
        path = [start_xy]
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        length = self.rng.randint(max(40, min(h, w) // 6), max(90, min(h, w) // 2))

        for _ in range(length):
            angle += self.rng.gauss(0.0, 0.18)
            step = self.rng.uniform(0.7, 2.1)
            dx = int(np.cos(angle) * step)
            dy = int(np.sin(angle) * step)
            nx = int(np.clip(path[-1][0] + dx, 0, w - 1))
            ny = int(np.clip(path[-1][1] + dy, 0, h - 1))
            if allowed_mask is not None and allowed_mask[ny, nx] == 0:
                continue
            path.append((nx, ny))

            if self.rng.random() < 0.045 and len(path) > 10:
                branch = self._branch(
                    start_xy=path[-1],
                    angle=angle + self.rng.uniform(0.4, 1.0),
                    length=max(10, length // 3),
                    shape_hw=(h, w),
                    allowed_mask=allowed_mask,
                )
                path.extend(branch)
        return path

    def apply_crack(
        self,
        image_bgr: np.ndarray,
        num_cracks: int = 2,
        allowed_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        result = image_bgr.copy().astype(np.float32)
        h, w = result.shape[:2]
        crack_mask = np.zeros((h, w), dtype=np.float32)

        ys = xs = None
        if allowed_mask is not None and np.count_nonzero(allowed_mask) > 0:
            ys, xs = np.where(allowed_mask > 0)

        for _ in range(max(1, num_cracks)):
            if xs is not None and ys is not None and len(xs) > 0:
                sel = self.rng.randint(0, len(xs) - 1)
                sx, sy = int(xs[sel]), int(ys[sel])
            else:
                sx = self.rng.randint(w // 4, max(w // 4 + 1, 3 * w // 4))
                sy = self.rng.randint(h // 4, max(h // 4 + 1, 3 * h // 4))

            path = self.generate_crack_path((sx, sy), (h, w), allowed_mask=allowed_mask)
            if len(path) < 2:
                continue

            for i, (x, y) in enumerate(path):
                base_w = max(1.0, self.rng.gauss(1.6, 0.45))
                taper = 1.0 - (i / max(1, len(path))) * 0.45
                width = max(1, int(base_w * taper))

                y1, y2 = max(0, y - 5), min(h, y + 6)
                x1, x2 = max(0, x - 5), min(w, x + 6)
                local_mean = float(np.mean(result[y1:y2, x1:x2]))
                crack_color = np.clip(local_mean * self.rng.uniform(0.28, 0.62), 0, 255)
                cv2.circle(result, (x, y), width, (crack_color, crack_color, crack_color), -1, lineType=cv2.LINE_AA)
                cv2.circle(crack_mask, (x, y), width, 1.0, -1, lineType=cv2.LINE_AA)

        # 경계 블렌딩 + 하이라이트
        mask_blurred = cv2.GaussianBlur(crack_mask, (0, 0), 0.7)
        highlight = cv2.GaussianBlur(crack_mask, (0, 0), 2.0) * self.rng.uniform(10.0, 18.0)
        result += highlight[..., np.newaxis]

        # 마스크 밖은 원본 유지 (mask_only 의도 보존)
        if allowed_mask is not None:
            keep = (allowed_mask > 0).astype(np.float32)[..., np.newaxis]
            result = result * keep + image_bgr.astype(np.float32) * (1.0 - keep)

        result = np.clip(result, 0, 255).astype(np.uint8)
        crack_binary = (mask_blurred > 0.28).astype(np.uint8)
        return result, crack_binary


def _iter_images(directory: Path):
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _draw_crack_like_line(
    image: np.ndarray,
    rng: random.Random,
    num_vertices: int,
    thickness: int,
    darkness: int,
    allowed_mask: np.ndarray | None = None,
) -> None:
    h, w = image.shape[:2]
    if allowed_mask is not None and np.count_nonzero(allowed_mask) > 0:
        ys, xs = np.where(allowed_mask > 0)
        sel = rng.randint(0, len(xs) - 1)
        x, y = int(xs[sel]), int(ys[sel])
    else:
        x = rng.randint(int(0.1 * w), int(0.9 * w))
        y = rng.randint(int(0.1 * h), int(0.9 * h))
    points = [(x, y)]

    for _ in range(max(2, num_vertices - 1)):
        dx = rng.randint(-max(8, w // 10), max(8, w // 10))
        dy = rng.randint(-max(8, h // 10), max(8, h // 10))
        nx = int(np.clip(points[-1][0] + dx, 0, w - 1))
        ny = int(np.clip(points[-1][1] + dy, 0, h - 1))
        if allowed_mask is not None and allowed_mask[ny, nx] == 0:
            continue
        points.append((nx, ny))

    if len(points) < 2:
        return

    # crack base (dark)
    crack_color = (darkness, darkness, darkness)
    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, crack_color, thickness, lineType=cv2.LINE_AA)

    # highlight around crack for realism
    if thickness >= 2:
        hi_color = (min(255, darkness + 50),) * 3
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, hi_color, 1, lineType=cv2.LINE_AA)

    # random branch
    if rng.random() < 0.55 and len(points) >= 3:
        pivot = points[rng.randint(1, len(points) - 2)]
        bx = int(np.clip(pivot[0] + rng.randint(-w // 8, w // 8), 0, w - 1))
        by = int(np.clip(pivot[1] + rng.randint(-h // 8, h // 8), 0, h - 1))
        cv2.line(image, pivot, (bx, by), crack_color, max(1, thickness - 1), lineType=cv2.LINE_AA)


def _add_surface_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)
    h, w = out.shape[:2]

    # subtle illumination shift
    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-8, 8)
    out = out * alpha + beta

    # mild gaussian noise
    noise_std = rng.uniform(1.5, 6.0)
    noise = np.random.normal(0.0, noise_std, size=(h, w, 3)).astype(np.float32)
    out = out + noise

    # occasional blur
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize_defect_image(
    image_bgr: np.ndarray,
    rng: random.Random,
    crack_count: int,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    base = max(1, min(h, w) // 140)

    for _ in range(crack_count):
        _draw_crack_like_line(
            out,
            rng=rng,
            num_vertices=rng.randint(4, 10),
            thickness=rng.randint(base, base + 2),
            darkness=rng.randint(20, 65),
            allowed_mask=allowed_mask,
        )

    out = _add_surface_noise(out, rng)
    return out


def synthesize_defect_image_realistic(
    image_bgr: np.ndarray,
    rng: random.Random,
    crack_count: int,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    gen = RealisticCrackGenerator(rng=rng)
    out, _ = gen.apply_crack(image_bgr=image_bgr, num_cracks=crack_count, allowed_mask=allowed_mask)
    out = _add_surface_noise(out, rng)
    return out


def run(
    input_dir: Path,
    output_dir: Path,
    mask_dir: Path | None,
    mask_only: bool,
    copies_per_image: int,
    crack_count_min: int,
    crack_count_max: int,
    seed: int,
    realistic: bool,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    images = list(_iter_images(input_dir))
    if not images:
        raise FileNotFoundError(f"입력 이미지가 없습니다: {input_dir}")

    created = 0
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"스킵: 로드 실패 {image_path}")
            continue
        allowed_mask = None
        if mask_only:
            if mask_dir is None:
                print(f"스킵(mask 없음): {image_path}")
                continue
            mpath = mask_dir / f"{image_path.stem}.png"
            if not mpath.exists():
                print(f"스킵(mask 파일 없음): {mpath}")
                continue
            allowed_mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if allowed_mask is None:
                print(f"스킵(mask 로드 실패): {mpath}")
                continue
            if allowed_mask.shape[:2] != img.shape[:2]:
                print(f"스킵(mask 크기 불일치): {mpath}")
                continue

        for i in range(copies_per_image):
            crack_count = rng.randint(crack_count_min, crack_count_max)
            if realistic:
                synth = synthesize_defect_image_realistic(
                    img, rng=rng, crack_count=crack_count, allowed_mask=allowed_mask
                )
            else:
                synth = synthesize_defect_image(img, rng=rng, crack_count=crack_count, allowed_mask=allowed_mask)
            out_name = f"{image_path.stem}_synthdefect_{i+1:02d}{image_path.suffix if image_path.suffix else '.jpg'}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), synth)
            created += 1

    print(f"완료: {created}개 생성 -> {output_dir}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="정상 이미지에 합성 결함(크랙) 추가")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="정상 이미지 입력 디렉터리")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="합성 결함 이미지 출력 디렉터리")
    parser.add_argument("--mask-dir", default=str(DEFAULT_MASK_DIR), help="ROI 마스크 디렉터리 (stem 동일한 .png)")
    parser.add_argument("--mask-only", action="store_true", help="마스크 영역(mask==1) 안에만 결함 추가")
    parser.add_argument("--copies-per-image", type=int, default=2, help="원본 1장당 생성 개수")
    parser.add_argument("--crack-count-min", type=int, default=1, help="이미지당 최소 크랙 개수")
    parser.add_argument("--crack-count-max", type=int, default=3, help="이미지당 최대 크랙 개수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--realistic", action="store_true", help="현실형 랜덤워크 크랙 생성기 사용")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
        mask_only=args.mask_only,
        copies_per_image=args.copies_per_image,
        crack_count_min=args.crack_count_min,
        crack_count_max=args.crack_count_max,
        seed=args.seed,
        realistic=args.realistic,
    )
