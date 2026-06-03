import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATES = (
    ROOT_DIR / "runs" / "segment" / "drone_seg_v1" / "weights" / "best.pt",
    ROOT_DIR / "runs" / "segment" / "runs" / "drone_seg_v1" / "weights" / "best.pt",
    ROOT_DIR / "yolo11n-seg.pt",
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "roi"
PART_ALIASES = {
    "class_4": "body",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_model_path(model_path: str | None) -> Path:
    if model_path:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("기본 세그멘테이션 모델을 찾을 수 없습니다.")


def _collect_images(image_path: str | None, input_dir: str | None, limit: int | None) -> list[Path]:
    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {path}")
        return [path]

    if input_dir:
        directory = Path(input_dir)
    else:
        val_dir = ROOT_DIR / "data" / "images" / "val"
        train_dir = ROOT_DIR / "data" / "images" / "train"
        directory = val_dir if val_dir.exists() else train_dir

    if not directory.exists():
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"입력 경로가 디렉터리가 아닙니다: {directory}")

    images = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"입력 디렉터리에 이미지가 없습니다: {directory}")
    return images[:limit] if limit else images


def _normalize_class_name(raw_name: str) -> str:
    return PART_ALIASES.get(raw_name, raw_name)


def crop_roi_from_images(
    images: list[Path],
    model_path: Path,
    output_dir: Path,
    conf: float,
    label: str,
    min_width: int,
    min_height: int,
    min_area: int,
    min_width_propeller: int,
    min_height_propeller: int,
    min_area_propeller: int,
    min_width_arm: int,
    min_height_arm: int,
    min_area_arm: int,
    min_width_body: int,
    min_height_body: int,
    min_area_body: int,
    save_mask: bool,
) -> None:
    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = model([str(path) for path in images], conf=conf)

    class_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    for image_path, result in zip(images, results):
        image = result.orig_img
        if image is None:
            image = cv2.imread(str(image_path))
        if image is None:
            print(f"스킵: 이미지 로드 실패 {image_path}")
            continue
        if result.boxes is None:
            print(f"{image_path.name}: 검출된 부품 없음")
            continue

        masks_xy = result.masks.xy if result.masks is not None else None

        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls)
            raw_cls_name = class_names.get(cls_id, f"class_{cls_id}")
            cls_name = _normalize_class_name(str(raw_cls_name))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            class_min_w = min_width
            class_min_h = min_height
            class_min_a = min_area
            if cls_name == "propeller":
                class_min_w = max(class_min_w, min_width_propeller)
                class_min_h = max(class_min_h, min_height_propeller)
                class_min_a = max(class_min_a, min_area_propeller)
            elif cls_name == "arm":
                class_min_w = max(class_min_w, min_width_arm)
                class_min_h = max(class_min_h, min_height_arm)
                class_min_a = max(class_min_a, min_area_arm)
            elif cls_name == "body":
                class_min_w = max(class_min_w, min_width_body)
                class_min_h = max(class_min_h, min_height_body)
                class_min_a = max(class_min_a, min_area_body)

            if box_w < class_min_w or box_h < class_min_h or box_area < class_min_a:
                print(
                    f"스킵(작은 ROI): {image_path.name} idx={idx} "
                    f"class={cls_name} size={box_w}x{box_h} area={box_area} "
                    f"(min={class_min_w}x{class_min_h}, area={class_min_a})"
                )
                continue

            roi = image[y1:y2, x1:x2]
            save_dir = output_dir / cls_name / label
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{image_path.stem}_{idx}.jpg"
            cv2.imwrite(str(save_path), roi)
            print(f"저장: {save_path}")

            if save_mask and masks_xy is not None and idx < len(masks_xy):
                full_mask = cv2.fillPoly(
                    np.zeros(image.shape[:2], dtype=np.uint8),
                    [np.int32([masks_xy[idx]])],
                    255,
                )
                roi_mask = full_mask[y1:y2, x1:x2]
                mask_dir = output_dir / cls_name / f"{label}_mask"
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{image_path.stem}_{idx}.png"
                cv2.imwrite(str(mask_path), roi_mask)
                print(f"마스크 저장: {mask_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="분할 결과 기반 ROI crop")
    parser.add_argument("--image", help="단일 입력 이미지 경로")
    parser.add_argument("--input-dir", help="입력 이미지 디렉터리")
    parser.add_argument("--model", help="세그멘테이션 모델 경로")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="ROI 저장 루트 경로")
    parser.add_argument("--conf", type=float, default=0.5, help="세그멘테이션 confidence threshold")
    parser.add_argument("--limit", type=int, help="디렉터리 모드에서 처리할 최대 이미지 수")
    parser.add_argument("--label", default="normal", help="저장 라벨 폴더명 (normal/defect 등)")
    parser.add_argument("--min-width", type=int, default=0, help="최소 ROI 너비(px), 미만이면 스킵")
    parser.add_argument("--min-height", type=int, default=0, help="최소 ROI 높이(px), 미만이면 스킵")
    parser.add_argument("--min-area", type=int, default=0, help="최소 ROI 면적(px^2), 미만이면 스킵")
    parser.add_argument("--min-width-propeller", type=int, default=0, help="propeller 최소 ROI 너비(px)")
    parser.add_argument("--min-height-propeller", type=int, default=0, help="propeller 최소 ROI 높이(px)")
    parser.add_argument("--min-area-propeller", type=int, default=0, help="propeller 최소 ROI 면적(px^2)")
    parser.add_argument("--min-width-arm", type=int, default=0, help="arm 최소 ROI 너비(px)")
    parser.add_argument("--min-height-arm", type=int, default=0, help="arm 최소 ROI 높이(px)")
    parser.add_argument("--min-area-arm", type=int, default=0, help="arm 최소 ROI 면적(px^2)")
    parser.add_argument("--min-width-body", type=int, default=0, help="body 최소 ROI 너비(px)")
    parser.add_argument("--min-height-body", type=int, default=0, help="body 최소 ROI 높이(px)")
    parser.add_argument("--min-area-body", type=int, default=0, help="body 최소 ROI 면적(px^2)")
    parser.add_argument("--save-mask", action="store_true", help="ROI별 부품 마스크를 별도 폴더에 저장")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    model_path = _resolve_model_path(args.model)
    images = _collect_images(image_path=args.image, input_dir=args.input_dir, limit=args.limit)
    crop_roi_from_images(
        images=images,
        model_path=model_path,
        output_dir=Path(args.output_dir),
        conf=args.conf,
        label=args.label,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
        min_width_propeller=args.min_width_propeller,
        min_height_propeller=args.min_height_propeller,
        min_area_propeller=args.min_area_propeller,
        min_width_arm=args.min_width_arm,
        min_height_arm=args.min_height_arm,
        min_area_arm=args.min_area_arm,
        min_width_body=args.min_width_body,
        min_height_body=args.min_height_body,
        min_area_body=args.min_area_body,
        save_mask=args.save_mask,
    )
