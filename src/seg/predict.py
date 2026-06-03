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
DEFAULT_COLORS = [
    (255, 0, 255),
    (0, 255, 0),
    (0, 165, 255),
    (255, 255, 0),
    (255, 0, 0),
]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "predict"


def _collect_images_from_dir(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file())


def _resolve_input_images(
    image_path: str | None,
    input_dir: str | None,
    limit: int | None,
) -> list[Path]:
    if image_path:
        resolved = Path(image_path)
        if resolved.exists():
            return [resolved]
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {resolved}")

    if input_dir:
        resolved_dir = Path(input_dir)
        if not resolved_dir.exists():
            raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {resolved_dir}")
        if not resolved_dir.is_dir():
            raise NotADirectoryError(f"입력 경로가 디렉터리가 아닙니다: {resolved_dir}")

        images = _collect_images_from_dir(resolved_dir)
        if not images:
            raise FileNotFoundError(f"입력 디렉터리에 이미지가 없습니다: {resolved_dir}")
        return images[:limit] if limit else images

    for candidate_dir in (ROOT_DIR / "data" / "images" / "val", ROOT_DIR / "data" / "images" / "train"):
        if candidate_dir.exists():
            images = _collect_images_from_dir(candidate_dir)
            if images:
                return images[:limit] if limit else images

    raise FileNotFoundError("기본 입력 이미지를 찾을 수 없습니다: data/images/val 또는 data/images/train")


def _resolve_model_path(model_path: str | None) -> Path:
    if model_path:
        resolved = Path(model_path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {resolved}")

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(f"기본 모델 파일을 찾을 수 없습니다:\n{searched}")


def _save_prediction(final_img: np.ndarray, image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{image_path.stem}_pred.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    return save_path


def _draw_legend(image: np.ndarray, labels: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not labels:
        return

    h, w = image.shape[:2]
    short_side = min(h, w)
    # 640 기준으로 스케일, 너무 작거나 커지지 않게 제한
    scale = float(np.clip(short_side / 640.0, 0.55, 1.25))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    thickness = max(1, int(round(2 * scale)))
    text_color = (255, 255, 255)
    x = max(8, int(round(12 * scale)))
    y = max(18, int(round(30 * scale)))
    swatch = max(8, int(round(14 * scale)))
    row_h = max(16, int(round(26 * scale)))
    pad = max(4, int(round(8 * scale)))
    top_pad = max(10, int(round(20 * scale)))

    # dark legend background for readability
    max_w = 0
    for label, _ in labels:
        text_w, _ = cv2.getTextSize(label, font, font_scale, thickness)[0]
        max_w = max(max_w, text_w)
    box_w = swatch + max(6, int(round(10 * scale))) + max_w + max(8, int(round(14 * scale)))
    box_h = row_h * len(labels) + max(6, int(round(10 * scale)))
    cv2.rectangle(image, (x - pad, y - top_pad), (x - pad + box_w, y - top_pad + box_h), (20, 20, 20), -1)

    for i, (label, color) in enumerate(labels):
        row_y = y + i * row_h
        swatch_y_offset = max(1, int(round(2 * scale)))
        text_x_gap = max(5, int(round(8 * scale)))
        cv2.rectangle(image, (x, row_y - swatch + swatch_y_offset), (x + swatch, row_y + swatch_y_offset), color, -1)
        cv2.putText(
            image,
            label,
            (x + swatch + text_x_gap, row_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def _render_prediction(image_path: Path, result, class_name_map: dict[int, str]) -> np.ndarray | None:
    img_bgr = result.orig_img
    if img_bgr is None:
        img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"OpenCV가 이미지를 읽지 못했습니다: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    if result.masks is None or result.boxes is None:
        print(f"{image_path.name}: 탐지된 부품 없음")
        return None

    masks = result.masks.xy
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    legend_entries: list[tuple[str, tuple[int, int, int]]] = []
    seen_labels: set[str] = set()

    for i, mask in enumerate(masks):
        cls_id = int(classes[i])
        label = class_name_map.get(cls_id, f"class_{cls_id}")
        conf = float(confidences[i])
        color = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]
        points = np.int32([mask])

        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
        if label not in seen_labels:
            legend_entries.append((label, color))
            seen_labels.add(label)
        print(f"{image_path.name}: [{label}] conf={conf:.2f}")

    alpha = 0.4
    final_img = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    _draw_legend(final_img, legend_entries)
    return final_img


def predict(
    image_path: str | None = None,
    model_path: str | None = None,
    input_dir: str | None = None,
    limit: int | None = None,
    save_dir: str | None = None,
    conf: float = 0.5,
    show: bool = False,
) -> None:
    images = _resolve_input_images(image_path=image_path, input_dir=input_dir, limit=limit)
    resolved_model_path = _resolve_model_path(model_path)
    model = YOLO(str(resolved_model_path))
    output_dir = Path(save_dir) if save_dir else DEFAULT_OUTPUT_DIR
    results = model([str(image) for image in images], conf=conf)

    class_names = model.names
    if isinstance(class_names, list):
        class_name_map = dict(enumerate(class_names))
    else:
        class_name_map = class_names

    for image, result in zip(images, results):
        final_img = _render_prediction(image, result, class_name_map)
        if final_img is None:
            continue

        save_path = _save_prediction(final_img, image, output_dir)
        print(f"저장 완료: {save_path}")

        if show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.imshow(final_img)
            plt.axis("off")
            plt.title(image.name)
            plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="드론 부품 분할 추론")
    parser.add_argument("--image", dest="image_path", help="단일 이미지 경로")
    parser.add_argument("--input-dir", help="여러 이미지를 읽을 디렉터리 경로")
    parser.add_argument("--model", dest="model_path", help="세그멘테이션 모델 경로")
    parser.add_argument("--limit", type=int, help="디렉터리에서 앞에서부터 테스트할 이미지 수")
    parser.add_argument("--save-dir", help="예측 결과 저장 디렉터리")
    parser.add_argument("--conf", type=float, default=0.5, help="추론 confidence threshold")
    parser.add_argument("--show", action="store_true", help="예측 결과를 화면에 표시")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    predict(
        image_path=args.image_path,
        model_path=args.model_path,
        input_dir=args.input_dir,
        limit=args.limit,
        save_dir=args.save_dir,
        conf=args.conf,
        show=args.show,
    )
