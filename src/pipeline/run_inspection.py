import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SEG_MODEL_CANDIDATES = (
    ROOT_DIR / "runs" / "segment" / "drone_seg_v1" / "weights" / "best.pt",
    ROOT_DIR / "runs" / "segment" / "runs" / "drone_seg_v1" / "weights" / "best.pt",
    ROOT_DIR / "yolo11n-seg.pt",
)
DEFAULT_CLASSIFIER_DIR = ROOT_DIR / "runs" / "classifier"
DEFAULT_THRESHOLD_DIR = ROOT_DIR / "runs" / "classifier" / "calibration"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "inspection"
PART_ALIASES = {
    "class_4": "body",
}
PARTS = {"propeller", "arm", "body"}
CLASS_COLORS = {
    "normal": (0, 200, 0),
    "defect": (0, 0, 255),
}


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_seg_model_path(path_str: str | None) -> Path:
    if path_str:
        path = Path(path_str)
        if path.exists():
            return path
        raise FileNotFoundError(f"세그멘테이션 모델 파일이 없습니다: {path}")
    for candidate in DEFAULT_SEG_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("세그멘테이션 모델 파일을 찾지 못했습니다.")


def _build_classifier_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _normalize_part_name(raw_name: str) -> str:
    name = str(raw_name)
    if name in PART_ALIASES:
        return PART_ALIASES[name]
    return name


def _load_classifier(part: str, ckpt_dir: Path, device: str):
    ckpt_path = ckpt_dir / f"{part}_resnet18_binary.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{part}] 분류기 체크포인트가 없습니다: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    class_names = checkpoint.get("class_names", ["normal", "defect"])
    model = _build_classifier_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def _preprocess_roi(roi_rgb: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(roi_rgb).unsqueeze(0)


def _predict_roi(model: nn.Module, class_names: list[str], roi_bgr: np.ndarray, device: str):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = _preprocess_roi(roi_rgb).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = class_names[pred_idx]
        pred_score = float(probs[pred_idx].item())
    return pred_idx, pred_label, pred_score, tensor


def _load_part_threshold(part: str, threshold_dir: Path | None, threshold_key: str) -> float | None:
    if threshold_dir is None:
        return None
    path = threshold_dir / f"{part}_defect_prob_thresholds.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if threshold_key == "suggested_threshold":
        value = payload.get("suggested_threshold")
    else:
        value = payload.get("thresholds", {}).get(threshold_key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _grad_cam(model: nn.Module, input_tensor: torch.Tensor, target_index: int, roi_bgr: np.ndarray) -> np.ndarray:
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients["value"] = grad_output[0].detach()

    target_layer = model.layer4[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits[0, target_index]
        score.backward(retain_graph=False)
    finally:
        handle_f.remove()
        handle_b.remove()

    acts = activations["value"]  # [1, C, H, W]
    grads = gradients["value"]  # [1, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    heatmap = cv2.resize(cam, (roi_bgr.shape[1], roi_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(roi_bgr, 0.6, heatmap, 0.4, 0)


def _draw_label(image_bgr: np.ndarray, x1: int, y1: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = image_bgr.shape[:2]
    base = min(h, w)
    font_scale = max(0.35, min(0.50, base / 2200.0))
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    y_text = max(text_h + 8, y1)
    box_y1 = y_text - text_h - baseline - 6
    box_y2 = y_text + baseline + 4
    box_x2 = x1 + text_w + 10

    cv2.rectangle(image_bgr, (x1, box_y1), (box_x2, box_y2), color, -1)
    text_org = (x1 + 5, y_text - 4)
    # thin outline first, then white text for readability on bright areas
    cv2.putText(image_bgr, text, text_org, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image_bgr, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def inspect_image(
    image_path: Path,
    seg_model_path: Path,
    classifier_dir: Path,
    output_dir: Path,
    conf: float,
    conf_propeller: float | None,
    conf_arm: float | None,
    conf_body: float | None,
    gradcam_all: bool,
    threshold_dir: Path | None,
    threshold_key: str,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"입력 이미지가 없습니다: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {image_path}")

    seg_model = YOLO(str(seg_model_path))
    part_conf_map = {
        "propeller": conf if conf_propeller is None else conf_propeller,
        "arm": conf if conf_arm is None else conf_arm,
        "body": conf if conf_body is None else conf_body,
    }
    infer_conf = min(part_conf_map.values())
    results = seg_model(str(image_path), conf=infer_conf)
    if not results:
        print("세그멘테이션 결과가 없습니다.")
        return

    result = results[0]
    if result.boxes is None:
        print("검출된 부품이 없습니다.")
        return

    class_name_map = seg_model.names if isinstance(seg_model.names, dict) else dict(enumerate(seg_model.names))
    classes = result.boxes.cls.cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    seg_scores = result.boxes.conf.cpu().numpy()

    device = _resolve_device()
    classifier_cache = {}
    image_out_dir = output_dir / image_path.stem
    roi_out_dir = image_out_dir / "roi"
    gradcam_out_dir = image_out_dir / "gradcam"
    roi_out_dir.mkdir(parents=True, exist_ok=True)
    gradcam_out_dir.mkdir(parents=True, exist_ok=True)

    annotated = image_bgr.copy()
    summary = {
        "image": str(image_path),
        "seg_model": str(seg_model_path),
        "detections": [],
    }

    for idx, (cls_id, box, seg_score) in enumerate(zip(classes, boxes, seg_scores)):
        raw_name = class_name_map.get(int(cls_id), f"class_{cls_id}")
        part = _normalize_part_name(raw_name)
        if part not in PARTS:
            print(f"알 수 없는 부품 클래스 스킵: {raw_name}")
            continue
        if float(seg_score) < part_conf_map[part]:
            continue

        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_bgr.shape[1], x2)
        y2 = min(image_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = image_bgr[y1:y2, x1:x2]

        roi_path = roi_out_dir / f"{idx:02d}_{part}.jpg"
        cv2.imwrite(str(roi_path), roi)

        if part not in classifier_cache:
            try:
                classifier_cache[part] = _load_classifier(part, classifier_dir, device=device)
            except FileNotFoundError as exc:
                print(f"[{part}] 분류기 없음으로 스킵: {exc}")
                continue
        classifier, class_names = classifier_cache[part]

        pred_idx, pred_label, pred_score, input_tensor = _predict_roi(classifier, class_names, roi, device=device)
        defect_prob = None
        threshold_used = _load_part_threshold(part, threshold_dir=threshold_dir, threshold_key=threshold_key)
        if "defect" in class_names:
            defect_idx = class_names.index("defect")
            with torch.no_grad():
                logits = classifier(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                defect_prob = float(probs[defect_idx].item())
            if threshold_used is not None:
                pred_label = "defect" if defect_prob >= threshold_used else "normal"
                pred_score = defect_prob if pred_label == "defect" else (1.0 - defect_prob)

        label_color = CLASS_COLORS.get(pred_label, (255, 165, 0))
        short_label = "D" if pred_label == "defect" else "N"
        if defect_prob is not None:
            draw_text = f"{part} {short_label} {defect_prob:.2f}"
        else:
            draw_text = f"{part} {short_label} {pred_score:.2f}"

        det_info = {
            "index": idx,
            "part": part,
            "seg_conf": float(seg_score),
            "pred_label": pred_label,
            "pred_score": pred_score,
            "defect_prob": defect_prob,
            "threshold_used": threshold_used,
            "roi_path": str(roi_path),
            "bbox_xyxy": [x1, y1, x2, y2],
        }

        if pred_label == "defect" or gradcam_all:
            gradcam = _grad_cam(classifier, input_tensor=input_tensor, target_index=pred_idx, roi_bgr=roi)
            gradcam_path = gradcam_out_dir / f"{idx:02d}_{part}_gradcam.jpg"
            cv2.imwrite(str(gradcam_path), gradcam)
            det_info["gradcam_path"] = str(gradcam_path)
            # Full-image view에서도 결함 근거가 보이도록 ROI 위치에 Grad-CAM을 합성한다.
            if pred_label == "defect":
                gh, gw = gradcam.shape[:2]
                ah, aw = (y2 - y1), (x2 - x1)
                if gh != ah or gw != aw:
                    gradcam = cv2.resize(gradcam, (aw, ah))
                roi_canvas = annotated[y1:y2, x1:x2]
                blended = cv2.addWeighted(roi_canvas, 0.25, gradcam, 0.75, 0)
                annotated[y1:y2, x1:x2] = blended
            if pred_label == "defect":
                print(f"[DEFECT] {part} score={pred_score:.2f} gradcam={gradcam_path}")
            else:
                print(f"[NORMAL] {part} score={pred_score:.2f} gradcam={gradcam_path}")
        else:
            print(f"[NORMAL] {part} score={pred_score:.2f}")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), label_color, 2)
        _draw_label(annotated, x1=x1, y1=y1, text=draw_text, color=label_color)

        summary["detections"].append(det_info)

    output_image_path = image_out_dir / f"{image_path.stem}_inspection.jpg"
    cv2.imwrite(str(output_image_path), annotated)
    summary["output_image"] = str(output_image_path)

    summary_path = image_out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"검사 결과 이미지: {output_image_path}")
    print(f"요약 파일: {summary_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="드론 부품 검사 파이프라인 (Seg -> ROI -> Classify -> Grad-CAM)")
    parser.add_argument("--image", required=True, help="드론 전체 이미지 경로")
    parser.add_argument("--seg-model", help="YOLO 세그멘테이션 모델 경로")
    parser.add_argument("--classifier-dir", default=str(DEFAULT_CLASSIFIER_DIR), help="부품별 분류기 모델 디렉터리")
    parser.add_argument(
        "--threshold-dir",
        default=str(DEFAULT_THRESHOLD_DIR),
        help="부품별 임계값 JSON 디렉터리 (없으면 argmax 판정)",
    )
    parser.add_argument(
        "--threshold-key",
        default="suggested_threshold",
        choices=["suggested_threshold", "p95_normal_defect_prob", "p99_normal_defect_prob", "p995_normal_defect_prob"],
        help="threshold-dir 사용 시 임계값 키",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="결과 저장 디렉터리")
    parser.add_argument("--conf", type=float, default=0.5, help="세그멘테이션 confidence threshold")
    parser.add_argument("--conf-propeller", type=float, help="propeller 클래스 전용 confidence threshold")
    parser.add_argument("--conf-arm", type=float, help="arm 클래스 전용 confidence threshold")
    parser.add_argument("--conf-body", type=float, help="body 클래스 전용 confidence threshold")
    parser.add_argument("--gradcam-all", action="store_true", help="normal 포함 모든 검출에 Grad-CAM 저장")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    inspect_image(
        image_path=Path(args.image),
        seg_model_path=_resolve_seg_model_path(args.seg_model),
        classifier_dir=Path(args.classifier_dir),
        output_dir=Path(args.output_dir),
        conf=args.conf,
        conf_propeller=args.conf_propeller,
        conf_arm=args.conf_arm,
        conf_body=args.conf_body,
        gradcam_all=args.gradcam_all,
        threshold_dir=Path(args.threshold_dir) if args.threshold_dir else None,
        threshold_key=args.threshold_key,
    )
