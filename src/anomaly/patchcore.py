import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_ROOT = ROOT_DIR / "data" / "roi"
DEFAULT_MODEL_DIR = ROOT_DIR / "runs" / "patchcore"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "patchcore" / "predict"
DEFAULT_CALIBRATION_DIR = ROOT_DIR / "runs" / "patchcore" / "calibration"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NORMAL_ALIASES = {"normal", "good", "ok"}


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _iter_images(directory: Path):
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"이미지 로드 실패: {path}")
    return image


def _make_overlay(image_bgr: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    score_map = np.clip(anomaly_map, 0.0, 1.0)
    heat = np.uint8(score_map * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.62, heat, 0.38, 0.0)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile 계산 대상이 비어있습니다.")
    if q < 0 or q > 100:
        raise ValueError(f"percentile은 0~100 범위여야 합니다: {q}")
    arr = np.array(values, dtype=np.float32)
    return float(np.percentile(arr, q))


def _heatmap_region_metrics(anomaly_map: np.ndarray, hot_thr: float) -> tuple[float, float]:
    if hot_thr < 0.0 or hot_thr > 1.0:
        raise ValueError(f"hot_thr는 0~1 범위여야 합니다: {hot_thr}")

    hot_mask = (anomaly_map >= hot_thr).astype(np.uint8)
    hot_ratio = float(hot_mask.mean())
    if hot_mask.sum() == 0:
        return hot_ratio, 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hot_mask, connectivity=8)
    if num_labels <= 1:
        return hot_ratio, 0.0

    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())  # skip background
    max_blob_ratio = float(largest_area / hot_mask.size)
    return hot_ratio, max_blob_ratio


class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = backbone
        self._feat2 = None
        self._feat3 = None

        self.backbone.layer2.register_forward_hook(self._hook2)
        self.backbone.layer3.register_forward_hook(self._hook3)

    def _hook2(self, _module, _inputs, output):
        self._feat2 = output

    def _hook3(self, _module, _inputs, output):
        self._feat3 = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.backbone(x)
        if self._feat2 is None or self._feat3 is None:
            raise RuntimeError("특징맵 추출 실패")
        feat2 = self._feat2
        feat3 = F.interpolate(self._feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        embedding = torch.cat([feat2, feat3], dim=1)  # [B, C, H, W]
        return embedding

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _ = self.backbone(x)
        if self._feat2 is None or self._feat3 is None:
            raise RuntimeError("특징맵 추출 실패")
        feat2 = self._feat2
        feat3_raw = self._feat3
        feat3 = F.interpolate(feat3_raw, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        embedding = torch.cat([feat2, feat3], dim=1)  # [B, C, H, W]
        return embedding, feat3_raw


@dataclass
class PatchCoreModel:
    part: str
    memory_bank: torch.Tensor  # [N, C]
    input_size: int = 224


class PatchCoreRunner:
    def __init__(self, device: str):
        self.device = device
        self.extractor = PatchFeatureExtractor().to(device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def embed_bgr(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.extractor(tensor)  # [1, C, H, W]
        return emb

    def make_memory_bank(
        self,
        image_paths: list[Path],
        sample_ratio: float,
        seed: int,
    ) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for p in image_paths:
            emb = self.embed_bgr(_load_bgr_image(p))  # [1, C, H, W]
            patches = emb.squeeze(0).permute(1, 2, 0).reshape(-1, emb.shape[1])  # [P, C]
            rows.append(patches.cpu())

        bank = torch.cat(rows, dim=0)  # [N, C]
        if sample_ratio >= 1.0:
            return bank

        rng = random.Random(seed)
        n = bank.shape[0]
        k = max(1, int(n * sample_ratio))
        indices = list(range(n))
        rng.shuffle(indices)
        selected = indices[:k]
        return bank[selected]

    def anomaly_from_bank(
        self,
        image_bgr: np.ndarray,
        memory_bank: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        emb = self.embed_bgr(image_bgr)  # [1, C, H, W]
        _, c, h, w = emb.shape
        patches = emb.squeeze(0).permute(1, 2, 0).reshape(-1, c)  # [P, C]

        bank = memory_bank.to(self.device)
        min_dists: list[torch.Tensor] = []
        chunk = 1024
        with torch.no_grad():
            for i in range(0, patches.shape[0], chunk):
                q = patches[i : i + chunk]
                dist = torch.cdist(q, bank)  # [Q, N]
                min_dist, _ = torch.min(dist, dim=1)
                min_dists.append(min_dist)
        d = torch.cat(min_dists, dim=0)  # [P]

        # PatchCore-like image score: top-k mean of patch distances.
        k = max(1, int(0.1 * d.numel()))
        topk_vals, _ = torch.topk(d, k=k, largest=True)
        score = float(torch.mean(topk_vals).item())

        amap = d.reshape(h, w).cpu().numpy()
        amap = amap - amap.min()
        amap = amap / (amap.max() + 1e-8)
        amap = cv2.resize(amap, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return score, amap

    def score_image(self, image_bgr: np.ndarray, memory_bank: torch.Tensor) -> float:
        score, _ = self.anomaly_from_bank(image_bgr=image_bgr, memory_bank=memory_bank)
        return score

    def gradcam_from_bank(self, image_bgr: np.ndarray, memory_bank: torch.Tensor) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)

        self.extractor.zero_grad(set_to_none=True)
        emb, feat3_raw = self.extractor.forward_with_features(tensor)
        _, c, h, w = emb.shape
        patches = emb.squeeze(0).permute(1, 2, 0).reshape(-1, c)  # [P, C]
        bank = memory_bank.to(self.device)

        # Differentiable PatchCore score: mean of top-k nearest-neighbor distances.
        dist = torch.cdist(patches, bank)  # [P, N]
        min_dist, _ = torch.min(dist, dim=1)  # [P]
        k = max(1, int(0.1 * min_dist.numel()))
        topk_vals, _ = torch.topk(min_dist, k=k, largest=True)
        score = torch.mean(topk_vals)

        grads = torch.autograd.grad(score, feat3_raw, retain_graph=False, create_graph=False)[0]  # [1, C3, H3, W3]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * feat3_raw, dim=1, keepdim=True))  # [1,1,H3,W3]
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return cam


def fit_part(
    part: str,
    roi_root: Path,
    model_dir: Path,
    sample_ratio: float,
    seed: int,
) -> Path:
    part_dir = roi_root / part
    normal_dirs = [d for d in part_dir.iterdir() if d.is_dir() and d.name.lower() in NORMAL_ALIASES] if part_dir.exists() else []
    if not normal_dirs:
        raise FileNotFoundError(f"[{part}] 정상 폴더를 찾을 수 없습니다: {part_dir}/{{normal|good|ok}}")

    image_paths: list[Path] = []
    for d in normal_dirs:
        image_paths.extend(list(_iter_images(d)))
    if not image_paths:
        raise FileNotFoundError(f"[{part}] 정상 이미지가 없습니다: {normal_dirs}")

    device = _resolve_device()
    runner = PatchCoreRunner(device=device)
    bank = runner.make_memory_bank(image_paths=image_paths, sample_ratio=sample_ratio, seed=seed)

    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"{part}_patchcore.pt"
    torch.save(
        {
            "part": part,
            "input_size": 224,
            "memory_bank": bank,
            "sample_ratio": sample_ratio,
            "num_normal_images": len(image_paths),
        },
        ckpt_path,
    )
    print(f"[{part}] 저장 완료: {ckpt_path} (memory={bank.shape})")
    return ckpt_path


def predict_part(
    part: str,
    ckpt_path: Path,
    image: Path | None,
    input_dir: Path | None,
    output_dir: Path,
    threshold: float | None,
    threshold_file: Path | None,
    viz: str,
    rule: str,
    hot_thr: float,
    hot_ratio_thr: float | None,
    max_blob_ratio_thr: float | None,
) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PatchCore 체크포인트를 찾을 수 없습니다: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    memory_bank: torch.Tensor = ckpt["memory_bank"]
    threshold_value = threshold
    if threshold_file:
        if not threshold_file.exists():
            raise FileNotFoundError(f"threshold 파일을 찾을 수 없습니다: {threshold_file}")
        with threshold_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold_value = float(payload["threshold"])
        print(f"threshold 로드: {threshold_value:.6f} ({threshold_file})")

    if image:
        image_paths = [image]
    else:
        if input_dir is None or not input_dir.exists():
            raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {input_dir}")
        image_paths = list(_iter_images(input_dir))
        if not image_paths:
            raise FileNotFoundError(f"입력 이미지가 없습니다: {input_dir}")

    device = _resolve_device()
    runner = PatchCoreRunner(device=device)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for p in image_paths:
        bgr = _load_bgr_image(p)
        score, patchcore_map = runner.anomaly_from_bank(bgr, memory_bank=memory_bank)
        hot_ratio, max_blob_ratio = _heatmap_region_metrics(patchcore_map, hot_thr=hot_thr)

        score_flag = threshold_value is not None and score > threshold_value
        area_conditions: list[bool] = []
        if hot_ratio_thr is not None:
            area_conditions.append(hot_ratio >= hot_ratio_thr)
        if max_blob_ratio_thr is not None:
            area_conditions.append(max_blob_ratio >= max_blob_ratio_thr)
        area_flag = bool(area_conditions) and all(area_conditions)

        verdict = None
        if rule == "score_only":
            if threshold_value is not None:
                verdict = "defect" if score_flag else "normal"
        elif rule == "score_or_area":
            if threshold_value is not None or area_conditions:
                verdict = "defect" if (score_flag or area_flag) else "normal"
        elif rule == "score_and_area":
            if threshold_value is not None and area_conditions:
                verdict = "defect" if (score_flag and area_flag) else "normal"
            elif threshold_value is not None:
                verdict = "defect" if score_flag else "normal"
            elif area_conditions:
                verdict = "defect" if area_flag else "normal"

        # Requested behavior: when defect is detected, visualize Grad-CAM instead of PatchCore heatmap.
        if viz == "gradcam" and verdict == "defect":
            viz_map = runner.gradcam_from_bank(bgr, memory_bank=memory_bank)
        else:
            viz_map = patchcore_map
        overlay = _make_overlay(bgr, viz_map)

        out_img = output_dir / f"{p.stem}_{part}_patchcore.jpg"
        cv2.imwrite(str(out_img), overlay)
        summary.append(
            {
                "image": str(p),
                "part": part,
                "score": score,
                "threshold": threshold_value,
                "verdict": verdict,
                "rule": rule,
                "rule_metrics": {
                    "hot_thr": hot_thr,
                    "hot_ratio": hot_ratio,
                    "max_blob_ratio": max_blob_ratio,
                    "hot_ratio_thr": hot_ratio_thr,
                    "max_blob_ratio_thr": max_blob_ratio_thr,
                    "score_flag": score_flag,
                    "area_flag": area_flag,
                },
                "visualization": "gradcam" if (viz == "gradcam" and verdict == "defect") else "patchcore",
                "result_image": str(out_img),
            }
        )
        if verdict:
            print(f"{p.name}: anomaly_score={score:.4f} ({verdict}) -> {out_img}")
        else:
            print(f"{p.name}: anomaly_score={score:.4f} -> {out_img}")

    with (output_dir / f"{part}_patchcore_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def calibrate_threshold(
    part: str,
    ckpt_path: Path,
    normal_dir: Path,
    output_dir: Path,
    percentile: float,
) -> Path:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PatchCore 체크포인트를 찾을 수 없습니다: {ckpt_path}")
    if not normal_dir.exists():
        raise FileNotFoundError(f"정상 샘플 디렉터리를 찾을 수 없습니다: {normal_dir}")

    image_paths = list(_iter_images(normal_dir))
    if not image_paths:
        raise FileNotFoundError(f"정상 샘플 이미지가 없습니다: {normal_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    memory_bank: torch.Tensor = ckpt["memory_bank"]
    runner = PatchCoreRunner(device=_resolve_device())

    scores: list[float] = []
    for p in image_paths:
        score = runner.score_image(_load_bgr_image(p), memory_bank=memory_bank)
        scores.append(score)

    threshold = _percentile(scores, percentile)
    payload = {
        "part": part,
        "num_samples": len(scores),
        "percentile": percentile,
        "threshold": threshold,
        "min_score": float(min(scores)),
        "max_score": float(max(scores)),
        "mean_score": float(sum(scores) / len(scores)),
        "normal_dir": str(normal_dir),
        "ckpt": str(ckpt_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{part}_threshold_p{int(percentile)}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[{part}] threshold 계산 완료: {threshold:.6f} "
        f"(p{percentile}, samples={len(scores)}) -> {out_file}"
    )
    return out_file


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchCore-style anomaly detection for drone parts")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="normal ROI로 PatchCore memory bank 학습")
    fit.add_argument("--part", required=True, help="부품명 (propeller|arm|body)")
    fit.add_argument("--roi-root", default=str(DEFAULT_ROI_ROOT), help="ROI 루트 디렉터리")
    fit.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="PatchCore 모델 저장 디렉터리")
    fit.add_argument("--sample-ratio", type=float, default=0.2, help="memory bank 샘플링 비율 (0~1]")
    fit.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    pred = sub.add_parser("predict", help="PatchCore 추론")
    pred.add_argument("--part", required=True, help="부품명 (propeller|arm|body)")
    pred.add_argument("--ckpt", help="PatchCore 체크포인트 경로")
    pred.add_argument("--image", help="단일 입력 이미지 경로")
    pred.add_argument("--input-dir", help="여러 입력 이미지 디렉터리")
    pred.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="추론 결과 저장 디렉터리")
    pred.add_argument("--threshold", type=float, help="anomaly threshold (초과 시 defect)")
    pred.add_argument("--threshold-file", help="calibrate로 생성한 threshold json 경로")
    pred.add_argument(
        "--rule",
        choices=["score_only", "score_or_area", "score_and_area"],
        default="score_only",
        help="판정 규칙: score_only(기본), score_or_area, score_and_area",
    )
    pred.add_argument("--hot-thr", type=float, default=0.7, help="히트맵 hot 픽셀 임계값(0~1)")
    pred.add_argument("--hot-ratio-thr", type=float, help="hot 픽셀 비율 임계값(예: 0.02)")
    pred.add_argument("--max-blob-ratio-thr", type=float, help="최대 연결영역 비율 임계값(예: 0.01)")
    pred.add_argument(
        "--viz",
        choices=["patchcore", "gradcam"],
        default="gradcam",
        help="시각화 방식 (gradcam은 defect 판정 샘플에만 적용)",
    )

    cal = sub.add_parser("calibrate", help="정상 샘플 score 분포로 threshold 계산")
    cal.add_argument("--part", required=True, help="부품명 (propeller|arm|body)")
    cal.add_argument("--ckpt", help="PatchCore 체크포인트 경로")
    cal.add_argument("--normal-dir", required=True, help="정상 샘플 이미지 디렉터리")
    cal.add_argument("--percentile", type=float, default=99.0, help="threshold percentile (기본 99)")
    cal.add_argument("--output-dir", default=str(DEFAULT_CALIBRATION_DIR), help="threshold 저장 디렉터리")

    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.command == "fit":
        fit_part(
            part=args.part,
            roi_root=Path(args.roi_root),
            model_dir=Path(args.model_dir),
            sample_ratio=args.sample_ratio,
            seed=args.seed,
        )
    elif args.command == "predict":
        if bool(args.image) == bool(args.input_dir):
            raise ValueError("--image 또는 --input-dir 중 하나만 지정해야 합니다.")
        if args.threshold is not None and args.threshold_file:
            raise ValueError("--threshold와 --threshold-file은 동시에 사용할 수 없습니다.")
        ckpt_path = Path(args.ckpt) if args.ckpt else Path(DEFAULT_MODEL_DIR) / f"{args.part}_patchcore.pt"
        predict_part(
            part=args.part,
            ckpt_path=ckpt_path,
            image=Path(args.image) if args.image else None,
            input_dir=Path(args.input_dir) if args.input_dir else None,
            output_dir=Path(args.output_dir),
            threshold=args.threshold,
            threshold_file=Path(args.threshold_file) if args.threshold_file else None,
            viz=args.viz,
            rule=args.rule,
            hot_thr=args.hot_thr,
            hot_ratio_thr=args.hot_ratio_thr,
            max_blob_ratio_thr=args.max_blob_ratio_thr,
        )
    elif args.command == "calibrate":
        ckpt_path = Path(args.ckpt) if args.ckpt else Path(DEFAULT_MODEL_DIR) / f"{args.part}_patchcore.pt"
        calibrate_threshold(
            part=args.part,
            ckpt_path=ckpt_path,
            normal_dir=Path(args.normal_dir),
            output_dir=Path(args.output_dir),
            percentile=args.percentile,
        )
