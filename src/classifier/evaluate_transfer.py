import argparse
from pathlib import Path

from evaluate import evaluate_all


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_DIR = ROOT_DIR / "data" / "roi"
DEFAULT_CKPT_DIR = ROOT_DIR / "runs" / "classifier" / "exp_maskonly_transfer_balanced"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "classifier" / "eval_transfer"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="전이학습(ResNet 분류기) 전용 평가")
    parser.add_argument("--roi-dir", default=str(DEFAULT_ROI_DIR), help="평가 데이터셋 루트 디렉터리")
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR), help="전이학습 체크포인트 디렉터리")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="평가 결과 저장 디렉터리")
    parser.add_argument("--threshold", type=float, help="고정 defect 확률 임계값 (미지정 시 argmax)")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "regnety_400mf"], help="평가 backbone")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    evaluate_all(
        roi_dir=Path(args.roi_dir),
        ckpt_dir=Path(args.ckpt_dir),
        out_dir=Path(args.out_dir),
        threshold=args.threshold,
        backbone=args.backbone,
    )
