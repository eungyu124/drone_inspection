import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_YAML = ROOT_DIR / "data" / "drone.yaml"
MODEL_WEIGHTS = ROOT_DIR / "yolo11n-seg.pt"
TRAIN_IMAGES_DIR = ROOT_DIR / "data" / "images" / "train"
VAL_IMAGES_DIR = ROOT_DIR / "data" / "images" / "val"


def _resolve_dataset_dirs(dataset_yaml: Path) -> tuple[Path, Path]:
    import yaml

    with dataset_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "")).expanduser()
    train_rel = Path(cfg.get("train", "images/train"))
    val_rel = Path(cfg.get("val", "images/val"))
    train_dir = train_rel if train_rel.is_absolute() else (root / train_rel)
    val_dir = val_rel if val_rel.is_absolute() else (root / val_rel)
    return train_dir, val_dir


def _validate_dataset(dataset_yaml: Path) -> None:
    train_dir, val_dir = _resolve_dataset_dirs(dataset_yaml)
    train_count = sum(1 for path in train_dir.iterdir() if path.is_file()) if train_dir.exists() else 0
    val_count = sum(1 for path in val_dir.iterdir() if path.is_file()) if val_dir.exists() else 0

    if train_count == 0:
        raise FileNotFoundError(f"학습 이미지가 없습니다: {train_dir}")

    if val_count == 0:
        raise FileNotFoundError(f"검증 이미지가 없습니다: {val_dir}")


def _resolve_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(
    model_weights: str | Path = MODEL_WEIGHTS,
    dataset_yaml: Path = DATASET_YAML,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 1,
    run_name: str = "drone_seg_v1",
    project_dir: Path = ROOT_DIR / "runs" / "segment",
    conf: float = 0.001,
) -> None:
    resolved_weights = Path(model_weights)
    if resolved_weights.exists():
        model_source = str(resolved_weights)
    else:
        # Allow Ultralytics model aliases (e.g. "yolov8n-seg.pt") and auto-download behavior.
        model_source = str(model_weights)

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"데이터셋 설정 파일을 찾을 수 없습니다: {dataset_yaml}")

    _validate_dataset(dataset_yaml)

    model = YOLO(model_source)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=str(project_dir),
        device=_resolve_device(),
        conf=conf,
        val=True,
        save=True,
        exist_ok=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO11n-seg 학습")
    parser.add_argument("--model", default=str(MODEL_WEIGHTS), help="초기 세그멘테이션 가중치 경로")
    parser.add_argument("--data", default=str(DATASET_YAML), help="데이터셋 YAML 경로")
    parser.add_argument("--epochs", type=int, default=30, help="학습 epoch 수")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=1, help="배치 크기")
    parser.add_argument("--run-name", default="drone_seg_v1", help="실험 이름")
    parser.add_argument("--project", default=str(ROOT_DIR / "runs" / "segment"), help="학습 결과 저장 경로")
    parser.add_argument("--conf", type=float, default=0.001, help="훈련 중 objectness conf 임계값")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train(
        model_weights=Path(args.model),
        dataset_yaml=Path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
        project_dir=Path(args.project),
        conf=args.conf,
    )
