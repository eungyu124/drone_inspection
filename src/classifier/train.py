import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_DIR = ROOT_DIR / "data" / "roi"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "classifier"
PARTS = ("propeller", "arm", "body")
NORMAL_ALIASES = {"good", "normal", "ok"}
DEFECT_ALIASES = {"defect", "bad", "ng", "abnormal"}


@dataclass
class Sample:
    path: Path
    label_idx: int  # 0: normal, 1: defect


class RoiBinaryDataset(Dataset):
    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample.label_idx, dtype=torch.long)
        return image, label


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_model(num_classes: int = 2, backbone: str = "resnet18") -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if backbone == "regnety_400mf":
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"지원하지 않는 backbone: {backbone}")


def _load_init_weights(model: nn.Module, init_weights: Path | None) -> None:
    if init_weights is None:
        return
    if not init_weights.exists():
        raise FileNotFoundError(f"초기 가중치 파일을 찾을 수 없습니다: {init_weights}")

    checkpoint = torch.load(init_weights, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    # Ignore classifier head mismatch and initialize backbone from checkpoint.
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"초기 가중치 로드: {init_weights}")
    if unexpected:
        print(f"경고: 예상치 못한 키 {len(unexpected)}개")
    if missing:
        print(f"참고: 미로드 키 {len(missing)}개 (대부분 fc 레이어)")


def _collect_samples(part_dir: Path) -> list[Sample]:
    if not part_dir.exists():
        return []

    samples: list[Sample] = []
    for class_dir in sorted(path for path in part_dir.iterdir() if path.is_dir()):
        class_name = class_dir.name.lower()
        if class_name in NORMAL_ALIASES:
            label_idx = 0
        elif class_name in DEFECT_ALIASES:
            label_idx = 1
        else:
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append(Sample(path=image_path, label_idx=label_idx))
    return samples


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_count += batch_size

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def train_part_classifier(
    part: str,
    roi_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    seed: int,
    init_weights: Path | None = None,
    backbone: str = "resnet18",
) -> None:
    part_dir = roi_dir / part
    samples = _collect_samples(part_dir)
    normal_count = sum(1 for s in samples if s.label_idx == 0)
    defect_count = sum(1 for s in samples if s.label_idx == 1)

    if normal_count == 0 or defect_count == 0:
        print(
            f"[{part}] 학습 스킵: normal={normal_count}, defect={defect_count}. "
            f"`{part_dir}`에 두 클래스가 모두 필요합니다."
        )
        return

    random.seed(seed)
    random.shuffle(samples)

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    full_train_dataset = RoiBinaryDataset(samples, transform=train_transform)
    full_val_dataset = RoiBinaryDataset(samples, transform=val_transform)

    val_size = max(1, int(len(samples) * val_ratio))
    train_size = len(samples) - val_size
    if train_size <= 0:
        raise RuntimeError(f"[{part}] 학습 데이터가 부족합니다. 샘플 수를 늘리거나 val_ratio를 낮춰주세요.")

    train_subset, _ = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    _, val_subset = random_split(
        full_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = _resolve_device()
    model = _build_model(num_classes=2, backbone=backbone).to(device)
    _load_init_weights(model, init_weights=init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    print(f"[{part}] train={train_size}, val={val_size}, normal={normal_count}, defect={defect_count}, device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_size_now = images.size(0)
            running_loss += loss.item() * batch_size_now
            running_correct += (preds == labels).sum().item()
            running_total += batch_size_now

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        val_loss, val_acc = _evaluate(model, val_loader, device=device)
        print(
            f"[{part}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"[{part}] 체크포인트 저장 실패: best state가 비어있습니다.")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{part}_{backbone}_binary.pt"
    torch.save(
        {
            "part": part,
            "arch": backbone,
            "input_size": 224,
            "class_names": ["normal", "defect"],
            "state_dict": best_state,
            "best_val_acc": best_val_acc,
        },
        ckpt_path,
    )
    print(f"[{part}] 저장 완료: {ckpt_path}")


def train_all(
    roi_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    seed: int,
    init_weights: Path | None = None,
    backbone: str = "resnet18",
) -> None:
    for part in PARTS:
        train_part_classifier(
            part=part,
            roi_dir=roi_dir,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            val_ratio=val_ratio,
            seed=seed,
            init_weights=init_weights,
            backbone=backbone,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="부품별 normal/defect 분류기 학습")
    parser.add_argument("--roi-dir", default=str(DEFAULT_ROI_DIR), help="ROI 데이터 루트 디렉터리")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="모델 저장 디렉터리")
    parser.add_argument("--epochs", type=int, default=15, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="검증셋 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--init-weights", help="사전학습 체크포인트 경로 (.pt)")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "regnety_400mf"],
        help="분류 backbone",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train_all(
        roi_dir=Path(args.roi_dir),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        init_weights=Path(args.init_weights) if args.init_weights else None,
        backbone=args.backbone,
    )
