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
DEFAULT_MVTEC_DIR = ROOT_DIR / "data" / "mvtec_ad"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "classifier"
DEFAULT_CATEGORIES = ("metal_nut", "screw", "cable", "tile", "grid", "leather")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BACKBONE_CHOICES = ("resnet18", "regnet_y_400mf", "regnet_y_800mf")


@dataclass
class Sample:
    path: Path
    label_idx: int  # 0 normal, 1 defect


class BinaryImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        image = image.convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(sample.label_idx, dtype=torch.long)


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_model(backbone: str, num_classes: int = 2) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if backbone == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if backbone == "regnet_y_800mf":
        model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"지원하지 않는 backbone: {backbone}")


def _iter_images(directory: Path):
    if not directory.exists():
        return
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _collect_mvtec_samples(
    mvtec_dir: Path,
    categories: tuple[str, ...],
    include_test_good: bool,
) -> list[Sample]:
    samples: list[Sample] = []

    for category in categories:
        cat_dir = mvtec_dir / category
        if not cat_dir.exists():
            print(f"카테고리 스킵(없음): {category}")
            continue

        # normal from train/good
        train_good = cat_dir / "train" / "good"
        for p in _iter_images(train_good) or []:
            samples.append(Sample(path=p, label_idx=0))

        # optional normal from test/good
        if include_test_good:
            test_good = cat_dir / "test" / "good"
            for p in _iter_images(test_good) or []:
                samples.append(Sample(path=p, label_idx=0))

        # defect from test/<defect_type>
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir() and path.name != "good"):
                for p in _iter_images(defect_dir) or []:
                    samples.append(Sample(path=p, label_idx=1))

    return samples


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
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
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def pretrain_mvtec(
    mvtec_dir: Path,
    out_dir: Path,
    categories: tuple[str, ...],
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    seed: int,
    include_test_good: bool,
) -> Path:
    samples = _collect_mvtec_samples(
        mvtec_dir=mvtec_dir,
        categories=categories,
        include_test_good=include_test_good,
    )
    if not samples:
        raise RuntimeError(f"MVTec 샘플이 없습니다: {mvtec_dir}")

    normal_count = sum(1 for s in samples if s.label_idx == 0)
    defect_count = sum(1 for s in samples if s.label_idx == 1)
    if normal_count == 0 or defect_count == 0:
        raise RuntimeError(f"클래스 불충분: normal={normal_count}, defect={defect_count}")

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

    full_train_dataset = BinaryImageDataset(samples=samples, transform=train_transform)
    full_val_dataset = BinaryImageDataset(samples=samples, transform=val_transform)

    val_size = max(1, int(len(samples) * val_ratio))
    train_size = len(samples) - val_size
    if train_size <= 0:
        raise RuntimeError("학습 데이터가 부족합니다. val_ratio를 낮춰주세요.")

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
    model = _build_model(backbone=backbone, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    print(
        f"MVTec pretrain 시작: categories={categories}, train={train_size}, val={val_size}, "
        f"normal={normal_count}, defect={defect_count}, device={device}"
    )
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
            f"[mvtec] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("사전학습 체크포인트 저장 실패")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"mvtec_pretrain_{backbone}_binary.pt"
    torch.save(
        {
            "arch": backbone,
            "input_size": 224,
            "class_names": ["normal", "defect"],
            "categories": list(categories),
            "state_dict": best_state,
            "best_val_acc": best_val_acc,
        },
        ckpt_path,
    )
    print(f"사전학습 저장 완료: {ckpt_path}")
    return ckpt_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVTec AD 기반 normal/defect 사전학습")
    parser.add_argument("--mvtec-dir", default=str(DEFAULT_MVTEC_DIR), help="MVTec AD 루트 디렉터리")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="체크포인트 저장 디렉터리")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="사용할 MVTec 카테고리 목록",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=BACKBONE_CHOICES,
        help="백본 모델 선택",
    )
    parser.add_argument("--epochs", type=int, default=10, help="사전학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="검증셋 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--exclude-test-good",
        action="store_true",
        help="test/good 이미지를 normal 학습 데이터에 포함하지 않음",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    pretrain_mvtec(
        mvtec_dir=Path(args.mvtec_dir),
        out_dir=Path(args.out_dir),
        categories=tuple(args.categories),
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        include_test_good=not args.exclude_test_good,
    )
