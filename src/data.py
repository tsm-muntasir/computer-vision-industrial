
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_images(folder: Path) -> list[Path]:
    """Return image files in a folder, including nested folders."""
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_training_images(category_dir: Path) -> list[dict]:
    """Discover normal-only images in the official training split."""
    good_dir = category_dir / "train" / "good"

    if not good_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found: {good_dir}"
        )

    records = []

    for image_path in find_images(good_dir):
        records.append(
            {
                "image_path": image_path,
                "label": 0,
                "mask_path": None,
                "category": "good",
            }
        )

    if not records:
        raise ValueError(f"No training images found in {good_dir}")

    return records


def discover_test_images(category_dir: Path) -> list[dict]:
    """Discover normal and defective test images with their masks."""
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    records = []

    class_dirs = sorted(
        path for path in test_dir.iterdir() if path.is_dir()
    )

    for class_dir in class_dirs:
        class_name = class_dir.name
        is_anomaly = class_name != "good"

        for image_path in find_images(class_dir):
            mask_path = None

            if is_anomaly:
                mask_path = (
                    ground_truth_dir
                    / class_name
                    / f"{image_path.stem}_mask.png"
                )

                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Ground-truth mask missing for {image_path}: "
                        f"{mask_path}"
                    )

            records.append(
                {
                    "image_path": image_path,
                    "label": int(is_anomaly),
                    "mask_path": mask_path,
                    "category": class_name,
                }
            )

    if not records:
        raise ValueError(f"No test images found in {test_dir}")

    return records


class IndustrialAnomalyDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_size: int = 256,
        training: bool = False,
        normalize: bool = True,
    ):
        self.records = records
        self.image_size = image_size
        self.training = training
        self.normalize = normalize

        image_transforms = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]

        if normalize:
            image_transforms.append(
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            )

        self.image_transform = transforms.Compose(image_transforms)

        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        image_path = Path(record["image_path"])

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            ) from error

        mask_path = record["mask_path"]

        if mask_path is None:
            mask = Image.new(
                "L", image.size, color=0
            )
        else:
            mask_path = Path(mask_path)

            with Image.open(mask_path) as source:
                mask = source.convert("L")

        # Apply the same random flip to image and mask.
        if self.training and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        image_tensor = self.image_transform(image)

        mask = self.mask_resize(mask)
        mask_array = np.array(mask, dtype=np.uint8).copy()
        mask_tensor = torch.from_numpy(
            (mask_array > 0).astype(np.float32)
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "label": torch.tensor(
                record["label"], dtype=torch.long
            ),
            "mask": mask_tensor,
            "path": str(image_path),
            "category": record["category"],
        }


def build_dataloaders(
    dataset_root="data/raw/mvtec_anomaly_detection",
    category="bottle",
    image_size=256,
    batch_size=16,
    validation_fraction=0.2,
    seed=42,
    num_workers=0,
    normalize=True,
):
    """
    Split only official normal training images into train and validation.
    Keep the official test set untouched.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    category_dir = Path(dataset_root) / category

    all_train_records = discover_training_images(category_dir)
    test_records = discover_test_images(category_dir)

    # Deterministic split for reproducibility.
    rng = random.Random(seed)
    shuffled_records = all_train_records.copy()
    rng.shuffle(shuffled_records)

    validation_count = max(
        1, round(len(shuffled_records) * validation_fraction)
    )

    val_records = shuffled_records[:validation_count]
    train_records = shuffled_records[validation_count:]

    if not train_records:
        raise ValueError("Training split is empty.")

    train_dataset = IndustrialAnomalyDataset(
        train_records,
        image_size=image_size,
        training=True,
        normalize=normalize,
    )

    val_dataset = IndustrialAnomalyDataset(
        val_records,
        image_size=image_size,
        training=False,
        normalize=normalize,
    )

    test_dataset = IndustrialAnomalyDataset(
        test_records,
        image_size=image_size,
        training=False,
        normalize=normalize,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
