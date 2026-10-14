
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_dataloaders,
)


def main():
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root="data/raw/mvtec_anomaly_detection",
        category="bottle",
        image_size=256,
        batch_size=8,
        validation_fraction=0.2,
        seed=42,
        num_workers=0,
    )

    train_paths = {
        record["image_path"]
        for record in train_loader.dataset.records
    }
    val_paths = {
        record["image_path"]
        for record in val_loader.dataset.records
    }
    test_paths = {
        record["image_path"]
        for record in test_loader.dataset.records
    }

    # No training/validation overlap, and no test leakage.
    assert train_paths.isdisjoint(val_paths), (
        "Training and validation images overlap."
    )
    assert train_paths.isdisjoint(test_paths), (
        "Training and test images overlap."
    )
    assert val_paths.isdisjoint(test_paths), (
        "Validation and test images overlap."
    )

    # Official training and validation images must be normal.
    assert all(
        record["label"] == 0
        for record in train_loader.dataset.records
    )
    assert all(
        record["label"] == 0
        for record in val_loader.dataset.records
    )

    batch = next(iter(train_loader))

    images = batch["image"]
    labels = batch["label"]
    masks = batch["mask"]

    print("=" * 60)
    print("PYTORCH DATA PIPELINE TEST")
    print("=" * 60)

    print(f"Training images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images:       {len(test_loader.dataset)}")

    print(f"\nImage batch shape: {tuple(images.shape)}")
    print(f"Label shape:       {tuple(labels.shape)}")
    print(f"Mask batch shape:  {tuple(masks.shape)}")

    print(f"Image dtype:       {images.dtype}")
    print(f"Label dtype:       {labels.dtype}")
    print(f"Mask dtype:        {masks.dtype}")

    print(f"Image min:         {images.min().item():.4f}")
    print(f"Image max:         {images.max().item():.4f}")
    print(f"Training labels:   {labels.tolist()}")

    assert images.ndim == 4
    assert images.shape[1:] == (3, 256, 256)
    assert labels.ndim == 1
    assert masks.shape == (images.shape[0], 1, 256, 256)
    assert torch.isfinite(images).all()
    assert set(labels.tolist()).issubset({0})

    # Verify the test set contains normal and anomalous labels.
    test_labels = {
        record["label"]
        for record in test_loader.dataset.records
    }
    assert test_labels.issubset({0, 1})
    assert 0 in test_labels, "Expected normal test images."
    assert 1 in test_labels, "Expected anomalous test images."

    # Save a visualization of the first training batch.
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    count = min(8, images.shape[0])
    columns = 4
    rows = (count + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3 * rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for index in range(count):
        image = images[index].cpu() * std + mean
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()

        ax = axes.flat[index]
        ax.imshow(image)
        ax.set_title(
            f"Label: {labels[index].item()}\n"
            f"Mask pixels: {int(masks[index].sum().item())}"
        )
        ax.axis("off")

    output_path = Path("reports/data_pipeline_batch.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to: {output_path}")
    print("\nAll data pipeline checks passed.")


if __name__ == "__main__":
    main()
