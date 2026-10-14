
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from src.data import build_dataloaders
from src.patchcore import ResNetFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

CATEGORY = "bottle"
IMAGE_SIZE = 256
BATCH_SIZE = 4
PATCHES_PER_IMAGE = 64
MAX_MEMORY_BANK_SIZE = 5000
SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("PATCHCORE-STYLE MEMORY BANK BUILDER")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Category: {CATEGORY}")

    train_loader, _, _ = build_dataloaders(
        dataset_root=str(DATA_ROOT),
        category=CATEGORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_fraction=0.2,
        seed=SEED,
        num_workers=0,
        normalize=True,
    )

    extractor = ResNetFeatureExtractor(
        pretrained=True
    ).to(device)
    extractor.eval()

    sampled_features = []

    with torch.inference_mode():
        for batch in tqdm(
            train_loader,
            desc="Extracting normal features",
        ):
            images = batch["image"].to(device)
            features = extractor(images)

            # Shape: (B, C, H, W)
            batch_size, channels, height, width = (
                features.shape
            )

            # Shape: (B, H*W, C)
            features = features.permute(
                0, 2, 3, 1
            ).reshape(batch_size, height * width, channels)

            # Randomly sample spatial patches from each image.
            for image_features in features:
                count = min(
                    PATCHES_PER_IMAGE,
                    image_features.shape[0],
                )

                indices = torch.randperm(
                    image_features.shape[0],
                    device=image_features.device,
                )[:count]

                sampled_features.append(
                    image_features[indices].cpu()
                )

    if not sampled_features:
        raise RuntimeError("No normal features were extracted.")

    candidates = torch.cat(sampled_features, dim=0)

    # Limit the bank size using a reproducible random sample.
    if len(candidates) > MAX_MEMORY_BANK_SIZE:
        generator = torch.Generator().manual_seed(SEED)
        indices = torch.randperm(
            len(candidates),
            generator=generator,
        )[:MAX_MEMORY_BANK_SIZE]

        memory_bank = candidates[indices]
    else:
        memory_bank = candidates

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ARTIFACT_DIR / "patchcore_memory_bank.pth"

    torch.save(
        {
            "memory_bank": memory_bank.contiguous(),
            "category": CATEGORY,
            "image_size": IMAGE_SIZE,
            "backbone": "resnet18",
            "feature_channels": int(memory_bank.shape[1]),
            "patches_per_image": PATCHES_PER_IMAGE,
            "max_memory_bank_size": MAX_MEMORY_BANK_SIZE,
            "seed": SEED,
        },
        output_path,
    )

    print("\nMemory bank created successfully.")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Candidate patches: {len(candidates)}")
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
