
from pathlib import Path
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.data import build_dataloaders
from src.model import ConvAutoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "autoencoder_best.pth"
REPORT_DIR = PROJECT_ROOT / "reports"

CATEGORY = "bottle"
IMAGE_SIZE = 256
BATCH_SIZE = 8
SEED = 42

# Initial threshold from normal validation pixels.
NORMAL_PIXEL_QUANTILE = 0.995


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_anomaly_map(images, reconstructions):
    """Return one reconstruction-error map per image."""
    error = (images - reconstructions).pow(2).mean(dim=1)

    # Light spatial smoothing reduces isolated noisy pixels.
    error = F.avg_pool2d(
        error.unsqueeze(1),
        kernel_size=5,
        stride=1,
        padding=2,
    ).squeeze(1)

    return error


def image_anomaly_scores(error_maps):
    """Average the highest-error 1% of pixels in each image."""
    flattened = error_maps.flatten(start_dim=1)
    k = max(1, int(np.ceil(flattened.shape[1] * 0.01)))

    return flattened.topk(k, dim=1).values.mean(dim=1)


@torch.inference_mode()
def collect_maps(model, loader, device, save_examples=False):
    model.eval()

    all_labels = []
    all_scores = []
    all_pixel_errors = []
    all_pixel_masks = []
    examples = []

    for batch in tqdm(loader, desc="Calculating anomaly maps"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        reconstructions = model(images)
        error_maps = make_anomaly_map(images, reconstructions)
        scores = image_anomaly_scores(error_maps)

        all_labels.extend(batch["label"].cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())

        # Flatten the maps and ground-truth masks for pixel metrics.
        all_pixel_errors.append(
            error_maps.cpu().numpy().reshape(-1)
        )
        all_pixel_masks.append(
            masks[:, 0].cpu().numpy().reshape(-1).astype(np.uint8)
        )

        if save_examples:
            for i in range(images.size(0)):
                label = int(batch["label"][i].item())

                # Save up to 3 normal and 5 anomalous examples.
                normal_count = sum(
                    item["label"] == 0 for item in examples
                )
                anomaly_count = sum(
                    item["label"] == 1 for item in examples
                )

                should_keep = (
                    (label == 0 and normal_count < 3)
                    or (label == 1 and anomaly_count < 5)
                )

                if should_keep:
                    examples.append(
                        {
                            "image": images[i].cpu().numpy().transpose(
                                1, 2, 0
                            ),
                            "mask": masks[i, 0].cpu().numpy(),
                            "error_map": error_maps[i].cpu().numpy(),
                            "label": label,
                            "category": batch["category"][i],
                        }
                    )

    return {
        "labels": np.asarray(all_labels, dtype=np.int64),
        "scores": np.asarray(all_scores, dtype=np.float64),
        "pixel_errors": np.concatenate(all_pixel_errors),
        "pixel_masks": np.concatenate(all_pixel_masks),
        "examples": examples,
    }


def calculate_image_metrics(labels, scores, threshold):
    predictions = (scores > threshold).astype(np.uint8)

    results = {
        "image_count": int(len(labels)),
        "anomaly_count": int(labels.sum()),
        "normal_count": int((labels == 0).sum()),
        "image_score_threshold": float(threshold),
        "image_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "image_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "image_f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
    }

    if len(np.unique(labels)) == 2:
        results["image_roc_auc"] = float(
            roc_auc_score(labels, scores)
        )
        results["image_average_precision"] = float(
            average_precision_score(labels, scores)
        )
    else:
        results["image_roc_auc"] = None
        results["image_average_precision"] = None

    return results


def calculate_pixel_metrics(pixel_masks, pixel_errors, threshold):
    predictions = (pixel_errors > threshold).astype(np.uint8)

    results = {
        "pixel_count": int(len(pixel_masks)),
        "positive_pixel_fraction": float(pixel_masks.mean()),
        "pixel_error_threshold": float(threshold),
        "pixel_precision": float(
            precision_score(
                pixel_masks, predictions, zero_division=0
            )
        ),
        "pixel_recall": float(
            recall_score(
                pixel_masks, predictions, zero_division=0
            )
        ),
        "pixel_f1": float(
            f1_score(
                pixel_masks, predictions, zero_division=0
            )
        ),
    }

    true_positive = int(
        np.logical_and(pixel_masks == 1, predictions == 1).sum()
    )
    false_positive = int(
        np.logical_and(pixel_masks == 0, predictions == 1).sum()
    )
    false_negative = int(
        np.logical_and(pixel_masks == 1, predictions == 0).sum()
    )

    denominator = true_positive + false_positive + false_negative

    results["pixel_iou"] = (
        float(true_positive / denominator)
        if denominator > 0
        else 0.0
    )

    if len(np.unique(pixel_masks)) == 2:
        results["pixel_roc_auc"] = float(
            roc_auc_score(pixel_masks, pixel_errors)
        )
        results["pixel_average_precision"] = float(
            average_precision_score(pixel_masks, pixel_errors)
        )
    else:
        results["pixel_roc_auc"] = None
        results["pixel_average_precision"] = None

    return results


def save_localization_grid(examples, threshold):
    if not examples:
        print("No examples available for visualization.")
        return

    fig, axes = plt.subplots(
        len(examples),
        5,
        figsize=(17, 3.5 * len(examples)),
        squeeze=False,
    )

    for row, item in enumerate(examples):
        original = np.clip(item["image"], 0, 1)
        gt_mask = item["mask"]
        error_map = item["error_map"]
        predicted_mask = error_map > threshold

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(
            f"Original: {item['category']}"
        )

        axes[row, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground-truth mask")

        axes[row, 2].imshow(error_map, cmap="inferno")
        axes[row, 2].set_title("Anomaly heatmap")

        axes[row, 3].imshow(
            predicted_mask, cmap="gray", vmin=0, vmax=1
        )
        axes[row, 3].set_title("Predicted mask")

        axes[row, 4].imshow(original)
        axes[row, 4].imshow(
            predicted_mask,
            cmap="Reds",
            alpha=0.4,
            vmin=0,
            vmax=1,
        )
        axes[row, 4].set_title("Predicted overlay")

        for col in range(5):
            axes[row, col].axis("off")

    plt.tight_layout()

    output_path = REPORT_DIR / "localization_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Localization grid saved: {output_path}")


def main():
    set_seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run `python -m src.train` first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Category: {CATEGORY}")

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=str(DATA_ROOT),
        category=CATEGORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_fraction=0.2,
        seed=SEED,
        num_workers=0,
        normalize=False,
    )

    # Training loader is unused during evaluation.
    del train_loader

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Estimate the pixel threshold from normal validation images only.
    val_results = collect_maps(model, val_loader, device)
    pixel_threshold = float(
        np.quantile(
            val_results["pixel_errors"],
            NORMAL_PIXEL_QUANTILE,
        )
    )

    # Use the same normal-only validation data to establish a baseline
    # image threshold. This is not optimized using anomaly labels.
    image_threshold = float(
        np.quantile(
            val_results["scores"],
            NORMAL_PIXEL_QUANTILE,
        )
    )

    print(f"\nPixel threshold: {pixel_threshold:.8f}")
    print(f"Image threshold: {image_threshold:.8f}")

    # Final evaluation against the official test set.
    test_results = collect_maps(
        model,
        test_loader,
        device,
        save_examples=True,
    )

    image_metrics = calculate_image_metrics(
        test_results["labels"],
        test_results["scores"],
        image_threshold,
    )

    pixel_metrics = calculate_pixel_metrics(
        test_results["pixel_masks"],
        test_results["pixel_errors"],
        pixel_threshold,
    )

    results = {
        "category": CATEGORY,
        "checkpoint": str(CHECKPOINT_PATH),
        "image_size": IMAGE_SIZE,
        "threshold_method": (
            "99.5th percentile of normal validation scores/errors; "
            "not optimized using test labels or masks"
        ),
        "image_level": image_metrics,
        "pixel_level": pixel_metrics,
    }

    output_json = REPORT_DIR / "localization_metrics.json"

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    save_localization_grid(
        test_results["examples"],
        pixel_threshold,
    )

    print("\n" + "=" * 60)
    print("IMAGE-LEVEL METRICS")
    print("=" * 60)

    for key, value in image_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("PIXEL-LEVEL METRICS")
    print("=" * 60)

    for key, value in pixel_metrics.items():
        print(f"{key}: {value}")

    print(f"\nMetrics saved to: {output_json}")


if __name__ == "__main__":
    main()
