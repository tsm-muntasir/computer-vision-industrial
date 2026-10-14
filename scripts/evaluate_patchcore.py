
from pathlib import Path
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.data import build_dataloaders
from src.patchcore import (
    PatchCoreStyleDetector,
    ResNetFeatureExtractor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
BANK_PATH = PROJECT_ROOT / "artifacts" / "patchcore_memory_bank.pth"
REPORT_DIR = PROJECT_ROOT / "reports"

CATEGORY = "bottle"
IMAGE_SIZE = 256
BATCH_SIZE = 4
SEED = 42
THRESHOLD_QUANTILE = 0.995


@torch.inference_mode()
def collect_results(detector, loader, device, keep_examples=False):
    labels = []
    image_scores = []
    pixel_errors = []
    pixel_masks = []
    examples = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"]

        anomaly_maps, scores = detector.predict(images)

        labels.extend(batch["label"].numpy().tolist())
        image_scores.extend(scores.cpu().numpy().tolist())

        pixel_errors.append(
            anomaly_maps.cpu().numpy().reshape(-1)
        )
        pixel_masks.append(
            masks[:, 0].numpy().reshape(-1).astype(np.uint8)
        )

        if keep_examples:
            for i in range(images.size(0)):
                if len(examples) >= 8:
                    break

                examples.append(
                    {
                        "image": images[i].cpu().numpy().transpose(
                            1, 2, 0
                        ),
                        "mask": masks[i, 0].numpy(),
                        "error_map": anomaly_maps[i].cpu().numpy(),
                        "label": int(batch["label"][i].item()),
                        "category": batch["category"][i],
                    }
                )

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "image_scores": np.asarray(image_scores, dtype=np.float64),
        "pixel_errors": np.concatenate(pixel_errors),
        "pixel_masks": np.concatenate(pixel_masks),
        "examples": examples,
    }


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def calculate_metrics(results, image_threshold, pixel_threshold):
    labels = results["labels"]
    scores = results["image_scores"]
    masks = results["pixel_masks"]
    errors = results["pixel_errors"]

    image_predictions = (scores > image_threshold).astype(np.uint8)
    pixel_predictions = (errors > pixel_threshold).astype(np.uint8)

    image_metrics = {
        "image_count": int(len(labels)),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int(labels.sum()),
        "roc_auc": safe_auc(labels, scores),
        "average_precision": (
            float(average_precision_score(labels, scores))
            if len(np.unique(labels)) == 2
            else None
        ),
        "precision_at_validation_threshold": float(
            precision_score(
                labels, image_predictions, zero_division=0
            )
        ),
        "recall_at_validation_threshold": float(
            recall_score(
                labels, image_predictions, zero_division=0
            )
        ),
        "f1_at_validation_threshold": float(
            f1_score(
                labels, image_predictions, zero_division=0
            )
        ),
    }

    true_positive = int(
        np.logical_and(masks == 1, pixel_predictions == 1).sum()
    )
    false_positive = int(
        np.logical_and(masks == 0, pixel_predictions == 1).sum()
    )
    false_negative = int(
        np.logical_and(masks == 1, pixel_predictions == 0).sum()
    )

    union = true_positive + false_positive + false_negative

    pixel_metrics = {
        "pixel_count": int(len(masks)),
        "roc_auc": safe_auc(masks, errors),
        "average_precision": (
            float(average_precision_score(masks, errors))
            if len(np.unique(masks)) == 2
            else None
        ),
        "precision_at_validation_threshold": float(
            precision_score(
                masks, pixel_predictions, zero_division=0
            )
        ),
        "recall_at_validation_threshold": float(
            recall_score(
                masks, pixel_predictions, zero_division=0
            )
        ),
        "f1_at_validation_threshold": float(
            f1_score(
                masks, pixel_predictions, zero_division=0
            )
        ),
        "iou_at_validation_threshold": (
            float(true_positive / union) if union else 0.0
        ),
    }

    return image_metrics, pixel_metrics


def save_examples(examples, pixel_threshold):
    if not examples:
        return

    fig, axes = plt.subplots(
        len(examples),
        5,
        figsize=(16, 3.2 * len(examples)),
        squeeze=False,
    )

    for row, item in enumerate(examples):
        original = np.clip(item["image"], 0, 1)
        mask = item["mask"]
        error_map = item["error_map"]
        predicted_mask = error_map > pixel_threshold

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Input: {item['category']}")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")

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
        axes[row, 4].set_title("Overlay")

        for col in range(5):
            axes[row, col].axis("off")

    plt.tight_layout()

    output_path = REPORT_DIR / "patchcore_localization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved: {output_path}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not BANK_PATH.exists():
        raise FileNotFoundError(
            f"Memory bank not found: {BANK_PATH}\n"
            "Run `python -m scripts.train_patchcore` first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, val_loader, test_loader = build_dataloaders(
        dataset_root=str(DATA_ROOT),
        category=CATEGORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_fraction=0.2,
        seed=SEED,
        num_workers=0,
        normalize=True,
    )

    bank_data = torch.load(
        BANK_PATH,
        map_location="cpu",
    )

    extractor = ResNetFeatureExtractor(
        pretrained=True
    ).to(device)
    extractor.eval()

    detector = PatchCoreStyleDetector(
        feature_extractor=extractor,
        memory_bank=bank_data["memory_bank"],
    )

    print(f"Device: {device}")
    print(f"Memory bank shape: {tuple(bank_data['memory_bank'].shape)}")

    # Validation consists only of normal images.
    val_results = collect_results(
        detector, val_loader, device
    )

    image_threshold = float(
        np.quantile(
            val_results["image_scores"],
            THRESHOLD_QUANTILE,
        )
    )

    pixel_threshold = float(
        np.quantile(
            val_results["pixel_errors"],
            THRESHOLD_QUANTILE,
        )
    )

    print(f"Image threshold: {image_threshold:.8f}")
    print(f"Pixel threshold: {pixel_threshold:.8f}")

    # Evaluate the official test split.
    test_results = collect_results(
        detector, test_loader, device, keep_examples=True
    )

    image_metrics, pixel_metrics = calculate_metrics(
        test_results,
        image_threshold,
        pixel_threshold,
    )

    output = {
        "method": "PatchCore-style nearest-neighbor detector",
        "category": CATEGORY,
        "backbone": "ResNet18 ImageNet pretrained",
        "memory_bank_size": int(
            bank_data["memory_bank"].shape[0]
        ),
        "threshold_quantile_on_normal_validation": (
            THRESHOLD_QUANTILE
        ),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_level": image_metrics,
        "pixel_level": pixel_metrics,
        "note": (
            "Thresholds are derived from normal validation data. "
            "ROC-AUC and average precision use official test labels "
            "only for final evaluation."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "patchcore_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    save_examples(
        test_results["examples"],
        pixel_threshold,
    )

    print("\nIMAGE-LEVEL METRICS")
    for key, value in image_metrics.items():
        print(f"{key}: {value}")

    print("\nPIXEL-LEVEL METRICS")
    for key, value in pixel_metrics.items():
        print(f"{key}: {value}")

    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
