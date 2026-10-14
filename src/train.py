
from pathlib import Path
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data import build_dataloaders
from src.model import ConvAutoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

CATEGORY = "bottle"
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruction_error(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
) -> torch.Tensor:
    """
    Return a per-pixel anomaly map.

    Input shapes:
        images:         (B, 3, H, W)
        reconstructions:(B, 3, H, W)

    Output shape:
        (B, H, W)
    """
    squared_error = (images - reconstructions).pow(2)
    return squared_error.mean(dim=1)


def image_scores(error_maps: torch.Tensor) -> torch.Tensor:
    """
    Average the largest 1% of pixel errors in each image.

    This can be more sensitive to small defects than averaging
    the error over every pixel.
    """
    batch_size = error_maps.shape[0]
    flattened = error_maps.reshape(batch_size, -1)

    k = max(1, int(np.ceil(flattened.shape[1] * 0.01)))

    return flattened.topk(k, dim=1).values.mean(dim=1)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    description="Processing",
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_images = 0

    progress = tqdm(loader, desc=description, leave=False)

    for batch in progress:
        images = batch["image"].to(
            device, non_blocking=True
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructions = model(images)
            loss = criterion(reconstructions, images)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        progress.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(total_images, 1)


@torch.inference_mode()
def collect_predictions(model, loader, device):
    model.eval()

    labels_all = []
    scores_all = []
    paths_all = []
    examples = []

    for batch in tqdm(loader, desc="Evaluating test set"):
        images = batch["image"].to(device)
        reconstructions = model(images)

        error_maps = reconstruction_error(
            images, reconstructions
        )
        scores = image_scores(error_maps)

        labels_all.extend(batch["label"].cpu().numpy().tolist())
        scores_all.extend(scores.cpu().numpy().tolist())
        paths_all.extend(batch["path"])

        # Retain a few examples for the visualization.
        if len(examples) < 8:
            remaining = 8 - len(examples)
            count = min(remaining, images.size(0))

            for i in range(count):
                examples.append(
                    {
                        "image": images[i].cpu(),
                        "reconstruction": reconstructions[i].cpu(),
                        "error_map": error_maps[i].cpu(),
                        "label": int(batch["label"][i].item()),
                        "category": batch["category"][i],
                        "path": batch["path"][i],
                    }
                )

    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(scores_all, dtype=np.float64),
        paths_all,
        examples,
    )


def save_loss_plot(train_losses, val_losses):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training")
    plt.plot(epochs, val_losses, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Autoencoder Reconstruction Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = REPORT_DIR / "training_loss.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Loss plot saved: {output_path}")


def save_prediction_grid(examples):
    if not examples:
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    count = len(examples)
    fig, axes = plt.subplots(
        count, 3, figsize=(11, 3.4 * count), squeeze=False
    )

    for row, example in enumerate(examples):
        original = (
            example["image"].permute(1, 2, 0).numpy()
        )
        reconstructed = (
            example["reconstruction"].permute(1, 2, 0).numpy()
        )
        error_map = example["error_map"].numpy()

        axes[row, 0].imshow(np.clip(original, 0, 1))
        axes[row, 0].set_title(
            f"Input: {example['category']}"
        )

        axes[row, 1].imshow(np.clip(reconstructed, 0, 1))
        axes[row, 1].set_title("Reconstruction")

        axes[row, 2].imshow(error_map, cmap="inferno")
        axes[row, 2].set_title("Pixel reconstruction error")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()

    output_path = REPORT_DIR / "test_predictions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Prediction visualization saved: {output_path}")


def main():
    set_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("INDUSTRIAL ANOMALY DETECTION - CNN AUTOENCODER")
    print("=" * 65)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device:      {device}")
    print(f"Category:    {CATEGORY}")
    print(f"Image size:  {IMAGE_SIZE}")
    print(f"Epochs:      {EPOCHS}")
    print(f"Batch size:  {BATCH_SIZE}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {DATA_ROOT}"
        )

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

    print(f"\nTraining images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images:       {len(test_loader.dataset)}")

    model = ConvAutoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    best_val_loss = float("inf")
    best_checkpoint = ARTIFACT_DIR / "autoencoder_best.pth"

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            description=f"Epoch {epoch}/{EPOCHS} - train",
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            description=f"Epoch {epoch}/{EPOCHS} - validation",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        # Select the model using normal-only validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "category": CATEGORY,
                    "image_size": IMAGE_SIZE,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "seed": SEED,
                },
                best_checkpoint,
            )

            print(f"  Saved best checkpoint: {best_checkpoint}")

    save_loss_plot(train_losses, val_losses)

    # Reload the best checkpoint before the final test evaluation.
    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Derive an initial threshold from normal validation images only.
    # This controls a normal-image false-alarm target approximately;
    # it is not an anomaly-optimal threshold.
    _, val_scores, _, _ = collect_predictions(
        model, val_loader, device
    )
    threshold = float(np.quantile(val_scores, 0.99))

    # Evaluate the official test set once, after training.
    y_true, y_scores, test_paths, examples = collect_predictions(
        model, test_loader, device
    )

    y_pred = (y_scores > threshold).astype(np.int64)

    metrics = {
        "category": CATEGORY,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_mse": float(checkpoint["best_val_loss"]),
        "validation_normal_score_p99_threshold": threshold,
        "test_image_count": int(len(y_true)),
        "test_anomaly_count": int(y_true.sum()),
        "test_normal_count": int((y_true == 0).sum()),
        "image_roc_auc": None,
        "threshold_accuracy": None,
        "threshold_precision": None,
        "threshold_recall": None,
        "threshold_f1": None,
        "threshold_note": (
            "Threshold derived from the 99th percentile of normal "
            "validation image scores. It is a baseline, not an "
            "anomaly-optimal threshold."
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["image_roc_auc"] = float(
            roc_auc_score(y_true, y_scores)
        )

    metrics["threshold_accuracy"] = float(
        accuracy_score(y_true, y_pred)
    )
    metrics["threshold_precision"] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    metrics["threshold_recall"] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    metrics["threshold_f1"] = float(
        f1_score(y_true, y_pred, zero_division=0)
    )

    metrics_path = REPORT_DIR / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    save_prediction_grid(examples)

    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Best model:    {best_checkpoint}")
    print("\nTraining and evaluation finished.")


if __name__ == "__main__":
    main()
