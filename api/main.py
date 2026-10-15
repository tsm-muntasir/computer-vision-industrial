
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.model import ConvAutoencoder
from src.patchcore import ResNetFeatureExtractor


# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_TYPE = os.getenv("MODEL_TYPE", "autoencoder").strip().lower()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUTOENCODER_PATH = PROJECT_ROOT / "artifacts" / "autoencoder_best.pth"
PATCHCORE_PATH = PROJECT_ROOT / "artifacts" / "patchcore_memory_bank.pth"

AUTOENCODER_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "localization_metrics.json"
)
PATCHCORE_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "patchcore_metrics.json"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# 2. Image preprocessing
# ---------------------------------------------------------

def build_transform(model_type: str):
    operations = [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
    ]

    if model_type == "patchcore":
        operations.append(
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    return transforms.Compose(operations)


# ---------------------------------------------------------
# 3. Read the model's calibrated evaluation threshold
# ---------------------------------------------------------

def find_numeric_value(
    data: Any,
    candidate_names: set[str],
) -> float | None:
    """Find a numeric field by name in a nested JSON object."""

    if isinstance(data, dict):
        # Prefer exact matching keys at the current level.
        for key, value in data.items():
            if key.lower() in candidate_names:
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    result = float(value)
                    if np.isfinite(result):
                        return result

        # Search nested objects.
        for value in data.values():
            result = find_numeric_value(value, candidate_names)
            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_numeric_value(item, candidate_names)
            if result is not None:
                return result

    return None


def load_thresholds(metrics_path: Path) -> tuple[float, float]:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {metrics_path}. "
            "Run the corresponding evaluation script first."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    image_threshold = find_numeric_value(
        metrics,
        {
            "image_score_threshold",
            "image_threshold",
            "validation_normal_score_p99_threshold",
            "threshold",
        },
    )

    pixel_threshold = find_numeric_value(
        metrics,
        {
            "pixel_error_threshold",
            "pixel_threshold",
        },
    )

    if image_threshold is None:
        raise ValueError(
            f"No image anomaly threshold found in {metrics_path}. "
            "Inspect the JSON structure and update the candidate keys."
        )

    # Pixel threshold is optional for image-level classification.
    if pixel_threshold is None:
        pixel_threshold = float("inf")

    return image_threshold, pixel_threshold


# ---------------------------------------------------------
# 4. Inference service
# ---------------------------------------------------------

class AnomalyInferenceService:
    def __init__(self, model_type: str):
        if model_type not in {"autoencoder", "patchcore"}:
            raise ValueError(
                "MODEL_TYPE must be 'autoencoder' or 'patchcore'."
            )

        self.model_type = model_type
        self.device = DEVICE
        self.transform = build_transform(model_type)

        self.image_threshold: float
        self.pixel_threshold: float

        if model_type == "autoencoder":
            self._load_autoencoder()
            metrics_path = AUTOENCODER_METRICS_PATH
        else:
            self._load_patchcore()
            metrics_path = PATCHCORE_METRICS_PATH

        (
            self.image_threshold,
            self.pixel_threshold,
        ) = load_thresholds(metrics_path)

    def _load_autoencoder(self):
        if not AUTOENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Autoencoder checkpoint missing: {AUTOENCODER_PATH}. "
                "Train the model before starting the API."
            )

        self.autoencoder = ConvAutoencoder().to(self.device)

        checkpoint = torch.load(
            AUTOENCODER_PATH,
            map_location=self.device,
        )

        # Support either a plain state_dict or a checkpoint dictionary.
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint),
            )
        else:
            state_dict = checkpoint

        # Also support checkpoints saved with DataParallel.
        if isinstance(state_dict, dict) and state_dict:
            if all(key.startswith("module.") for key in state_dict):
                state_dict = {
                    key.removeprefix("module."): value
                    for key, value in state_dict.items()
                }

        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.eval()

    def _load_patchcore(self):
        if not PATCHCORE_PATH.exists():
            raise FileNotFoundError(
                f"PatchCore memory bank missing: {PATCHCORE_PATH}. "
                "Build it using scripts.train_patchcore first."
            )

        # Uses the same pretrained feature extractor as training.
        self.feature_extractor = ResNetFeatureExtractor(
            pretrained=True
        ).to(self.device)
        self.feature_extractor.eval()

        checkpoint = torch.load(
            PATCHCORE_PATH,
            map_location="cpu",
        )

        if isinstance(checkpoint, torch.Tensor):
            memory_bank = checkpoint
        elif isinstance(checkpoint, dict):
            memory_bank = None

            for key in (
                "memory_bank",
                "bank",
                "features",
                "patch_features",
            ):
                if key in checkpoint and isinstance(
                    checkpoint[key], torch.Tensor
                ):
                    memory_bank = checkpoint[key]
                    break

            if memory_bank is None:
                raise ValueError(
                    "Could not find a tensor named 'memory_bank', "
                    "'bank', 'features', or 'patch_features' in the "
                    "PatchCore checkpoint. Inspect the save format in "
                    "scripts/train_patchcore.py and adjust this loader."
                )
        else:
            raise ValueError("Unsupported PatchCore checkpoint format.")

        if memory_bank.ndim != 2:
            raise ValueError(
                "Expected memory bank shape (number_of_patches, "
                "feature_dimensions), but received "
                f"{tuple(memory_bank.shape)}."
            )

        self.memory_bank = memory_bank.float().to(self.device)

    # -----------------------------------------------------
    # 5. Calculate an anomaly map and image-level score
    # -----------------------------------------------------

    @torch.inference_mode()
    def predict_tensor(self, image_tensor: torch.Tensor):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        if self.model_type == "autoencoder":
            reconstruction = self.autoencoder(image_tensor)

            # Per-pixel mean squared reconstruction error.
            anomaly_map = (
                image_tensor - reconstruction
            ).pow(2).mean(dim=1, keepdim=True)

        else:
            feature_map = self.feature_extractor(image_tensor)

            # Feature map: [1, channels, height, width].
            if feature_map.ndim != 4:
                raise RuntimeError(
                    "Feature extractor must return a 4D tensor."
                )

            _, channels, height, width = feature_map.shape

            patches = (
                feature_map[0]
                .permute(1, 2, 0)
                .reshape(-1, channels)
            )

            # Compute nearest-neighbour distances in chunks to
            # limit peak memory usage.
            patch_distances = []

            for start in range(0, len(patches), 256):
                patch_chunk = patches[start:start + 256]
                distances = torch.cdist(
                    patch_chunk,
                    self.memory_bank,
                    p=2,
                )
                nearest_distances = distances.min(dim=1).values
                patch_distances.append(nearest_distances)

            distances = torch.cat(patch_distances)

            anomaly_map = distances.reshape(
                1, 1, height, width
            )

            anomaly_map = F.interpolate(
                anomaly_map,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Smooth small variations, matching the localization
        # evaluation approach used in this project.
        anomaly_map = F.avg_pool2d(
            anomaly_map,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        # Use the mean of the highest-scoring 1% of pixels.
        flattened = anomaly_map.flatten(start_dim=1)
        top_k = max(1, int(flattened.shape[1] * 0.01))
        image_score = flattened.topk(top_k, dim=1).values.mean()

        return (
            float(image_score.item()),
            anomaly_map[0, 0].detach().cpu().numpy(),
        )

    # -----------------------------------------------------
    # 6. Create a heatmap image and encode it as Base64
    # -----------------------------------------------------

    @staticmethod
    def encode_heatmap(anomaly_map: np.ndarray) -> str:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colormaps

        finite_map = np.nan_to_num(
            anomaly_map,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        low = float(finite_map.min())
        high = float(finite_map.max())

        if high > low:
            normalized = (finite_map - low) / (high - low)
        else:
            normalized = np.zeros_like(finite_map)

        colored = colormaps["inferno"](normalized)[..., :3]
        colored = (colored * 255).astype(np.uint8)

        heatmap_image = Image.fromarray(colored, mode="RGB")

        buffer = io.BytesIO()
        heatmap_image.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict_image(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        image_tensor = self.transform(image)

        score, anomaly_map = self.predict_tensor(image_tensor)

        return {
            "model": self.model_type,
            "anomaly_score": score,
            "image_threshold": self.image_threshold,
            "is_anomaly": score > self.image_threshold,
            "image_size": {
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            },
            "heatmap_format": "png",
            "heatmap_base64": self.encode_heatmap(anomaly_map),
            "pixel_threshold": (
                self.pixel_threshold
                if np.isfinite(self.pixel_threshold)
                else None
            ),
            "note": (
                "Threshold-based decision; validate the threshold "
                "on representative production data."
            ),
        }


# ---------------------------------------------------------
# 7. FastAPI application and endpoints
# ---------------------------------------------------------

app = FastAPI(
    title="Industrial Anomaly Detection API",
    description=(
        "Detect unusual bottle images using a CNN autoencoder "
        "or a PatchCore-style feature detector."
    ),
    version="1.0.0",
)

service = AnomalyInferenceService(MODEL_TYPE)


@app.get("/")
def home():
    return {
        "message": "Industrial Anomaly Detection API is running.",
        "model": service.model_type,
        "device": str(service.device),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": service.model_type,
        "device": str(service.device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, WebP, or BMP image.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(content)) as uploaded_image:
            uploaded_image.load()
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        result = service.predict_image(image)
    except Exception as exc:
        # Do not expose internal stack traces to API clients.
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}. "
                   "Check the server logs.",
        ) from exc

    result["filename"] = Path(file.filename or "upload").name
    return result
