
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    """
    Frozen ImageNet-pretrained ResNet18 feature extractor.

    Input:
        (B, 3, 256, 256), ImageNet-normalized images

    Output:
        (B, 384, 32, 32), combined local feature maps
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT if pretrained else None
        )
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # We use a frozen feature extractor, not a classifier.
        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features2 = self.layer2(x)
        features3 = self.layer3(features2)

        # Align the spatial resolution of both feature maps.
        features3 = F.interpolate(
            features3,
            size=features2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        features = torch.cat(
            [features2, features3],
            dim=1,
        )

        # Normalize each local feature vector.
        features = F.normalize(
            features, p=2, dim=1, eps=1e-6
        )

        return features


class PatchCoreStyleDetector:
    """
    Nearest-neighbor anomaly detector using a normal feature bank.

    The bank is created from normal training images only.
    """

    def __init__(self, feature_extractor, memory_bank):
        self.feature_extractor = feature_extractor
        self.memory_bank = memory_bank

    @torch.inference_mode()
    def predict(self, images, bank_chunk_size=1024):
        """
        Returns:
            anomaly_maps: (B, H, W)
            image_scores: (B,)
        """
        features = self.feature_extractor(images)

        batch_size, channels, height, width = features.shape

        # (B, C, H, W) -> (B*H*W, C)
        patch_features = (
            features.permute(0, 2, 3, 1)
            .reshape(-1, channels)
        )

        bank = self.memory_bank.to(
            device=patch_features.device,
            dtype=patch_features.dtype,
        )

        # Calculate minimum Euclidean distance for every patch.
        minimum_distances = []

        for start in range(
            0, len(patch_features), 256
        ):
            query = patch_features[start:start + 256]
            best_distances = torch.full(
                (len(query),),
                float("inf"),
                device=query.device,
            )

            # Chunk the bank to limit peak memory use.
            for bank_start in range(
                0, len(bank), bank_chunk_size
            ):
                bank_chunk = bank[
                    bank_start:bank_start + bank_chunk_size
                ]

                distances = torch.cdist(
                    query, bank_chunk, p=2
                )

                chunk_min = distances.min(dim=1).values
                best_distances = torch.minimum(
                    best_distances, chunk_min
                )

            minimum_distances.append(best_distances)

        patch_scores = torch.cat(minimum_distances)

        # Restore each image's patch scores to a spatial map.
        anomaly_maps = patch_scores.reshape(
            batch_size, 1, height, width
        )

        # Upsample the feature map to image resolution.
        anomaly_maps = F.interpolate(
            anomaly_maps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Light smoothing can reduce isolated noisy pixels.
        anomaly_maps = F.avg_pool2d(
            anomaly_maps.unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2,
        ).squeeze(1)

        # Score each image using its highest-scoring 1% of pixels.
        flattened = anomaly_maps.flatten(start_dim=1)
        k = max(
            1,
            int(flattened.shape[1] * 0.01),
        )

        image_scores = flattened.topk(
            k, dim=1
        ).values.mean(dim=1)

        return anomaly_maps, image_scores
