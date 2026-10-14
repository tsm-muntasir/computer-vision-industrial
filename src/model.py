
import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for industrial anomaly detection.

    Expected input:  (batch_size, 3, 256, 256)
    Expected output: (batch_size, 3, 256, 256)
    Pixel values:    approximately [0, 1]
    """

    def __init__(self):
        super().__init__()

        # Encoder: progressively reduce spatial dimensions.
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 256, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # Decoder: progressively reconstruct the image.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction


if __name__ == "__main__":
    model = ConvAutoencoder()
    dummy_input = torch.rand(2, 3, 256, 256)
    output = model(dummy_input)

    print("Input shape: ", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters:  ", sum(p.numel() for p in model.parameters()))

    assert output.shape == dummy_input.shape
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0

    print("Model shape test passed.")
