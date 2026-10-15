
import torch

from src.model import ConvAutoencoder


def test_autoencoder_output_shape():
    """The autoencoder should preserve the input image dimensions."""
    model = ConvAutoencoder()
    model.eval()

    sample = torch.rand(1, 3, 256, 256)

    with torch.inference_mode():
        reconstruction = model(sample)

    assert reconstruction.shape == sample.shape


def test_autoencoder_output_range():
    """The decoder should return pixel values in the [0, 1] range."""
    model = ConvAutoencoder()
    model.eval()

    sample = torch.rand(1, 3, 256, 256)

    with torch.inference_mode():
        reconstruction = model(sample)

    assert torch.isfinite(reconstruction).all()
    assert reconstruction.min().item() >= 0.0
    assert reconstruction.max().item() <= 1.0
