import torch

from src.model import ConvAutoencoder, reconstruction_scores, residual_map


def test_autoencoder_preserves_image_shape():
    model = ConvAutoencoder(latent_channels=32)
    inputs = torch.rand(2, 3, 128, 128)
    outputs = model(inputs)
    assert outputs.shape == inputs.shape


def test_scores_and_residual_shapes():
    inputs = torch.zeros(2, 3, 128, 128)
    reconstructions = torch.ones_like(inputs)

    scores = reconstruction_scores(inputs, reconstructions)
    residuals = residual_map(inputs, reconstructions)

    assert scores.shape == (2,)
    assert residuals.shape == (2, 128, 128)
    assert torch.allclose(scores, torch.ones_like(scores))
