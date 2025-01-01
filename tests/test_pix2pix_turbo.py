import torch
from src.pix2pixturbo import Pix2PixConfig, Pix2PixTurbo


def test_model_dimensions():
    # Initialize model and move to device
    config = Pix2PixConfig()
    model = Pix2PixTurbo(config)
    model.set_eval()

    # Create test input
    batch_size = 1
    test_image = torch.randn(batch_size, 3, 256, 256).to(model.device)
    test_prompt = "test prompt"

    # Run forward pass
    with torch.no_grad():
        output = model(
            control_image=test_image,
            prompt=test_prompt,
            deterministic=True
        )

    # Assert expected dimensions
    assert output.shape == test_image.shape, f"Output shape {output.shape} != expected shape {test_image.shape}"
