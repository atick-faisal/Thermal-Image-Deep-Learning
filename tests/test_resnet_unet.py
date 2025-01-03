import torch
import pytest
from src.unet import ResNetUNet  # Assuming model.py contains the ResNetUNet class

@pytest.fixture
def model():
    return ResNetUNet(n_classes=3, pretrained=False)

def test_single_image(model):
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 3, 256, 256)
    assert not torch.isnan(output).any()

def test_batch_processing(model):
    x = torch.randn(4, 3, 256, 256)
    output = model(x)
    assert output.shape == (4, 3, 256, 256)
    assert not torch.isnan(output).any()

def test_output_range(model):
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert torch.all(output >= -1) and torch.all(output <= 1)  # Due to tanh activation