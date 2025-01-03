import pytest
import torch
from src.unet import DenseNetUNet


@pytest.fixture
def model():
    return DenseNetUNet(in_channels=3, pretrained=False)


def test_dimensions():
    model = DenseNetUNet(in_channels=3, pretrained=False)
    model.eval()

    sizes = [(1, 3, 256, 256), (4, 3, 256, 256)]

    for size in sizes:
        x = torch.randn(size)
        output = model(x)
        assert output.shape == size
        assert not torch.isnan(output).any()


def test_output_range():
    model = DenseNetUNet(in_channels=3, pretrained=False)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    output = model(x)

    assert torch.all(output >= -1) and torch.all(output <= 1)


def test_pretrained_weights():
    model = DenseNetUNet(pretrained=True)
    model_no_pretrain = DenseNetUNet(pretrained=False)

    # Check if weights are different
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                  model_no_pretrain.named_parameters()):
        if "encoder" in n1:
            assert not torch.allclose(p1, p2)


def test_gradient_flow():
    model = DenseNetUNet(pretrained=False)
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    output = model(x)
    loss = output.mean()
    loss.backward()

    # Check if gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None