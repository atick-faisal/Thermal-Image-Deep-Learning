import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from src.metrics import (
    Metrics,
    normalize_image_range,
    validate_inputs,
    calculate_metrics
)


@pytest.fixture
def identical_images():
    """Fixture for identical images in [0, 1] range"""
    return np.ones((256, 256, 3)) * 0.5, np.ones((256, 256, 3)) * 0.5


@pytest.fixture
def different_range_images():
    """Fixture for same images in different ranges"""
    img1 = np.ones((256, 256, 3)) * 0.5  # [0, 1] range
    img2 = np.ones((256, 256, 3)) * 0.0  # [-1, 1] range after conversion
    return img1, img2


@pytest.fixture
def random_images():
    """Fixture for random images"""
    rng = np.random.default_rng(42)  # Set seed for reproducibility
    img1 = rng.random((256, 256, 3))
    img2 = rng.random((256, 256, 3))
    return img1, img2


def test_normalize_image_range():
    """Test image range normalization"""
    # Test [-1, 1] range image
    img = np.array([-1, 0, 1])
    normalized = normalize_image_range(img)
    assert_almost_equal(normalized, np.array([0, 0.5, 1]))

    # Test [0, 1] range image
    img = np.array([0, 0.5, 1])
    normalized = normalize_image_range(img)
    assert_array_equal(normalized, img)


def test_validate_inputs():
    """Test input validation"""
    # Test valid inputs
    img1 = np.ones((256, 256, 3))
    img2 = np.ones((256, 256, 3))
    val1, val2 = validate_inputs(img1, img2)
    assert_array_equal(val1, img1)
    assert_array_equal(val2, img2)

    # Test different shapes
    img2 = np.ones((224, 224, 3))
    with pytest.raises(ValueError, match="Images must have the same dimensions"):
        validate_inputs(img1, img2)

    # Test invalid dimensions
    img1 = np.ones((256, 256))
    img2 = np.ones((256, 256))
    with pytest.raises(ValueError, match="Images must be RGB with shape"):
        validate_inputs(img1, img2)


def test_identical_images(identical_images):
    """Test metrics with identical images"""
    img1, img2 = identical_images
    metrics = calculate_metrics(img1, img2)

    assert isinstance(metrics, Metrics)
    assert_almost_equal(metrics.ssim, 1.0, decimal=4)
    assert metrics.psnr == float('inf')
    assert_almost_equal(metrics.mae, 0.0, decimal=4)


def test_different_images(different_range_images):
    """Test metrics with different images"""
    img1, img2 = different_range_images
    metrics = calculate_metrics(img1, img2)

    assert isinstance(metrics, Metrics)
    assert metrics.ssim < 1.0
    assert metrics.psnr < float('inf')
    assert metrics.mae > 0


def test_random_images(random_images):
    """Test metrics with random images"""
    img1, img2 = random_images
    metrics = calculate_metrics(img1, img2)

    assert isinstance(metrics, Metrics)
    assert 0 <= metrics.ssim <= 1
    assert metrics.psnr > 0
    assert 0 <= metrics.mae <= 1
    assert metrics.mutual_info >= 0


def test_edge_cases():
    """Test edge cases"""
    # Zero images
    zero_img = np.zeros((256, 256, 3))
    metrics = calculate_metrics(zero_img, zero_img)

    assert isinstance(metrics, Metrics)
    assert_almost_equal(metrics.ssim, 1.0, decimal=4)
    assert metrics.psnr == float('inf')
    assert_almost_equal(metrics.mae, 0.0, decimal=4)
