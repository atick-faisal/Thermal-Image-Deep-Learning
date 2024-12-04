from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score


@dataclass
class Metrics:
    """Data class to store image comparison metrics"""
    ssim: float
    psnr: float
    mutual_info: float
    mae: float
    temp_mae: float

    def __str__(self) -> str:
        """Pretty string representation of metrics"""
        return (
            f"SSIM: {self.ssim:.4f}\n"
            f"PSNR: {self.psnr:.4f} dB\n"
            f"Mutual Information: {self.mutual_info:.4f}\n"
            f"MAE: {self.mae:.4f}"
            f"Temperature MAE: {self.temp_mae:.4f}"
        )


def normalize_image_range(image: NDArray) -> NDArray:
    """
    Detect and normalize image range to [0, 1].

    Args:
        image: Input image array

    Returns:
        Normalized image array in [0, 1] range
    """
    # Check if image is in [-1, 1] range
    if np.any(image < 0):
        image = (image + 1) / 2
    return image


def validate_inputs(img1: NDArray, img2: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Validate and prepare input images.

    Args:
        img1: First input image
        img2: Second input image

    Returns:
        Tuple of normalized images

    Raises:
        ValueError: If images have different shapes or invalid dimensions
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    if len(img1.shape) != 3 or img1.shape[2] != 3:
        raise ValueError("Images must be RGB with shape (H, W, 3)")

    return normalize_image_range(img1), normalize_image_range(img2)


def calculate_ssim(img1: NDArray,
                   img2: NDArray,
                   channel_axis: int = 2) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two RGB images
    using scikit-image implementation.

    Args:
        img1: First image array
        img2: Second image array
        channel_axis: Axis index for color channels

    Returns:
        SSIM value between 0 and 1
    """
    img1, img2 = validate_inputs(img1, img2)

    return float(ssim(
        img1,
        img2,
        channel_axis=channel_axis,
        data_range=1.0,
        gaussian_weights=True,
        use_sample_covariance=False
    ))


def calculate_psnr(img1: NDArray,
                   img2: NDArray,
                   max_value: Optional[float] = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two RGB images.

    Args:
        img1: First image array
        img2: Second image array
        max_value: Maximum possible pixel value (default: None, auto-detected)

    Returns:
        PSNR value in decibels
    """
    img1, img2 = validate_inputs(img1, img2)

    if max_value is None:
        max_value = 1.0

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return float(20 * np.log10(max_value) - 10 * np.log10(mse))


def calculate_mutual_info(img1: NDArray,
                          img2: NDArray,
                          bins: int = 256) -> float:
    """
    Calculate Mutual Information between two RGB images
    using scikit-learn implementation.

    Args:
        img1: First image array
        img2: Second image array
        bins: Number of histogram bins

    Returns:
        Average mutual information across RGB channels
    """
    img1, img2 = validate_inputs(img1, img2)

    mi_sum = 0
    for channel in range(3):
        # Convert to histogram bins
        hist1, _ = np.histogram(img1[..., channel], bins=bins, range=(0, 1))
        hist2, _ = np.histogram(img2[..., channel], bins=bins, range=(0, 1))

        # Calculate mutual information for this channel
        mi_sum += mutual_info_score(hist1, hist2)

    return float(mi_sum / 3)


def calculate_mae(img1: NDArray, img2: NDArray) -> float:
    """
    Calculate Mean Absolute Error (MAE) between two RGB images.

    Args:
        img1: First image array
        img2: Second image array

    Returns:
        MAE value between 0 and 1
    """
    img1, img2 = validate_inputs(img1, img2)
    return float(np.mean(np.abs(img1 - img2)))


def calculate_temp_mae(img1: NDArray, img2: NDArray) -> float:
    """
    Calculate Mean Absolute Error (MAE) between two RGB images.

    Args:
        img1: First image array
        img2: Second image array

    Returns:
        Temperature MAE value between 0 and 1
    """
    img1, img2 = validate_inputs(img1, img2)

    # Convert to degrees Celsius
    img1 = img1 * (40 - 25) + 25
    img2 = img2 * (40 - 25) + 25

    return float(np.mean(np.abs(img1 - img2)))


def calculate_metrics(img1: NDArray, img2: NDArray, win_size: int = 7) -> Metrics:
    """
    Calculate all image comparison metrics between two RGB images.

    Args:
        img1: First image array
        img2: Second image array
        win_size: Window size for SSIM calculation

    Returns:
        ImageMetrics object containing all calculated metrics
    """
    # Validate and normalize inputs once
    img1, img2 = validate_inputs(img1, img2)

    return Metrics(
        ssim=calculate_ssim(img1, img2),
        psnr=calculate_psnr(img1, img2),
        mutual_info=calculate_mutual_info(img1, img2),
        mae=calculate_mae(img1, img2),
        temp_mae=calculate_temp_mae(img1, img2)
    )


def calculate_batch_metrics(img1_batch: NDArray, img2_batch: NDArray) -> Metrics:
    """
    Calculate mean image comparison metrics between two batches of RGB images.

    Args:
        img1_batch: Input image arrays
        img2_batch: Target image arrays

    Returns:
        ImageMetrics object containing all calculated metrics
    """
    batch_size = img1_batch.shape[0]
    metrics: List[Metrics] = []

    # Calculate metrics for each image pair in the batch
    for i in range(batch_size):
        img1 = img1_batch[i]
        img2 = img2_batch[i]

        if img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        # Calculate metrics for this pair
        metric = calculate_metrics(img1, img2)
        metrics.append(metric)

    # Calculate mean of all metrics
    return Metrics(
        ssim=float(np.mean([m.ssim for m in metrics])),
        psnr=float(np.mean([m.psnr for m in metrics])),
        mutual_info=float(np.mean([m.mutual_info for m in metrics])),
        mae=float(np.mean([m.mae for m in metrics])),
        temp_mae=float(np.mean([m.temp_mae for m in metrics]))
    )
