# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

from dataclasses import dataclass
from typing import Literal
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from skimage.metrics import normalized_mutual_information, structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from torch import Tensor


@dataclass
class Metrics:
    """Data class to store image comparison metrics"""
    ssim: float = 0.0
    psnr: float = 0.0
    mutual_info: float = 0.0
    mae: float = 0.0
    temp_mae: float = 0.0

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


def calculate_mutual_info(
        img1: NDArray,
        img2: NDArray,
) -> float:
    """
    Calculate Mutual Information between two RGB images
    using scikit-learn implementation.

    Args:
        img1: First image array
        img2: Second image array

    Returns:
        Average mutual information across RGB channels
    """
    img1, img2 = validate_inputs(img1, img2)

    return normalized_mutual_information(img1, img2)


def calculate_normalized_mi(
        img1: NDArray,
        img2: NDArray,
        method: Literal['min', 'geometric', 'arithmetic', 'max'] = 'min'
) -> float:
    """
    Calculate Normalized Mutual Information between two RGB images
    with different normalization options.

    Args:
        img1: First image array
        img2: Second image array
        method: Normalization method:
               'min': MI / min(H(X), H(Y))
               'geometric': MI / sqrt(H(X) * H(Y))
               'arithmetic': MI / ((H(X) + H(Y)) / 2)
               'max': MI / max(H(X), H(Y))

    Returns:
        Normalized mutual information value between 0 and 1
    """
    img1, img2 = validate_inputs(img1, img2)

    def entropy(x: NDArray) -> float:
        """Calculate Shannon entropy."""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs))

    def mutual_info(x: NDArray, y: NDArray) -> float:
        """Calculate mutual information."""
        return mutual_info_score(x.ravel(), y.ravel())

    # Calculate MI and entropies
    mi = mutual_info(img1, img2)
    h1 = entropy(img1)
    h2 = entropy(img2)

    # Normalize based on selected method
    if method == 'min':
        nmi = mi / min(h1, h2)
    elif method == 'geometric':
        nmi = mi / np.sqrt(h1 * h2)
    elif method == 'arithmetic':
        nmi = mi / ((h1 + h2) / 2)
    elif method == 'max':
        nmi = mi / max(h1, h2)
    else:
        raise ValueError("Invalid normalization method")

    # Ensure value is between 0 and 1
    return float(np.clip(nmi, 0, 1))


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


def calculate_metrics(img1: NDArray, img2: NDArray) -> Metrics:
    """
    Calculate all image comparison metrics between two RGB images.

    Args:
        img1: First image array
        img2: Second image array

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


def get_batched_psnr(
        output: Tensor,
        target: Tensor,
):
    """
    Calculate the  PSNR
    Args:
        output: Predicted images
        target: Ground truth images
    Returns:
        Negative PSNR value (scalar tensor)
    """
    # Ensure max_val is on the same device as the inputs
    max_val: Tensor = torch.tensor(1.0).to(output.device)

    # [-1, 1] -> [0, 1]
    output = output * 0.5 + 0.5
    target = target * 0.5 + 0.5

    # Calculate MSE per image
    mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])

    # Handle cases where mse is 0
    zero_mask = (mse == 0)
    mse = torch.where(zero_mask, torch.tensor(1e-8).to(mse.device), mse)

    # Calculate PSNR
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)

    # Set PSNR to a large value where mse was 0
    psnr = torch.where(zero_mask, torch.tensor(100.0).to(psnr.device), psnr)

    return psnr.view(-1, 1, 1)


class BatchSimilarityMetrics(torch.nn.Module):
    """
    Module to calculate MI, SSIM, and MS-SSIM for batches of images in [-1, 1] range.
    Returns metrics in shape (batch_size, 3, 1) where the metrics are ordered as:
    [MI, SSIM, MS-SSIM]
    """

    def __init__(
            self,
            window_size: int = 11,
            num_bins: int = 256,
            mi_method: str = 'min'
    ):
        super().__init__()
        self.window_size = window_size
        self.num_bins = num_bins
        self.mi_method = mi_method

        # Initialize Gaussian window for SSIM
        gaussian_1d = self._gaussian_window(window_size)
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d).float()
        self.register_buffer(
            'gaussian_window',
            gaussian_2d.expand(3, 1, window_size, window_size)
        )

    @staticmethod
    def _gaussian_window(window_size: int, sigma: Tensor = torch.tensor(1.5)) -> torch.Tensor:
        """Generate 1D Gaussian window."""
        gauss = torch.Tensor([
            torch.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _calculate_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate normalized mutual information for a batch of images in [-1, 1] range."""
        batch_size = x.shape[0]
        mi_values = torch.zeros(batch_size, device=x.device)

        for i in range(batch_size):
            # Flatten and normalize to [0, 1] for histogram calculation
            x_flat = (x[i].flatten() + 1) / 2
            y_flat = (y[i].flatten() + 1) / 2

            # Joint histogram
            joint_hist = torch.histogramdd(
                torch.stack([x_flat.cpu(), y_flat.cpu()], dim=1),
                bins=self.num_bins,
                range=[0.0, 1.0, 0.0, 1.0]
            )[0]

            # Calculate probabilities
            joint_probs = joint_hist.to(x.device) / len(x_flat)
            p_x = joint_probs.sum(dim=1)
            p_y = joint_probs.sum(dim=0)

            # Calculate entropy
            h_x = -torch.sum(p_x[p_x > 0] * torch.log2(p_x[p_x > 0]))
            h_y = -torch.sum(p_y[p_y > 0] * torch.log2(p_y[p_y > 0]))

            # Calculate MI
            mi = torch.zeros(1, device=x.device)
            nonzero_mask = joint_probs > 0
            if nonzero_mask.any():
                joint_probs_nz = joint_probs[nonzero_mask]
                p_xy = joint_probs_nz * torch.log2(
                    joint_probs_nz /
                    (p_x.unsqueeze(1).expand_as(joint_probs)[nonzero_mask] *
                     p_y.unsqueeze(0).expand_as(joint_probs)[nonzero_mask])
                )
                mi = p_xy.sum()

            # Normalize MI based on method
            if self.mi_method == 'min':
                norm = torch.min(h_x, h_y)
            elif self.mi_method == 'geometric':
                norm = torch.sqrt(h_x * h_y)
            elif self.mi_method == 'arithmetic':
                norm = (h_x + h_y) / 2
            else:  # max
                norm = torch.max(h_x, h_y)

            mi_values[i] = mi / norm if norm > 0 else torch.tensor(0.)

        return torch.clip(mi_values, 0, 1)

    def _calculate_ssim(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            C1: float = 0.01 ** 2,
            C2: float = 0.03 ** 2
    ) -> torch.Tensor:
        """
        Calculate SSIM for a batch of images in [-1, 1] range.
        Note: C1 and C2 are adjusted for [-1, 1] range.
        """
        pad = self.window_size // 2

        # Calculate means
        mu1 = F.conv2d(x, self.gaussian_window, padding=pad, groups=x.shape[1])
        mu2 = F.conv2d(y, self.gaussian_window, padding=pad, groups=y.shape[1])

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = F.conv2d(x * x, self.gaussian_window, padding=pad, groups=x.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(y * y, self.gaussian_window, padding=pad, groups=y.shape[1]) - mu2_sq
        sigma12 = F.conv2d(x * y, self.gaussian_window, padding=pad, groups=x.shape[1]) - mu1_mu2

        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean([1, 2, 3])

    def _calculate_msssim(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            weights=None
    ) -> torch.Tensor:
        """Calculate MS-SSIM for a batch of images in [-1, 1] range."""
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        batch_size = x.shape[0]
        msssim_values = torch.ones(batch_size, device=x.device)

        for i, weight in enumerate(weights):
            ssim_val = self._calculate_ssim(x, y)
            msssim_values *= ssim_val ** weight

            if i < len(weights) - 1:
                x = F.avg_pool2d(x, kernel_size=2)
                y = F.avg_pool2d(y, kernel_size=2)

        return msssim_values

    @staticmethod
    def _calculate_psnr(
            x: Tensor,
            y: Tensor,
            max_psnr: float = 40.0
    ) -> Tensor:
        """
        Calculate the  PSNR
        Args:
            x: Predicted images
            y: Ground truth images
        Returns:
            Negative PSNR value (scalar tensor)
        """
        # Ensure max_val is on the same device as the inputs
        max_val: Tensor = torch.tensor(1.0).to(x.device)

        # [-1, 1] -> [0, 1]
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5

        # Calculate MSE per image
        mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])

        # Handle cases where mse is 0
        zero_mask = (mse == 0)
        mse = torch.where(zero_mask, torch.tensor(1e-8).to(mse.device), mse)

        # Calculate PSNR
        psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)

        # Set PSNR to a large value where mse was 0
        return torch.where(zero_mask, torch.tensor(100.0).to(psnr.device), psnr) / max_psnr

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate all metrics for batch of images in [-1, 1] range.

        Args:
            x: First batch of images [B, C, H, W] in range [-1, 1]
            y: Second batch of images [B, C, H, W] in range [-1, 1]

        Returns:
            Tensor of shape [B, 3, 1] containing [MI, SSIM, MS-SSIM] for each item
        """
        # Input validation
        if torch.min(x) < -1.0 or torch.max(x) > 1.0 or torch.min(y) < -1.0 or torch.max(y) > 1.0:
            raise ValueError("Input images must be in range [-1, 1]")

        # Calculate all metrics
        mi_vals = self._calculate_mi(x, y)
        ssim_vals = self._calculate_ssim(x, y)
        # msssim_vals = self._calculate_msssim(x, y)
        psnr_vals = self._calculate_psnr(x, y)

        # Stack and reshape to required format
        metrics = torch.stack([mi_vals, ssim_vals, psnr_vals], dim=1)
        return metrics.unsqueeze(-1)  # Shape: [B, 3, 1]


def calculate_batch_metrics_corenet(
        batch1: torch.Tensor,
        batch2: torch.Tensor,
        window_size: int = 11,
        num_bins: int = 256,
        mi_method: str = 'min'
) -> torch.Tensor:
    """
    Convenient function to calculate metrics for a batch of images in [-1, 1] range.

    Args:
        batch1: First batch of images [B, C, H, W] in range [-1, 1]
        batch2: Second batch of images [B, C, H, W] in range [-1, 1]
        window_size: Size of Gaussian window for SSIM
        num_bins: Number of bins for MI calculation
        mi_method: MI normalization method ('min', 'geometric', 'arithmetic', 'max')

    Returns:
        Tensor of shape [B, 3, 1] containing [MI, SSIM, MS-SSIM] for each image pair
    """
    metric_calculator = BatchSimilarityMetrics(
        window_size=window_size,
        num_bins=num_bins,
        mi_method=mi_method
    ).to(batch1.device)

    with torch.no_grad():
        metrics = metric_calculator(batch1, batch2)

    return metrics
