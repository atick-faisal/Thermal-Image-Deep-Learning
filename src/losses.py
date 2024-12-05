# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class Losses:
    generator_mse_loss: float = 0.0
    generator_psnr_loss: float = 0.0
    total_generator_loss: float = 0.0
    total_discriminator_loss: float = 0.0

    def __str__(self):
        return (
            f"G MSE Loss: {self.generator_mse_loss:.4f}\n"
            f"G PSNR Loss: {self.generator_psnr_loss:.4f}\n"
            f"G Total Loss: {self.total_generator_loss:.4f}"
            f"D Total Loss: {self.total_discriminator_loss:.4f}"
        )


class PSNRLoss(nn.Module):
    def __init__(self, max_val: float = 1.0) -> None:
        """
        Args:
            max_val: Maximum value of the signal (1.0 for normalized images)
        """
        super().__init__()
        self.max_val = torch.tensor(max_val)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the negative PSNR (we minimize this value)
        Args:
            output: Predicted images
            target: Ground truth images
        Returns:
            Negative PSNR value (scalar tensor)
        """
        # Ensure max_val is on the same device as the inputs
        self.max_val = self.max_val.to(output.device)

        # [-1, 1] -> [0, 1]
        output = output * 0.5 + 0.5
        target = target * 0.5 + 0.5

        # Calculate MSE per image
        mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])

        # Handle cases where mse is 0
        zero_mask = (mse == 0)
        mse = torch.where(zero_mask, torch.tensor(1e-8).to(mse.device), mse)

        # Calculate PSNR
        psnr = 20 * torch.log10(self.max_val) - 10 * torch.log10(mse)

        # Set PSNR to a large value where mse was 0
        psnr = torch.where(zero_mask, torch.tensor(100.0).to(psnr.device), psnr)

        return -torch.mean(psnr)  # Negative because we want to maximize PSNR


def compute_unet_losses(
        output: Tensor,
        target: Tensor,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        mse_weight: float,
        psnr_weight: float
) -> Tuple[Tensor, Losses]:
    """
    Compute the combined loss from MSE and PSNR losses.

    Args:
        output: Predicted images
        target: Ground truth images
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        mse_weight: Weight for MSE loss
        psnr_weight: Weight for PSNR loss
    """
    mse_loss = mse_criterion(output, target)
    psnr_loss = psnr_criterion(output, target)

    # Combine losses
    total_loss = mse_weight * mse_loss + psnr_weight * psnr_loss

    losses = Losses(
        generator_mse_loss=mse_loss.item(),
        generator_psnr_loss=psnr_loss.item(),
        total_generator_loss=total_loss.item()
    )

    return total_loss, losses


def compute_corenet_losses(
        generated: Tensor,
        target: Tensor,
        ones: Tensor,
        max_psnr: Tensor,
        psnr_generated: Tensor,
        psnr_target: Tensor,
        l1_criterion: nn.Module,
        psnr_criterion: nn.Module,
        l1_weight: float,
        psnr_weight: float
) -> Tuple[Tensor, Tensor, Losses]:
    """
    Compute the combined loss from MSE and PSNR losses.

    Args:
        generated: Generated images
        target: Ground truth images
        ones: Tensor of ones for the discriminator loss
        max_psnr: Maximum PSNR value
        psnr_generated: Predicted PSNR for generated images
        psnr_target: Predicted PSNR for target images
        l1_criterion: L1 loss function
        psnr_criterion: PSNR loss function
        l1_weight: Weight for L1 loss
        psnr_weight: Weight for PSNR loss
    """


    l1_loss_g = l1_criterion(psnr_generated, ones)
    psnr_loss_g = psnr_criterion(generated, target)

    total_generator_loss = l1_weight * l1_loss_g + psnr_weight * psnr_loss_g

    l1_loss_actual = l1_criterion(psnr_target, ones)
    l1_loss_predicted = l1_criterion(psnr_generated, psnr_target / max_psnr)


    losses = Losses()

    return total_generator_loss, total_discriminator_loss, losses
