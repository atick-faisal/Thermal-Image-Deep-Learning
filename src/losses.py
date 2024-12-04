from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class Losses:
    mse_loss: float
    psnr_loss: float
    total_loss: float

    def __str__(self):
        return (
            f"MSE Loss: {self.mse_loss:.4f}\n"
            f"PSNR Loss: {self.psnr_loss:.4f}\n"
            f"Total Loss: {self.total_loss:.4f}"
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
        # output = output * 0.5 + 0.5
        # target = target * 0.5 + 0.5

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


def compute_losses(
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
        mse_loss=mse_loss.item(),
        psnr_loss=psnr_loss.item(),
        total_loss=total_loss.item()
    )

    return total_loss, losses
