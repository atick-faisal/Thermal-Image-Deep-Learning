# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from losses import Losses, compute_losses
from metrics import Metrics
from src.metrics import calculate_batch_metrics


def train_unet(
        regressor: nn.Module,
        critic: nn.Module | None,
        train_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace
) -> Tuple[Losses, Metrics]:
    """
    Train the model for one epoch.

    Args:
        regressor: Model to train
        critic: Critic model for adversarial training
        train_loader: Training data loader
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        optimizer: Optimizer for training
        device: Device to run the model on
        epoch: Current epoch number
        args: Command-line arguments
    """
    regressor.train()
    total_losses = Losses(mse_loss=0.0, psnr_loss=0.0, total_loss=0.0)
    total_metrics = Metrics(ssim=0.0, psnr=0.0, mutual_info=0.0, mae=0.0, temp_mae=0.0)

    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        input_images, target_images = input_images.to(device), target_images.to(device)

        optimizer.zero_grad()
        output = regressor(input_images)

        total_loss, losses = compute_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        total_loss.backward()
        optimizer.step()

        metrics = calculate_batch_metrics(
            output.detach().cpu().numpy(),
            target_images.detach().cpu().numpy()
        )

        # Update running losses
        total_losses.mse_loss += losses.mse_loss
        total_losses.psnr_loss += losses.psnr_loss
        total_losses.total_loss += total_loss

        # Update running metrics
        total_metrics.ssim += metrics.ssim
        total_metrics.psnr += metrics.psnr
        total_metrics.mutual_info += metrics.mutual_info
        total_metrics.mae += metrics.mae
        total_metrics.temp_mae += metrics.temp_mae

        pbar.set_postfix(metrics.__dict__)

    # Compute averages
    avg_losses = Losses(
        mse_loss=total_losses.mse_loss / len(train_loader),
        psnr_loss=total_losses.psnr_loss / len(train_loader),
        total_loss=total_losses.total_loss / len(train_loader)
    )

    avg_metrics = Metrics(
        ssim=total_metrics.ssim / len(train_loader),
        psnr=total_metrics.psnr / len(train_loader),
        mutual_info=total_metrics.mutual_info / len(train_loader),
        mae=total_metrics.mae / len(train_loader),
        temp_mae=total_metrics.temp_mae / len(train_loader)
    )

    return avg_losses, avg_metrics


@torch.no_grad()
def validate_unet(
        regressor: nn.Module,
        critic: nn.Module | None,
        val_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace
) -> Tuple[Losses, Metrics, Tensor]:
    """
    Validate the model on the validation set.

    Args:
        regressor: Model to validate
        critic: Critic model for adversarial training
        val_loader: Validation data loader
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        device: Device to run the model on
        epoch: Current epoch number
        args: Command-line arguments
    """
    regressor.eval()

    total_losses = Losses(mse_loss=0.0, psnr_loss=0.0, total_loss=0.0)
    total_metrics = Metrics(ssim=0.0, psnr=0.0, mutual_info=0.0, mae=0.0, temp_mae=0.0)

    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')

    vis_images: List[Tensor] = []

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        input_images, target_images = input_images.to(device), target_images.to(device)
        output = regressor(input_images)

        _, losses = compute_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        metrics = calculate_batch_metrics(
            output.cpu().numpy(),
            target_images.cpu().numpy()
        )

        # Update running losses
        total_losses.mse_loss += losses.mse_loss
        total_losses.psnr_loss += losses.psnr_loss
        total_losses.total_loss += losses.total_loss

        # Update running metrics
        total_metrics.ssim += metrics.ssim
        total_metrics.psnr += metrics.psnr
        total_metrics.mutual_info += metrics.mutual_info
        total_metrics.mae += metrics.mae
        total_metrics.temp_mae += metrics.temp_mae

        if batch_idx == 0:
            vis_images = [
                input_images[:args.num_samples].cpu(),
                target_images[:args.num_samples].cpu(),
                output[:args.num_samples].cpu(),
            ]

        pbar.set_postfix(metrics.__dict__)

    # Compute averages
    avg_losses = Losses(
        mse_loss=total_losses.mse_loss / len(val_loader),
        psnr_loss=total_losses.psnr_loss / len(val_loader),
        total_loss=total_losses.total_loss / len(val_loader)
    )

    avg_metrics = Metrics(
        ssim=total_metrics.ssim / len(val_loader),
        psnr=total_metrics.psnr / len(val_loader),
        mutual_info=total_metrics.mutual_info / len(val_loader),
        mae=total_metrics.mae / len(val_loader),
        temp_mae=total_metrics.temp_mae / len(val_loader)
    )

    vis_grid = make_grid(torch.cat(vis_images, dim=0), nrow=args.num_samples)

    return avg_losses, avg_metrics, vis_grid
