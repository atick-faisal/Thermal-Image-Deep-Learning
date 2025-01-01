# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from losses import Losses, compute_unet_losses
from metrics import Metrics
from src.metrics import calculate_batch_metrics, calculate_batch_metrics_corenet


def train_unet(
        generator: nn.Module,
        train_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        optimizer_g: Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
        **kwargs
) -> Tuple[Losses, Metrics]:
    """
    Train the model for one epoch.

    Args:
        generator: Model to train
        train_loader: Training data loader
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        optimizer_g: Optimizer for training
        device: Device to run the model on
        epoch: Current epoch number
        args: Command-line arguments
    """
    generator.train()
    total_losses = Losses()
    total_metrics = Metrics()

    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        input_images, target_images = input_images.to(device), target_images.to(device)

        optimizer_g.zero_grad()
        output = generator(input_images)

        total_loss, losses = compute_unet_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        total_loss.backward()
        optimizer_g.step()

        metrics = calculate_batch_metrics(
            output.detach().cpu().numpy(),
            target_images.detach().cpu().numpy()
        )

        # Update running losses
        total_losses.generator_mse_loss += losses.generator_mse_loss
        total_losses.generator_psnr_loss += losses.generator_psnr_loss
        total_losses.total_generator_loss += total_loss

        # Update running metrics
        total_metrics.ssim += metrics.ssim
        total_metrics.psnr += metrics.psnr
        total_metrics.mutual_info += metrics.mutual_info
        total_metrics.mae += metrics.mae
        total_metrics.temp_mae += metrics.temp_mae

        pbar.set_postfix(metrics.__dict__)

    # Compute averages
    avg_losses = Losses(
        generator_mse_loss=total_losses.generator_mse_loss / len(train_loader),
        generator_psnr_loss=total_losses.generator_psnr_loss / len(train_loader),
        total_generator_loss=total_losses.total_generator_loss / len(train_loader)
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
        generator: nn.Module,
        val_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
        **kwargs
) -> Tuple[Losses, Metrics, Tensor]:
    """
    Validate the model on the validation set.

    Args:
        generator: Model to validate
        val_loader: Validation data loader
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        device: Device to run the model on
        epoch: Current epoch number
        args: Command-line arguments
    """
    generator.eval()

    total_losses = Losses()
    total_metrics = Metrics()

    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')

    vis_images: List[Tensor] = []

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        input_images, target_images = input_images.to(device), target_images.to(device)
        output = generator(input_images)

        _, losses = compute_unet_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        metrics = calculate_batch_metrics(
            output.cpu().numpy(),
            target_images.cpu().numpy()
        )

        # Update running losses
        total_losses.generator_mse_loss += losses.generator_mse_loss
        total_losses.generator_psnr_loss += losses.generator_psnr_loss
        total_losses.total_generator_loss += losses.total_generator_loss

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
        generator_mse_loss=total_losses.generator_mse_loss / len(val_loader),
        generator_psnr_loss=total_losses.generator_psnr_loss / len(val_loader),
        total_generator_loss=total_losses.total_generator_loss / len(val_loader)
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


def train_corenet(
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        l1_criterion: nn.Module,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
        **kwargs
) -> Tuple[Losses, Metrics]:
    """
    Train the model for one epoch.

    Args:
        generator: Model to train
        discriminator: Critic model for adversarial training
        train_loader: Training data loader
        l1_criterion: L1 loss function
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        optimizer_g: Optimizer for the regressor
        optimizer_d: Optimizer for the critic
        device: Device to run the model on
        epoch: Current epoch number
        args: Command-line arguments
    """
    generator.train()
    discriminator.train()

    total_losses = Losses()
    total_metrics = Metrics()

    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        ones = torch.ones(input_images.size(0), args.corenet_out_channels, 1, device=device)

        input_images, target_images = input_images.to(device), target_images.to(device)

        # ... Train Generator ...
        optimizer_g.zero_grad()
        g_out = generator(input_images)
        d_out_generated = discriminator(g_out)

        loss_g_l1 = l1_criterion(d_out_generated, ones)
        loss_g_psnr = psnr_criterion(g_out, target_images)
        loss_g_mse = mse_criterion(g_out, target_images)

        total_loss_g = args.l1_weight * loss_g_l1 + args.psnr_weight * loss_g_psnr \
                       + args.mse_weight * loss_g_mse

        total_loss_g.backward()
        optimizer_g.step()

        # ... Train Discriminator ...
        optimizer_d.zero_grad()
        d_out_target = discriminator(target_images)
        loss_d_target_l1 = l1_criterion(d_out_target, ones)

        d_out_generated = discriminator(g_out.detach())

        with torch.no_grad():
            # actual_psnr = get_batched_psnr(g_out, target_images)
            # actual_psnr_normalized = (actual_psnr / args.max_psnr)
            actual_metrics = calculate_batch_metrics_corenet(g_out, target_images)

        loss_d_predicted_l1 = l1_criterion(d_out_generated, actual_metrics)

        total_loss_d = loss_d_target_l1 + loss_d_predicted_l1

        total_loss_d.backward()
        optimizer_d.step()

        metrics = calculate_batch_metrics(
            g_out.detach().cpu().numpy(),
            target_images.detach().cpu().numpy()
        )

        # Update running losses
        total_losses.generator_psnr_loss += loss_g_psnr.item()
        total_losses.generator_mse_loss += loss_g_mse.item()
        total_losses.total_generator_loss += total_loss_g.item()
        total_losses.total_discriminator_loss += total_loss_d.item()

        # Update running metrics
        total_metrics.ssim += metrics.ssim
        total_metrics.psnr += metrics.psnr
        total_metrics.mutual_info += metrics.mutual_info
        total_metrics.mae += metrics.mae
        total_metrics.temp_mae += metrics.temp_mae

        pbar.set_postfix(metrics.__dict__)

    # Compute averages
    avg_losses = Losses(
        generator_psnr_loss=total_losses.generator_psnr_loss / len(train_loader),
        generator_mse_loss=total_losses.generator_mse_loss / len(train_loader),
        total_generator_loss=total_losses.total_generator_loss / len(train_loader),
        total_discriminator_loss=total_losses.total_discriminator_loss / len(train_loader)
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
def validate_corenet(
        generator: nn.Module,
        discriminator: nn.Module,
        val_loader: DataLoader,
        l1_criterion: nn.Module,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
        **kwargs
) -> Tuple[Losses, Metrics, Tensor]:
    """
    Validate the model on the validation set.

    Args:
        generator: Model to validate
        discriminator: Critic model for adversarial training
        val_loader: Validation data loader
        l1_criterion: L1 loss function
        mse_criterion: MSE loss function
        psnr_criterion: PSNR loss function
        device: Device to run the model on
        epoch: Current epoch number
        args: Command-line arguments
    """
    generator.eval()
    discriminator.eval()

    total_losses = Losses()
    total_metrics = Metrics()

    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')

    vis_images: List[Tensor] = []

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        ones = torch.ones(input_images.size(0), args.corenet_out_channels, 1, device=device)

        input_images, target_images = input_images.to(device), target_images.to(device)

        g_out = generator(input_images)
        d_out_generated = discriminator(g_out)

        loss_g_l1 = l1_criterion(d_out_generated, ones)
        loss_g_psnr = psnr_criterion(g_out, target_images)
        loss_g_mse = mse_criterion(g_out, target_images)

        total_loss_g = args.l1_weight * loss_g_l1 + args.psnr_weight * loss_g_psnr \
                       + args.mse_weight * loss_g_mse

        d_out_target = discriminator(target_images)
        loss_d_target_l1 = l1_criterion(d_out_target, ones)

        d_out_generated = discriminator(g_out)

        # actual_psnr = get_batched_psnr(g_out, target_images)
        # actual_psnr_normalized = (actual_psnr / args.max_psnr)

        actual_metrics = calculate_batch_metrics_corenet(g_out, target_images)

        loss_d_predicted_l1 = l1_criterion(d_out_generated, actual_metrics)

        total_loss_d = loss_d_target_l1 + loss_d_predicted_l1

        metrics = calculate_batch_metrics(
            g_out.cpu().numpy(),
            target_images.cpu().numpy()
        )

        # Update running losses
        total_losses.generator_psnr_loss += loss_g_psnr.item()
        total_losses.generator_mse_loss += loss_g_mse.item()
        total_losses.total_generator_loss += total_loss_g.item()
        total_losses.total_discriminator_loss += total_loss_d.item()

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
                g_out[:args.num_samples].cpu(),
            ]

        pbar.set_postfix(metrics.__dict__)

    # Compute averages
    avg_losses = Losses(
        generator_psnr_loss=total_losses.generator_psnr_loss / len(val_loader),
        generator_mse_loss=total_losses.generator_mse_loss / len(val_loader),
        total_generator_loss=total_losses.total_generator_loss / len(val_loader),
        total_discriminator_loss=total_losses.total_discriminator_loss / len(val_loader)
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


def train_pix2pix_turbo(
        model: nn.Module,
        train_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
        prompt: Optional[str] = None
) -> Tuple[Losses, Metrics]:
    """Train Pix2PixTurbo for one epoch."""
    model.set_train()
    total_losses = Losses()
    total_metrics = Metrics()

    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(
            control_image=input_images,
            prompt=prompt or "Convert this image",
            deterministic=True
        )

        # Calculate losses
        total_loss, losses = compute_unet_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Calculate metrics
        metrics = calculate_batch_metrics(
            output.detach().cpu().numpy(),
            target_images.detach().cpu().numpy()
        )

        # Update running totals
        total_losses.generator_mse_loss += losses.generator_mse_loss
        total_losses.generator_psnr_loss += losses.generator_psnr_loss
        total_losses.total_generator_loss += total_loss.item()

        total_metrics.ssim += metrics.ssim
        total_metrics.psnr += metrics.psnr
        total_metrics.mutual_info += metrics.mutual_info
        total_metrics.mae += metrics.mae
        total_metrics.temp_mae += metrics.temp_mae

        pbar.set_postfix(metrics.__dict__)

    # Calculate averages
    n_batches = len(train_loader)
    avg_losses = Losses(
        generator_mse_loss=total_losses.generator_mse_loss / n_batches,
        generator_psnr_loss=total_losses.generator_psnr_loss / n_batches,
        total_generator_loss=total_losses.total_generator_loss / n_batches
    )

    avg_metrics = Metrics(
        ssim=total_metrics.ssim / n_batches,
        psnr=total_metrics.psnr / n_batches,
        mutual_info=total_metrics.mutual_info / n_batches,
        mae=total_metrics.mae / n_batches,
        temp_mae=total_metrics.temp_mae / n_batches
    )

    return avg_losses, avg_metrics
