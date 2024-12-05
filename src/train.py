# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from dataloader import create_dataloaders
from losses import PSNRLoss
from src.utils import get_generator, get_discriminator, get_trainer_and_validator

current_dir: str = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description='UNet Regressor Training')

    parser.add_argument('--model', type=str, default='corenet',
                        choices=['unet', 'attention_unet', 'self_unet', 'corenet'],
                        help='regressor name')

    # dataset
    parser.add_argument('--data-root', type=str, default='../data/dataset/small',
                        help='root directory for data')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr-generator', type=float, default=0.0003,
                        help='generator learning rate')
    parser.add_argument('--lr-discriminator', type=float, default=0.0005,
                        help='discriminator learning rate')
    parser.add_argument('--l1-lambda', type=float, default=1e-5, help='L1 regularization lambda')
    parser.add_argument('--l2-lambda', type=float, default=1e-4, help='L2 regularization lambda')
    parser.add_argument('--init-features', type=int, default=64,
                        help='initial number of features in UNet')

    # Loss weights
    parser.add_argument('--mse-weight', type=float, default=1.0, help='weight for MSE loss')
    parser.add_argument('--psnr-weight', type=float, default=0.05, help='weight for PSNR loss')
    parser.add_argument('--l1-weight', type=float, default=1.0, help='weight for L1 loss')

    # max psnr
    parser.add_argument('--max-psnr', type=float, default=40.0, help='max psnr value')

    # [Rest of the arguments remain the same]
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='save checkpoint every N epochs')
    parser.add_argument('--log-interval', type=int, default=1, help='log metrics every N batches')
    parser.add_argument('--vis-interval', type=int, default=10,
                        help='visualize examples every N epochs')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='number of examples to visualize')
    parser.add_argument('--wandb-dir', type=str, default='../wandb', help='wandb directory')
    parser.add_argument('--wandb-project', type=str, default='rgb2ir',
                        help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--wandb-name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers for data loading')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to regressor checkpoint  to load')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(os.path.join(current_dir, args.checkpoint_dir), exist_ok=True)
    os.makedirs(os.path.join(current_dir, args.wandb_dir), exist_ok=True)

    wandb.init(
        dir=os.path.join(current_dir, args.wandb_dir),
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    generator = get_generator(args).to(device)
    discriminator = get_discriminator(args)

    if discriminator is not None:
        discriminator = discriminator.to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'], strict=True)

    wandb.watch(generator)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_generator)
    optimizer_d: Optimizer | None = None

    scheduler_g = CosineAnnealingLR(
        optimizer_g,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scheduler_d: LRScheduler | None = None

    if discriminator is not None:
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_discriminator)
        scheduler_d = CosineAnnealingLR(
            optimizer_d,
            T_max=args.epochs,
            eta_min=1e-6
        )

    mse_criterion = nn.MSELoss()
    psnr_criterion = PSNRLoss()
    l1_criterion = nn.L1Loss()

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True
    )

    best_val_psnr: float = float(0)

    trainer, validator = get_trainer_and_validator(args)

    for epoch in range(args.epochs):
        train_losses, train_metrics = trainer(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            mse_criterion=mse_criterion,
            l1_criterion=l1_criterion,
            psnr_criterion=psnr_criterion,

            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            args=args,
        )
        val_losses, val_metrics, val_images = validator(
            generator=generator,
            discriminator=discriminator,
            val_loader=val_loader,
            mse_criterion=mse_criterion,
            l1_criterion=l1_criterion,
            psnr_criterion=psnr_criterion,
            device=device,
            epoch=epoch,
            args=args
        )

        wandb.log({
            'epoch': epoch,
            'train_mse_loss': train_losses.generator_mse_loss,
            'train_psnr_loss': train_losses.generator_psnr_loss,
            'train_total_loss': train_losses.total_generator_loss,
            'val_mse_loss': val_losses.generator_mse_loss,
            'val_psnr_loss': val_losses.generator_psnr_loss,
            'val_total_loss': val_losses.total_generator_loss,
            'train_ssim': train_metrics.ssim,
            'train_psnr': train_metrics.psnr,
            'train_mutual_info': train_metrics.mutual_info,
            'train_mae': train_metrics.mae,
            'train_temp_mae': train_metrics.temp_mae,
            'val_ssim': val_metrics.ssim,
            'val_psnr': val_metrics.psnr,
            'val_mutual_info': val_metrics.mutual_info,
            'val_mae': val_metrics.mae,
            'val_temp_mae': val_metrics.temp_mae,
        })

        if epoch % args.vis_interval == 0:
            wandb.log({
                'val_images': wandb.Image(
                    val_images,
                    caption=f"Top: Input | Middle: Target | Bottom: Predicted (Epoch {epoch})"
                )
            })

        # Save best generator based on total validation loss
        if val_metrics.psnr > best_val_psnr:
            best_val_psnr = val_metrics.psnr
            checkpoint_path = os.path.join(current_dir, args.checkpoint_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(current_dir, args.checkpoint_dir,
                                           f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

        scheduler_g.step()

        if scheduler_d is not None:
            scheduler_d.step()

    wandb.finish()


if __name__ == '__main__':
    main()
