import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import create_dataloaders
from losses import PSNRLoss
from src.utils import get_regressor, get_critic, get_trainer_and_validator

current_dir: str = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description='UNet Regressor Training')

    parser.add_argument('--model', type=str, default='attention_unet', required=True,
                        choices=['unet', 'attention_unet', 'self_unet'], help='regressor name')

    # dataset
    parser.add_argument('--data-root', type=str, default='../data/dataset/rgb2ir',
                        help='root directory for data')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--l1-lambda', type=float, default=1e-5, help='L1 regularization lambda')
    parser.add_argument('--l2-lambda', type=float, default=1e-4, help='L2 regularization lambda')
    parser.add_argument('--init-features', type=int, default=64,
                        help='initial number of features in UNet')

    # Loss weights
    parser.add_argument('--mse-weight', type=float, default=1.0, help='weight for MSE loss')
    parser.add_argument('--psnr-weight', type=float, default=0.1, help='weight for PSNR loss')

    # [Rest of the arguments remain the same]
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='save checkpoint every N epochs')
    parser.add_argument('--log-interval', type=int, default=10, help='log metrics every N batches')
    parser.add_argument('--vis-interval', type=int, default=5,
                        help='visualize examples every N epochs')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='number of examples to visualize')
    parser.add_argument('--wandb-dir', type=str, default='../wandb', help='wandb directory')
    parser.add_argument('--wandb-project', type=str, default='unet-regressor',
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

    regressor = get_regressor(args).to(device)
    critic = get_critic(args)

    if critic is not None:
        critic = critic.to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        regressor.load_state_dict(checkpoint['model_state_dict'], strict=True)

    wandb.watch(regressor)

    optimizer = torch.optim.Adam(regressor.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    mse_criterion = nn.MSELoss()
    psnr_criterion = PSNRLoss()

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # normalize=isinstance(regressor, SelfUNetRegressor)
    )

    best_val_psnr: float = float(0)

    trainer, validator = get_trainer_and_validator(args)

    for epoch in range(args.epochs):
        train_losses, train_metrics = trainer(
            regressor=regressor,
            critic=critic,
            train_loader=train_loader,
            mse_criterion=mse_criterion,
            psnr_criterion=psnr_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args
        )
        val_losses, val_metrics, val_images = validator(
            regressor=regressor,
            critic=critic,
            val_loader=val_loader,
            mse_criterion=mse_criterion,
            psnr_criterion=psnr_criterion,
            device=device,
            epoch=epoch,
            args=args
        )

        wandb.log({
            'epoch': epoch,
            'train_mse_loss': train_losses.mse_loss,
            'train_psnr_loss': train_losses.psnr_loss,
            'train_total_loss': train_losses.total_loss,
            'val_mse_loss': val_losses.mse_loss,
            'val_psnr_loss': val_losses.psnr_loss,
            'val_total_loss': val_losses.total_loss,
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

        # Save best regressor based on total validation loss
        if val_metrics.psnr > best_val_psnr:
            best_val_psnr = val_metrics.psnr
            checkpoint_path = os.path.join(current_dir, args.checkpoint_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(current_dir, args.checkpoint_dir,
                                           f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

        scheduler.step()

    wandb.finish()


if __name__ == '__main__':
    main()
