# Copyright (c) 2024 Atick Faisal
# Licensed under the MIT License - see LICENSE file for details

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import create_dataloaders
from metrics import calculate_batch_metrics, Metrics
from utils import get_generator


def main() -> None:
    parser = argparse.ArgumentParser(description='Model Testing Script')

    # Model arguments
    parser.add_argument('--model', type=str, default='corenet',
                        choices=['unet', 'attention_unet', 'self_unet', 'corenet'],
                        help='model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint to load')

    # Dataset arguments
    parser.add_argument('--data-root', type=str, required=True,
                        help='root directory for test data')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='directory to save test results')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for testing')

    # Model configuration
    parser.add_argument('--init-features', type=int, default=64,
                        help='initial number of features in model')
    parser.add_argument('--corenet-out-channels', type=int, default=3,
                        help='corenet out channels')

    # Other arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    real_dir = output_dir / 'real'
    predicted_dir = output_dir / 'predicted'

    for dir_path in [output_dir, real_dir, predicted_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    generator = get_generator(args).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['model_state_dict'], strict=True)
    generator.eval()

    # Create data loader
    _, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True
    )

    # Initialize metrics
    total_metrics = Metrics()

    print("Starting evaluation...")

    with torch.no_grad():
        for idx, (input_images, target_images) in enumerate(tqdm(val_loader)):
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            # Generate predictions
            predicted_images = generator(input_images)

            # Calculate metrics
            metrics = calculate_batch_metrics(predicted_images.cpu().numpy(),
                                              target_images.cpu().numpy())

            # Update running metrics
            total_metrics.ssim += metrics.ssim
            total_metrics.psnr += metrics.psnr
            total_metrics.mutual_info += metrics.mutual_info
            total_metrics.mae += metrics.mae
            total_metrics.temp_mae += metrics.temp_mae

            # Save images
            for b in range(input_images.size(0)):
                image_name = f"image_{idx * args.batch_size + b}"

                # Save real image
                save_image(
                    target_images[b],
                    real_dir / f"{image_name}_real.png",
                    normalize=True
                )

                # Save predicted image
                save_image(
                    predicted_images[b],
                    predicted_dir / f"{image_name}_predicted.png",
                    normalize=True
                )

    # Calculate average metrics
    avg_metrics = Metrics(
        ssim=total_metrics.ssim / len(val_loader),
        psnr=total_metrics.psnr / len(val_loader),
        mutual_info=total_metrics.mutual_info / len(val_loader),
        mae=total_metrics.mae / len(val_loader),
        temp_mae=total_metrics.temp_mae / len(val_loader)
    )

    print(f"Average SSIM: {avg_metrics.ssim}")
    print(f"Average PSNR: {avg_metrics.psnr}")
    print(f"Average Mutual Information: {avg_metrics.mutual_info}")
    print(f"Average MAE: {avg_metrics.mae}")
    print(f"Average Temp MAE: {avg_metrics.temp_mae}")


if __name__ == '__main__':
    main()
