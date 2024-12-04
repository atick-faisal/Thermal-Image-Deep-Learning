import os
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose


class ImagePairDataset(Dataset):
    def __init__(
            self,
            root_dir: str | Path,
            split: str,
            transform: Callable[[Image.Image], Tensor]
    ) -> None:
        """
        Args:
            root_dir: Root directory containing train and val folders
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
        """
        self.root_dir: Path = Path(root_dir)
        self.split: str = split
        self.transform: Callable[[Image.Image], Tensor] = transform

        # Get paths for input and target directories
        self.input_dir: Path = self.root_dir / 'RGB' / split
        self.target_dir: Path = self.root_dir / 'IR' / split

        # Get list of image files
        self.image_files: List[str] = sorted([
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Verify that all input images have corresponding target images
        for img_file in self.image_files:
            assert (self.target_dir / img_file).exists(), \
                f"Target image not found for {img_file}"

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_name: str = self.image_files[idx]

        # Read input and target images
        input_path: Path = self.input_dir / img_name
        target_path: Path = self.target_dir / img_name

        input_image: Image.Image = Image.open(input_path).convert('RGB')
        target_image: Image.Image = Image.open(target_path).convert('RGB')

        input_image_transformed = self.transform(input_image)
        target_image_transformed = self.transform(target_image)

        return input_image_transformed, target_image_transformed


def create_dataloaders(
        data_root: str | Path,
        batch_size: int,
        num_workers: int = 4,
        normalize: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from the given data root directory.

    Args:
        normalize: Normalize data
        data_root: Path to root directory containing train and val folders
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    transform: Compose = transforms.Compose([
        transforms.CenterCrop(320),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    if normalize:
        transform.transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    # Create datasets
    train_dataset: ImagePairDataset = ImagePairDataset(
        root_dir=data_root,
        split='train',
        transform=transform
    )

    val_dataset: ImagePairDataset = ImagePairDataset(
        root_dir=data_root,
        split='val',
        transform=transform
    )

    # Create dataloaders
    train_loader: DataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader: DataLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

# Example usage:
# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     data_root: str = os.path.join(current_dir, '../data/dataset/rgb2ir')
#     train_loader, val_loader = create_dataloaders(
#         data_root=data_root,
#         batch_size=32,
#         num_workers=4
#     )
#
#     # Print dataset sizes
#     print(f"Training samples: {len(train_loader.dataset)}")
#     print(f"Validation samples: {len(val_loader.dataset)}")
#
#     # Test loading a batch
#     for inputs, targets in train_loader:
#         print(f"Input batch shape: {inputs.shape}")
#         print(f"Target batch shape: {targets.shape}")
#         break
