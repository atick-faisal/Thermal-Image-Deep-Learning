from PIL import Image
from torch.utils.data import DataLoader

from src.dataloader import create_dataloaders


class TestCreateDataloaders:

    # Creates train and val dataloaders with default parameters and valid data directory
    def test_create_dataloaders_with_valid_directory(self, tmp_path):
        # Create test directory structure
        rgb_train = tmp_path / "RGB" / "train"
        rgb_val = tmp_path / "RGB" / "val"
        ir_train = tmp_path / "IR" / "train"
        ir_val = tmp_path / "IR" / "val"

        for img_dir in [rgb_train, rgb_val, ir_train, ir_val]:
            img_dir.mkdir(parents=True)

        # Add sample images
        sample_img = Image.new('RGB', (320, 320))
        sample_img.save(rgb_train / "img1.jpg")
        sample_img.save(ir_train / "img1.jpg")
        sample_img.save(rgb_val / "img2.jpg")
        sample_img.save(ir_val / "img2.jpg")

        # Test dataloader creation
        train_loader, val_loader = create_dataloaders(
            data_root=tmp_path,
            batch_size=2
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert len(train_loader.dataset) == 1
        assert len(val_loader.dataset) == 1