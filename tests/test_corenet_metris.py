from typing import Tuple

import pytest
import torch

from src.metrics import BatchSimilarityMetrics


# Import your metrics calculator here
# from your_module import BatchSimilarityMetrics, calculate_batch_metrics

class TestBatchSimilarityMetrics:
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @pytest.fixture
    def metric_calculator(self, device):
        return BatchSimilarityMetrics().to(device)

    @staticmethod
    def generate_test_batch(
            batch_size: int = 2,
            channels: int = 3,
            height: int = 32,
            width: int = 32,
            device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test batch pairs with controlled differences."""
        # Create base images
        batch1 = torch.rand(batch_size, channels, height, width, device=device) * 2 - 1

        # Create slightly modified versions
        batch2 = batch1.clone()
        noise = (torch.rand_like(batch1) * 0.2 - 0.1)  # Small random noise
        batch2 = torch.clamp(batch2 + noise, -1, 1)

        return batch1, batch2

    def test_output_shape(self, metric_calculator, device):
        """Test if output shape is correct."""
        batch1, batch2 = self.generate_test_batch(device=device)
        metrics = metric_calculator(batch1, batch2)

        assert metrics.shape == (2, 3, 1)
        assert metrics.device.type == device

    def test_value_range(self, metric_calculator, device):
        """Test if all metrics are in [0, 1] range."""
        batch1, batch2 = self.generate_test_batch(device=device)
        metrics = metric_calculator(batch1, batch2)

        assert torch.all(metrics >= 0)
        assert torch.all(metrics <= 1)

    def test_identical_images(self, metric_calculator, device):
        """Test metrics for identical images."""
        batch1, _ = self.generate_test_batch(device=device)
        metrics = metric_calculator(batch1, batch1)

        # All metrics should be very close to 1 for identical images
        assert torch.all(metrics > 0.99)

    def test_completely_different_images(self, metric_calculator, device):
        """Test metrics for completely different images."""
        batch1 = torch.ones(2, 3, 32, 32, device=device)
        batch2 = -torch.ones(2, 3, 32, 32, device=device)
        metrics = metric_calculator(batch1, batch2)

        # Metrics should be low for completely different images
        assert torch.all(metrics < 0.5)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, metric_calculator, device, batch_size):
        """Test if function works with different batch sizes."""
        batch1, batch2 = self.generate_test_batch(batch_size=batch_size, device=device)
        metrics = metric_calculator(batch1, batch2)

        assert metrics.shape == (batch_size, 3, 1)

    @pytest.mark.parametrize("size", [(16, 16), (32, 32), (64, 64)])
    def test_different_image_sizes(self, metric_calculator, device, size):
        """Test if function works with different image sizes."""
        height, width = size
        batch1, batch2 = self.generate_test_batch(
            height=height, width=width, device=device
        )
        metrics = metric_calculator(batch1, batch2)

        assert metrics.shape == (2, 3, 1)

    def test_input_range_validation(self, metric_calculator, device):
        """Test if function properly validates input range."""
        # Create data outside [-1, 1] range
        batch1 = torch.rand(2, 3, 32, 32, device=device) * 3 - 1.5
        batch2 = torch.rand(2, 3, 32, 32, device=device) * 3 - 1.5

        with pytest.raises(ValueError):
            _ = metric_calculator(batch1, batch2)

    def test_different_noise_levels(self, metric_calculator, device):
        """Test metric sensitivity to different noise levels."""
        batch1, _ = self.generate_test_batch(device=device)
        noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        previous_metrics: torch.Tensor | None = None

        for noise in noise_levels:
            batch2 = batch1.clone()
            noise_tensor = (torch.rand_like(batch1) * 2 - 1) * noise
            batch2 = torch.clamp(batch2 + noise_tensor, -1, 1)

            metrics = metric_calculator(batch1, batch2)

            if previous_metrics is not None:
                # Metrics should generally decrease with more noise
                assert torch.all(metrics <= previous_metrics)

            previous_metrics = metrics

    def test_batch_consistency(self, metric_calculator, device):
        """Test if metrics are consistent across batch items."""
        # Create batch with identical pairs
        batch1 = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
        batch2 = batch1.clone()

        # Add identical noise to both pairs
        noise = (torch.rand(1, 3, 32, 32, device=device) * 0.2 - 0.1)
        batch2 = torch.clamp(batch2 + noise, -1, 1)

        metrics = metric_calculator(batch1, batch2)

        # Metrics should be very similar for identical pairs
        assert torch.allclose(metrics[0], metrics[1], rtol=1e-2)

    # @pytest.mark.parametrize("mi_method", ['min', 'geometric', 'arithmetic', 'max'])
    # def test_mi_methods(self, device, mi_method):
    #     """Test different MI normalization methods."""
    #     calculator = BatchSimilarityMetrics(mi_method=mi_method).to(device)
    #     batch1, batch2 = self.generate_test_batch(device=device)
    #
    #     metrics = calculator(batch1, batch2)
    #
    #     assert metrics.shape == (2, 3, 1)
    #     assert torch.all(metrics >= 0)
    #     assert torch.all(metrics <= 1)
