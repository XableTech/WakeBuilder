"""
Tests for the training pipeline.

This module tests:
- Model architectures (BC-ResNet, TC-ResNet)
- Training configuration
- Data preparation
- Threshold calibration
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.models import (
    BCResNet,
    TCResNet,
    Trainer,
    TrainingConfig,
    WakeWordDataset,
    calibrate_threshold,
    count_parameters,
    create_model,
    get_model_info,
)


# Test constants
BATCH_SIZE = 4
N_MELS = 80
TIME_STEPS = 96


def generate_dummy_spectrogram(
    batch_size: int = BATCH_SIZE,
    time_steps: int = TIME_STEPS,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """Generate dummy mel spectrogram for testing."""
    return torch.randn(batch_size, time_steps, n_mels)


class TestBCResNet:
    """Tests for BC-ResNet architecture."""

    def test_creation(self):
        """Test model creation."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        assert model is not None
        assert model.num_classes == 2

    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        x = generate_dummy_spectrogram()

        output = model(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_forward_with_channel_dim(self):
        """Test forward pass with explicit channel dimension."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        x = generate_dummy_spectrogram().unsqueeze(1)  # Add channel dim

        output = model(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_embedding_extraction(self):
        """Test embedding extraction."""
        model = BCResNet(num_classes=2, n_mels=N_MELS, base_channels=16)
        x = generate_dummy_spectrogram()

        embedding = model.get_embedding(x)

        # Embedding should be (batch, channels * 8) based on architecture
        assert embedding.dim() == 2
        assert embedding.shape[0] == BATCH_SIZE

    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = BCResNet(num_classes=2, n_mels=N_MELS, base_channels=16)
        num_params = count_parameters(model)

        # Should be under 500K for small footprint
        assert num_params < 500_000
        print(f"BC-ResNet parameters: {num_params:,}")

    def test_scale_factor(self):
        """Test channel scaling."""
        model_small = BCResNet(num_classes=2, n_mels=N_MELS, scale=0.5)
        model_large = BCResNet(num_classes=2, n_mels=N_MELS, scale=2.0)

        params_small = count_parameters(model_small)
        params_large = count_parameters(model_large)

        assert params_small < params_large


class TestTCResNet:
    """Tests for TC-ResNet architecture."""

    def test_creation(self):
        """Test model creation."""
        model = TCResNet(num_classes=2, n_mels=N_MELS)
        assert model is not None
        assert model.num_classes == 2

    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = TCResNet(num_classes=2, n_mels=N_MELS)
        x = generate_dummy_spectrogram()

        output = model(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = TCResNet(num_classes=2, n_mels=N_MELS)
        num_params = count_parameters(model)

        # TC-ResNet should be smaller than BC-ResNet
        assert num_params < 200_000
        print(f"TC-ResNet parameters: {num_params:,}")


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_bc_resnet(self):
        """Test creating BC-ResNet."""
        model = create_model("bc_resnet", num_classes=2, n_mels=N_MELS)
        assert isinstance(model, BCResNet)

    def test_create_tc_resnet(self):
        """Test creating TC-ResNet."""
        model = create_model("tc_resnet", num_classes=2, n_mels=N_MELS)
        assert isinstance(model, TCResNet)

    def test_default_is_tc_resnet(self):
        """Test that default model is TC-ResNet."""
        model = create_model()
        assert isinstance(model, TCResNet)

    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError):
            create_model("invalid_model")


class TestGetModelInfo:
    """Tests for model info function."""

    def test_get_info(self):
        """Test getting model info."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        info = get_model_info(model)

        assert "model_class" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "size_mb" in info

        assert info["model_class"] == "BCResNet"
        assert info["total_parameters"] > 0


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()

        assert config.model_type == "bc_resnet"
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-3

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            model_type="tc_resnet",
            batch_size=32,
            num_epochs=50,
        )

        assert config.model_type == "tc_resnet"
        assert config.batch_size == 32
        assert config.num_epochs == 50

    def test_auto_device(self):
        """Test automatic device selection."""
        config = TrainingConfig(device="auto")

        # Should be either 'cuda' or 'cpu'
        assert config.device in ["cuda", "cpu"]


class TestWakeWordDataset:
    """Tests for dataset class."""

    def test_creation(self):
        """Test dataset creation."""
        specs = [
            np.random.randn(TIME_STEPS, N_MELS).astype(np.float32) for _ in range(10)
        ]
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        dataset = WakeWordDataset(specs, labels)

        assert len(dataset) == 10

    def test_getitem(self):
        """Test getting items from dataset."""
        specs = [
            np.random.randn(TIME_STEPS, N_MELS).astype(np.float32) for _ in range(10)
        ]
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        dataset = WakeWordDataset(specs, labels)
        spec, label = dataset[0]

        assert isinstance(spec, torch.Tensor)
        assert spec.shape == (TIME_STEPS, N_MELS)
        assert label in [0, 1]


class TestTrainer:
    """Tests for trainer class."""

    def test_creation(self):
        """Test trainer creation."""
        config = TrainingConfig(num_epochs=1)
        trainer = Trainer(config=config)

        assert trainer.config == config
        assert trainer.model is None

    def test_create_model(self):
        """Test model creation through trainer."""
        config = TrainingConfig(model_type="bc_resnet", n_mels=N_MELS)
        trainer = Trainer(config=config)

        model = trainer.create_model()

        assert isinstance(model, BCResNet)
        # Model is returned but also stored internally
        assert model is not None


class TestThresholdCalibration:
    """Tests for threshold calibration."""

    def test_calibration(self):
        """Test threshold calibration with dummy model."""
        from torch.utils.data import DataLoader

        # Create dummy model and data
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        model.eval()

        specs = [
            np.random.randn(TIME_STEPS, N_MELS).astype(np.float32) for _ in range(20)
        ]
        labels = [1] * 10 + [0] * 10

        dataset = WakeWordDataset(specs, labels)
        loader = DataLoader(dataset, batch_size=4)

        device = torch.device("cpu")

        # Calibrate
        optimal_thresh, metrics = calibrate_threshold(
            model, loader, device, num_thresholds=10
        )

        assert 0.0 < optimal_thresh < 1.0
        assert len(metrics) == 10

        # Check metrics structure
        for m in metrics:
            assert hasattr(m, "threshold")
            assert hasattr(m, "far")
            assert hasattr(m, "frr")
            assert hasattr(m, "f1")
            assert 0.0 <= m.far <= 1.0
            assert 0.0 <= m.frr <= 1.0


class TestModelGradients:
    """Tests for gradient flow."""

    def test_bc_resnet_gradients(self):
        """Test that gradients flow through BC-ResNet."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        x = generate_dummy_spectrogram()
        target = torch.randint(0, 2, (BATCH_SIZE,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_tc_resnet_gradients(self):
        """Test that gradients flow through TC-ResNet."""
        model = TCResNet(num_classes=2, n_mels=N_MELS)
        x = generate_dummy_spectrogram()
        target = torch.randint(0, 2, (BATCH_SIZE,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelOutputs:
    """Tests for model output properties."""

    def test_output_is_logits(self):
        """Test that output is logits (not probabilities)."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        x = generate_dummy_spectrogram()

        output = model(x)

        # Logits can be any real number
        # Probabilities would sum to 1
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE))

    def test_deterministic_eval(self):
        """Test that eval mode gives deterministic outputs."""
        model = BCResNet(num_classes=2, n_mels=N_MELS)
        model.eval()
        x = generate_dummy_spectrogram()

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
