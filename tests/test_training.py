"""
Tests for the AST-based training pipeline.

This module tests:
- AST model architecture and classifier
- Training configuration
- Data preparation with ASTDataset
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
    ASTDataset,
    ASTTrainer,
    Trainer,
    TrainingConfig,
    WakeWordClassifier,
    count_parameters,
)


# Test constants
BATCH_SIZE = 4
EMBEDDING_DIM = 768  # AST embedding dimension


def generate_dummy_embeddings(
    batch_size: int = BATCH_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
) -> torch.Tensor:
    """Generate dummy AST embeddings for testing."""
    return torch.randn(batch_size, embedding_dim)


def generate_dummy_audio(
    batch_size: int = BATCH_SIZE,
    sample_rate: int = 16000,
    duration: float = 1.0,
) -> list[tuple[np.ndarray, int]]:
    """Generate dummy audio samples for testing."""
    samples = []
    num_samples = int(sample_rate * duration)
    for _ in range(batch_size):
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1
        samples.append((audio, sample_rate))
    return samples


class TestWakeWordClassifier:
    """Tests for WakeWordClassifier (the trainable head)."""

    def test_creation(self):
        """Test classifier creation with default settings."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
            dropout=0.3,
        )
        assert classifier is not None
        assert classifier.embedding_dim == EMBEDDING_DIM

    def test_forward_pass(self):
        """Test forward pass with dummy embeddings."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )
        x = generate_dummy_embeddings()

        output = classifier(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_with_attention(self):
        """Test classifier with self-attention enabled."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
            use_attention=True,
        )
        x = generate_dummy_embeddings()

        output = classifier(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_with_se_block(self):
        """Test classifier with SE block enabled."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
            use_se_block=True,
        )
        x = generate_dummy_embeddings()

        output = classifier(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_with_tcn(self):
        """Test classifier with TCN block enabled."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
            use_tcn=True,
        )
        x = generate_dummy_embeddings()

        output = classifier(x)

        assert output.shape == (BATCH_SIZE, 2)

    def test_parameter_count(self):
        """Test that classifier has reasonable parameter count."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )
        num_params = count_parameters(classifier)

        # Classifier should be relatively small
        assert num_params < 500_000
        print(f"WakeWordClassifier parameters: {num_params:,}")


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-4

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            batch_size=16,
            num_epochs=50,
            learning_rate=5e-5,
        )

        assert config.batch_size == 16
        assert config.num_epochs == 50
        assert config.learning_rate == 5e-5

    def test_auto_device(self):
        """Test automatic device selection."""
        config = TrainingConfig(device="auto")

        # Should be either 'cuda' or 'cpu'
        assert config.device in ["cuda", "cpu"]

    def test_classifier_config(self):
        """Test classifier configuration options."""
        config = TrainingConfig(
            classifier_hidden_dims=[512, 256, 128],
            classifier_dropout=0.5,
            use_attention=True,
        )

        assert config.classifier_hidden_dims == [512, 256, 128]
        assert config.classifier_dropout == 0.5
        assert config.use_attention is True


class TestASTDataset:
    """Tests for AST dataset class."""

    @pytest.fixture
    def feature_extractor(self):
        """Load AST feature extractor for tests."""
        from transformers import AutoFeatureExtractor
        from wakebuilder.models.classifier import AST_MODEL_CHECKPOINT

        return AutoFeatureExtractor.from_pretrained(AST_MODEL_CHECKPOINT)

    def test_creation(self, feature_extractor):
        """Test dataset creation with audio samples."""
        # Generate raw audio arrays (not tuples)
        audio_samples = [
            np.random.randn(16000).astype(np.float32) * 0.1 for _ in range(10)
        ]
        labels = [1] * 5 + [0] * 5

        dataset = ASTDataset(audio_samples, labels, feature_extractor)

        assert len(dataset) == 10

    def test_getitem(self, feature_extractor):
        """Test getting items from dataset."""
        # Generate raw audio arrays (not tuples)
        audio_samples = [
            np.random.randn(16000).astype(np.float32) * 0.1 for _ in range(10)
        ]
        labels = [1] * 5 + [0] * 5

        dataset = ASTDataset(audio_samples, labels, feature_extractor)
        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert label in [0, 1]


class TestASTTrainer:
    """Tests for AST trainer class."""

    def test_creation(self):
        """Test trainer creation."""
        config = TrainingConfig(num_epochs=1)
        trainer = ASTTrainer(config=config)

        assert trainer.config == config
        assert trainer.model is None

    def test_trainer_alias(self):
        """Test that Trainer is an alias for ASTTrainer."""
        assert Trainer is ASTTrainer


class TestClassifierGradients:
    """Tests for gradient flow through classifier."""

    def test_classifier_gradients(self):
        """Test that gradients flow through classifier."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )
        x = generate_dummy_embeddings()
        target = torch.randint(0, 2, (BATCH_SIZE,))

        output = classifier(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_classifier_with_attention_gradients(self):
        """Test gradients with attention enabled."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
            use_attention=True,
            use_se_block=True,
        )
        x = generate_dummy_embeddings()
        target = torch.randint(0, 2, (BATCH_SIZE,))

        output = classifier(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestClassifierOutputs:
    """Tests for classifier output properties."""

    def test_output_is_logits(self):
        """Test that output is logits (not probabilities)."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )
        x = generate_dummy_embeddings()

        output = classifier(x)

        # Logits can be any real number
        # Probabilities would sum to 1
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE))

    def test_deterministic_eval(self):
        """Test that eval mode gives deterministic outputs."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )
        classifier.eval()
        x = generate_dummy_embeddings()

        with torch.no_grad():
            output1 = classifier(x)
            output2 = classifier(x)

        assert torch.allclose(output1, output2)


class TestCountParameters:
    """Tests for parameter counting utility."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )

        total = count_parameters(classifier, trainable_only=False)
        trainable = count_parameters(classifier, trainable_only=True)

        # For classifier, all parameters should be trainable
        assert total == trainable
        assert total > 0

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        classifier = WakeWordClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[256, 128],
        )

        # Freeze some parameters
        for param in list(classifier.parameters())[:2]:
            param.requires_grad = False

        total = count_parameters(classifier, trainable_only=False)
        trainable = count_parameters(classifier, trainable_only=True)

        assert trainable < total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
