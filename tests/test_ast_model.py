#!/usr/bin/env python3
"""
Test script to verify AST model loading and inference.

This script tests:
1. Loading the AST base model from Hugging Face
2. Creating the WakeWordClassifier
3. Running inference on sample audio
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch


@pytest.fixture(scope="module")
def ast_model():
    """Fixture to load and return the AST model."""
    from wakebuilder.models.classifier import (
        ASTWakeWordModel,
    )

    model = ASTWakeWordModel(
        freeze_base=True,
        classifier_hidden_dims=[256, 128],
        classifier_dropout=0.3,
    )
    return model


@pytest.fixture(scope="module")
def feature_extractor_and_inputs():
    """Fixture to create feature extractor and test inputs."""
    from transformers import AutoFeatureExtractor
    from wakebuilder.models.classifier import AST_MODEL_CHECKPOINT

    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_CHECKPOINT)

    # Create dummy audio (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    return feature_extractor, inputs


def test_ast_model_loading(ast_model):
    """Test loading the AST model."""
    from wakebuilder.models.classifier import get_model_info

    print("=" * 60)
    print("Testing AST Model Loading")
    print("=" * 60)

    print("   [OK] Model loaded successfully!")

    # Get model info
    info = get_model_info(ast_model)
    print("\n2. Model Information:")
    print(f"   - Base model: {info['base_model']}")
    print(f"   - Embedding dimension: {info['embedding_dim']}")
    print(f"   - Total parameters: {info['total_parameters']:,}")
    print(f"   - Trainable parameters: {info['trainable_parameters']:,}")
    print(f"   - Frozen parameters: {info['frozen_parameters']:,}")
    print(f"   - Model size: {info['size_mb']:.1f} MB")
    print(f"   - Trainable size: {info['trainable_size_mb']:.3f} MB")

    # Assertions
    assert ast_model is not None
    assert info["embedding_dim"] > 0
    assert info["total_parameters"] > 0


def test_feature_extraction(feature_extractor_and_inputs):
    """Test AST feature extraction."""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)

    feature_extractor, inputs = feature_extractor_and_inputs

    print("\n1. Feature extractor loaded!")
    print(f"   Input shape: {inputs['input_values'].shape}")
    print("   [OK] Feature extraction successful!")

    # Assertions
    assert feature_extractor is not None
    assert "input_values" in inputs
    assert len(inputs["input_values"].shape) == 3  # (batch, mel_bins, time_steps)


def test_inference(ast_model, feature_extractor_and_inputs):
    """Test model inference."""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)

    _, inputs = feature_extractor_and_inputs
    ast_model.eval()

    print("\n1. Running inference...")
    with torch.inference_mode():
        outputs = ast_model(inputs["input_values"])

    print(f"   Output shape: {outputs.shape}")
    print(f"   Output logits: {outputs[0].tolist()}")

    # Get probabilities
    probs = torch.softmax(outputs, dim=-1)
    print(f"   Probabilities: {probs[0].tolist()}")
    print(f"   - Negative class: {probs[0, 0].item():.4f}")
    print(f"   - Positive class: {probs[0, 1].item():.4f}")

    print("\n   [OK] Inference successful!")

    # Assertions
    assert outputs.shape[0] == 1
    assert outputs.shape[1] == 2
    assert 0 <= probs[0, 0].item() <= 1
    assert 0 <= probs[0, 1].item() <= 1


def test_trainer_import():
    """Test trainer module imports."""
    print("\n" + "=" * 60)
    print("Testing Trainer Module")
    print("=" * 60)

    print("\n1. Importing trainer module...")
    from wakebuilder.models.trainer import (
        TrainingConfig,
    )

    print("   [OK] Trainer module imported!")

    print("\n2. Creating training config...")
    config = TrainingConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
    )
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Device: {config.device}")
    print("   [OK] Config created!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WakeBuilder AST Model Test Suite")
    print("=" * 60)

    try:
        # Test 1: Model loading
        model = test_ast_model_loading()

        # Test 2: Feature extraction
        feature_extractor, inputs = test_feature_extraction()

        # Test 3: Inference
        test_inference(model, inputs)

        # Test 4: Trainer imports
        test_trainer_import()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe AST-based wake word system is ready to use.")
        print("You can now train models using the web interface or API.")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
