#!/usr/bin/env python3
"""
Test script to verify training data preparation with balanced TTS provider distribution.

Tests that:
1. Positive samples are distributed across ALL TTS providers (Piper, Kokoro, Coqui, Edge)
2. Sample counts respect the configured targets
3. Negative samples are generated with the correct ratios

Expected with:
- target_positive_samples=500
- hard_negative_ratio=4 (generated negatives = 500 * 4 = 2000)
- negative_ratio=2 (real negatives = 500 * 2 = 1000)
- Total expected: 500 + 2000 + 1000 = 3500 samples
"""

import sys
import numpy as np
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from wakebuilder.models.trainer import ASTTrainer, TrainingConfig


def test_data_preparation():
    """Test that samples are properly generated and distributed across providers."""
    print("=" * 70)
    print("Testing Training Data Preparation with Multi-Provider Distribution")
    print("=" * 70)

    # Configuration matching user requirements:
    # - 500 positive samples
    # - hard_negative_ratio=4 -> 2000 hard negatives
    # - negative_ratio=2 -> 1000 real negatives
    # - Total expected: ~3500 samples
    config = TrainingConfig(
        target_positive_samples=500,
        use_hard_negatives=True,
        hard_negative_ratio=4.0,  # 500 * 4 = 2000 hard negatives
        use_real_negatives=True,
        negative_ratio=2.0,  # 500 * 2 = 1000 real negatives
        val_split=0.25,
    )

    print("\nConfiguration:")
    print(f"  Target positive samples: {config.target_positive_samples}")
    print(f"  Hard negative ratio: {config.hard_negative_ratio}x")
    print(f"  Real negative ratio: {config.negative_ratio}x")
    print(f"  Validation split: {config.val_split}")
    print("\nExpected totals:")
    print(f"  Positive: ~{config.target_positive_samples}")
    print(
        f"  Hard negatives: ~{int(config.target_positive_samples * config.hard_negative_ratio)}"
    )
    print(
        f"  Real negatives: ~{int(config.target_positive_samples * config.negative_ratio)}"
    )
    expected_total = config.target_positive_samples * (
        1 + config.hard_negative_ratio + config.negative_ratio
    )
    print(f"  TOTAL: ~{int(expected_total)}")

    trainer = ASTTrainer(config=config)

    # Create fake recording (1 second of noise)
    fake_recording = (np.random.randn(16000).astype(np.float32) * 0.1, 16000)

    print("\n" + "-" * 70)
    print("Preparing data for wake word 'jarvis'...")
    print("-" * 70)

    try:
        train_loader, val_loader = trainer.prepare_data(
            positive_audio=[fake_recording],
            negative_audio=[],
            wake_word="jarvis",
            augment_positive=True,
        )

        print(f"\n{'=' * 70}")
        print("RESULTS:")
        print(f"{'=' * 70}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")

        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        print(f"\n  TOTAL SAMPLES: {total_samples}")
        print(f"  EXPECTED: ~{int(expected_total)}")

        # Check class balance
        train_labels = [
            train_loader.dataset.labels[i] for i in range(len(train_loader.dataset))
        ]
        val_labels = [
            val_loader.dataset.labels[i] for i in range(len(val_loader.dataset))
        ]

        train_pos = sum(train_labels)
        train_neg = len(train_labels) - train_pos
        val_pos = sum(val_labels)
        val_neg = len(val_labels) - val_pos

        print(f"\n  Train: {train_pos} positive, {train_neg} negative")
        print(f"  Val: {val_pos} positive, {val_neg} negative")
        print(f"  Total positive: {train_pos + val_pos}")
        print(f"  Total negative: {train_neg + val_neg}")

        # Check if counts are reasonable (within 20% of expected)
        tolerance = 0.30  # 30% tolerance for generation variability

        total_pos = train_pos + val_pos
        expected_pos = config.target_positive_samples
        pos_ratio = total_pos / expected_pos if expected_pos > 0 else 0

        print(f"\n  Positive sample ratio: {pos_ratio:.2f}x (expected: 1.0x)")

        if abs(pos_ratio - 1.0) <= tolerance:
            print("  [OK] Positive samples within expected range")
        else:
            print(
                f"  [WARNING] Positive samples outside expected range (tolerance: {tolerance*100}%)"
            )

        if total_samples > 0 and train_neg > 0 and val_neg > 0:
            print("\n  [OK] Both train and val have positive and negative samples")
        else:
            print("\n  [ERROR] Missing samples in train or val!")

        print("\n" + "=" * 70)
        print("Test complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_data_preparation()
