#!/usr/bin/env python3
"""
Test script to verify training data preparation with hard negatives.
"""

import sys
import numpy as np
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0])

from src.wakebuilder.models.trainer import ASTTrainer, TrainingConfig


def test_data_preparation():
    """Test that hard negatives are properly generated and split."""
    print("=" * 60)
    print("Testing Training Data Preparation")
    print("=" * 60)
    
    # Create a minimal config
    config = TrainingConfig(
        target_positive_samples=100,  # Small for testing
        use_hard_negatives=True,
        use_real_negatives=False,  # Skip real negatives for speed
        val_split=0.25,
    )
    
    trainer = ASTTrainer(config=config)
    
    # Create fake recording (1 second of noise)
    fake_recording = (np.random.randn(16000).astype(np.float32) * 0.1, 16000)
    
    print("\nPreparing data for wake word 'samix'...")
    print("-" * 40)
    
    try:
        train_loader, val_loader = trainer.prepare_data(
            positive_audio=[fake_recording],
            negative_audio=[],
            wake_word="samix",
            augment_positive=True,
        )
        
        print(f"\nResults:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        
        # Check class balance
        train_labels = [train_loader.dataset.labels[i] for i in range(len(train_loader.dataset))]
        val_labels = [val_loader.dataset.labels[i] for i in range(len(val_loader.dataset))]
        
        train_pos = sum(train_labels)
        train_neg = len(train_labels) - train_pos
        val_pos = sum(val_labels)
        val_neg = len(val_labels) - val_pos
        
        print(f"\n  Train: {train_pos} positive, {train_neg} negative")
        print(f"  Val: {val_pos} positive, {val_neg} negative")
        
        # Check if we have enough hard negatives
        if train_neg > 0 and val_neg > 0:
            print("\n  [OK] Both train and val have negative samples")
        else:
            print("\n  [ERROR] Missing negative samples!")
            
        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_preparation()
