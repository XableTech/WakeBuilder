#!/usr/bin/env python
"""
Improved training script for wake word models.

This script implements several improvements to address discrimination issues:
1. Focal Loss - Better handling of hard examples
2. Higher hard negative ratio - More phonetically similar words
3. Deeper classifier - Better discrimination capacity
4. Optional attention - Better feature selection

Usage:
    uv run python scripts/train_improved.py --wake-word "samix" --recordings recordings/

Key improvements over default training:
- Focal loss with gamma=2.0 for hard example mining
- Hard negative ratio of 4.0x (more similar-sounding words)
- Deeper classifier [512, 256, 128] for better discrimination
- Optional self-attention for feature refinement
- Higher label smoothing (0.3) to prevent overconfidence
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_recordings(recordings_dir: Path) -> list[tuple[np.ndarray, int]]:
    """Load audio recordings from a directory."""
    recordings = []
    
    for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
        for audio_file in recordings_dir.glob(ext):
            try:
                audio, sr = sf.read(audio_file)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                recordings.append((audio.astype(np.float32), sr))
                print(f"  Loaded: {audio_file.name} ({len(audio)/sr:.2f}s)")
            except Exception as e:
                print(f"  Failed to load {audio_file}: {e}")
    
    return recordings


def main():
    parser = argparse.ArgumentParser(description="Train improved wake word model")
    parser.add_argument("--wake-word", type=str, required=True,
                       help="Wake word to train")
    parser.add_argument("--recordings", type=str, required=True,
                       help="Directory containing recording files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: models/custom/<wake_word>)")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=100,
                       help="Maximum training epochs (default: 100)")
    parser.add_argument("--positive-samples", type=int, default=4000,
                       help="Target positive samples (default: 4000)")
    parser.add_argument("--hard-negative-ratio", type=float, default=4.0,
                       help="Hard negative ratio (default: 4.0x)")
    parser.add_argument("--negative-ratio", type=float, default=2.0,
                       help="Real negative ratio (default: 2.0x)")
    
    # Model options
    parser.add_argument("--use-attention", action="store_true",
                       help="Enable self-attention in classifier")
    parser.add_argument("--classifier-dims", type=str, default="512,256,128",
                       help="Classifier hidden dimensions (default: 512,256,128)")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate (default: 0.5)")
    
    # Loss options
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--label-smoothing", type=float, default=0.3,
                       help="Label smoothing (default: 0.3)")
    
    args = parser.parse_args()
    
    # Parse classifier dimensions
    classifier_dims = [int(x) for x in args.classifier_dims.split(",")]
    
    # Setup paths
    recordings_dir = Path(args.recordings)
    if not recordings_dir.exists():
        print(f"Error: Recordings directory not found: {recordings_dir}")
        return 1
    
    output_dir = Path(args.output) if args.output else Path(f"models/custom/{args.wake_word.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IMPROVED WAKE WORD TRAINING")
    print("=" * 70)
    print(f"Wake word: '{args.wake_word}'")
    print(f"Recordings: {recordings_dir}")
    print(f"Output: {output_dir}")
    print()
    print("Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Positive samples: {args.positive_samples}")
    print(f"  Hard negative ratio: {args.hard_negative_ratio}x")
    print(f"  Real negative ratio: {args.negative_ratio}x")
    print(f"  Classifier dims: {classifier_dims}")
    print(f"  Use attention: {args.use_attention}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Focal gamma: {args.focal_gamma}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print()
    
    # Load recordings
    print("Loading recordings...")
    recordings = load_recordings(recordings_dir)
    
    if not recordings:
        print("Error: No recordings found!")
        return 1
    
    print(f"Loaded {len(recordings)} recordings")
    print()
    
    # Import training modules
    from src.wakebuilder.models.trainer import ASTTrainer, TrainingConfig
    from src.wakebuilder.config import Config
    
    # Create training config with improvements
    config = TrainingConfig(
        # Classifier architecture - deeper for better discrimination
        classifier_hidden_dims=classifier_dims,
        classifier_dropout=args.dropout,
        freeze_base=True,
        
        # Training hyperparameters
        batch_size=32,
        num_epochs=args.epochs,
        learning_rate=1e-4,
        weight_decay=1e-3,
        warmup_epochs=10,
        
        # Regularization - higher values for better generalization
        label_smoothing=args.label_smoothing,
        mixup_alpha=0.5,
        
        # Focal loss for hard example mining
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=args.focal_gamma,
        
        # Attention
        use_attention=args.use_attention,
        
        # Early stopping
        patience=10,
        min_delta=1e-4,
        
        # Data settings
        val_split=0.25,
        target_positive_samples=args.positive_samples,
        use_tts_positives=True,
        
        # Negative settings - higher ratios for better discrimination
        use_real_negatives=True,
        max_real_negatives=0,
        use_hard_negatives=True,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
    )
    
    # Create trainer
    trainer = ASTTrainer(config=config, output_dir=output_dir)
    
    # Prepare data
    print("Preparing training data...")
    train_loader, val_loader = trainer.prepare_data(
        positive_audio=recordings,
        negative_audio=[],  # Will be loaded from cache
        wake_word=args.wake_word,
        augment_positive=True,
    )
    
    # Train
    print("\nStarting training...")
    model = trainer.train(train_loader, val_loader, args.wake_word)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run diagnostic: uv run python scripts/diagnose_model.py")
    print("  2. Test in UI: uv run uvicorn wakebuilder.backend.main:app --host 127.0.0.1 --port 8000")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
