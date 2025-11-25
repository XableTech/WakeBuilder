"""
Training pipeline for WakeBuilder wake word models.

This module provides a complete training pipeline including:
- Data loading and batching
- Training loop with validation
- Early stopping and learning rate scheduling
- Threshold calibration for FAR/FRR optimization
- Model checkpointing and export

Designed to train models that compete with commercial solutions like Porcupine.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..audio import AudioPreprocessor, DataAugmenter, NegativeExampleGenerator
from ..config import Config
from .classifier import count_parameters, create_model


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    model_type: str = "bc_resnet"
    n_mels: int = 80
    base_channels: int = 16
    scale: float = 1.0

    # Training - conservative settings for stability
    batch_size: int = 32
    num_epochs: int = 150  # More epochs with early stopping
    learning_rate: float = 3e-4  # Conservative LR
    weight_decay: float = 1e-2  # Strong regularization
    warmup_epochs: int = 15  # Longer warmup for stability

    # Regularization - strong to prevent overfitting on small data
    dropout: float = 0.4  # Higher dropout
    label_smoothing: float = 0.1  # Moderate smoothing
    mixup_alpha: float = 0.4  # Stronger mixup

    # SpecAugment parameters (time/frequency masking)
    spec_augment: bool = True
    time_mask_param: int = 20  # Max time mask width
    freq_mask_param: int = 10  # Max frequency mask width
    num_time_masks: int = 2
    num_freq_masks: int = 2

    # Class weighting - prioritize reducing false positives
    negative_class_weight: float = 3.0  # Penalize false positives more strongly

    # Early stopping
    patience: int = 25  # More patience
    min_delta: float = 1e-4

    # Data
    val_split: float = 0.2
    num_workers: int = 0  # Windows compatibility

    # Device
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    learning_rate: float = 0.0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0


def spec_augment(
    spec: torch.Tensor,
    time_mask_param: int = 20,
    freq_mask_param: int = 10,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> torch.Tensor:
    """
    Apply SpecAugment to a spectrogram.
    
    SpecAugment masks random time and frequency bands to improve generalization.
    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR"
    
    Args:
        spec: Spectrogram tensor of shape (time, freq) or (batch, time, freq)
        time_mask_param: Maximum width of time mask
        freq_mask_param: Maximum width of frequency mask
        num_time_masks: Number of time masks to apply
        num_freq_masks: Number of frequency masks to apply
    
    Returns:
        Augmented spectrogram
    """
    spec = spec.clone()
    
    # Handle different input shapes
    if spec.dim() == 2:
        time_dim, freq_dim = spec.shape
    else:
        time_dim, freq_dim = spec.shape[-2], spec.shape[-1]
    
    # Apply time masks
    for _ in range(num_time_masks):
        t = min(np.random.randint(0, time_mask_param + 1), time_dim - 1)
        if t > 0:
            t0 = np.random.randint(0, time_dim - t)
            if spec.dim() == 2:
                spec[t0:t0 + t, :] = 0
            else:
                spec[..., t0:t0 + t, :] = 0
    
    # Apply frequency masks
    for _ in range(num_freq_masks):
        f = min(np.random.randint(0, freq_mask_param + 1), freq_dim - 1)
        if f > 0:
            f0 = np.random.randint(0, freq_dim - f)
            if spec.dim() == 2:
                spec[:, f0:f0 + f] = 0
            else:
                spec[..., :, f0:f0 + f] = 0
    
    return spec


class WakeWordDataset(Dataset):
    """
    Dataset for wake word training.

    Stores mel spectrograms and labels for efficient batch loading.
    Supports SpecAugment for training data augmentation.
    """

    def __init__(
        self,
        spectrograms: list[np.ndarray],
        labels: list[int],
        augment: bool = False,
        spec_augment_config: Optional[dict] = None,
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment
        self.spec_augment_config = spec_augment_config or {}

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()
        
        # Apply SpecAugment during training
        if self.augment and self.spec_augment_config:
            spec_tensor = spec_augment(
                spec_tensor,
                time_mask_param=self.spec_augment_config.get("time_mask_param", 20),
                freq_mask_param=self.spec_augment_config.get("freq_mask_param", 10),
                num_time_masks=self.spec_augment_config.get("num_time_masks", 2),
                num_freq_masks=self.spec_augment_config.get("num_freq_masks", 2),
            )

        return spec_tensor, label


class Trainer:
    """
    Trainer for wake word detection models.

    Handles the complete training pipeline including data preparation,
    training loop, validation, and model export.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir) if output_dir else Path(Config.MODELS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(self.config.device)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.metrics = TrainingMetrics()
        self.history: list[dict] = []

        # Preprocessor for converting audio to spectrograms
        self.preprocessor = AudioPreprocessor(n_mels=self.config.n_mels)

    def prepare_data(
        self,
        positive_audio: list[tuple[np.ndarray, int]],
        negative_audio: list[tuple[np.ndarray, int]],
        wake_word: str,
        augment_positive: bool = True,
        generate_negatives: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.

        Args:
            positive_audio: List of (audio, sample_rate) for positive examples
            negative_audio: List of (audio, sample_rate) for negative examples
            wake_word: The wake word being trained
            augment_positive: Whether to augment positive examples
            generate_negatives: Whether to generate synthetic negatives

        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("Preparing training data...")

        all_spectrograms: list[np.ndarray] = []
        all_labels: list[int] = []

        # Process positive examples
        print(f"  Processing {len(positive_audio)} positive recordings...")
        if augment_positive:
            augmenter = DataAugmenter(target_sample_rate=16000)
            for audio, sr in positive_audio:
                for sample in augmenter.augment_audio(audio, sr):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(1)
        else:
            for audio, sr in positive_audio:
                spec = self.preprocessor.process_audio(audio, sr)
                all_spectrograms.append(spec)
                all_labels.append(1)

        num_positive = len(all_spectrograms)
        print(f"    Generated {num_positive} positive samples")

        # Process negative examples
        print(f"  Processing {len(negative_audio)} negative recordings...")
        for audio, sr in negative_audio:
            spec = self.preprocessor.process_audio(audio, sr)
            all_spectrograms.append(spec)
            all_labels.append(0)

        # Generate synthetic negatives - MANY MORE for better real-world performance
        # Target ratio: ~10:1 negative to positive for robust false positive rejection
        if generate_negatives:
            print("  Generating synthetic negatives (this may take a moment)...", flush=True)
            neg_gen = NegativeExampleGenerator(target_sample_rate=16000)
            print(f"    TTS available: {neg_gen.tts_available}", flush=True)

            # Silence and noise - CRITICAL for preventing false positives in quiet environments
            print("    Generating silence samples...", flush=True)
            silence_count = 0
            try:
                for sample in neg_gen.generate_silence(num_samples=600):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)
                    silence_count += 1
            except Exception as e:
                print(f"      Error generating silence: {e}", flush=True)
            print(f"      Generated {silence_count} silence samples", flush=True)

            print("    Generating noise samples...", flush=True)
            noise_count = 0
            try:
                for sample in neg_gen.generate_pure_noise(num_samples=400):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)
                    noise_count += 1
            except Exception as e:
                print(f"      Error generating noise: {e}", flush=True)
            print(f"      Generated {noise_count} noise samples", flush=True)

            # TTS-based negatives (if available) - CRITICAL for reducing false positives
            if neg_gen.tts_available:
                # Phonetically similar words - most important for reducing false positives
                print("    Generating phonetically similar negatives...")
                count = 0
                for sample in neg_gen.generate_phonetically_similar(
                    wake_word, num_voices=5, add_noise=True
                ):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)
                    count += 1
                    if count >= 800:  # Many similar words
                        break
                print(f"      Generated {count} phonetically similar samples")

                # Random speech - general negative examples (CRITICAL for noisy environments)
                print("    Generating random speech negatives...")
                count = 0
                for sample in neg_gen.generate_random_speech(
                    num_samples=500, num_voices=5, add_noise=True
                ):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)
                    count += 1
                    if count >= 1000:  # Many random speech samples for robust rejection
                        break
                print(f"      Generated {count} random speech samples")
            else:
                # If TTS not available, generate more noise variants
                print("    TTS not available, generating more noise variants...")
                for sample in neg_gen.generate_pure_noise(num_samples=800):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)

        num_negative = len(all_spectrograms) - num_positive
        print(f"    Total negative samples: {num_negative}")

        # Balance classes
        print(f"\n  Dataset: {num_positive} positive, {num_negative} negative")

        # Store data stats for UI
        self.data_stats = {
            "num_recordings": len(positive_audio),
            "num_positive_samples": num_positive,
            "num_negative_samples": num_negative,
            "total_samples": len(all_spectrograms),
        }

        # Stratified split to ensure balanced val set
        all_labels_arr = np.array(all_labels)
        pos_indices = np.where(all_labels_arr == 1)[0]
        neg_indices = np.where(all_labels_arr == 0)[0]
        
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)
        
        val_pos_size = max(int(len(pos_indices) * self.config.val_split), 10)
        val_neg_size = max(int(len(neg_indices) * self.config.val_split), 20)
        
        val_indices = np.concatenate([pos_indices[:val_pos_size], neg_indices[:val_neg_size]])
        train_indices = np.concatenate([pos_indices[val_pos_size:], neg_indices[val_neg_size:]])
        
        np.random.shuffle(val_indices)
        np.random.shuffle(train_indices)

        train_specs = [all_spectrograms[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_specs = [all_spectrograms[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]

        self.data_stats["num_train_samples"] = len(train_specs)
        self.data_stats["num_val_samples"] = len(val_specs)
        
        print(f"  Train: {len(train_specs)} (pos: {sum(train_labels)}, neg: {len(train_labels) - sum(train_labels)})")
        print(f"  Val: {len(val_specs)} (pos: {sum(val_labels)}, neg: {len(val_labels) - sum(val_labels)})")

        # SpecAugment config for training
        spec_augment_config = None
        if self.config.spec_augment:
            spec_augment_config = {
                "time_mask_param": self.config.time_mask_param,
                "freq_mask_param": self.config.freq_mask_param,
                "num_time_masks": self.config.num_time_masks,
                "num_freq_masks": self.config.num_freq_masks,
            }
            print(f"  SpecAugment enabled: time_mask={self.config.time_mask_param}, freq_mask={self.config.freq_mask_param}")

        # Create datasets with SpecAugment for training
        train_dataset = WakeWordDataset(
            train_specs, train_labels, augment=True, spec_augment_config=spec_augment_config
        )
        val_dataset = WakeWordDataset(val_specs, val_labels, augment=False)

        # Weighted sampler with higher weight for negatives to reduce false positives
        train_labels_arr = np.array(train_labels)
        class_counts = np.bincount(train_labels_arr)
        # Apply negative class weight to penalize false positives more
        class_weights = np.array([
            self.config.negative_class_weight / class_counts[0],  # Negative class
            1.0 / class_counts[1],  # Positive class
        ])
        sample_weights = class_weights[train_labels_arr]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )
        print(f"  Class weights: negative={class_weights[0]:.4f}, positive={class_weights[1]:.4f}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def create_model(self) -> nn.Module:
        """Create the model based on configuration."""
        # Build kwargs based on model type
        kwargs = {}
        if self.config.model_type == "bc_resnet":
            kwargs["base_channels"] = self.config.base_channels
            kwargs["scale"] = self.config.scale
        elif self.config.model_type == "tc_resnet":
            kwargs["width_mult"] = self.config.scale

        model = create_model(
            model_type=self.config.model_type,
            num_classes=2,
            n_mels=self.config.n_mels,
            **kwargs,
        )

        num_params = count_parameters(model)
        print(f"\nModel: {self.config.model_type}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Size: {num_params * 4 / 1024:.1f} KB")

        return model.to(self.device)

    def setup_training(self, model: nn.Module, num_batches: int) -> None:
        """Setup optimizer and scheduler."""
        self.model = model

        # Optimizer with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # OneCycleLR scheduler for better convergence
        total_steps = max(num_batches * self.config.num_epochs, 10)  # Minimum 10 steps
        warmup_pct = min(
            self.config.warmup_epochs / max(self.config.num_epochs, 1), 0.3
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy="cos",
        )

    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> tuple[float, float]:
        """Train for one epoch with class-weighted loss."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Class weights for loss function - penalize false positives more
        # Weight[0] = negative class (higher to reduce false accepts)
        # Weight[1] = positive class
        class_weights = torch.tensor(
            [self.config.negative_class_weight, 1.0], 
            device=self.device
        )

        for batch_idx, (specs, labels) in enumerate(train_loader):
            specs = specs.to(self.device)
            labels = labels.to(self.device)

            # Apply mixup with probability 0.5
            if self.config.mixup_alpha > 0 and np.random.random() > 0.5:
                specs, labels_a, labels_b, lam = self.mixup_data(
                    specs, labels, self.config.mixup_alpha
                )

                # Forward pass
                outputs = self.model(specs)

                # Mixup loss with class weights
                loss = lam * F.cross_entropy(
                    outputs, labels_a, 
                    weight=class_weights,
                    label_smoothing=self.config.label_smoothing
                ) + (1 - lam) * F.cross_entropy(
                    outputs, labels_b, 
                    weight=class_weights,
                    label_smoothing=self.config.label_smoothing
                )
            else:
                # Standard forward pass with class weights
                outputs = self.model(specs)
                loss = F.cross_entropy(
                    outputs, labels, 
                    weight=class_weights,
                    label_smoothing=self.config.label_smoothing
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for specs, labels in val_loader:
            specs = specs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(specs)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()

        # Precision, recall, F1 for positive class
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        wake_word: str,
    ) -> nn.Module:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            wake_word: The wake word being trained

        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        # Create model
        self.model = self.create_model()

        # Setup optimizer and scheduler
        self.setup_training(self.model, len(train_loader))

        best_model_state = None
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update metrics
            self.metrics.epoch = epoch
            self.metrics.train_loss = train_loss
            self.metrics.train_acc = train_acc
            self.metrics.val_loss = val_metrics["loss"]
            self.metrics.val_acc = val_metrics["accuracy"]
            self.metrics.val_precision = val_metrics["precision"]
            self.metrics.val_recall = val_metrics["recall"]
            self.metrics.val_f1 = val_metrics["f1"]
            self.metrics.learning_rate = self.optimizer.param_groups[0]["lr"]

            # Save history
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "lr": self.metrics.learning_rate,
                }
            )

            # Check for improvement
            if val_metrics["loss"] < self.metrics.best_val_loss - self.config.min_delta:
                self.metrics.best_val_loss = val_metrics["loss"]
                self.metrics.epochs_without_improvement = 0
                best_model_state = self.model.state_dict().copy()
            else:
                self.metrics.epochs_without_improvement += 1

            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                f"Train: {train_loss:.4f} ({train_acc:.1%}) | "
                f"Val: {val_metrics['loss']:.4f} ({val_metrics['accuracy']:.1%}) | "
                f"F1: {val_metrics['f1']:.3f} | "
                f"LR: {self.metrics.learning_rate:.2e} | "
                f"{epoch_time:.1f}s"
            )

            # Early stopping
            if self.metrics.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.metrics.best_val_loss:.4f}")

        return self.model

    def save_model(
        self,
        wake_word: str,
        threshold: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Save the trained model with metadata.

        Args:
            wake_word: The wake word this model detects
            threshold: Detection threshold
            metadata: Additional metadata to save

        Returns:
            Path to saved model
        """
        # Create model directory
        model_name = wake_word.lower().replace(" ", "_")
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = model_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_type": self.config.model_type,
                "n_mels": self.config.n_mels,
                "base_channels": self.config.base_channels,
                "scale": self.config.scale,
            },
            model_path,
        )

        # Save metadata
        meta = {
            "wake_word": wake_word,
            "threshold": threshold,
            "model_type": self.config.model_type,
            "parameters": count_parameters(self.model),
            "training_config": {
                "batch_size": self.config.batch_size,
                "num_epochs": self.metrics.epoch + 1,
                "learning_rate": self.config.learning_rate,
            },
            "metrics": {
                "best_val_loss": self.metrics.best_val_loss,
                "val_accuracy": self.metrics.val_acc,
                "val_f1": self.metrics.val_f1,
            },
        }
        if metadata:
            meta.update(metadata)

        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Save training history
        history_path = model_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nModel saved to: {model_dir}")
        return model_dir


# ============================================================================
# Threshold Calibration (Task 2.9)
# ============================================================================


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold."""

    threshold: float
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    accuracy: float
    precision: float
    recall: float
    f1: float


def calibrate_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_thresholds: int = 100,
) -> tuple[float, list[ThresholdMetrics]]:
    """
    Calibrate detection threshold by computing FAR/FRR at different thresholds.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run inference on
        num_thresholds: Number of threshold values to test

    Returns:
        Tuple of (optimal_threshold, list of ThresholdMetrics)
    """
    model.eval()

    # Collect all predictions
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            outputs = model(specs)
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Test different thresholds
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    metrics_list = []

    for thresh in thresholds:
        predictions = (all_probs >= thresh).astype(int)

        # Calculate metrics
        tp = ((predictions == 1) & (all_labels == 1)).sum()
        tn = ((predictions == 0) & (all_labels == 0)).sum()
        fp = ((predictions == 1) & (all_labels == 0)).sum()
        fn = ((predictions == 0) & (all_labels == 1)).sum()

        # FAR: False positives / Total negatives
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # FRR: False negatives / Total positives
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        accuracy = (tp + tn) / len(all_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics_list.append(
            ThresholdMetrics(
                threshold=thresh,
                far=far,
                frr=frr,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    # Find optimal threshold for wake word detection
    # We STRONGLY prioritize low FAR (false activations are very annoying)
    # Target: FAR < 1% with acceptable FRR < 10%
    best_idx = len(metrics_list) // 2  # Default to 0.5
    best_score = float("inf")

    for i, m in enumerate(metrics_list):
        # Heavy penalty for FAR - we want very few false activations
        # Score = 10*FAR + FRR, but only consider if FAR < 5%
        if m.far > 0.05:  # Skip thresholds with too high FAR
            continue
        score = 10 * m.far + m.frr
        if score < best_score:
            best_score = score
            best_idx = i

    # If no threshold met FAR < 5%, find the one with lowest FAR
    if best_score == float("inf"):
        min_far = float("inf")
        for i, m in enumerate(metrics_list):
            if m.far < min_far:
                min_far = m.far
                best_idx = i

    optimal_threshold = metrics_list[best_idx].threshold
    
    # Ensure threshold is at least 0.6 for safety
    if optimal_threshold < 0.6:
        for i, m in enumerate(metrics_list):
            if m.threshold >= 0.6:
                optimal_threshold = m.threshold
                break

    return optimal_threshold, metrics_list


def print_threshold_report(
    optimal_threshold: float,
    metrics_list: list[ThresholdMetrics],
) -> None:
    """Print a report of threshold calibration results."""
    print("\n" + "=" * 60)
    print("Threshold Calibration Report")
    print("=" * 60)

    # Find metrics at optimal threshold
    optimal_metrics = None
    for m in metrics_list:
        if abs(m.threshold - optimal_threshold) < 0.01:
            optimal_metrics = m
            break

    if optimal_metrics:
        print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
        print(f"  FAR (False Acceptance Rate): {optimal_metrics.far:.2%}")
        print(f"  FRR (False Rejection Rate): {optimal_metrics.frr:.2%}")
        print(f"  Accuracy: {optimal_metrics.accuracy:.2%}")
        print(f"  Precision: {optimal_metrics.precision:.2%}")
        print(f"  Recall: {optimal_metrics.recall:.2%}")
        print(f"  F1 Score: {optimal_metrics.f1:.3f}")

    # Print table of key thresholds
    print("\nThreshold Analysis:")
    print("-" * 60)
    print(f"{'Threshold':>10} {'FAR':>8} {'FRR':>8} {'Accuracy':>10} {'F1':>8}")
    print("-" * 60)

    key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in key_thresholds:
        for m in metrics_list:
            if abs(m.threshold - thresh) < 0.01:
                print(
                    f"{m.threshold:>10.2f} {m.far:>8.2%} {m.frr:>8.2%} "
                    f"{m.accuracy:>10.2%} {m.f1:>8.3f}"
                )
                break
