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

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Regularization
    dropout: float = 0.2
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Data
    val_split: float = 0.15
    num_workers: int = 0  # Windows compatibility

    # Augmentation during training
    augment_on_fly: bool = True

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


class WakeWordDataset(Dataset):
    """
    Dataset for wake word training.

    Stores mel spectrograms and labels for efficient batch loading.
    """

    def __init__(
        self,
        spectrograms: list[np.ndarray],
        labels: list[int],
        augment: bool = False,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augment = augment
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()

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

        # Generate synthetic negatives
        if generate_negatives:
            print("  Generating synthetic negatives...")
            neg_gen = NegativeExampleGenerator(target_sample_rate=16000)

            # Silence and noise
            for sample in neg_gen.generate_silence(num_samples=50):
                spec = self.preprocessor.process_audio(sample.audio, 16000)
                all_spectrograms.append(spec)
                all_labels.append(0)

            for sample in neg_gen.generate_pure_noise(num_samples=50):
                spec = self.preprocessor.process_audio(sample.audio, 16000)
                all_spectrograms.append(spec)
                all_labels.append(0)

            # TTS-based negatives (if available)
            if neg_gen.tts_available:
                count = 0
                for sample in neg_gen.generate_phonetically_similar(
                    wake_word, num_voices=3, add_noise=True
                ):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)
                    count += 1
                    if count >= 200:
                        break

                count = 0
                for sample in neg_gen.generate_random_speech(
                    num_samples=100, num_voices=3, add_noise=True
                ):
                    spec = self.preprocessor.process_audio(sample.audio, 16000)
                    all_spectrograms.append(spec)
                    all_labels.append(0)
                    count += 1
                    if count >= 200:
                        break

        num_negative = len(all_spectrograms) - num_positive
        print(f"    Total negative samples: {num_negative}")

        # Balance classes
        print(f"\n  Dataset: {num_positive} positive, {num_negative} negative")

        # Split into train/val
        indices = np.random.permutation(len(all_spectrograms))
        val_size = int(len(indices) * self.config.val_split)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_specs = [all_spectrograms[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_specs = [all_spectrograms[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]

        print(f"  Train: {len(train_specs)}, Val: {len(val_specs)}")

        # Create datasets
        train_dataset = WakeWordDataset(train_specs, train_labels, augment=True)
        val_dataset = WakeWordDataset(val_specs, val_labels, augment=False)

        # Weighted sampler for class imbalance
        train_labels_arr = np.array(train_labels)
        class_counts = np.bincount(train_labels_arr)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels_arr]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )

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
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (specs, labels) in enumerate(train_loader):
            specs = specs.to(self.device)
            labels = labels.to(self.device)

            # Apply mixup
            if self.config.mixup_alpha > 0 and np.random.random() > 0.5:
                specs, labels_a, labels_b, lam = self.mixup_data(
                    specs, labels, self.config.mixup_alpha
                )

                # Forward pass
                outputs = self.model(specs)

                # Mixup loss
                loss = lam * F.cross_entropy(
                    outputs, labels_a, label_smoothing=self.config.label_smoothing
                ) + (1 - lam) * F.cross_entropy(
                    outputs, labels_b, label_smoothing=self.config.label_smoothing
                )
            else:
                # Standard forward pass
                outputs = self.model(specs)
                loss = F.cross_entropy(
                    outputs, labels, label_smoothing=self.config.label_smoothing
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

    # Find optimal threshold (minimize FAR + FRR, or maximize F1)
    # For wake word detection, we typically want low FAR with acceptable FRR
    best_idx = 0
    best_score = float("inf")

    for i, m in enumerate(metrics_list):
        # Weight FAR more heavily (false activations are more annoying)
        score = 2 * m.far + m.frr
        if score < best_score:
            best_score = score
            best_idx = i

    optimal_threshold = metrics_list[best_idx].threshold

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
