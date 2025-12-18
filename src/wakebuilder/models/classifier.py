"""
Wake word classifier using Audio Spectrogram Transformer (AST).

This module provides a transfer learning approach using the pre-trained
MIT/ast-finetuned-speech-commands-v2 model as a frozen feature extractor,
with a trainable classifier head on top.

Architecture:
- Base Model: AST (MIT/ast-finetuned-speech-commands-v2) - FROZEN
- Classifier: 2-3 layer feedforward neural network - TRAINABLE

The AST model is purpose-built for audio classification and is pre-trained
on the Speech Commands dataset with 35 different wake words, providing an
excellent foundation for transfer learning.

References:
- AST Paper: Gong et al., "AST: Audio Spectrogram Transformer"
- Model: https://huggingface.co/MIT/ast-finetuned-speech-commands-v2
"""

from typing import Optional, Tuple
from pathlib import Path
import logging
 
import torch
import torch.nn as nn
from transformers import ASTModel, AutoFeatureExtractor

logger = logging.getLogger(__name__)

# Model checkpoint for AST
AST_MODEL_CHECKPOINT = "MIT/ast-finetuned-speech-commands-v2"


class SelfAttentionPooling(nn.Module):
    """
    Self-attention pooling layer for embedding refinement.
    
    This helps the model focus on discriminative features in the embedding.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embedding_dim)
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)  # Residual connection
        return x.squeeze(1)  # (batch, embedding_dim)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    
    This learns to emphasize important features and suppress less useful ones
    by modeling channel interdependencies. Particularly effective after
    attention layers to refine feature selection.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    
    Args:
        dim: Input/output dimension
        reduction: Reduction ratio for the bottleneck (default: 4)
    """
    
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(dim // reduction, 32)  # Minimum 32 to avoid too small
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        scale = self.fc(x)
        return x * scale


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) block for capturing local patterns.
    
    This applies 1D convolutions to capture local temporal patterns in the
    embedding space that attention might miss. Uses dilated convolutions
    for larger receptive fields without increasing parameters.
    
    Reference: Bai et al., "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling" (2018)
    
    Args:
        dim: Input/output dimension
        kernel_size: Convolution kernel size (default: 3)
        dilation: Dilation factor for larger receptive field (default: 1)
    """
    
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        # Use LayerNorm instead of BatchNorm to handle batch size 1
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, dim),  # GroupNorm with 1 group = LayerNorm, works with batch size 1
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, dim),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        # Add channel dimension for conv1d: (batch, dim, 1)
        x_conv = x.unsqueeze(-1)
        
        # Apply convolution
        out = self.conv(x_conv)
        
        # Residual connection
        out = out + x_conv
        out = self.activation(out)
        out = self.dropout(out)
        
        # Remove channel dimension
        return out.squeeze(-1)


class WakeWordClassifier(nn.Module):
    """
    Wake word classifier head that sits on top of AST embeddings.
    
    This is a feedforward network with batch normalization and residual-like
    connections that takes AST embeddings and outputs binary classification.
    
    Architecture:
        Input (768) -> LayerNorm -> [Optional: SelfAttention] -> [Optional: SE Block] ->
        [Optional: TCN] -> Linear(256) -> GELU -> Dropout -> Linear(128) -> GELU -> Dropout -> Linear(2)
    
    Enhancements over basic MLP:
    - LayerNorm on input for stable training
    - Optional self-attention for better feature discrimination
    - Optional SE block for channel attention (emphasizes important features)
    - Optional TCN block for local pattern capture
    - GELU activation (smoother than ReLU, better for transformers)
    - Batch normalization between layers for faster convergence
    
    Args:
        embedding_dim: Dimension of AST embeddings (768 for AST)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        num_classes: Number of output classes (2 for binary classification)
        use_attention: Whether to use self-attention pooling
        use_se_block: Whether to use Squeeze-and-Excitation block
        use_tcn: Whether to use Temporal Convolutional Network block
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        num_classes: int = 2,
        use_attention: bool = False,
        use_se_block: bool = False,
        use_tcn: bool = False,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_se_block = use_se_block
        self.use_tcn = use_tcn
        
        # Input normalization for stable training
        self.input_norm = nn.LayerNorm(embedding_dim)
        
        # Optional self-attention for better discrimination
        if use_attention:
            self.attention = SelfAttentionPooling(embedding_dim, num_heads=4)
        else:
            self.attention = None
        
        # Optional SE block after attention for channel refinement
        if use_se_block:
            self.se_block = SqueezeExcitation(embedding_dim, reduction=4)
        else:
            self.se_block = None
        
        # Optional TCN block for local pattern capture
        if use_tcn:
            self.tcn_block = TemporalConvBlock(embedding_dim, kernel_size=3)
        else:
            self.tcn_block = None
        
        # Build classifier layers
        layers = []
        in_dim = embedding_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm works with any batch size including 1
                nn.GELU(),  # GELU works better with transformer embeddings
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            embeddings: AST embeddings of shape (batch, embedding_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Normalize input embeddings
        x = self.input_norm(embeddings)
        
        # Apply attention if enabled
        if self.attention is not None:
            x = self.attention(x)
        
        # Apply SE block if enabled (after attention)
        if self.se_block is not None:
            x = self.se_block(x)
        
        # Apply TCN block if enabled
        if self.tcn_block is not None:
            x = self.tcn_block(x)
        
        return self.classifier(x)


class ASTWakeWordModel(nn.Module):
    """
    Complete wake word detection model using AST as the base.
    
    This model combines:
    1. AST base model (frozen) for feature extraction
    2. WakeWordClassifier (trainable) for binary classification
    
    The AST model is loaded from Hugging Face and kept frozen during training.
    Only the classifier head is trained.
    
    Args:
        freeze_base: Whether to freeze the AST base model (default: True)
        classifier_hidden_dims: Hidden dimensions for classifier
        classifier_dropout: Dropout for classifier
        use_attention: Whether to use self-attention in classifier
        use_se_block: Whether to use Squeeze-and-Excitation block
        use_tcn: Whether to use Temporal Convolutional Network block
    """
    
    def __init__(
        self,
        freeze_base: bool = True,
        classifier_hidden_dims: Optional[list[int]] = None,
        classifier_dropout: float = 0.3,
        use_attention: bool = False,
        use_se_block: bool = False,
        use_tcn: bool = False,
    ):
        super().__init__()
        
        self.freeze_base = freeze_base
        
        # Load AST base model
        logger.info(f"Loading AST model from {AST_MODEL_CHECKPOINT}...")
        self.base_model = ASTModel.from_pretrained(AST_MODEL_CHECKPOINT)
        
        # Get embedding dimension from config
        self.embedding_dim = self.base_model.config.hidden_size  # 768
        
        # Freeze base model if specified
        if freeze_base:
            logger.info("Freezing AST base model parameters")
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()
        
        # Create classifier head
        self.classifier = WakeWordClassifier(
            embedding_dim=self.embedding_dim,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
            num_classes=2,
            use_attention=use_attention,
            use_se_block=use_se_block,
            use_tcn=use_tcn,
        )
        
        logger.info(f"Model initialized with embedding_dim={self.embedding_dim}, use_attention={use_attention}, use_se_block={use_se_block}, use_tcn={use_tcn}")
    
    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from audio using AST.
        
        Args:
            input_values: Preprocessed audio features from ASTFeatureExtractor
                         Shape: (batch, max_length, num_mel_bins)
            
        Returns:
            Pooled embeddings of shape (batch, embedding_dim)
        """
        # Get AST outputs
        # Use no_grad() instead of inference_mode() to allow backprop through classifier
        # inference_mode() creates tensors that cannot be used in autograd at all
        if self.freeze_base:
            with torch.no_grad():
                outputs = self.base_model(input_values=input_values)
            # Clone the output to detach from the no_grad context
            embeddings = outputs.pooler_output.clone()
        else:
            outputs = self.base_model(input_values=input_values)
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            input_values: Preprocessed audio features from ASTFeatureExtractor
                         Shape: (batch, max_length, num_mel_bins)
            
        Returns:
            Logits of shape (batch, 2)
        """
        embeddings = self.get_embeddings(input_values)
        logits = self.classifier(embeddings)
        return logits
    
    def predict_proba(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            input_values: Preprocessed audio features
            
        Returns:
            Probabilities of shape (batch, 2)
        """
        logits = self.forward(input_values)
        return torch.softmax(logits, dim=-1)
    
    def predict(
        self, 
        input_values: torch.Tensor, 
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with threshold.
        
        Args:
            input_values: Preprocessed audio features
            threshold: Detection threshold for positive class
            
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Binary predictions (0 or 1)
            - confidence_scores: Confidence for positive class
        """
        probs = self.predict_proba(input_values)
        confidence = probs[:, 1]  # Probability of positive class
        predictions = (confidence >= threshold).long()
        return predictions, confidence
    
    def train(self, mode: bool = True):
        """
        Set training mode.
        
        Note: Base model stays in eval mode if frozen.
        """
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self


class ASTFeatureExtractorWrapper:
    """
    Wrapper for AST feature extractor with consistent interface.
    
    This handles audio preprocessing for the AST model, converting
    raw audio waveforms to mel spectrograms in the format expected by AST.
    """
    
    def __init__(self, sampling_rate: int = 16000):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: Expected audio sampling rate (16000 Hz for AST)
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            AST_MODEL_CHECKPOINT
        )
        self.sampling_rate = sampling_rate
    
    def __call__(
        self,
        audio: torch.Tensor | list,
        sampling_rate: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> dict:
        """
        Process audio into AST input format.
        
        Args:
            audio: Raw audio waveform(s). Can be:
                   - torch.Tensor of shape (samples,) or (batch, samples)
                   - List of numpy arrays or tensors
            sampling_rate: Audio sampling rate (default: self.sampling_rate)
            return_tensors: Return format ("pt" for PyTorch)
            
        Returns:
            Dictionary with 'input_values' key containing processed features
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Process through feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
        )
        
        return inputs


def load_ast_model(
    model_path: Optional[Path] = None,
    device: str = "cpu",
    freeze_base: bool = True,
) -> ASTWakeWordModel:
    """
    Load an AST wake word model.
    
    Args:
        model_path: Path to saved classifier weights (None for fresh model)
        device: Device to load model on
        freeze_base: Whether to freeze the base AST model
        
    Returns:
        Loaded ASTWakeWordModel
    """
    model = ASTWakeWordModel(freeze_base=freeze_base)
    
    if model_path is not None and Path(model_path).exists():
        logger.info(f"Loading classifier weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load only classifier weights
        if 'classifier_state_dict' in checkpoint:
            model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Try to load full model state (backward compatibility)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    return model


def save_wake_word_model(
    model: ASTWakeWordModel,
    save_path: Path,
    wake_word: str,
    threshold: float = 0.65,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save a trained wake word model.
    
    Args:
        model: Trained ASTWakeWordModel
        save_path: Path to save the model
        wake_word: The wake word this model detects
        threshold: Recommended detection threshold
        metadata: Additional metadata to save
    """
    import json
    from datetime import datetime
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count parameters
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    checkpoint = {
        'classifier_state_dict': model.classifier.state_dict(),
        'wake_word': wake_word,
        'threshold': threshold,
        'embedding_dim': model.embedding_dim,
        'classifier_hidden_dims': model.classifier.hidden_dims,
        'classifier_dropout': getattr(model.classifier, 'dropout', 0.3),
        'use_attention': getattr(model.classifier, 'use_attention', False),
        'use_se_block': getattr(model.classifier, 'use_se_block', False),
        'use_tcn': getattr(model.classifier, 'use_tcn', False),
        'sample_rate': 16000,
        'model_version': '2.0',
        'base_model': AST_MODEL_CHECKPOINT,
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Also save metadata.json for the model listing API
    metadata_path = save_path.parent / "metadata.json"
    metadata_json = {
        'wake_word': wake_word,
        'threshold': threshold,
        'model_type': 'ast',
        'embedding_dim': model.embedding_dim,
        'classifier_hidden_dims': model.classifier.hidden_dims,
        'use_attention': getattr(model.classifier, 'use_attention', False),
        'sample_rate': 16000,
        'model_version': '2.0',
        'base_model': AST_MODEL_CHECKPOINT,
        'parameters': trainable_params,
        'total_parameters': total_params,
        'created_at': datetime.now().isoformat(),
    }
    
    if metadata:
        # Copy relevant fields to metadata.json
        if 'metrics' in metadata:
            metadata_json['metrics'] = metadata['metrics']
        if 'threshold_analysis' in metadata:
            metadata_json['threshold_analysis'] = metadata['threshold_analysis']
        if 'training_config' in metadata:
            metadata_json['training_config'] = metadata['training_config']
        if 'data_stats' in metadata:
            metadata_json['data_stats'] = metadata['data_stats']
        if 'training_time_seconds' in metadata:
            metadata_json['training_time_seconds'] = metadata['training_time_seconds']
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_json, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_info(model: ASTWakeWordModel) -> dict:
    """
    Get model information.
    
    Args:
        model: ASTWakeWordModel instance
        
    Returns:
        Dictionary with model information
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    return {
        "model_class": model.__class__.__name__,
        "base_model": AST_MODEL_CHECKPOINT,
        "embedding_dim": model.embedding_dim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "size_mb": total_params * 4 / (1024 * 1024),
        "trainable_size_mb": trainable_params * 4 / (1024 * 1024),
    }
