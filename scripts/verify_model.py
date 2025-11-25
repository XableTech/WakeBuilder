#!/usr/bin/env python3
"""Verify exported model can be loaded and used."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from wakebuilder.models import create_model, get_model_info


def main():
    model_dir = Path("models/hi_alexa")

    # Load metadata
    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)

    print("Model Metadata:")
    print(f"  Wake word: {meta['wake_word']}")
    print(f"  Model type: {meta['model_type']}")
    print(f"  Threshold: {meta['threshold']:.3f}")
    print(f"  Parameters: {meta['parameters']:,}")
    print(f"  Val accuracy: {meta['metrics']['val_accuracy']:.2%}")
    print(f"  Val F1: {meta['metrics']['val_f1']:.3f}")

    # Load model
    ckpt = torch.load(model_dir / "model.pt", weights_only=True)
    model = create_model(ckpt["model_type"], n_mels=ckpt["n_mels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    info = get_model_info(model)
    print(f"\nModel loaded successfully!")
    print(f"  Class: {info['model_class']}")
    print(f"  Size: {info['size_mb']:.2f} MB")

    # Test inference
    dummy_input = torch.randn(1, 96, 80)
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)

    print(f"\nTest inference:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Wake word probability: {probs[0, 1].item():.3f}")


if __name__ == "__main__":
    main()
