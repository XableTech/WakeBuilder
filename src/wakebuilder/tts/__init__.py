"""
TTS (Text-to-Speech) module for WakeBuilder.

This module provides text-to-speech functionality using Piper TTS
for generating synthetic voice samples during wake word training.
"""

from .generator import TTSGenerator, VoiceInfo, list_available_voices

__all__ = [
    "TTSGenerator",
    "VoiceInfo",
    "list_available_voices",
]
