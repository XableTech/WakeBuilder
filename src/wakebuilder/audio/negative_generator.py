"""
Negative example generation for WakeBuilder.

This module generates negative training examples - audio samples that
should NOT trigger wake word detection. These include:
- Phonetically similar words
- Random speech
- Silence
- Pure noise
"""

import random
from typing import Iterator, Optional

import numpy as np

from ..tts import TTSGenerator
from .augmentation import (
    AugmentedSample,
    NoiseLoader,
    mix_audio_with_noise,
    pad_or_trim_audio,
)


# Common English words for random speech generation
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "hello", "okay", "yes", "please",
    "thank", "sorry", "right", "left", "stop", "start", "open", "close",
]  # fmt: skip

# Phonetically similar word patterns for common wake words
PHONETIC_SIMILAR_PATTERNS: dict[str, list[str]] = {
    "computer": ["commuter", "compute", "compote", "come here", "calm down"],
    "assistant": ["assist", "insistent", "resistant", "persistent", "consistent"],
    "alexa": ["elect", "alex", "flexor", "elects", "relax"],
    "hey": ["hay", "say", "day", "way", "pay", "may", "lay"],
    "hello": ["yellow", "fellow", "mellow", "bellow", "below"],
    "listen": ["listing", "glistening", "christen", "glisten", "missing"],
    "wake": ["make", "take", "bake", "lake", "sake", "fake", "cake"],
    "voice": ["choice", "noise", "boys", "toys", "joys", "poise"],
    "system": ["sister", "mister", "blister", "twister", "whisper"],
}


def get_phonetically_similar_words(wake_word: str) -> list[str]:
    """
    Get phonetically similar words for a wake word.

    Args:
        wake_word: The wake word to find similar words for.

    Returns:
        List of phonetically similar words/phrases.
    """
    wake_word_lower = wake_word.lower().strip()
    similar_words: list[str] = []

    # Check predefined patterns
    for key, words in PHONETIC_SIMILAR_PATTERNS.items():
        if key in wake_word_lower or wake_word_lower in key:
            similar_words.extend(words)

    # Generate variations by character substitution
    substitutions = [
        ("c", "k"), ("k", "c"), ("s", "z"), ("z", "s"),
        ("f", "v"), ("v", "f"), ("i", "e"), ("e", "i"),
        ("a", "e"), ("e", "a"), ("t", "d"), ("d", "t"),
        ("p", "b"), ("b", "p"), ("m", "n"), ("n", "m"),
    ]  # fmt: skip

    for old, new in substitutions:
        if old in wake_word_lower:
            similar_words.append(wake_word_lower.replace(old, new, 1))

    # Add partial matches for multi-word wake words
    words = wake_word_lower.split()
    if len(words) == 2:
        similar_words.extend(words)

    # Remove duplicates and the original word
    similar_words = list(set(similar_words))
    if wake_word_lower in similar_words:
        similar_words.remove(wake_word_lower)

    return similar_words


def generate_random_phrases(num_phrases: int = 50) -> list[str]:
    """Generate random phrases for negative examples."""
    phrases = []
    for _ in range(num_phrases):
        length = random.randint(1, 5)
        phrase = " ".join(random.choices(COMMON_WORDS, k=length))
        phrases.append(phrase)
    return phrases


class NegativeExampleGenerator:
    """Generator for negative training examples."""

    def __init__(
        self,
        tts_generator: Optional[TTSGenerator] = None,
        noise_loader: Optional[NoiseLoader] = None,
        target_sample_rate: int = 16000,
        target_duration: float = 1.0,
    ):
        """Initialize the negative example generator."""
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_length = int(target_sample_rate * target_duration)

        self._tts: Optional[TTSGenerator] = None
        if tts_generator is not None:
            self._tts = tts_generator
        else:
            try:
                self._tts = TTSGenerator(target_sample_rate=target_sample_rate)
            except (ImportError, FileNotFoundError):
                self._tts = None

        self._noise = noise_loader if noise_loader else NoiseLoader()

    @property
    def tts_available(self) -> bool:
        """Check if TTS is available."""
        return self._tts is not None

    @property
    def noise_available(self) -> bool:
        """Check if noise samples are available."""
        return self._noise.num_samples > 0

    def generate_phonetically_similar(
        self,
        wake_word: str,
        num_voices: Optional[int] = None,
        add_noise: bool = True,
    ) -> Iterator[AugmentedSample]:
        """Generate negative examples from phonetically similar words."""
        if not self.tts_available or self._tts is None:
            return

        similar_words = get_phonetically_similar_words(wake_word)
        if not similar_words:
            similar_words = ["something", "nothing", "anything", "everything"]

        voices = self._tts.voice_names
        if num_voices is not None and num_voices < len(voices):
            voices = voices[:num_voices]

        for word in similar_words:
            for voice_name in voices:
                try:
                    audio, _ = self._tts.synthesize(word, voice_name=voice_name)
                    audio = pad_or_trim_audio(audio, self.target_length)

                    yield AugmentedSample(
                        audio=audio,
                        sample_rate=self.target_sample_rate,
                        label=0,
                        metadata={
                            "source": "phonetic_similar",
                            "text": word,
                            "voice": voice_name,
                        },
                    )

                    if add_noise and self.noise_available:
                        snr_db = random.choice([-20, -15, -10, -5])
                        noise = self._noise.get_random_noise(
                            self.target_duration, self.target_sample_rate
                        )
                        noisy = mix_audio_with_noise(audio, noise, snr_db)

                        yield AugmentedSample(
                            audio=noisy,
                            sample_rate=self.target_sample_rate,
                            label=0,
                            metadata={
                                "source": "phonetic_similar",
                                "text": word,
                                "snr_db": snr_db,
                            },
                        )
                except Exception:
                    continue

    def generate_random_speech(
        self,
        num_samples: int = 100,
        num_voices: Optional[int] = None,
        add_noise: bool = True,
    ) -> Iterator[AugmentedSample]:
        """Generate negative examples from random speech."""
        if not self.tts_available or self._tts is None:
            return

        phrases = generate_random_phrases(num_samples)

        voices = self._tts.voice_names
        if num_voices is not None and num_voices < len(voices):
            voices = voices[:num_voices]

        for phrase in phrases:
            voice_name = random.choice(voices)

            try:
                audio, _ = self._tts.synthesize(phrase, voice_name=voice_name)
                audio = pad_or_trim_audio(audio, self.target_length)

                yield AugmentedSample(
                    audio=audio,
                    sample_rate=self.target_sample_rate,
                    label=0,
                    metadata={
                        "source": "random_speech",
                        "text": phrase,
                        "voice": voice_name,
                    },
                )

                if add_noise and self.noise_available and random.random() > 0.5:
                    snr_db = random.choice([-20, -15, -10, -5])
                    noise = self._noise.get_random_noise(
                        self.target_duration, self.target_sample_rate
                    )
                    noisy = mix_audio_with_noise(audio, noise, snr_db)

                    yield AugmentedSample(
                        audio=noisy,
                        sample_rate=self.target_sample_rate,
                        label=0,
                        metadata={
                            "source": "random_speech",
                            "text": phrase,
                            "snr_db": snr_db,
                        },
                    )
            except Exception:
                continue

    def generate_silence(self, num_samples: int = 20) -> Iterator[AugmentedSample]:
        """Generate silence/near-silence negative examples."""
        for i in range(num_samples):
            noise_floor = random.uniform(0.0001, 0.005)
            audio = np.random.randn(self.target_length).astype(np.float32) * noise_floor

            yield AugmentedSample(
                audio=audio,
                sample_rate=self.target_sample_rate,
                label=0,
                metadata={"source": "silence", "noise_floor": noise_floor},
            )

    def generate_pure_noise(self, num_samples: int = 30) -> Iterator[AugmentedSample]:
        """Generate pure noise negative examples."""
        for i in range(num_samples):
            if self.noise_available:
                # Use loaded noise samples
                audio = self._noise.get_random_noise(
                    self.target_duration, self.target_sample_rate
                )
                # Random amplitude scaling
                scale = random.uniform(0.3, 0.8)
                audio = audio * scale
                source = "loaded_noise"
            else:
                # Generate synthetic noise
                noise_type = random.choice(["white", "pink", "brown"])
                audio = self._generate_synthetic_noise(noise_type)
                source = f"synthetic_{noise_type}"

            yield AugmentedSample(
                audio=audio,
                sample_rate=self.target_sample_rate,
                label=0,
                metadata={"source": source, "sample_idx": i},
            )

    def _generate_synthetic_noise(self, noise_type: str) -> np.ndarray:
        """Generate synthetic noise of specified type."""
        if noise_type == "white":
            audio = np.random.randn(self.target_length).astype(np.float32)
        elif noise_type == "pink":
            # Simple pink noise via filtering
            white = np.random.randn(self.target_length)
            # Apply simple 1/f filter approximation
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            from scipy.signal import lfilter

            audio = lfilter(b, a, white).astype(np.float32)
        elif noise_type == "brown":
            # Brown noise via cumulative sum
            white = np.random.randn(self.target_length)
            audio = np.cumsum(white).astype(np.float32)
        else:
            audio = np.random.randn(self.target_length).astype(np.float32)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * random.uniform(0.3, 0.7)

        return audio

    def generate_all_negatives(
        self,
        wake_word: str,
        num_random_speech: int = 50,
        num_silence: int = 10,
        num_noise: int = 20,
        num_voices: Optional[int] = 3,
    ) -> Iterator[AugmentedSample]:
        """
        Generate all types of negative examples.

        Args:
            wake_word: Wake word for phonetically similar generation.
            num_random_speech: Number of random speech samples.
            num_silence: Number of silence samples.
            num_noise: Number of pure noise samples.
            num_voices: Number of TTS voices to use.

        Yields:
            AugmentedSample objects with label=0.
        """
        # Phonetically similar
        yield from self.generate_phonetically_similar(
            wake_word, num_voices=num_voices, add_noise=True
        )

        # Random speech
        yield from self.generate_random_speech(
            num_samples=num_random_speech, num_voices=num_voices, add_noise=True
        )

        # Silence
        yield from self.generate_silence(num_samples=num_silence)

        # Pure noise
        yield from self.generate_pure_noise(num_samples=num_noise)
