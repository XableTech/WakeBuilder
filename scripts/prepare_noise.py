#!/usr/bin/env python3
"""
Prepare background noise samples for WakeBuilder data augmentation.

This script generates and downloads various noise samples that will be
mixed with wake word audio during training to improve model robustness.

Noise types include:
- Synthetic noise (white, pink, brown)
- Environmental sounds (downloaded from free sources)
- Silence samples
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.config import Config

# Sample rate for all noise files
SAMPLE_RATE = 16000

# Duration of each noise sample in seconds
NOISE_DURATION = 10.0

# Free sound URLs for environmental noise (public domain / CC0)
# Using Freesound.org API-free samples and other free sources
ENVIRONMENTAL_NOISE_URLS: list[str] = [
    # Note: These are placeholder URLs - in production, you would use
    # actual free sound sources or generate synthetic environmental noise
]


def generate_white_noise(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate white noise.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude scaling (0-1).

    Returns:
        Numpy array of noise samples.
    """
    n_samples = int(duration * sample_rate)
    noise = np.random.randn(n_samples).astype(np.float32)
    noise = noise / np.abs(noise).max() * amplitude
    return noise


def generate_pink_noise(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate pink noise (1/f noise).

    Pink noise has equal energy per octave, making it more natural sounding.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude scaling (0-1).

    Returns:
        Numpy array of noise samples.
    """
    n_samples = int(duration * sample_rate)

    # Apply 1/f filter using Voss-McCartney algorithm
    # This is a simplified version
    n_rows = 16
    n_cols = n_samples // n_rows + 1

    # Create random values
    array = np.random.randn(n_rows, n_cols)

    # Cumulative sum along rows
    pink = np.cumsum(array, axis=1)

    # Reshape and trim
    pink = pink.flatten()[:n_samples]

    # Normalize
    pink = pink - pink.mean()
    pink = pink / np.abs(pink).max() * amplitude

    return pink.astype(np.float32)


def generate_brown_noise(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate brown (Brownian) noise.

    Brown noise has more energy at lower frequencies.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude scaling (0-1).

    Returns:
        Numpy array of noise samples.
    """
    n_samples = int(duration * sample_rate)

    # Generate white noise
    white = np.random.randn(n_samples)

    # Integrate (cumulative sum) to get brown noise
    brown = np.cumsum(white)

    # Normalize
    brown = brown - brown.mean()
    brown = brown / np.abs(brown).max() * amplitude

    return brown.astype(np.float32)


def generate_silence(
    duration: float,
    sample_rate: int,
    noise_floor: float = 0.001,
) -> np.ndarray:
    """
    Generate near-silence with minimal noise floor.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        noise_floor: Very low amplitude noise to simulate real silence.

    Returns:
        Numpy array of near-silent samples.
    """
    n_samples = int(duration * sample_rate)
    silence = np.random.randn(n_samples).astype(np.float32) * noise_floor
    return silence


def generate_office_ambience(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.3,
) -> np.ndarray:
    """
    Generate synthetic office ambience noise.

    Combines low-frequency rumble with occasional higher frequency components.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude scaling (0-1).

    Returns:
        Numpy array of noise samples.
    """
    n_samples = int(duration * sample_rate)

    # Base: low-frequency rumble (HVAC-like)
    t = np.linspace(0, duration, n_samples)
    rumble = np.sin(2 * np.pi * 60 * t) * 0.3  # 60 Hz hum
    rumble += np.sin(2 * np.pi * 120 * t) * 0.15  # 120 Hz harmonic

    # Add some pink noise for texture
    pink = generate_pink_noise(duration, sample_rate, 0.2)

    # Combine
    office = rumble + pink

    # Normalize
    office = office / np.abs(office).max() * amplitude

    return office.astype(np.float32)


def generate_street_ambience(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.4,
) -> np.ndarray:
    """
    Generate synthetic street ambience noise.

    Combines traffic-like rumble with occasional transients.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude scaling (0-1).

    Returns:
        Numpy array of noise samples.
    """
    n_samples = int(duration * sample_rate)

    # Base: brown noise for traffic rumble
    brown = generate_brown_noise(duration, sample_rate, 0.5)

    # Add some white noise for high-frequency content
    white = generate_white_noise(duration, sample_rate, 0.1)

    # Add occasional "car pass" events (amplitude modulation)
    n_events = int(duration / 2)  # One event every 2 seconds on average
    events = np.zeros(n_samples)

    for _ in range(n_events):
        # Random position and width
        pos = np.random.randint(0, n_samples)
        width = int(sample_rate * np.random.uniform(0.5, 2.0))

        # Gaussian envelope
        start = max(0, pos - width // 2)
        end = min(n_samples, pos + width // 2)
        x = np.linspace(-2, 2, end - start)
        envelope = np.exp(-(x**2))

        events[start:end] += envelope * np.random.uniform(0.3, 0.8)

    # Combine
    street = brown + white + events * brown

    # Normalize
    street = street / np.abs(street).max() * amplitude

    return street.astype(np.float32)


def generate_cafe_ambience(
    duration: float,
    sample_rate: int,
    amplitude: float = 0.35,
) -> np.ndarray:
    """
    Generate synthetic cafe/restaurant ambience.

    Simulates background chatter and clinking sounds.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude scaling (0-1).

    Returns:
        Numpy array of noise samples.
    """
    n_samples = int(duration * sample_rate)

    # Base: pink noise for general ambience
    pink = generate_pink_noise(duration, sample_rate, 0.4)

    # Add babble-like modulation
    t = np.linspace(0, duration, n_samples)

    # Multiple slow modulations to simulate conversation rhythm
    mod1 = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi)
    mod2 = 0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t + np.random.rand() * 2 * np.pi)
    mod3 = 0.5 + 0.5 * np.sin(2 * np.pi * 1.2 * t + np.random.rand() * 2 * np.pi)

    modulation = (mod1 + mod2 + mod3) / 3

    # Apply modulation
    cafe = pink * modulation

    # Add occasional clink sounds (high-frequency transients)
    n_clinks = int(duration * 0.5)  # 0.5 clinks per second
    for _ in range(n_clinks):
        pos = np.random.randint(0, n_samples)
        # Short high-frequency burst
        clink_len = int(sample_rate * 0.05)
        if pos + clink_len < n_samples:
            clink = np.random.randn(clink_len) * np.exp(-np.linspace(0, 5, clink_len))
            cafe[pos : pos + clink_len] += clink * 0.2

    # Normalize
    cafe = cafe / np.abs(cafe).max() * amplitude

    return cafe.astype(np.float32)


def save_noise_sample(
    audio: np.ndarray,
    output_path: Path,
    sample_rate: int,
) -> None:
    """
    Save a noise sample to a WAV file.

    Args:
        audio: Audio samples.
        output_path: Path to save the file.
        sample_rate: Sample rate in Hz.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)


def create_noise_index(noise_dir: Path) -> None:
    """
    Create an index file listing all available noise samples.

    Args:
        noise_dir: Directory containing noise files.
    """
    samples = []

    for wav_file in sorted(noise_dir.glob("*.wav")):
        try:
            info = sf.info(wav_file)
            sample_info = {
                "name": wav_file.stem,
                "file": wav_file.name,
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
            }
            samples.append(sample_info)
        except Exception:
            continue

    # Save index
    index_path = noise_dir / "noise_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": samples,
                "count": len(samples),
            },
            f,
            indent=2,
        )

    print(f"\n[OK] Created noise index: {index_path.name}")
    print(f"     Total samples available: {len(samples)}")


def main() -> int:
    """Main execution function."""
    print("=" * 70)
    print("WakeBuilder - Noise Sample Preparation Script")
    print("=" * 70)

    # Configuration
    config = Config()
    noise_dir = Path(config.DATA_DIR) / "noise"

    print(f"\nNoise directory: {noise_dir}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration per sample: {NOISE_DURATION} seconds")

    # Ensure directory exists
    noise_dir.mkdir(parents=True, exist_ok=True)

    # Define noise types to generate
    noise_generators = [
        ("white_noise", generate_white_noise),
        ("pink_noise", generate_pink_noise),
        ("brown_noise", generate_brown_noise),
        ("silence", lambda d, sr, a=0.001: generate_silence(d, sr, a)),
        ("office_ambience", generate_office_ambience),
        ("street_ambience", generate_street_ambience),
        ("cafe_ambience", generate_cafe_ambience),
    ]

    # Generate multiple variations of each noise type
    variations_per_type = 3
    successful = 0
    failed = 0

    print(
        f"\nGenerating {len(noise_generators) * variations_per_type} noise samples..."
    )

    for noise_name, generator in noise_generators:
        print(f"\n  Generating: {noise_name}")

        for i in range(variations_per_type):
            try:
                # Generate noise with slight amplitude variation
                amplitude = 0.3 + np.random.rand() * 0.4  # 0.3 to 0.7

                if noise_name == "silence":
                    audio = generator(NOISE_DURATION, SAMPLE_RATE)
                else:
                    audio = generator(NOISE_DURATION, SAMPLE_RATE, amplitude)

                # Save
                filename = f"{noise_name}_{i + 1:02d}.wav"
                output_path = noise_dir / filename
                save_noise_sample(audio, output_path, SAMPLE_RATE)

                print(f"    [OK] {filename}")
                successful += 1

            except Exception as e:
                print(f"    [ERROR] {noise_name}_{i + 1:02d}: {e}")
                failed += 1

    # Create index
    print("\n" + "=" * 70)
    print("Creating noise index...")
    create_noise_index(noise_dir)

    # Summary
    print("\n" + "=" * 70)
    print("Generation Summary")
    print("=" * 70)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")

    if failed > 0:
        print("\n[WARN] Some noise samples failed to generate.")
        return 1

    print("\n[OK] All noise samples generated successfully!")
    print(f"\nNoise samples are stored in: {noise_dir}")
    print("\nNext steps:")
    print("  1. Implement data augmentation module")
    print("  2. Test noise mixing with: uv run pytest tests/test_augmentation.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
