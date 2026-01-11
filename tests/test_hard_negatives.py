#!/usr/bin/env python3
"""
Test script to verify hard negative generation for wake words.

This script tests the phonetically similar word generation to ensure
it produces the critical hard negatives needed to prevent false positives.

The algorithm is purely based on the wake word itself - no hardcoded names.
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from wakebuilder.audio.negative_generator import get_phonetically_similar_words


def test_wake_word(wake_word: str) -> None:
    """Test hard negative generation for a wake word."""
    print(f"\n{'='*60}")
    print(f"Testing wake word: '{wake_word}'")
    print(f"{'='*60}")

    similar = get_phonetically_similar_words(wake_word)
    wake_lower = wake_word.lower()

    print(
        f"\nGenerated {len(similar)} similar words/phrases (all derived from '{wake_word}'):"
    )

    # Categorize by type
    prefixes = []
    suffixes = []
    edits = []
    phonetic = []

    for w in similar:
        # Check if it's a prefix of wake word
        if wake_lower.startswith(w) or any(
            wake_lower.startswith(w[:i]) for i in range(2, len(w))
        ):
            prefixes.append(w)
        # Check if it's a suffix
        elif wake_lower.endswith(w) or w in wake_lower:
            suffixes.append(w)
        # Check if it's an edit (similar length, few char differences)
        elif abs(len(w) - len(wake_lower)) <= 2:
            edits.append(w)
        else:
            phonetic.append(w)

    print(f"\n  CRITICAL - Prefixes + extensions ({len(prefixes)}):")
    for w in prefixes[:12]:
        print(f"    - {w}")
    if len(prefixes) > 12:
        print(f"    ... and {len(prefixes) - 12} more")

    print(f"\n  HIGH - Suffixes/substrings ({len(suffixes)}):")
    for w in suffixes[:8]:
        print(f"    - {w}")
    if len(suffixes) > 8:
        print(f"    ... and {len(suffixes) - 8} more")

    print(f"\n  HIGH - Edit variations ({len(edits)}):")
    for w in edits[:8]:
        print(f"    - {w}")
    if len(edits) > 8:
        print(f"    ... and {len(edits) - 8} more")

    print(f"\n  MEDIUM - Phonetic variations ({len(phonetic)}):")
    for w in phonetic[:8]:
        print(f"    - {w}")
    if len(phonetic) > 8:
        print(f"    ... and {len(phonetic) - 8} more")

    # Verify key properties
    print("\n  Verification:")

    # All prefixes should be present
    for i in range(2, len(wake_lower)):
        prefix = wake_lower[:i]
        found = prefix in similar
        status = "[OK]" if found else "[MISSING!]"
        if not found or i <= 4:  # Show first few or missing ones
            print(f"    {status} prefix '{prefix}'")


def main():
    # Test various wake words
    test_words = [
        "jarvis",
        "alexa",
        "computer",
        "hey siri",
        "phoenix",
        "hey google",
    ]

    for word in test_words:
        test_wake_word(word)

    print("\n" + "=" * 60)
    print("Hard negative generation test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
