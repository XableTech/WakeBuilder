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


# Common English words for random speech generation - expanded for better coverage
COMMON_WORDS = [
    # Basic words
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
    # Common commands and phrases (likely to cause false positives)
    "hey", "hi", "play", "pause", "next", "volume", "music", "call",
    "send", "message", "set", "timer", "alarm", "remind", "weather",
    "lights", "turn", "off", "on", "help", "search", "find", "show",
    "tell", "ask", "what's", "where", "when", "why", "how", "who",
    # Names and greetings
    "sam", "alex", "max", "hey there", "hi there", "good morning",
    "good night", "goodbye", "see you", "thanks", "welcome",
    # Filler words and sounds
    "um", "uh", "hmm", "ah", "oh", "well", "so", "like", "you know",
    "actually", "basically", "literally", "seriously", "really",
    # Conversational phrases that might trigger false positives
    "I think", "you know what", "let me", "can you", "could you",
    "would you", "should I", "what if", "how about", "let's go",
    "come on", "wait a minute", "hold on", "never mind", "forget it",
    "that's right", "exactly", "absolutely", "definitely", "probably",
    "maybe", "perhaps", "certainly", "of course", "sure thing",
    # Background chatter simulation
    "did you hear", "I was thinking", "the other day", "you see",
    "anyway", "by the way", "speaking of", "as I was saying",
    "to be honest", "in my opinion", "I believe", "it seems like",
    # Exclamations and loud phrases (critical for preventing false positives on loud speech)
    "wow", "whoa", "yes", "no", "stop", "go", "wait", "look",
    "watch out", "be careful", "hurry up", "come here", "over here",
    "what the", "oh my god", "oh no", "oh yes", "oh wow",
    "are you kidding", "no way", "get out", "shut up", "come on",
    "let's go", "right now", "do it", "got it", "I got it",
    "hello there", "hey you", "excuse me", "pardon me", "sorry",
    # Short utterances that might be confused
    "huh", "what", "yeah", "nope", "yep", "nah", "meh", "ugh",
    "oops", "ouch", "yay", "boo", "shh", "psst", "hey hey",
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
    
    CRITICAL: This function generates "hard negatives" - words that sound
    similar to the wake word but should NOT trigger detection. These are
    essential for teaching the model to discriminate between the exact
    wake word and similar-sounding words.
    
    This is a purely algorithmic approach - NO hardcoded names or words.
    All variations are derived from the wake word itself.
    
    For multi-word wake words (e.g., "hey jarvis"), this also generates:
    - Individual words alone ("hey", "jarvis") - CRITICAL for preventing
      partial matches from triggering detection
    - Words with common prefixes ("hi jarvis", "hey jarvie")
    - Swapped/missing word combinations

    Args:
        wake_word: The wake word to find similar words for.

    Returns:
        List of phonetically similar words/phrases (sorted by importance).
    """
    wake_word_lower = wake_word.lower().strip()
    
    # Use lists to preserve priority order, sets for deduplication
    partial_words: list[str] = []         # Individual words from multi-word wake words - HIGHEST priority
    pure_prefixes: list[str] = []         # Exact prefixes (sa, sam, sami) - HIGHEST priority
    prefix_extensions: list[str] = []     # Prefixes with endings (sama, sami, samy)
    high_priority: list[str] = []         # Suffixes, edits
    medium_priority: list[str] = []       # Phonetic variations
    seen: set[str] = set()                # For deduplication
    
    # =========================================================================
    # CRITICAL: Multi-word wake word handling (e.g., "hey jarvis")
    # The model must learn to REJECT individual words and only accept the
    # complete phrase. This is the #1 cause of false positives for multi-word
    # wake words!
    # =========================================================================
    words = wake_word_lower.split()
    if len(words) >= 2:
        # Add each individual word as a hard negative (HIGHEST priority)
        # e.g., for "open sesame": add "open" and "sesame" as negatives
        for word in words:
            if word not in seen and len(word) >= 2:
                partial_words.append(word)
                seen.add(word)
        
        # Generate phonetic variations for each word position
        # This is purely algorithmic - no hardcoded word lists
        for word_idx, word in enumerate(words):
            other_words = words[:word_idx] + words[word_idx+1:]
            
            if len(word) >= 2:
                # 1. Prefixes of this word (e.g., "ope", "open" for "open")
                for i in range(2, len(word)):
                    prefix = word[:i]
                    if word_idx == 0:
                        phrase = f"{prefix} {' '.join(other_words)}"
                    else:
                        phrase = f"{' '.join(words[:word_idx])} {prefix}"
                        if word_idx < len(words) - 1:
                            phrase += f" {' '.join(words[word_idx+1:])}"
                    if phrase not in seen:
                        partial_words.append(phrase)
                        seen.add(phrase)
                
                # 2. Single character substitutions at end (phonetic variations)
                vowels = "aeiou"
                consonants = "bcdfghjklmnpqrstvwxyz"
                
                # Replace last character
                for char in vowels + "y":
                    if char != word[-1]:
                        modified = word[:-1] + char
                        if word_idx == 0:
                            phrase = f"{modified} {' '.join(other_words)}"
                        else:
                            phrase = f"{' '.join(words[:word_idx])} {modified}"
                            if word_idx < len(words) - 1:
                                phrase += f" {' '.join(words[word_idx+1:])}"
                        if phrase not in seen and phrase != wake_word_lower:
                            partial_words.append(phrase)
                            seen.add(phrase)
                
                # 3. Common ending substitutions
                if len(word) > 3:
                    for ending in ["ie", "y", "a", "o", "er", "is", "us", "en", "on"]:
                        modified = word[:-1] + ending
                        if modified != word:
                            if word_idx == 0:
                                phrase = f"{modified} {' '.join(other_words)}"
                            else:
                                phrase = f"{' '.join(words[:word_idx])} {modified}"
                                if word_idx < len(words) - 1:
                                    phrase += f" {' '.join(words[word_idx+1:])}"
                            if phrase not in seen and phrase != wake_word_lower:
                                partial_words.append(phrase)
                                seen.add(phrase)
        
        # Add reversed/swapped word order
        if len(words) == 2:
            swapped = f"{words[1]} {words[0]}"
            if swapped not in seen:
                partial_words.append(swapped)
                seen.add(swapped)
        
        # Add just the last word repeated (common confusion)
        if len(words) == 2:
            repeated = f"{words[1]} {words[1]}"
            if repeated not in seen:
                partial_words.append(repeated)
                seen.add(repeated)
    
    # =========================================================================
    # CRITICAL: Prefix matches (e.g., "jarv" from "jarvis")
    # These cause the most false positives!
    # Pure prefixes MUST come first in the list!
    # =========================================================================
    if len(wake_word_lower) > 2:
        # First: Add PURE prefixes in order (shortest to longest)
        # These are the most critical - "ja", "jar", "jarv" for "jarvis"
        for i in range(2, len(wake_word_lower)):
            prefix = wake_word_lower[:i]
            if prefix not in seen:
                pure_prefixes.append(prefix)
                seen.add(prefix)
        
        # Second: Add prefix extensions (prefixes + common endings)
        word_endings = ["a", "e", "i", "o", "y", "er", "ie", "ey", "ay", 
                       "an", "en", "in", "on", "ar", "or", "ir",
                       "la", "ra", "na", "ta", "da", "ma",
                       "ly", "ry", "ny", "ty", "dy", "my"]
        for i in range(2, len(wake_word_lower)):
            prefix = wake_word_lower[:i]
            for ending in word_endings:
                extended = prefix + ending
                if extended not in seen and extended != wake_word_lower:
                    prefix_extensions.append(extended)
                    seen.add(extended)
    
    # =========================================================================
    # HIGH PRIORITY: Suffix matches and edit-distance-1 variations
    # =========================================================================
    if len(wake_word_lower) > 2:
        # All suffixes (e.g., "rvis" from "jarvis")
        for i in range(1, len(wake_word_lower) - 1):
            suffix = wake_word_lower[i:]
            if suffix not in seen:
                high_priority.append(suffix)
                seen.add(suffix)
        
        # Single character deletions (e.g., "javis" from "jarvis")
        for i in range(len(wake_word_lower)):
            deleted = wake_word_lower[:i] + wake_word_lower[i+1:]
            if deleted not in seen and deleted != wake_word_lower:
                high_priority.append(deleted)
                seen.add(deleted)
        
        # Single character duplications (e.g., "jarrvis" from "jarvis")
        for i in range(len(wake_word_lower)):
            doubled = wake_word_lower[:i] + wake_word_lower[i] + wake_word_lower[i:]
            if doubled not in seen:
                high_priority.append(doubled)
                seen.add(doubled)
        
        # Adjacent character swaps (e.g., "ajrvis" from "jarvis")
        for i in range(len(wake_word_lower) - 1):
            swapped = wake_word_lower[:i] + wake_word_lower[i+1] + wake_word_lower[i] + wake_word_lower[i+2:]
            if swapped not in seen:
                high_priority.append(swapped)
                seen.add(swapped)
        
        # Single character insertions at each position
        vowels = "aeiou"
        for i in range(len(wake_word_lower) + 1):
            for char in vowels:
                inserted = wake_word_lower[:i] + char + wake_word_lower[i:]
                if inserted not in seen:
                    high_priority.append(inserted)
                    seen.add(inserted)
    
    # =========================================================================
    # HIGH PRIORITY: Systematic character substitutions for last N characters
    # This catches words like "jarvey" vs "jarvis" (same prefix, different ending)
    # Pure algorithmic approach: no hardcoded lists
    # 
    # IMPORTANT: Exclude endings that contain the original last char or last 2 chars
    # e.g., for "jarvis": exclude "jarvies" (contains s), "jarviis" (contains is)
    # =========================================================================
    
    # Helper function to generate substitutions for a single word
    def generate_ending_substitutions(word: str, result_list: list, seen_set: set):
        if len(word) < 3:
            return
            
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        
        # Original endings to exclude from generated words
        original_last_char = word[-1]
        original_last_2chars = word[-2:] if len(word) >= 2 else ""
        
        def is_valid_new_word(new_word: str) -> bool:
            """Check that new word's ending doesn't contain original last char or last 2 chars."""
            # Get the portion that was substituted (compare to original)
            # For safety, check the entire new ending section
            check_len = min(4, len(new_word))  # Check last 4 chars of new word
            new_ending = new_word[-check_len:] if len(new_word) >= check_len else new_word
            
            # Exclude if new ending contains the original last character
            if original_last_char in new_ending:
                return False
            # Exclude if new ending contains the original last 2 consecutive characters
            if len(original_last_2chars) == 2 and original_last_2chars in new_ending:
                return False
            return True
        
        # Number of trailing characters to substitute (up to 3, or word length - 2)
        num_trailing = min(3, len(word) - 2)
        
        # Single character substitutions for each of the last N positions
        for pos_from_end in range(1, num_trailing + 1):
            pos = len(word) - pos_from_end
            original_char = word[pos]
            
            for new_char in alphabet:
                if new_char != original_char:
                    substituted = word[:pos] + new_char + word[pos+1:]
                    if is_valid_new_word(substituted):
                        if substituted not in seen_set and substituted != word:
                            result_list.append(substituted)
                            seen_set.add(substituted)
        
        # Two-character ending substitutions (last 2 chars replaced)
        if len(word) >= 4:
            prefix = word[:-2]
            
            def add_2char_ending(ending: str):
                new_word = prefix + ending
                if is_valid_new_word(new_word):
                    if new_word not in seen_set and new_word != word:
                        result_list.append(new_word)
                        seen_set.add(new_word)
            
            # Pattern 1: consonant + vowel (e.g., "ra", "la", "na")
            for c in consonants:
                for v in vowels:
                    add_2char_ending(c + v)
            
            # Pattern 2: vowel + consonant (e.g., "ar", "er", "in")
            for v in vowels:
                for c in consonants:
                    add_2char_ending(v + c)
            
            # Pattern 3: vowel + vowel (e.g., "ia", "ea", "io")
            for v1 in vowels:
                for v2 in vowels:
                    add_2char_ending(v1 + v2)
        
        # Three-character ending substitutions (last 3 chars replaced)
        if len(word) >= 5:
            prefix = word[:-3]
            
            def add_3char_ending(ending: str):
                new_word = prefix + ending
                if is_valid_new_word(new_word):
                    if new_word not in seen_set and new_word != word:
                        result_list.append(new_word)
                        seen_set.add(new_word)
            
            # Pattern: vowel + consonant + vowel (e.g., "ira", "ara", "ona")
            for v1 in vowels:
                for c in consonants:
                    for v2 in vowels:
                        add_3char_ending(v1 + c + v2)
            
            # Pattern: consonant + vowel + consonant (e.g., "son", "man", "ler")
            for c1 in consonants:
                for v in vowels:
                    for c2 in consonants:
                        add_3char_ending(c1 + v + c2)
    
    # Apply to single word or each word in multi-word wake word
    words_to_process = wake_word_lower.split()
    if len(words_to_process) == 1:
        # Single word wake word
        generate_ending_substitutions(wake_word_lower, high_priority, seen)
    else:
        # Multi-word wake word (e.g., "hey siri")
        # CRITICAL: Add each individual word as a high-priority negative
        # This ensures the model learns to reject partial matches like just "hey" or just "siri"
        for word in words_to_process:
            if word not in seen and len(word) > 1:
                # Add the word itself as a critical negative (highest priority)
                pure_prefixes.append(word)
                seen.add(word)
        
        # Generate substitutions for each word separately
        for i, word in enumerate(words_to_process):
            if len(word) >= 3:
                # Generate substitutions for this word
                word_subs: list[str] = []
                word_seen: set[str] = set()
                generate_ending_substitutions(word, word_subs, word_seen)
                
                # Reconstruct full phrase with substituted word
                for sub_word in word_subs:
                    new_phrase_words = words_to_process.copy()
                    new_phrase_words[i] = sub_word
                    new_phrase = " ".join(new_phrase_words)
                    if new_phrase not in seen and new_phrase != wake_word_lower:
                        high_priority.append(new_phrase)
                        seen.add(new_phrase)
                
                # Also add the substituted word alone as a negative
                for sub_word in word_subs:
                    if sub_word not in seen:
                        high_priority.append(sub_word)
                        seen.add(sub_word)

    # =========================================================================
    # MEDIUM PRIORITY: Phonetic substitutions
    # These are sound-alike character replacements
    # =========================================================================
    phonetic_substitutions = [
        # Consonant confusions
        ("c", "k"), ("k", "c"), ("s", "z"), ("z", "s"),
        ("f", "v"), ("v", "f"), ("t", "d"), ("d", "t"),
        ("p", "b"), ("b", "p"), ("m", "n"), ("n", "m"),
        ("g", "k"), ("k", "g"), ("j", "g"), ("g", "j"),
        # Vowel confusions
        ("i", "e"), ("e", "i"), ("a", "e"), ("e", "a"),
        ("a", "o"), ("o", "a"), ("u", "o"), ("o", "u"),
        ("i", "y"), ("y", "i"), ("e", "y"), ("y", "e"),
        # Digraph confusions
        ("th", "t"), ("t", "th"), ("sh", "s"), ("s", "sh"),
        ("ch", "k"), ("k", "ch"), ("ch", "sh"), ("sh", "ch"),
        ("ck", "k"), ("k", "ck"), ("ck", "c"), ("c", "ck"),
        ("ph", "f"), ("f", "ph"), ("gh", "f"), ("f", "gh"),
        # X sound variations
        ("x", "ks"), ("ks", "x"), ("x", "cks"), ("cks", "x"),
        ("x", "cs"), ("cs", "x"), ("x", "z"), ("z", "x"),
        # Common ending confusions
        ("ix", "icks"), ("ix", "ics"), ("ix", "ik"), ("ix", "ic"),
        ("er", "or"), ("or", "er"), ("er", "ar"), ("ar", "er"),
        ("le", "el"), ("el", "le"), ("re", "er"), ("er", "re"),
    ]

    for old, new in phonetic_substitutions:
        if old in wake_word_lower:
            # Replace first occurrence
            replaced = wake_word_lower.replace(old, new, 1)
            if replaced not in seen and replaced != wake_word_lower:
                medium_priority.append(replaced)
                seen.add(replaced)
            # Replace all occurrences
            replaced_all = wake_word_lower.replace(old, new)
            if replaced_all not in seen and replaced_all != wake_word_lower:
                medium_priority.append(replaced_all)
                seen.add(replaced_all)

    # =========================================================================
    # Handle multi-word wake words (e.g., "hey siri")
    # =========================================================================
    words = wake_word_lower.split()
    if len(words) == 2:
        # Each word alone is a critical negative - add to pure_prefixes for priority
        for word in words:
            if word not in seen and len(word) > 1:
                pure_prefixes.append(word)
                seen.add(word)
        # Common filler words between
        for filler in ["and", "or", "the", "a", "to", "for"]:
            phrase = f"{words[0]} {filler} {words[1]}"
            if phrase not in seen:
                medium_priority.append(phrase)
                seen.add(phrase)
    
    # =========================================================================
    # Add common short words that often cause false positives
    # These are words people say frequently that might trigger detection
    # =========================================================================
    common_short_words = [
        # Affirmations/responses
        "ok", "okay", "yes", "no", "yeah", "yep", "nope", "sure", "right",
        # Greetings
        "hi", "hey", "hello", "bye", "yo",
        # Filler words
        "um", "uh", "ah", "oh", "hmm", "huh", "wow", "ow", "ooh", "aah",
        # Common short words
        "so", "go", "do", "see", "be", "me", "we", "he", "she",
        "it", "is", "as", "at", "to", "or", "if", "up", "on", "in",
        # Question words
        "what", "who", "why", "how", "when", "where",
        # Common verbs
        "stop", "start", "wait", "look", "come", "here", "there",
    ]
    
    for word in common_short_words:
        if word not in seen and word != wake_word_lower:
            medium_priority.append(word)
            seen.add(word)
    
    # =========================================================================
    # Add common speech patterns around the wake word
    # =========================================================================
    speech_prefixes = ["hey ", "hi ", "oh ", "say ", "the ", "a "]
    speech_suffixes = [" please", " now", " here", "s", "'s", "ed", "ing"]
    
    for prefix in speech_prefixes:
        phrase = prefix + wake_word_lower
        if phrase not in seen:
            medium_priority.append(phrase)
            seen.add(phrase)
    for suffix in speech_suffixes:
        phrase = wake_word_lower + suffix
        if phrase not in seen:
            medium_priority.append(phrase)
            seen.add(phrase)

    # =========================================================================
    # Combine in strict priority order
    # Order: partial_words > pure_prefixes > prefix_extensions > high_priority > medium_priority
    # =========================================================================
    all_words = []
    
    # 0. Partial words first (for multi-word wake words) - MOST CRITICAL
    # e.g., "hey" and "jarvis" alone for "hey jarvis"
    for word in partial_words:
        if word and len(word) > 1 and word != wake_word_lower:
            all_words.append(word)
    
    # 1. Pure prefixes (sa, sam, sami) - VERY CRITICAL
    for word in pure_prefixes:
        if word and len(word) > 1 and word != wake_word_lower:
            all_words.append(word)
    
    # 2. Prefix extensions (sama, sami, samy, etc.)
    for word in prefix_extensions:
        if word and len(word) > 1 and word != wake_word_lower:
            all_words.append(word)
    
    # 3. High priority (suffixes, edits)
    for word in high_priority:
        if word and len(word) > 1 and word != wake_word_lower:
            all_words.append(word)
    
    # 4. Medium priority (phonetic variations)
    for word in medium_priority:
        if word and len(word) > 1 and word != wake_word_lower:
            all_words.append(word)
    
    # =========================================================================
    # Final filter: Remove words where the SUBSTITUTED ending contains the
    # original last char or last 2 consecutive chars
    # This prevents words like "samiex" (contains x) or "samiix" (contains ix)
    # Handle both single-word and multi-word wake words
    # =========================================================================
    def get_forbidden_patterns(word: str) -> tuple[str, str, int]:
        """Get the last char, last 2 chars, and word length."""
        return (word[-1], word[-2:] if len(word) >= 2 else "", len(word))
    
    def has_forbidden_ending(candidate: str, original_word: str, last_char: str, last_2chars: str) -> bool:
        """
        Check if candidate's ending contains forbidden patterns.
        
        Rules:
        1. If candidate is a pure prefix of original (shorter), keep it
        2. If candidate ends with the original last char, exclude it
        3. If candidate ends with the original last 2 chars, exclude it
        """
        # If candidate is shorter than original, it's likely a prefix - keep it
        if len(candidate) < len(original_word) - 1:
            return False
        
        # Check the last 2 characters of the candidate
        candidate_last_2 = candidate[-2:] if len(candidate) >= 2 else candidate
        candidate_last_1 = candidate[-1] if candidate else ""
        
        # Exclude if candidate ends with the original last character
        if candidate_last_1 == last_char:
            return True
        
        # Exclude if candidate's last 2 chars contain the original last 2 chars
        if len(last_2chars) == 2 and last_2chars in candidate_last_2:
            return True
        
        # Also check if the original last char appears in the last 2 positions
        # This catches cases like "samiex" where x is in position -2
        if last_char in candidate_last_2:
            return True
            
        return False
    
    # Get forbidden patterns from wake word (handle multi-word)
    wake_words = wake_word_lower.split()
    forbidden_patterns = [get_forbidden_patterns(w) for w in wake_words]
    
    # Filter the results
    filtered_words = []
    for candidate in all_words:
        # For multi-word candidates, check each word part
        candidate_parts = candidate.split()
        
        # Only apply strict filtering to single-word wake words
        # or to the corresponding parts of multi-word wake words
        if len(wake_words) == 1:
            # Single word wake word - apply filter to all candidates
            last_char, last_2chars, orig_len = forbidden_patterns[0]
            original_word = wake_words[0]
            
            if has_forbidden_ending(candidate, original_word, last_char, last_2chars):
                continue
        else:
            # Multi-word wake word - only filter if candidate has same structure
            # and the corresponding part has forbidden ending
            should_exclude = False
            
            # CRITICAL: Never filter out the exact individual words from the wake word
            # These are essential negatives to prevent partial match detection
            if candidate in wake_words:
                # This is an exact word from the wake word - always keep it
                pass
            elif len(candidate_parts) == len(wake_words) and any(part == wake_words[i] for i, part in enumerate(candidate_parts)):
                # At least one part is exactly the same as the original
                # This is a greeting alternative like "hi jarvis" - always keep it
                # These are critical for multi-word wake word discrimination
                pass
            elif len(candidate_parts) == len(wake_words):
                # Same structure - check each part against its corresponding original
                # But SKIP filtering if the part is EXACTLY the same as the original
                # (e.g., "hi jarvis" should keep "jarvis" unchanged)
                for i, part in enumerate(candidate_parts):
                    original_word = wake_words[i]
                    
                    # Skip filtering if this part is exactly the original word
                    if part == original_word:
                        continue
                    
                    last_char, last_2chars, orig_len = forbidden_patterns[i]
                    if has_forbidden_ending(part, original_word, last_char, last_2chars):
                        should_exclude = True
                        break
            elif len(candidate_parts) == 1:
                # Single word from multi-word wake word - check against all patterns
                # But skip if it's one of the original words
                for i, (last_char, last_2chars, orig_len) in enumerate(forbidden_patterns):
                    original_word = wake_words[i]
                    if has_forbidden_ending(candidate, original_word, last_char, last_2chars):
                        should_exclude = True
                        break
            
            if should_exclude:
                continue
        
        filtered_words.append(candidate)
    
    return filtered_words


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
        """Generate negative examples from random speech with volume variations."""
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

                # Normalize audio first
                max_val = np.abs(audio).max()
                if max_val > 0.01:
                    audio = audio / max_val * 0.9
                
                # Apply random amplitude scaling to simulate quiet to loud speech
                # This helps the model learn to reject loud non-wake-word speech
                amplitude_scale = random.uniform(0.3, 1.5)  # 0.3x to 1.5x volume
                scaled_audio = np.clip(audio * amplitude_scale, -1.0, 1.0).astype(np.float32)

                yield AugmentedSample(
                    audio=scaled_audio,
                    sample_rate=self.target_sample_rate,
                    label=0,
                    metadata={
                        "source": "random_speech",
                        "text": phrase,
                        "voice": voice_name,
                        "amplitude_scale": amplitude_scale,
                    },
                )

                if add_noise and self.noise_available and random.random() > 0.5:
                    snr_db = random.choice([-20, -15, -10, -5])
                    noise = self._noise.get_random_noise(
                        self.target_duration, self.target_sample_rate
                    )
                    noisy = mix_audio_with_noise(scaled_audio, noise, snr_db)

                    yield AugmentedSample(
                        audio=noisy,
                        sample_rate=self.target_sample_rate,
                        label=0,
                        metadata={
                            "source": "random_speech",
                            "text": phrase,
                            "snr_db": snr_db,
                            "amplitude_scale": amplitude_scale,
                        },
                    )
            except Exception:
                continue

    def generate_silence(self, num_samples: int = 20) -> Iterator[AugmentedSample]:
        """Generate silence/near-silence negative examples with various characteristics."""
        for i in range(num_samples):
            # Vary the noise floor significantly - from near-zero to low ambient
            noise_floor = random.uniform(0.00001, 0.01)
            
            # Different types of silence/ambient
            silence_type = random.choice(["pure", "hum", "hiss", "room"])
            
            if silence_type == "pure":
                # Near-perfect silence with tiny random noise
                audio = np.random.randn(self.target_length).astype(np.float32) * noise_floor
            elif silence_type == "hum":
                # Low frequency hum (like electrical hum at 50/60Hz)
                t = np.linspace(0, self.target_duration, self.target_length)
                hum_freq = random.choice([50, 60, 100, 120])
                audio = np.sin(2 * np.pi * hum_freq * t).astype(np.float32) * noise_floor * 5
                audio += np.random.randn(self.target_length).astype(np.float32) * noise_floor
            elif silence_type == "hiss":
                # High frequency hiss (like air conditioning)
                audio = np.random.randn(self.target_length).astype(np.float32)
                # Simple high-pass effect
                audio = np.diff(audio, prepend=audio[0]).astype(np.float32) * noise_floor * 3
            else:  # room
                # Room tone - mix of frequencies
                audio = np.random.randn(self.target_length).astype(np.float32) * noise_floor
                # Add some low frequency rumble
                t = np.linspace(0, self.target_duration, self.target_length)
                rumble = np.sin(2 * np.pi * random.uniform(20, 80) * t).astype(np.float32)
                audio += rumble * noise_floor * 2

            yield AugmentedSample(
                audio=audio,
                sample_rate=self.target_sample_rate,
                label=0,
                metadata={"source": f"silence_{silence_type}", "noise_floor": noise_floor},
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
        try:
            if noise_type == "white":
                audio = np.random.randn(self.target_length).astype(np.float32)
            elif noise_type == "pink":
                # Simple pink noise via filtering
                white = np.random.randn(self.target_length)
                # Apply simple 1/f filter approximation
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                try:
                    from scipy.signal import lfilter
                    audio = lfilter(b, a, white).astype(np.float32)
                except ImportError:
                    # Fallback to white noise if scipy not available
                    audio = white.astype(np.float32)
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
        except Exception:
            # Fallback to simple white noise on any error
            return np.random.randn(self.target_length).astype(np.float32) * 0.5

    def generate_loud_sounds(self, num_samples: int = 100) -> Iterator[AugmentedSample]:
        """Generate loud non-speech sounds to prevent false positives on screams, claps, etc."""
        for i in range(num_samples):
            sound_type = random.choice(["burst", "sweep", "impulse", "modulated"])
            
            if sound_type == "burst":
                # Sudden loud burst (like a clap or bang)
                audio = np.zeros(self.target_length, dtype=np.float32)
                burst_start = random.randint(0, self.target_length // 2)
                burst_len = random.randint(1000, 5000)
                burst_end = min(burst_start + burst_len, self.target_length)
                audio[burst_start:burst_end] = np.random.randn(burst_end - burst_start).astype(np.float32)
                # Apply envelope
                envelope = np.exp(-np.linspace(0, 5, burst_end - burst_start))
                audio[burst_start:burst_end] *= envelope
                
            elif sound_type == "sweep":
                # Frequency sweep (like a whistle or siren)
                t = np.linspace(0, self.target_duration, self.target_length)
                start_freq = random.uniform(200, 1000)
                end_freq = random.uniform(1000, 4000)
                freq = np.linspace(start_freq, end_freq, self.target_length)
                audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
                
            elif sound_type == "impulse":
                # Sharp impulse (like a door slam)
                audio = np.zeros(self.target_length, dtype=np.float32)
                num_impulses = random.randint(1, 3)
                for _ in range(num_impulses):
                    pos = random.randint(0, self.target_length - 100)
                    audio[pos:pos+100] = np.random.randn(100).astype(np.float32) * random.uniform(0.5, 1.0)
                    
            else:  # modulated
                # Amplitude modulated noise (like a scream or yell)
                t = np.linspace(0, self.target_duration, self.target_length)
                carrier = np.random.randn(self.target_length).astype(np.float32)
                mod_freq = random.uniform(5, 20)  # Modulation frequency
                modulator = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
                audio = (carrier * modulator).astype(np.float32)
            
            # Normalize to high amplitude (loud sounds)
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * random.uniform(0.7, 1.0)  # High amplitude
            
            yield AugmentedSample(
                audio=audio,
                sample_rate=self.target_sample_rate,
                label=0,
                metadata={"source": f"loud_{sound_type}", "sample_idx": i},
            )

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
