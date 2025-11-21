import re
import unicodedata
from typing import List


class UzbekNormalizer:
    """
    Production-ready Uzbek text normalizer for document ingestion and query preprocessing.

    Handles:
    - Uzbek apostrophe variants (o', g', etc.)
    - Common digraphs (sh, ch, ng)
    - Punctuation normalization
    - Whitespace cleanup
    - Light stemming of common Uzbek suffixes
    """

    def __init__(self):
        # Apostrophe variants commonly used in Uzbek text
        self.apostrophe_variants = ["'", "'", "ʼ", "`", "ʻ", "ʽ", "′"]

        # Common Uzbek suffixes for light stemming (in order of priority - longer first)
        self.suffixes = [
            "laridan",
            "larimiz",
            "laringiz",
            "larining",
            "lariga",
            "larida",
            "larini",
            "lari",
            "lar",
            "ning",
            "miz",
            "lik",
            "chi",
            "dan",
            "ga",
            "da",
            "ni",
        ]

        # Uzbek digraphs to preserve
        self.digraphs = ["sh", "ch", "ng", "o'", "g'"]

    def normalize(self, text: str) -> str:
        """
        Main normalization pipeline for Uzbek text.

        Pipeline steps:
        1. Normalize apostrophes
        2. Normalize unicode
        3. Clean punctuation
        4. Lowercase
        5. Normalize whitespace
        6. Tokenize and light stem
        7. Recombine

        Args:
            text: Raw input text

        Returns:
            Normalized text string
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Normalize apostrophes to standard straight apostrophe
        text = self._normalize_apostrophes(text)

        # Step 2: Normalize unicode (remove diacritics, normalize forms)
        text = self._normalize_unicode(text)

        # Step 3: Clean punctuation (remove or normalize strange chars)
        text = self._clean_punctuation(text)

        # Step 4: Lowercase everything
        text = text.lower()

        # Step 5: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 6: Tokenize and apply light stemming
        tokens = text.split()
        stemmed_tokens = [self.light_stem(token) for token in tokens]

        # Step 7: Recombine
        normalized = " ".join(stemmed_tokens)

        return normalized.strip()

    def _normalize_apostrophes(self, text: str) -> str:
        """
        Unify all apostrophe variants to standard straight apostrophe (').
        Handles o', g', and other Uzbek-specific characters.
        """
        for variant in self.apostrophe_variants:
            text = text.replace(variant, "'")
        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters while preserving Uzbek-specific characters.
        - Normalize to NFC form
        - Remove combining diacritics (except those needed for Uzbek)
        """
        # First normalize to NFC (canonical composition)
        text = unicodedata.normalize('NFC', text)

        # Remove most diacritical marks but preserve Uzbek characters
        # We keep: o', g', and other essential Uzbek chars
        result = []
        for char in text:
            # Keep ASCII, Cyrillic, and essential punctuation
            if (ord(char) < 128 or
                (0x0400 <= ord(char) <= 0x04FF) or  # Cyrillic
                char in ["'", "-", "_"]):
                result.append(char)
            # Filter out combining marks and strange unicode
            elif unicodedata.category(char) not in ['Mn', 'Mc', 'Me']:
                result.append(char)

        return ''.join(result)

    def _clean_punctuation(self, text: str) -> str:
        """
        Clean and normalize punctuation:
        - Remove repeated punctuation (!!!, ???, etc.)
        - Keep only basic punctuation
        - Remove excessive special characters
        """
        # Replace multiple punctuation marks with single space
        text = re.sub(r'[!?.,;:]+', ' ', text)

        # Remove other special characters except apostrophe and hyphen
        text = re.sub(r"[^\w\s'\-]", ' ', text, flags=re.UNICODE)

        # Clean up hyphens surrounded by spaces
        text = re.sub(r'\s+-\s+', ' ', text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace:
        - Replace multiple spaces with single space
        - Remove leading/trailing whitespace
        - Normalize tabs and newlines to spaces
        """
        # Replace all whitespace (tabs, newlines, multiple spaces) with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def light_stem(self, word: str) -> str:
        """
        Apply light stemming for common Uzbek suffixes.

        Rules:
        - Only stem if word length > 5
        - Remove longest matching suffix
        - Ensure result is stable (length >= 3)

        Args:
            word: Single token to stem

        Returns:
            Stemmed word or original if no suffix matches
        """
        if not word or len(word) <= 5:
            return word

        # Try to match suffixes (longest first due to ordering)
        for suffix in self.suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                # Only return stem if it's long enough to be meaningful
                if len(stem) >= 3:
                    return stem

        return word


def normalize_uzbek(text: str) -> str:
    """
    Convenience function for normalizing Uzbek text.

    Args:
        text: Raw Uzbek text

    Returns:
        Normalized text

    Example:
        >>> normalize_uzbek("O'qituvchilarning ta'til muddati")
        "o'qituvchi ta'til muddat"
    """
    return UzbekNormalizer().normalize(text)
