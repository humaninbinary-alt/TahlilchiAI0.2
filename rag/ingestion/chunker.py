import re
from typing import List, Dict, Any, Optional


class Chunker:
    """
    Production-ready text chunker with Uzbek/Russian/English awareness.

    Features:
    - Heading detection (ALL CAPS, #, Section/Bo'lim/Раздел, numbered)
    - Bullet list detection
    - Paragraph boundary splitting
    - Fallback token-based chunking (300-500 words)
    - Integration with UzbekNormalizer
    """

    # Maximum words per chunk for fallback splitting
    MAX_CHUNK_WORDS = 500
    MIN_CHUNK_WORDS = 300

    # Minimum chunk size in characters (smaller chunks are discarded)
    MIN_CHUNK_LENGTH = 20

    def __init__(self, normalizer):
        """
        Initialize chunker with a text normalizer.

        Args:
            normalizer: UzbekNormalizer instance for text preprocessing
        """
        self.normalizer = normalizer

        # Heading patterns (Uzbek/Russian/English)
        self.heading_patterns = [
            # Numbered headings: 1., 1.1, 2), etc.
            re.compile(r'^\s*\d+[\.\)]\s+'),
            re.compile(r'^\s*\d+\.\d+[\.\)]\s+'),
            # Section keywords
            re.compile(r'^\s*(Section|Bo\'lim|Раздел|SECTION|BO\'LIM|РАЗДЕЛ)\s+\d*', re.IGNORECASE),
            # Hash headings
            re.compile(r'^\s*#+\s+'),
        ]

        # Bullet patterns
        self.bullet_patterns = [
            re.compile(r'^\s*[-\*•●■▪▫◦⦿⦾]\s+'),
            re.compile(r'^\s*—\s+'),  # Em dash
        ]

    def chunk_text(self, text: str, document_title: str, language: str) -> List[Dict[str, Any]]:
        """
        Split text into logical chunks with metadata.

        Pipeline:
        1. Normalize text using provided normalizer
        2. Split into sections based on headings
        3. Further split based on paragraphs and bullets
        4. Apply fallback token-based chunking if needed
        5. Remove tiny chunks and add metadata

        Args:
            text: Raw text to chunk
            document_title: Title of the source document
            language: Language code (e.g., 'uz', 'ru', 'en')

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []

        # Step 1: Normalize the full text first
        normalized_text = self.normalizer.normalize(text)

        # Step 2: Split into logical sections
        sections = self._split_by_headings(text)  # Use original text for heading detection

        # Step 3: Process each section
        all_chunks = []
        for section in sections:
            section_heading = section.get('heading')
            section_text = section.get('text', '')

            if not section_text.strip():
                continue

            # Normalize section text
            normalized_section = self.normalizer.normalize(section_text)

            # Split section into paragraph-based chunks
            paragraph_chunks = self._split_by_paragraphs(normalized_section)

            # Apply token-based splitting if chunks are too large
            final_chunks = []
            for chunk_text in paragraph_chunks:
                if self._count_words(chunk_text) > self.MAX_CHUNK_WORDS:
                    # Split large chunk into smaller pieces
                    sub_chunks = self._split_by_tokens(chunk_text)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk_text)

            # Add chunks with metadata
            for chunk_text in final_chunks:
                if len(chunk_text) >= self.MIN_CHUNK_LENGTH:
                    all_chunks.append({
                        'text': chunk_text,
                        'heading': section_heading
                    })

        # Step 4: Add final metadata and indexing
        chunks_with_metadata = []
        for i, chunk in enumerate(all_chunks):
            chunks_with_metadata.append({
                'text': chunk['text'],
                'metadata': {
                    'chunk_index': i,
                    'document_title': document_title,
                    'language': language,
                    'section_title': chunk['heading']
                }
            })

        return chunks_with_metadata

    def _split_by_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sections based on heading detection.

        Returns:
            List of dicts with 'heading' and 'text' keys
        """
        lines = text.split('\n')
        sections = []
        current_section = {'heading': None, 'text': ''}

        for line in lines:
            # Check if line is a heading
            is_heading = False
            heading_text = None

            # Check for ALL CAPS headings (at least 3 words, all uppercase)
            words = line.strip().split()
            if len(words) >= 3 and line.strip().isupper() and len(line.strip()) > 10:
                is_heading = True
                heading_text = line.strip()

            # Check for pattern-based headings
            if not is_heading:
                for pattern in self.heading_patterns:
                    if pattern.match(line):
                        is_heading = True
                        heading_text = line.strip()
                        break

            if is_heading:
                # Save current section if it has content
                if current_section['text'].strip():
                    sections.append(current_section)

                # Start new section
                current_section = {'heading': heading_text, 'text': ''}
            else:
                # Add line to current section
                current_section['text'] += line + '\n'

        # Add final section
        if current_section['text'].strip():
            sections.append(current_section)

        # If no sections were created, return entire text as one section
        if not sections:
            return [{'heading': None, 'text': text}]

        return sections

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text by paragraph boundaries and bullet lists.

        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = []

        lines = text.split('\n')
        for line in lines:
            stripped = line.strip()

            # Empty line indicates paragraph boundary
            if not stripped:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                continue

            # Check for bullet points
            is_bullet = False
            for pattern in self.bullet_patterns:
                if pattern.match(line):
                    is_bullet = True
                    break

            if is_bullet:
                # Start new chunk on bullet if we have content
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []

            current_chunk.append(stripped)

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Fallback splitting: divide text into chunks of 300-500 words.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_word_count = 0

        for word in words:
            current_chunk.append(word)
            current_word_count += 1

            # Create chunk when we reach target size
            if current_word_count >= self.MIN_CHUNK_WORDS:
                # Look for sentence boundary in next few words
                if current_word_count >= self.MAX_CHUNK_WORDS or word.endswith(('.', '!', '?')):
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0

        # Add remaining words as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
