import os
import re
import unicodedata
from pathlib import Path


class ExtractionError(Exception):
    """Custom exception raised when text extraction fails."""
    pass


class TextExtractor:
    """
    Production-ready text extractor for PDF, DOCX, and TXT files.

    Supports:
    - PDF extraction via pdfminer
    - DOCX extraction via python-docx
    - TXT extraction with proper encoding handling
    - Unicode normalization and whitespace cleanup
    """

    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

    def extract(self, path: str) -> str:
        """
        Extract text from a file based on its extension.

        Args:
            path: File path to extract text from

        Returns:
            Clean, normalized text string

        Raises:
            ExtractionError: If file doesn't exist, unsupported format, or extraction fails
        """
        # Validate file exists
        if not os.path.exists(path):
            raise ExtractionError(f"File not found: {path}")

        # Detect file type by extension
        file_path = Path(path)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ExtractionError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Extract based on file type
        try:
            if extension == '.pdf':
                text = self._extract_pdf(path)
            elif extension == '.docx':
                text = self._extract_docx(path)
            elif extension == '.txt':
                text = self._extract_txt(path)
            else:
                raise ExtractionError(f"Unsupported extension: {extension}")

            # Normalize and clean the extracted text
            text = self._clean_text(text)

            return text

        except ExtractionError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap any other exceptions
            raise ExtractionError(f"Failed to extract text from {path}: {str(e)}") from e

    def _extract_pdf(self, path: str) -> str:
        """Extract text from PDF file using pdfminer."""
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(path)
            if not text:
                raise ExtractionError(f"No text extracted from PDF: {path}")
            return text
        except ImportError:
            raise ExtractionError(
                "pdfminer.six is required for PDF extraction. "
                "Install with: pip install pdfminer.six"
            )

    def _extract_docx(self, path: str) -> str:
        """Extract text from DOCX file using python-docx."""
        try:
            import docx
            doc = docx.Document(path)
            paragraphs = [para.text for para in doc.paragraphs]
            text = '\n'.join(paragraphs)
            if not text.strip():
                raise ExtractionError(f"No text extracted from DOCX: {path}")
            return text
        except ImportError:
            raise ExtractionError(
                "python-docx is required for DOCX extraction. "
                "Install with: pip install python-docx"
            )

    def _extract_txt(self, path: str) -> str:
        """Extract text from plain text file with encoding detection."""
        # Try UTF-8 first (most common)
        encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']

        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    text = f.read()
                    if text:
                        return text
            except (UnicodeDecodeError, UnicodeError):
                continue

        # If all encodings fail, try with errors='ignore'
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                if not text.strip():
                    raise ExtractionError(f"No text content in file: {path}")
                return text
        except Exception as e:
            raise ExtractionError(f"Failed to read text file {path}: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        - Normalize unicode to NFC form
        - Remove excessive whitespace
        - Remove control characters
        - Normalize line breaks
        """
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize('NFC', text)

        # Remove control characters (except newlines and tabs)
        text = ''.join(char for char in text if char == '\n' or char == '\t' or not unicodedata.category(char).startswith('C'))

        # Normalize different types of line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean up excessive spaces on each line
        lines = text.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(cleaned_lines)

        # Final strip
        text = text.strip()

        return text
