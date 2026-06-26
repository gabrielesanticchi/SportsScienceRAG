"""PDF extraction and section detection for academic papers."""

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import fitz  # PyMuPDF

from sports_science_search.constants import MIN_SECTIONS_FOR_AUTO_DETECTION, SECTION_PATTERNS
from sports_science_search.exceptions import PDFExtractionError, SectionDetectionError
from sports_science_search.models import Paper


class PDFExtractor:
    """Extracts text and metadata from academic PDFs using PyMuPDF."""

    def __init__(self, manual_mappings_path: Optional[Path] = None, strict: bool = True):
        """
        Initialize PDF extractor.

        Args:
            manual_mappings_path: Path to section_mappings.json
            strict: If True (default), raise SectionDetectionError when sections
                cannot be auto-detected and no manual mapping exists. If False,
                fall back to a single "Body" section spanning the whole document
                (a warning is logged) so processing can continue.
        """
        self._manual_mappings = self._load_manual_mappings(manual_mappings_path)
        self.strict = strict

    @property
    def manual_mappings(self) -> Dict[str, Any]:
        """Read-only access to manual section mappings."""
        return self._manual_mappings

    def _load_manual_mappings(self, path: Optional[Path]) -> Dict[str, Any]:
        """Load manual section mappings from JSON file.

        Args:
            path: Path to section mappings JSON file

        Returns:
            Dictionary mapping PDF filenames to section boundaries,
            or empty dict if file doesn't exist or cannot be parsed

        Note:
            Exceptions are suppressed; returns empty dict on any failure.
        """
        if path is None or not path.exists():
            return {}

        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Failed to load manual mappings from {path}: {e}")
            return {}

    def extract(self, pdf_path: Path) -> Paper:
        """
        Extract text, detect sections, extract metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Paper object with extracted data

        Raises:
            SectionDetectionError: If sections cannot be detected and no manual mapping exists
            PDFExtractionError: If PDF cannot be read or parsed
        """
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        try:
            # Extract full text
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            if not text.strip():
                raise PDFExtractionError(f"No text extracted from {pdf_path.name}")

            # Detect sections
            sections = self._detect_sections(text, pdf_path.name)

            # Extract metadata
            metadata = self._extract_metadata(pdf_path, text)

            # Create Paper object
            paper = Paper(
                filename=pdf_path.name,
                title=metadata.get("title", pdf_path.stem),
                text=text,
                sections=sections,
                metadata=metadata
            )

            return paper

        except SectionDetectionError:
            # Re-raise section detection errors
            raise
        except Exception as e:
            raise PDFExtractionError(
                f"Failed to extract {pdf_path.name}: {str(e)}"
            )

    def _detect_sections(self, text: str, filename: str) -> Dict[str, str]:
        """
        Auto-detect paper sections using regex patterns.
        Falls back to manual mapping from section_mappings.json.

        Args:
            text: Full text of paper
            filename: PDF filename for manual mapping lookup

        Returns:
            Dict mapping section names to section text

        Raises:
            SectionDetectionError: If detection fails and no manual mapping available
        """
        # Try auto-detection first
        lines = text.split('\n')
        section_positions: Dict[str, int] = {}  # section_name → line_index

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for section_name, pattern in SECTION_PATTERNS.items():
                if re.match(pattern, line_stripped):
                    section_positions[section_name] = i
                    break

        # If we found at least MIN_SECTIONS sections, consider auto-detection successful
        if len(section_positions) >= MIN_SECTIONS_FOR_AUTO_DETECTION:
            sections = {}

            # Sort sections by their position in the text to preserve document order
            sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
            section_names = [name for name, _ in sorted_sections]
            section_indices = [idx for _, idx in sorted_sections]

            for idx, section_name in enumerate(section_names):
                start_line = section_indices[idx]
                end_line = section_indices[idx + 1] if idx + 1 < len(section_indices) else len(lines)

                section_text = '\n'.join(lines[start_line:end_line])
                sections[section_name] = section_text.strip()

            return sections

        # Auto-detection failed, try manual mapping
        if filename in self.manual_mappings:
            # TODO(Task 5): Implement page-based text extraction using start_page/end_page
            # For now, return placeholder to enable testing of auto-detection fallback
            sections = {}
            for section_name in self.manual_mappings[filename].keys():
                sections[section_name] = f"[Manual mapping: {section_name}]"
            return sections

        # Both auto-detection and manual mapping failed.
        # In lenient mode, fall back to a single whole-document section so the
        # paper can still be chunked and indexed.
        if not self.strict:
            logging.warning(
                "Section auto-detection found only %d section(s) in %s "
                "(need at least %d); falling back to single 'Body' section.",
                len(section_positions), filename, MIN_SECTIONS_FOR_AUTO_DETECTION,
            )
            return {"Body": text.strip()}

        raise SectionDetectionError(
            f"Could not detect sections in {filename}. "
            f"Found {len(section_positions)} sections (need at least 3). "
            f"Please add manual mapping to section_mappings.json"
        )

    def _extract_metadata(self, pdf_path: Path, text: str) -> Dict[str, Any]:
        """
        Extract title, authors, year, DOI, journal from PDF metadata and text.

        Args:
            pdf_path: Path to PDF file
            text: Full text of PDF

        Returns:
            Dict with keys: authors, year, DOI, journal, page_count, title

        Raises:
            None (gracefully handles all extraction failures with initialized defaults)
        """
        # Extract PDF metadata (page count, title, authors)
        page_count, pdf_title, pdf_authors = self._read_pdf_metadata(pdf_path)

        # Extract year from filename (e.g., malone2017.pdf)
        year = self._extract_year(pdf_path)

        # Extract DOI from text
        doi = self._extract_doi(text)

        # Extract title from text fallback if not in PDF metadata
        title = pdf_title or self._infer_title_from_text(text)

        # Extract authors from text fallback if not in PDF metadata
        authors = pdf_authors or self._infer_authors_from_text(text)

        return {
            "authors": authors,
            "year": year,
            "DOI": doi,
            "journal": None,  # TODO(Task X): Implement journal extraction
            "page_count": page_count,
            "title": title,
        }

    def _read_pdf_metadata(self, pdf_path: Path) -> tuple[int, Optional[str], Optional[str]]:
        """
        Read page count and metadata from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (page_count, title, authors). Returns (0, None, None) on error.
        """
        try:
            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count
                pdf_metadata = doc.metadata
                title = pdf_metadata.get("title") if pdf_metadata else None
                author = pdf_metadata.get("author") if pdf_metadata else None
                return page_count, title, author
        except (fitz.FileDataError, fitz.FileNotFoundError) as exc:
            logging.warning("Could not read PDF metadata from %s: %s", pdf_path, exc)
            return 0, None, None
        except Exception as exc:
            logging.warning("Unexpected error reading PDF %s: %s", pdf_path, exc)
            return 0, None, None

    def _extract_year(self, pdf_path: Path) -> Optional[int]:
        """
        Extract year from PDF filename.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Year if found and valid (1900-current year + 1), else None.
        """
        MIN_YEAR = 1900
        MAX_YEAR = datetime.date.today().year + 1

        year_match = re.search(r'(\d{4})', pdf_path.stem)
        if year_match:
            try:
                year = int(year_match.group(1))
                if MIN_YEAR <= year <= MAX_YEAR:
                    return year
            except ValueError:
                pass
        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """
        Extract DOI from text.

        Args:
            text: Full text of PDF

        Returns:
            DOI string if found, else None. Strips trailing punctuation.
        """
        # Pattern: 10.xxxx/... but exclude trailing punctuation
        doi_match = re.search(r'10\.\d{4,}/[^\s.,;)\]]+', text)
        if doi_match:
            return doi_match.group(0)
        return None

    def _infer_title_from_text(self, text: str) -> Optional[str]:
        """
        Infer title from first 10 lines of text (best effort heuristic).

        Args:
            text: Full text of PDF

        Returns:
            Inferred title if found, else None. Uses longest line heuristic.
        """
        if not text:
            return None

        first_lines = text.split('\n')[:10]
        title_candidate = max(first_lines, key=len) if first_lines else ""

        if len(title_candidate) > 10:
            logging.debug("Title inferred from text heuristic for: %s", title_candidate[:50])
            return title_candidate.strip()

        return None

    def _infer_authors_from_text(self, text: str) -> Optional[str]:
        """
        Infer authors from first 20 lines of text (best effort heuristic).

        Args:
            text: Full text of PDF

        Returns:
            Inferred authors if found, else None. Searches for "et al." patterns.
        """
        if not text:
            return None

        # Look for common author patterns in first 20 lines
        for line in text.split('\n')[:20]:
            if re.search(r'[A-Z][a-z]+\s+et\s+al\.', line):
                logging.debug("Authors inferred from text heuristic: %s", line[:80])
                return line.strip()

        return None
