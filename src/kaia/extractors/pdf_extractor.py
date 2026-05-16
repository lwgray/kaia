"""
Extract and chunk PDF research papers.

This extractor parses PDF files and creates chunks based on detected
section structure (Abstract, Introduction, Methods, etc.), preserving
hierarchical context and metadata like paper title and cluster topic.

Large sections are recursively split into smaller chunks with overlap
to ensure no chunk exceeds the token limit and cross-boundary context
is preserved.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

from kaia.models import ChunkMetadata, ChunkType, PdfChunk

# Token encoder for measuring chunk sizes
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_enc.encode(text))


class PdfExtractor:
    """
    Extract chunks from PDF research papers.

    Creates chunks for:
    - Paper abstract
    - Major sections (Introduction, Methods, Experiments, etc.)
    - Subsections within major sections
    - References (as a single chunk)
    - Figure/table captions

    Large sections are recursively sub-chunked to stay within
    ``max_chunk_tokens`` with ``overlap_tokens`` of context overlap.
    """

    # Chunking parameters
    MAX_CHUNK_TOKENS = 800
    OVERLAP_TOKENS = 150

    # Numbered section headings: "1 Introduction", "2.1 Background", "A Appendix"
    NUMBERED_SECTION_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+"  # number or letter prefix
        r"([A-Z][A-Za-z\s:,\-–—]+)$"              # title in title case
    )

    # Known major section headings (all-caps, commonly found in papers)
    KNOWN_SECTIONS = {
        "ABSTRACT",
        "INTRODUCTION",
        "MOTIVATION",
        "BACKGROUND",
        "RELATED WORK",
        "METHODS",
        "METHOD",
        "APPROACH",
        "EXPERIMENTS",
        "RESULTS",
        "DISCUSSION",
        "CONCLUSION",
        "CONCLUSIONS",
        "REFERENCES",
        "ACKNOWLEDGEMENTS",
        "ACKNOWLEDGMENTS",
        "APPENDIX",
        "SUPPLEMENTARY MATERIAL",
    }

    # Figure/Table captions like "Figure 1: ...", "Table 2. ..."
    CAPTION_PATTERN = re.compile(
        r"^(Figure|Table|Fig\.)\s*(\d+)[.:]\s*(.+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        max_chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
        use_llm_extraction: bool = True,
    ):
        """
        Parameters
        ----------
        max_chunk_tokens : int | None
            Maximum tokens per chunk. Defaults to 800.
        overlap_tokens : int | None
            Token overlap between consecutive sub-chunks. Defaults to 150.
        use_llm_extraction : bool
            If True, try pymupdf4llm for better text extraction.
            Falls back to raw PyMuPDF if unavailable.
        """
        if max_chunk_tokens is not None:
            self.MAX_CHUNK_TOKENS = max_chunk_tokens
        if overlap_tokens is not None:
            self.OVERLAP_TOKENS = overlap_tokens

        self._use_llm_extraction = use_llm_extraction
        self._pymupdf4llm = None
        if use_llm_extraction:
            try:
                import pymupdf4llm

                self._pymupdf4llm = pymupdf4llm
            except ImportError:
                pass

    def extract_from_file(
        self, file_path: Path, repository: str = "papers"
    ) -> list[PdfChunk]:
        """
        Extract all chunks from a PDF file.

        Parameters
        ----------
        file_path : Path
            Path to PDF file
        repository : str
            Repository tag for chunks (default "papers" — distinguishes
            research PDFs from code repos for filtering)

        Returns
        -------
        list[PdfChunk]
            All extracted chunks
        """
        full_text = self._extract_text(file_path)

        if not full_text.strip():
            return []

        title = self._extract_title(full_text)
        cluster = self._extract_cluster(file_path)
        sections = self._split_into_sections(full_text)

        chunks: list[PdfChunk] = []
        for section_title, section_body, hierarchy in sections:
            content = section_body.strip()
            if not content or len(content) < 50:
                continue

            # Sub-chunk large sections
            sub_chunks = self._split_large_section(content)

            for idx, sub_content in enumerate(sub_chunks):
                sub_hierarchy = hierarchy.copy()
                if len(sub_chunks) > 1:
                    sub_hierarchy.append(f"Part {idx + 1}/{len(sub_chunks)}")

                chunk = self._create_chunk(
                    title=title,
                    section_title=section_title,
                    content=sub_content,
                    hierarchy=sub_hierarchy,
                    cluster=cluster,
                    file_path=file_path,
                    repository=repository,
                )
                chunks.append(chunk)

        # Extract figure/table captions as separate chunks
        captions = self._extract_captions(full_text)
        for caption_label, caption_text in captions:
            header = f"Paper: {title}\n"
            if cluster:
                header = f"Topic: {cluster}\n{header}"
            caption_content = f"{header}{caption_label}: {caption_text}"

            chunks.append(
                PdfChunk(
                    content=caption_content,
                    chunk_type=ChunkType.PDF_SECTION,
                    metadata=ChunkMetadata(
                        repository=repository,
                        file_path=str(file_path),
                        section_hierarchy=[caption_label],
                        keywords=[title, caption_label],
                        importance=0.6,
                    ),
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    # Line numbers commonly found in review-format papers (e.g., "000", "123")
    LINE_NUMBER_PATTERN = re.compile(r"^\d{2,4}$")

    # Page number lines (just a number, typically at bottom of page)
    PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,3}$")

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF, preferring pymupdf4llm when available."""
        if self._pymupdf4llm is not None:
            try:
                return self._extract_text_llm(file_path)
            except Exception:
                # Fall back to raw extraction
                pass

        return self._extract_text_raw(file_path)

    def _extract_text_llm(self, file_path: Path) -> str:
        """
        Extract text using pymupdf4llm for better layout handling.

        Handles multi-column layouts, tables, and equations better than
        raw PyMuPDF text extraction.
        """
        md_text = self._pymupdf4llm.to_markdown(str(file_path))

        # pymupdf4llm returns markdown — strip markdown formatting artifacts
        # but keep structure (headings become detectable by our section parser)
        cleaned = self._clean_markdown_text(md_text)
        return cleaned

    @staticmethod
    def _clean_markdown_text(md_text: str) -> str:
        """Clean pymupdf4llm markdown output for our section parser."""
        lines = []
        for line in md_text.splitlines():
            # Convert markdown headings to plain text (our parser detects caps/numbered)
            stripped = line.strip()
            if stripped.startswith("#"):
                # Remove markdown heading markers
                stripped = stripped.lstrip("#").strip()
                # If it was a heading, keep it as-is (our parser will detect it)
                lines.append(stripped)
            else:
                # Remove bold/italic markers but keep text
                cleaned = stripped.replace("**", "").replace("__", "")
                lines.append(cleaned)
        return "\n".join(lines)

    def _extract_text_raw(self, file_path: Path) -> str:
        """Extract text from all pages using raw PyMuPDF, stripping noise."""
        doc = fitz.open(str(file_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text)
        doc.close()
        raw = "\n".join(pages)

        # Detect repeated header line (appears on most pages)
        header_line = self._detect_repeated_header(raw)

        cleaned_lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            # Strip review-format line numbers
            if self.LINE_NUMBER_PATTERN.match(stripped):
                continue
            # Strip repeated page headers
            if header_line and stripped == header_line:
                continue
            # Strip standalone page numbers
            if self.PAGE_NUMBER_PATTERN.match(stripped):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _detect_repeated_header(self, text: str) -> str | None:
        """Find a line that repeats on many pages (e.g., conference header)."""
        lines = text.splitlines()
        counts: dict[str, int] = {}
        for line in lines:
            s = line.strip()
            if len(s) > 20:  # Headers are usually substantial
                counts[s] = counts.get(s, 0) + 1
        # A line appearing 3+ times is likely a page header
        for line, count in counts.items():
            if count >= 3:
                return line
        return None

    # ------------------------------------------------------------------
    # Sub-chunking with overlap
    # ------------------------------------------------------------------

    def _split_large_section(self, content: str) -> list[str]:
        """
        Split a section into sub-chunks if it exceeds max_chunk_tokens.

        Uses paragraph boundaries as natural split points. Adds overlap
        between consecutive chunks so cross-boundary content isn't lost.

        Parameters
        ----------
        content : str
            Section content to potentially split

        Returns
        -------
        list[str]
            One or more chunks (original if small enough)
        """
        if _count_tokens(content) <= self.MAX_CHUNK_TOKENS:
            return [content]

        # Split into paragraphs (double newline or single newline with blank)
        paragraphs = re.split(r"\n\s*\n", content)
        # If too few paragraph breaks, split on single newlines
        if len(paragraphs) <= 1:
            paragraphs = content.split("\n")

        return self._merge_paragraphs_with_overlap(paragraphs)

    def _merge_paragraphs_with_overlap(self, paragraphs: list[str]) -> list[str]:
        """
        Merge paragraphs into chunks respecting max tokens, with overlap.

        Parameters
        ----------
        paragraphs : list[str]
            Paragraphs to merge

        Returns
        -------
        list[str]
            Merged chunks with overlap
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = _count_tokens(para)

            # If a single paragraph exceeds max, split it by sentences
            if para_tokens > self.MAX_CHUNK_TOKENS:
                # Flush current
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # Split oversized paragraph by sentences
                sentence_chunks = self._split_by_sentences(para)
                chunks.extend(sentence_chunks)
                continue

            # Would adding this paragraph exceed the limit?
            if current_tokens + para_tokens > self.MAX_CHUNK_TOKENS and current_parts:
                chunks.append("\n\n".join(current_parts))
                # Start new chunk with overlap from the tail of the previous
                overlap_parts = self._get_overlap_tail(current_parts)
                current_parts = overlap_parts
                current_tokens = _count_tokens("\n\n".join(current_parts)) if current_parts else 0

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks if chunks else ["\n\n".join(paragraphs)]

    def _get_overlap_tail(self, parts: list[str]) -> list[str]:
        """
        Get tail paragraphs from the previous chunk to use as overlap.

        Returns enough trailing paragraphs to fill ~overlap_tokens.
        """
        overlap: list[str] = []
        tokens = 0
        for part in reversed(parts):
            part_tokens = _count_tokens(part)
            if tokens + part_tokens > self.OVERLAP_TOKENS:
                break
            overlap.insert(0, part)
            tokens += part_tokens
        return overlap

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Split a long paragraph into chunks at sentence boundaries.

        Used as a fallback when a single paragraph exceeds max tokens.
        """
        # Split on sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = _count_tokens(sentence)
            if current_tokens + s_tokens > self.MAX_CHUNK_TOKENS and current:
                chunks.append(" ".join(current))
                # Overlap: keep last sentence(s)
                overlap: list[str] = []
                overlap_tokens = 0
                for s in reversed(current):
                    st = _count_tokens(s)
                    if overlap_tokens + st > self.OVERLAP_TOKENS:
                        break
                    overlap.insert(0, s)
                    overlap_tokens += st
                current = overlap
                current_tokens = overlap_tokens
            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    # Title / cluster / heading detection
    # ------------------------------------------------------------------

    def _extract_title(self, text: str) -> str:
        """
        Extract the paper title from the first lines of text.

        Heuristic: the title is typically the first non-empty, non-header
        line(s) before the abstract or author info.
        """
        lines = text.strip().splitlines()
        title_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if title_lines:
                    break
                continue
            # Skip common header noise
            lower = stripped.lower()
            if any(
                skip in lower
                for skip in [
                    "under review",
                    "published",
                    "conference",
                    "workshop",
                    "arxiv",
                    "preprint",
                ]
            ):
                continue
            # Stop at author lines (contain multiple commas or "anonymous")
            if "anonymous" in lower or stripped.count(",") >= 2:
                break
            # Stop at abstract
            if lower == "abstract":
                break
            title_lines.append(stripped)
            # Titles are usually 1-3 lines
            if len(title_lines) >= 3:
                break

        return " ".join(title_lines) if title_lines else "Unknown Title"

    def _extract_cluster(self, file_path: Path) -> str | None:
        """
        Extract cluster/topic from directory structure.

        Expects paths like:
        clusters/01_Multi-Agent_Coordination_Communication/paper.pdf
        """
        parts = file_path.parts
        for part in parts:
            # Match cluster directory names like "01_Multi-Agent_..."
            if re.match(r"^\d{2}_", part):
                # Convert underscores to spaces, strip leading number
                return re.sub(r"^\d{2}_", "", part).replace("_", " ")
        return None

    def _is_section_heading(self, line: str) -> tuple[str, int] | None:
        """
        Check if a line is a section heading.

        Returns (heading_title, level) or None.
        """
        stripped = line.strip()
        if not stripped:
            return None

        # Check for numbered heading: "1 Introduction" or "2.1 Background"
        numbered = self.NUMBERED_SECTION_PATTERN.match(stripped)
        if numbered:
            number = numbered.group(1)
            title = numbered.group(2).strip()
            level = number.count(".") + 1
            return (title, level)

        # Check for all-caps lines that match known sections
        upper = stripped.upper()
        if upper == stripped and len(stripped) > 3:
            # Exact match to known sections
            if upper in self.KNOWN_SECTIONS:
                return (stripped.title(), 1)

            # All-caps lines with certain patterns are likely subsections
            # e.g., "MULTI-AGENT IMITATION LEARNING", "PART I: ..."
            if (
                len(stripped) > 5
                and len(stripped) < 80
                and stripped.replace(" ", "").replace("-", "").replace(":", "").replace("–", "").replace("—", "").isalpha()
            ):
                return (stripped.title(), 2)

        return None

    def _split_into_sections(
        self, text: str
    ) -> list[tuple[str, str, list[str]]]:
        """
        Split PDF text into sections based on detected headings.

        Returns list of (section_title, section_content, hierarchy).
        """
        lines = text.splitlines()
        sections: list[tuple[str, str, list[str]]] = []
        current_title = "Preamble"
        current_lines: list[str] = []
        current_hierarchy: list[str] = []
        current_level = 0

        for line in lines:
            result = self._is_section_heading(line)

            if result:
                heading, level = result

                # Save previous section
                if current_lines:
                    body = "\n".join(current_lines)
                    sections.append(
                        (current_title, body, current_hierarchy.copy())
                    )

                # Update hierarchy
                if level <= current_level:
                    current_hierarchy = current_hierarchy[: level - 1]
                current_hierarchy.append(heading)
                current_level = level
                current_title = heading
                current_lines = []
            else:
                current_lines.append(line)

        # Save final section
        if current_lines:
            body = "\n".join(current_lines)
            sections.append(
                (current_title, body, current_hierarchy.copy())
            )

        # If no sections were detected, return the whole text as one chunk
        if len(sections) <= 1 and not any(
            s[0] != "Preamble" for s in sections
        ):
            return [("Full Text", text, ["Full Text"])]

        return sections

    # ------------------------------------------------------------------
    # Captions
    # ------------------------------------------------------------------

    def _extract_captions(
        self, text: str
    ) -> list[tuple[str, str]]:
        """
        Extract figure and table captions from the text.

        Returns list of (label, caption_text) tuples.
        Captions often span multiple lines, so we collect continuation lines.
        """
        lines = text.splitlines()
        captions: list[tuple[str, str]] = []
        current_label = None
        current_text: list[str] = []

        for line in lines:
            stripped = line.strip()
            match = self.CAPTION_PATTERN.match(stripped)

            if match:
                # Save previous caption
                if current_label and current_text:
                    captions.append(
                        (current_label, " ".join(current_text))
                    )

                fig_type = match.group(1)
                fig_num = match.group(2)
                caption_start = match.group(3)

                # Normalize "Fig." to "Figure"
                if fig_type.lower().startswith("fig"):
                    fig_type = "Figure"
                current_label = f"{fig_type} {fig_num}"
                current_text = [caption_start]
            elif current_label and stripped and not stripped[0].isupper():
                # Continuation line of a caption (lowercase start)
                current_text.append(stripped)
            elif current_label:
                # End of caption
                captions.append(
                    (current_label, " ".join(current_text))
                )
                current_label = None
                current_text = []

        # Save last caption
        if current_label and current_text:
            captions.append(
                (current_label, " ".join(current_text))
            )

        return captions

    # ------------------------------------------------------------------
    # Chunk creation
    # ------------------------------------------------------------------

    def _create_chunk(
        self,
        title: str,
        section_title: str,
        content: str,
        hierarchy: list[str],
        cluster: str | None,
        file_path: Path,
        repository: str = "papers",
    ) -> PdfChunk:
        """Create a PdfChunk from a paper section."""
        # Build rich content with paper context
        header = f"Paper: {title}\nSection: {' > '.join(hierarchy)}\n\n"
        if cluster:
            header = f"Topic: {cluster}\n{header}"
        chunk_content = header + content

        # Determine importance based on section type
        importance = 0.7
        lower_title = section_title.lower()
        if lower_title in ("abstract", "preamble"):
            importance = 0.95
        elif lower_title in ("introduction", "motivation", "conclusion"):
            importance = 0.9
        elif "method" in lower_title or "approach" in lower_title:
            importance = 0.85
        elif "experiment" in lower_title or "result" in lower_title:
            importance = 0.8
        elif lower_title == "references":
            importance = 0.4

        keywords = [title]
        if cluster:
            keywords.append(cluster)

        return PdfChunk(
            content=chunk_content,
            chunk_type=ChunkType.PDF_SECTION,
            metadata=ChunkMetadata(
                repository=repository,
                file_path=str(file_path),
                section_hierarchy=hierarchy if hierarchy else [],
                keywords=keywords[:10],
                importance=importance,
            ),
        )
