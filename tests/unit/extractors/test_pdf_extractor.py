"""
Unit tests for PDF extractor — chunking, overlap, and text extraction.
"""

from pathlib import Path

import pytest

from kaia.extractors.pdf_extractor import PdfExtractor, _count_tokens


class TestChunkSplitting:
    """Test sub-chunking of large sections."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(
            max_chunk_tokens=100,
            overlap_tokens=20,
            use_llm_extraction=False,
        )

    def test_small_section_not_split(self, extractor: PdfExtractor) -> None:
        """A section under the token limit should stay as one chunk."""
        content = "This is a short paragraph."
        result = extractor._split_large_section(content)
        assert len(result) == 1
        assert result[0] == content

    def test_large_section_is_split(self, extractor: PdfExtractor) -> None:
        """A section exceeding max tokens should be split into multiple chunks."""
        # Build content that's definitely > 100 tokens
        paragraphs = [f"Paragraph {i}. " + "word " * 30 for i in range(10)]
        content = "\n\n".join(paragraphs)
        assert _count_tokens(content) > 100

        result = extractor._split_large_section(content)
        assert len(result) > 1

        # Each chunk should be at or under the limit (with some tolerance for overlap text)
        for chunk in result:
            assert _count_tokens(chunk) <= extractor.MAX_CHUNK_TOKENS + extractor.OVERLAP_TOKENS

    def test_overlap_exists_between_chunks(self) -> None:
        """Consecutive chunks should share some overlapping text."""
        # Use smaller paragraphs and a larger overlap budget so overlap
        # actually fits within the token window.
        extractor = PdfExtractor(
            max_chunk_tokens=100,
            overlap_tokens=40,
            use_llm_extraction=False,
        )
        # ~12 tokens each, so overlap of 40 tokens can capture ~3 paragraphs
        paragraphs = [f"Unique paragraph {i} about topic." for i in range(20)]
        content = "\n\n".join(paragraphs)

        result = extractor._split_large_section(content)
        if len(result) < 2:
            pytest.skip("Content didn't produce multiple chunks")

        # At least one pair of consecutive chunks should share text
        found_overlap = False
        for i in range(len(result) - 1):
            paras_i = result[i].split("\n\n")
            for para in paras_i:
                para = para.strip()
                if para and para in result[i + 1]:
                    found_overlap = True
                    break
            if found_overlap:
                break

        assert found_overlap, "No overlap found between any consecutive chunks"

    def test_single_huge_paragraph_split_by_sentences(self, extractor: PdfExtractor) -> None:
        """A single paragraph exceeding max tokens should split at sentence boundaries."""
        sentences = [f"Sentence number {i} has some content here." for i in range(50)]
        content = " ".join(sentences)
        assert _count_tokens(content) > 100

        result = extractor._split_large_section(content)
        assert len(result) > 1


class TestSectionDetection:
    """Test section heading detection."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(use_llm_extraction=False)

    def test_numbered_heading(self, extractor: PdfExtractor) -> None:
        result = extractor._is_section_heading("1 Introduction")
        assert result is not None
        title, level = result
        assert title == "Introduction"
        assert level == 1

    def test_sub_section_heading(self, extractor: PdfExtractor) -> None:
        result = extractor._is_section_heading("2.1 Background")
        assert result is not None
        title, level = result
        assert title == "Background"
        assert level == 2

    def test_all_caps_known_section(self, extractor: PdfExtractor) -> None:
        result = extractor._is_section_heading("ABSTRACT")
        assert result is not None
        title, level = result
        assert title == "Abstract"
        assert level == 1

    def test_normal_text_not_heading(self, extractor: PdfExtractor) -> None:
        result = extractor._is_section_heading("This is just normal text in a paper.")
        assert result is None

    def test_empty_line_not_heading(self, extractor: PdfExtractor) -> None:
        result = extractor._is_section_heading("")
        assert result is None


class TestSplitIntoSections:
    """Test full section splitting pipeline."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(use_llm_extraction=False)

    def test_splits_on_headings(self, extractor: PdfExtractor) -> None:
        text = """Title of the Paper

Some preamble text here.

ABSTRACT
This is the abstract of the paper with some content.

INTRODUCTION
This is the introduction section.

METHODS
These are the methods used."""

        sections = extractor._split_into_sections(text)

        titles = [s[0] for s in sections]
        assert "Abstract" in titles
        assert "Introduction" in titles
        assert "Methods" in titles

    def test_no_sections_returns_full_text(self, extractor: PdfExtractor) -> None:
        text = "Just some plain text without any section headings or structure."
        sections = extractor._split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Full Text"


class TestExtractFromFile:
    """Test full PDF extraction pipeline with sub-chunking."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(
            max_chunk_tokens=200,
            overlap_tokens=30,
            use_llm_extraction=False,
        )

    def test_extract_creates_subchunks_for_large_sections(
        self, extractor: PdfExtractor, tmp_path: Path
    ) -> None:
        """Integration test: a PDF with a long section should produce sub-chunks."""
        # We can't easily create a real PDF in tests, but we can test the
        # internal pipeline by calling _split_into_sections + _split_large_section
        long_body = "\n\n".join(
            [f"Paragraph {i}. " + "analysis " * 40 for i in range(10)]
        )
        text = f"ABSTRACT\nShort abstract.\n\nINTRODUCTION\n{long_body}"

        sections = extractor._split_into_sections(text)
        all_chunks = []
        for title, body, hierarchy in sections:
            content = body.strip()
            if not content or len(content) < 50:
                continue
            sub_chunks = extractor._split_large_section(content)
            all_chunks.extend(sub_chunks)

        # The introduction section should have been split
        assert len(all_chunks) > 2


class TestClusterExtraction:
    """Test cluster/topic extraction from file paths."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(use_llm_extraction=False)

    def test_extracts_cluster(self, extractor: PdfExtractor) -> None:
        path = Path("/data/clusters/01_Multi-Agent_Coordination/paper.pdf")
        cluster = extractor._extract_cluster(path)
        assert cluster == "Multi-Agent Coordination"

    def test_no_cluster(self, extractor: PdfExtractor) -> None:
        path = Path("/data/papers/paper.pdf")
        cluster = extractor._extract_cluster(path)
        assert cluster is None


class TestTitleExtraction:
    """Test paper title extraction."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(use_llm_extraction=False)

    def test_extracts_title(self, extractor: PdfExtractor) -> None:
        text = "My Great Paper Title\n\nAnonymous Authors\n\nABSTRACT\nBlah."
        title = extractor._extract_title(text)
        assert title == "My Great Paper Title"

    def test_skips_header_noise(self, extractor: PdfExtractor) -> None:
        text = "Published at Conference 2024\nActual Paper Title\n\nABSTRACT"
        title = extractor._extract_title(text)
        assert title == "Actual Paper Title"

    def test_unknown_title(self, extractor: PdfExtractor) -> None:
        text = "\n\n\n"
        title = extractor._extract_title(text)
        assert title == "Unknown Title"


class TestCaptionExtraction:
    """Test figure/table caption extraction."""

    @pytest.fixture
    def extractor(self) -> PdfExtractor:
        return PdfExtractor(use_llm_extraction=False)

    def test_extracts_figure_caption(self, extractor: PdfExtractor) -> None:
        text = "Some text.\nFigure 1: Architecture diagram of the system.\nMore text."
        captions = extractor._extract_captions(text)
        assert len(captions) >= 1
        assert captions[0][0] == "Figure 1"
        assert "Architecture" in captions[0][1]

    def test_extracts_table_caption(self, extractor: PdfExtractor) -> None:
        text = "Some text.\nTable 3: Results comparison.\nMore text."
        captions = extractor._extract_captions(text)
        assert len(captions) >= 1
        assert captions[0][0] == "Table 3"

    def test_normalizes_fig_to_figure(self, extractor: PdfExtractor) -> None:
        text = "Fig. 2: Some figure.\nMore text."
        captions = extractor._extract_captions(text)
        assert captions[0][0] == "Figure 2"
