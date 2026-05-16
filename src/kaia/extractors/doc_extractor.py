"""
Extract and chunk markdown documentation.

This extractor parses markdown files and creates chunks based on section
structure, preserving hierarchical context and extracting code examples.
"""

import re
from pathlib import Path

from kaia.models import ChunkMetadata, ChunkType, DocChunk


class DocExtractor:
    """
    Extract chunks from markdown documentation files.

    Creates chunks for:
    - Documentation sections (by headers)
    - Code examples within documentation
    """

    def extract_from_file(
        self, file_path: Path, repository: str = "unknown"
    ) -> list[DocChunk]:
        """
        Extract all chunks from a markdown file.

        Parameters
        ----------
        file_path : Path
            Path to markdown file
        repository : str
            Repository name (e.g., "marcus", "cato")

        Returns
        -------
        list[DocChunk]
            All extracted chunks (sections, code examples)
        """
        content = file_path.read_text()
        chunks: list[DocChunk] = []

        sections = self._parse_sections(content, file_path, repository)
        chunks.extend(sections)

        code_examples = self._extract_code_examples(content, file_path, repository)
        chunks.extend(code_examples)

        return chunks

    def _parse_sections(
        self, content: str, file_path: Path, repository: str
    ) -> list[DocChunk]:
        """
        Parse markdown into section-based chunks.

        Parameters
        ----------
        content : str
            Markdown content
        file_path : Path
            Source file path

        Returns
        -------
        list[DocChunk]
            Section chunks
        """
        chunks: list[DocChunk] = []
        lines = content.splitlines()

        current_section = []
        current_hierarchy: list[str] = []
        current_level = 0

        for i, line in enumerate(lines):
            # Check if this is a header
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if header_match:
                # Save previous section if it exists
                if current_section:
                    section_content = "\n".join(current_section)
                    if section_content.strip():
                        chunks.append(
                            self._create_section_chunk(
                                section_content,
                                current_hierarchy.copy(),
                                file_path,
                                repository,
                            )
                        )

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Update hierarchy
                if level <= current_level:
                    # Going up or same level - pop back
                    current_hierarchy = current_hierarchy[: level - 1]
                current_hierarchy.append(title)
                current_level = level

                current_section = [line]
            else:
                current_section.append(line)

        # Save final section
        if current_section:
            section_content = "\n".join(current_section)
            if section_content.strip():
                chunks.append(
                    self._create_section_chunk(
                        section_content,
                        current_hierarchy,
                        file_path,
                        repository,
                    )
                )

        return chunks

    def _create_section_chunk(
        self,
        content: str,
        hierarchy: list[str],
        file_path: Path,
        repository: str,
    ) -> DocChunk:
        """
        Create a documentation section chunk.

        Parameters
        ----------
        content : str
            Section content
        hierarchy : list[str]
            Section hierarchy (e.g., ["ERROR_HANDLING", "RETRY_PATTERNS"])
        file_path : Path
            Source file path

        Returns
        -------
        DocChunk
            Documentation chunk
        """
        # Extract keywords from content (simple approach)
        # Look for ALL_CAPS words and common patterns
        keywords = re.findall(r"\b[A-Z_]{3,}\b", content)
        keywords = list(set(keywords))[:10]  # Limit to 10 unique keywords

        # Higher importance for CLAUDE.md and top-level sections
        importance = 0.7
        if "CLAUDE.md" in str(file_path):
            importance = 0.9
        if len(hierarchy) == 1:
            importance += 0.1

        return DocChunk(
            content=content,
            chunk_type=ChunkType.DOC_SECTION,
            metadata=ChunkMetadata(
                repository=repository,
                file_path=str(file_path),
                section_hierarchy=hierarchy if hierarchy else [],
                keywords=keywords if keywords else [],
                importance=min(1.0, importance),
            ),
        )

    def _extract_code_examples(
        self, content: str, file_path: Path, repository: str
    ) -> list[DocChunk]:
        """
        Extract code examples from markdown.

        Parameters
        ----------
        content : str
            Markdown content
        file_path : Path
            Source file path

        Returns
        -------
        list[DocChunk]
            Code example chunks
        """
        chunks: list[DocChunk] = []

        # Find code blocks with ```language syntax
        pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2)

            # Only index substantial code examples (more than 2 lines)
            if code.count("\n") < 2:
                continue

            # Find context around the code block (preceding text)
            start_pos = match.start()
            context_start = max(0, start_pos - 200)
            context = content[context_start:start_pos].strip()

            # Extract last sentence or paragraph as description
            description_lines = context.split("\n")[-3:]
            description = " ".join(description_lines).strip()

            chunk_content = f"Language: {language}\n\n{code}"
            if description:
                chunk_content = f"{description}\n\n{chunk_content}"

            chunks.append(
                DocChunk(
                    content=chunk_content,
                    chunk_type=ChunkType.DOC_CODE_EXAMPLE,
                    metadata=ChunkMetadata(
                        repository=repository,
                        file_path=str(file_path),
                        keywords=(
                            [language, "example", "code"]
                            if language
                            else ["example", "code"]
                        ),
                        importance=0.8,
                    ),
                )
            )

        return chunks
