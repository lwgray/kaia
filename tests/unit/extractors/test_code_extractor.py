"""
Unit tests for Python code extraction.
"""

from pathlib import Path

import pytest

from kaia.extractors.code_extractor import CodeExtractor
from kaia.models import ChunkType


class TestCodeExtractor:
    """Test suite for CodeExtractor"""

    @pytest.fixture
    def extractor(self) -> CodeExtractor:
        """Create a CodeExtractor instance"""
        return CodeExtractor()

    def test_extract_function_chunks(
        self, extractor: CodeExtractor, sample_python_file: Path
    ) -> None:
        """Test extracting function-level chunks from Python file"""
        chunks = extractor.extract_from_file(sample_python_file)

        # Should find the method and standalone function
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(function_chunks) == 2  # request_next_task + standalone_function

        # Check the method chunk
        method_chunk = next(
            c
            for c in function_chunks
            if c.metadata.function_name == "request_next_task"
        )
        assert "request_next_task" in method_chunk.content
        assert method_chunk.metadata.function_name == "request_next_task"
        assert method_chunk.metadata.class_name == "TaskCoordinator"
        assert "Request next available task" in method_chunk.content  # Has docstring
        assert method_chunk.metadata.line_start == 6  # Function starts at line 6
        assert method_chunk.metadata.file_path == str(sample_python_file)

    def test_extract_class_chunk(
        self, extractor: CodeExtractor, sample_python_file: Path
    ) -> None:
        """Test extracting class-level overview chunk"""
        chunks = extractor.extract_from_file(sample_python_file)

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1

        chunk = class_chunks[0]
        assert "TaskCoordinator" in chunk.content
        assert chunk.metadata.class_name == "TaskCoordinator"
        assert "Coordinates task assignment" in chunk.content  # Has docstring
        assert chunk.metadata.importance == 0.8  # Classes are important
        assert "request_next_task" in chunk.metadata.keywords  # Method in keywords

    def test_extract_module_chunk(
        self, extractor: CodeExtractor, sample_python_file: Path
    ) -> None:
        """Test extracting module-level summary chunk"""
        chunks = extractor.extract_from_file(sample_python_file)

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE]
        assert len(module_chunks) == 1

        chunk = module_chunks[0]
        assert "Sample module for testing" in chunk.content  # Module docstring
        assert "TaskCoordinator" in chunk.content  # Class name
        assert "standalone_function" in chunk.content  # Function name
        assert chunk.metadata.importance == 0.6

    def test_extract_all_chunk_types(
        self, extractor: CodeExtractor, sample_python_file: Path
    ) -> None:
        """Test that all three chunk types are extracted"""
        chunks = extractor.extract_from_file(sample_python_file)

        chunk_types = {c.chunk_type for c in chunks}
        assert ChunkType.MODULE in chunk_types
        assert ChunkType.CLASS in chunk_types
        assert ChunkType.FUNCTION in chunk_types

    def test_private_function_importance(
        self, extractor: CodeExtractor, tmp_path: Path
    ) -> None:
        """Test that private functions have lower importance"""
        file = tmp_path / "private.py"
        file.write_text('''
def public_function():
    """Public function"""
    pass

def _private_function():
    """Private function"""
    pass

def __dunder_function__():
    """Dunder function"""
    pass
''')

        chunks = extractor.extract_from_file(file)
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        public = next(
            c for c in function_chunks if c.metadata.function_name == "public_function"
        )
        private = next(
            c
            for c in function_chunks
            if c.metadata.function_name == "_private_function"
        )
        dunder = next(
            c
            for c in function_chunks
            if c.metadata.function_name == "__dunder_function__"
        )

        assert public.metadata.importance == 0.7  # Default
        assert private.metadata.importance == 0.5  # Lower for private
        assert dunder.metadata.importance == 0.4  # Lowest for dunder

    def test_extract_from_empty_file(
        self, extractor: CodeExtractor, tmp_path: Path
    ) -> None:
        """Test extracting from an empty Python file"""
        file = tmp_path / "empty.py"
        file.write_text("")

        chunks = extractor.extract_from_file(file)

        # Should only have module chunk
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.MODULE

    def test_extract_from_file_with_syntax_error(
        self, extractor: CodeExtractor, tmp_path: Path
    ) -> None:
        """Test handling of files with syntax errors"""
        file = tmp_path / "bad_syntax.py"
        file.write_text("def broken(\n    # Missing closing paren")

        chunks = extractor.extract_from_file(file)

        # Should return empty list for files with syntax errors
        assert chunks == []

    def test_async_function_extraction(
        self, extractor: CodeExtractor, tmp_path: Path
    ) -> None:
        """Test that async functions are extracted correctly"""
        file = tmp_path / "async.py"
        file.write_text('''
async def async_function():
    """An async function"""
    await something()

class AsyncClass:
    """Class with async method"""

    async def async_method(self):
        """An async method"""
        pass
''')

        chunks = extractor.extract_from_file(file)
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        assert len(function_chunks) == 2
        assert any(
            c.metadata.function_name == "async_function" for c in function_chunks
        )
        assert any(c.metadata.function_name == "async_method" for c in function_chunks)

        # Async method should have class name
        async_method = next(
            c for c in function_chunks if c.metadata.function_name == "async_method"
        )
        assert async_method.metadata.class_name == "AsyncClass"

    def test_metadata_line_numbers(
        self, extractor: CodeExtractor, sample_python_file: Path
    ) -> None:
        """Test that line numbers are correctly captured"""
        chunks = extractor.extract_from_file(sample_python_file)

        for chunk in chunks:
            # Module chunks don't have line numbers
            if chunk.chunk_type == ChunkType.MODULE:
                continue

            assert chunk.metadata.line_start is not None
            assert chunk.metadata.line_end is not None
            assert chunk.metadata.line_end >= chunk.metadata.line_start

    def test_file_path_in_metadata(
        self, extractor: CodeExtractor, sample_python_file: Path
    ) -> None:
        """Test that file path is stored in metadata"""
        chunks = extractor.extract_from_file(sample_python_file)

        for chunk in chunks:
            assert chunk.metadata.file_path == str(sample_python_file)
