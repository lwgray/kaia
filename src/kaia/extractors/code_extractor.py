"""
Extract and chunk Python code using AST parsing.

This extractor parses Python files and creates multiple chunks at different
granularities (function, class, module) to enable flexible retrieval.
"""

import ast
from pathlib import Path

from kaia.models import ChunkMetadata, ChunkType, CodeChunk


class CodeExtractor:
    """
    Extract chunks from Python source files using AST parsing.

    The extractor creates chunks at three levels:
    1. Function-level: Individual functions/methods with docstrings
    2. Class-level: Class overview with method signatures
    3. Module-level: Module docstring with class/function summaries
    """

    def extract_from_file(
        self, file_path: Path, repository: str = "unknown"
    ) -> list[CodeChunk]:
        """
        Extract all chunks from a Python file.

        Parameters
        ----------
        file_path : Path
            Path to Python source file
        repository : str
            Repository name (e.g., "marcus", "cato")

        Returns
        -------
        list[CodeChunk]
            All extracted chunks (module, classes, functions)
        """
        source = file_path.read_text()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Skip files with syntax errors
            return []

        chunks: list[CodeChunk] = []

        # Module-level chunk
        module_chunk = self._extract_module_chunk(tree, file_path, source, repository)
        chunks.append(module_chunk)

        # Extract classes and their methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Class overview chunk
                chunks.append(
                    self._extract_class_chunk(node, file_path, source, repository)
                )

                # Extract methods from class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(
                            self._extract_function_chunk(
                                item,
                                file_path,
                                source,
                                repository,
                                class_name=node.name,
                            )
                        )

        # Extract top-level functions (not inside classes)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(
                    self._extract_function_chunk(node, file_path, source, repository)
                )

        return chunks

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        source: str,
        repository: str,
        class_name: str | None = None,
    ) -> CodeChunk:
        """
        Extract a single function/method as a chunk.

        Parameters
        ----------
        node : ast.FunctionDef | ast.AsyncFunctionDef
            AST node for the function
        file_path : Path
            Source file path
        source : str
            Complete source code
        repository : str
            Repository name
        class_name : str | None
            Parent class name if this is a method

        Returns
        -------
        CodeChunk
            Function chunk with metadata
        """
        # Get source lines for this function
        lines = source.splitlines()
        if node.end_lineno is None:
            # Fallback if end_lineno is not available
            func_source = "\n".join(lines[node.lineno - 1 :])
        else:
            func_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])

        # Determine importance based on naming patterns
        importance = 0.7  # Default for functions
        if node.name.startswith("_") and not node.name.startswith("__"):
            importance = 0.5  # Private methods less important
        elif node.name.startswith("__"):
            importance = 0.4  # Dunder methods even less important

        return CodeChunk(
            content=func_source,
            chunk_type=ChunkType.FUNCTION,
            metadata=ChunkMetadata(
                repository=repository,
                file_path=str(file_path),
                function_name=node.name,
                class_name=class_name,
                line_start=node.lineno,
                line_end=node.end_lineno,
                importance=importance,
            ),
        )

    def _extract_class_chunk(
        self, node: ast.ClassDef, file_path: Path, source: str, repository: str
    ) -> CodeChunk:
        """
        Extract class overview as a chunk.

        Parameters
        ----------
        node : ast.ClassDef
            AST node for the class
        file_path : Path
            Source file path
        source : str
            Complete source code
        repository : str
            Repository name

        Returns
        -------
        CodeChunk
            Class overview chunk
        """
        lines = source.splitlines()

        # Get class signature
        class_start = [lines[node.lineno - 1]]

        # Get class docstring
        docstring = ast.get_docstring(node) or ""
        if docstring:
            class_start.append(f'    """{docstring}"""')

        # Get method signatures (first line of each method)
        method_sigs = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_line = lines[item.lineno - 1].strip()
                method_sigs.append(f"    {method_line}")

        class_overview = "\n".join(class_start + [""] + method_sigs)

        # Extract method names for metadata
        method_names = [
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        return CodeChunk(
            content=class_overview,
            chunk_type=ChunkType.CLASS,
            metadata=ChunkMetadata(
                repository=repository,
                file_path=str(file_path),
                class_name=node.name,
                line_start=node.lineno,
                line_end=node.end_lineno,
                keywords=method_names,  # Store method names as keywords
                importance=0.8,  # Classes are architecturally important
            ),
        )

    def _extract_module_chunk(
        self, tree: ast.Module, file_path: Path, source: str, repository: str
    ) -> CodeChunk:
        """
        Extract module-level summary.

        Parameters
        ----------
        tree : ast.Module
            AST tree for the module
        file_path : Path
            Source file path
        source : str
            Complete source code
        repository : str
            Repository name

        Returns
        -------
        CodeChunk
            Module summary chunk
        """
        # Get module docstring
        module_doc = ast.get_docstring(tree) or f"Module: {file_path.name}"

        # Extract top-level names
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        functions = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        # Create summary
        summary_parts = [module_doc]
        if classes:
            summary_parts.append(f"\nClasses: {', '.join(classes)}")
        if functions:
            summary_parts.append(f"Functions: {', '.join(functions)}")

        summary = "\n".join(summary_parts)

        return CodeChunk(
            content=summary,
            chunk_type=ChunkType.MODULE,
            metadata=ChunkMetadata(
                repository=repository,
                file_path=str(file_path),
                keywords=classes + functions,
                importance=0.6,  # Module summaries moderately important
            ),
        )
