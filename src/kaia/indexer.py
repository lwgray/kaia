"""
Main indexing pipeline orchestrator.

Coordinates extraction from multiple sources (code, docs, git, PDFs) and
indexes them into the vector store.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from kaia.extractors.code_extractor import CodeExtractor
from kaia.extractors.doc_extractor import DocExtractor
from kaia.extractors.git_extractor import GitExtractor
from kaia.extractors.pdf_extractor import PdfExtractor
from kaia.vector_store import VectorStore


class MarcusIndexer:
    """
    Orchestrates indexing of repository codebases and PDF research papers.

    Coordinates multiple extractors to build a comprehensive index
    of code, documentation, git history, and research papers.
    """

    def __init__(
        self,
        repo_roots: list[Path] | None = None,
        pdf_paths: list[Path] | None = None,
    ):
        """
        Initialize indexer.

        Parameters
        ----------
        repo_roots : list[Path] | None
            Paths to repository roots to index
        pdf_paths : list[Path] | None
            Paths to directories of PDF research papers to index
        """
        self.repo_roots = repo_roots or []
        self.pdf_paths = pdf_paths or []
        self.vector_store = VectorStore()
        self.console = Console()

        self.code_extractor = CodeExtractor()
        self.doc_extractor = DocExtractor()
        self.git_extractor = GitExtractor()
        self.pdf_extractor = PdfExtractor()

    def index_all(self) -> dict[str, Any]:
        """
        Index all configured sources.

        Returns
        -------
        dict[str, Any]
            Statistics about indexed content
        """
        stats: dict[str, Any] = {
            "code": {"chunks": 0, "files": 0},
            "docs": {"chunks": 0, "files": 0},
            "git": {"chunks": 0},
            "pdfs": {"chunks": 0, "files": 0},
            "repos": {},
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            # Index each repository
            for repo_root in self.repo_roots:
                repo_name = repo_root.name
                self.console.print(
                    f"\n[bold]Indexing repo: {repo_name}[/bold] ({repo_root})"
                )
                repo_stats = self._index_repo(repo_root, repo_name, progress)
                stats["repos"][repo_name] = repo_stats
                stats["code"]["chunks"] += repo_stats["code"]["chunks"]
                stats["code"]["files"] += repo_stats["code"]["files"]
                stats["docs"]["chunks"] += repo_stats["docs"]["chunks"]
                stats["docs"]["files"] += repo_stats["docs"]["files"]
                stats["git"]["chunks"] += repo_stats["git"]["chunks"]

            # Index PDFs from all paths
            for pdf_path in self.pdf_paths:
                if not pdf_path.exists():
                    self.console.print(
                        f"[yellow]Warning: PDF path not found: {pdf_path}[/yellow]"
                    )
                    continue
                self.console.print(
                    f"\n[bold]Indexing PDFs: {pdf_path.name}[/bold] ({pdf_path})"
                )
                pdf_files = sum(1 for _ in pdf_path.rglob("*.pdf"))
                if pdf_files > 0:
                    pdf_task = progress.add_task(
                        f"Indexing PDFs from {pdf_path.name}...",
                        total=pdf_files,
                    )
                    pdf_chunks = self._index_pdfs(pdf_path, progress, pdf_task)
                    stats["pdfs"]["chunks"] += len(pdf_chunks)
                    stats["pdfs"]["files"] += pdf_files

        stats["total_chunks"] = (
            stats["code"]["chunks"]
            + stats["docs"]["chunks"]
            + stats["git"]["chunks"]
            + stats["pdfs"]["chunks"]
        )

        return stats

    def _index_repo(
        self, repo_root: Path, repo_name: str, progress: Progress
    ) -> dict[str, Any]:
        """Index a single repository (code, docs, git)."""
        repo_stats: dict[str, Any] = {}

        py_files = self._count_files(repo_root, "*.py")
        md_files = self._count_files(repo_root, "*.md")

        # Index Python code
        code_task = progress.add_task(
            f"Indexing {repo_name} Python code...", total=py_files
        )
        code_chunks = self._index_code(repo_root, repo_name, progress, code_task)
        repo_stats["code"] = {"chunks": len(code_chunks), "files": py_files}

        # Index documentation
        doc_task = progress.add_task(
            f"Indexing {repo_name} documentation...", total=md_files
        )
        doc_chunks = self._index_docs(repo_root, repo_name, progress, doc_task)
        repo_stats["docs"] = {"chunks": len(doc_chunks), "files": md_files}

        # Index git history
        git_task = progress.add_task(
            f"Indexing {repo_name} git history...", total=1
        )
        git_chunks = self._index_git(repo_root, repo_name)
        repo_stats["git"] = {"chunks": len(git_chunks)}
        progress.update(git_task, advance=1)

        return repo_stats

    def _index_code(
        self, repo_root: Path, repo_name: str, progress: Progress, task_id: Any
    ) -> list[Any]:
        """Index all Python files in a repository."""
        chunks = []
        for py_file in repo_root.rglob("*.py"):
            if self._should_skip(py_file):
                continue
            try:
                file_chunks = self.code_extractor.extract_from_file(
                    py_file, repository=repo_name
                )
                chunks.extend(file_chunks)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to extract from {py_file}: {e}[/yellow]"
                )
            finally:
                progress.update(task_id, advance=1)

        if chunks:
            self.vector_store.add_chunks(chunks)
        return chunks

    def _index_docs(
        self, repo_root: Path, repo_name: str, progress: Progress, task_id: Any
    ) -> list[Any]:
        """Index all markdown documentation in a repository."""
        chunks = []
        for md_file in repo_root.rglob("*.md"):
            if self._should_skip(md_file):
                continue
            try:
                file_chunks = self.doc_extractor.extract_from_file(
                    md_file, repository=repo_name
                )
                chunks.extend(file_chunks)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to extract from {md_file}: {e}[/yellow]"
                )
            finally:
                progress.update(task_id, advance=1)

        if chunks:
            self.vector_store.add_chunks(chunks)
        return chunks

    def _index_git(self, repo_root: Path, repo_name: str) -> list[Any]:
        """Index git history for a repository."""
        try:
            chunks = self.git_extractor.extract_recent_commits(
                repo_root, limit=100, repository=repo_name
            )
            if chunks:
                self.vector_store.add_chunks(chunks)
            return chunks
        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Failed to extract git history: {e}[/yellow]"
            )
            return []

    def _index_pdfs(
        self, pdf_path: Path, progress: Progress, task_id: Any
    ) -> list[Any]:
        """Index all PDF files from a directory."""
        chunks = []
        for pdf_file in pdf_path.rglob("*.pdf"):
            try:
                file_chunks = self.pdf_extractor.extract_from_file(
                    pdf_file, repository="papers"
                )
                chunks.extend(file_chunks)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to extract from {pdf_file}: {e}[/yellow]"
                )
            finally:
                progress.update(task_id, advance=1)

        if chunks:
            self.vector_store.add_chunks(chunks)
        return chunks

    @staticmethod
    def _should_skip(file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            ".venv",
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".eggs",
        ]
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)

    @staticmethod
    def _count_files(repo_root: Path, pattern: str) -> int:
        """Count files matching pattern, excluding skipped dirs."""
        skip_patterns = [
            ".venv",
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".eggs",
        ]
        count = 0
        for f in repo_root.rglob(pattern):
            if not any(p in str(f) for p in skip_patterns):
                count += 1
        return count
