"""
Extract and chunk git history.

This extractor parses git commits to understand code evolution, design
decisions, and why changes were made.
"""

from datetime import datetime
from pathlib import Path

import git

from kaia.models import ChunkMetadata, ChunkType, CommitChunk


class GitExtractor:
    """
    Extract chunks from git commit history.

    Creates chunks for:
    - Individual commits with their changes
    - File evolution summaries
    """

    def extract_recent_commits(
        self,
        repo_path: Path,
        limit: int = 100,
        repository: str = "unknown",
    ) -> list[CommitChunk]:
        """
        Extract recent commits from repository.

        Parameters
        ----------
        repo_path : Path
            Path to git repository
        limit : int
            Number of recent commits to extract
        repository : str
            Repository name (e.g., "marcus", "cato")

        Returns
        -------
        list[CommitChunk]
            Commit chunks
        """
        try:
            repo = git.Repo(repo_path)
        except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
            return []

        chunks: list[CommitChunk] = []

        commits = list(repo.iter_commits("HEAD", max_count=limit))

        for commit in commits:
            chunk = self._create_commit_chunk(commit, repository)
            chunks.append(chunk)

        return chunks

    def _create_commit_chunk(
        self, commit: git.Commit, repository: str
    ) -> CommitChunk:
        """
        Create a chunk from a git commit.

        Parameters
        ----------
        commit : git.Commit
            Git commit object

        Returns
        -------
        CommitChunk
            Commit chunk
        """
        # Get commit message
        message = commit.message.strip()

        # Get changed files
        changed_files = []
        try:
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
                changed_files = [d.a_path or d.b_path for d in diffs]
        except Exception:
            # Handle commits without parents (initial commit)
            pass

        # Build commit summary
        summary_parts = [
            f"Commit: {commit.hexsha[:7]}",
            f"Author: {commit.author.name}",
            f"Date: {datetime.fromtimestamp(commit.committed_date).isoformat()}",
            f"\nMessage:\n{message}",
        ]

        if changed_files:
            summary_parts.append(
                f"\nChanged files:\n"
                + "\n".join(f"  - {f}" for f in changed_files[:10])
            )
            if len(changed_files) > 10:
                summary_parts.append(f"  ... and {len(changed_files) - 10} more")

        content = "\n".join(summary_parts)

        # Determine importance based on commit type
        importance = 0.5
        message_lower = message.lower()
        if any(word in message_lower for word in ["feat:", "feature:", "add:"]):
            importance = 0.7
        elif any(word in message_lower for word in ["fix:", "bug:"]):
            importance = 0.6
        elif any(word in message_lower for word in ["refactor:", "perf:"]):
            importance = 0.5

        # Extract keywords from commit message
        keywords = []
        if changed_files:
            # Use file paths as keywords
            keywords = [Path(f).stem for f in changed_files[:5]]

        return CommitChunk(
            content=content,
            chunk_type=ChunkType.GIT_COMMIT,
            metadata=ChunkMetadata(
                repository=repository,
                commit_hash=commit.hexsha[:7],
                author=commit.author.name,
                keywords=keywords if keywords else [],
                importance=importance,
            ),
        )
