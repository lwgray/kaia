"""
MCP server exposing Kaia RAG capabilities.

Provides tools for semantic search over indexed codebases (marcus, cato,
marcus-mini, posidonius) and research papers, enabling Dr. Kaia Chen to
quickly look up implementation details, architectural patterns, and prior
art across all repositories.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from kaia.vector_store import VectorStore


# Tool name aliases — old marcus-only names kept for back-compat.
TOOL_ALIASES = {
    "search_marcus_architecture": "search_codebase",
    "query_implementation_details": "query_implementation",
    "find_usage_examples": "find_usage",
    "search_research_papers": "search_papers",
}


class KaiaMCPServer:
    """
    MCP server for Kaia multi-repo RAG system.

    Exposes search across all indexed repositories (code, docs, git history)
    and research papers. Each tool accepts an optional ``repository`` filter
    to scope results to one repo (e.g., "marcus", "cato", "marcus-mini",
    "posidonius", "papers").
    """

    def __init__(self, repo_roots: list[Path] | None = None):
        """
        Parameters
        ----------
        repo_roots : list[Path] | None
            Configured repository roots — used only for display/logging.
            Actual indexed data lives in the vector store.
        """
        self.repo_roots = repo_roots or []
        self.vector_store = VectorStore()
        self.server = Server("kaia")
        self._register_tools()

    @property
    def known_repos(self) -> list[str]:
        """Repository names derived from configured roots."""
        return [p.name for p in self.repo_roots]

    def _register_tools(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            repos_hint = (
                f" Indexed repos: {', '.join(self.known_repos)}, papers."
                if self.known_repos
                else ""
            )
            return [
                Tool(
                    name="search_codebase",
                    description=(
                        "Search indexed codebases for architectural patterns, "
                        "implementation details, design decisions, or code "
                        "examples." + repos_hint
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Search query (e.g., 'task coordination', "
                                    "'error handling pattern')"
                                ),
                            },
                            "top_k": {
                                "type": "number",
                                "description": "Number of results (default 10)",
                                "default": 10,
                            },
                            "repository": {
                                "type": "string",
                                "description": (
                                    "Optional: scope to a single repo "
                                    "(e.g., 'marcus', 'cato', 'marcus-mini', "
                                    "'posidonius'). Omit to search all."
                                ),
                            },
                            "file_filter": {
                                "type": "string",
                                "description": (
                                    "Optional: exact file path filter "
                                    "(e.g., 'src/core/coordinator.py')"
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="query_implementation",
                    description=(
                        "Get implementation details for a class, function, or "
                        "module across indexed codebases." + repos_hint
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "component": {
                                "type": "string",
                                "description": (
                                    "Component name (e.g., 'TaskCoordinator', "
                                    "'request_next_task')"
                                ),
                            },
                            "component_type": {
                                "type": "string",
                                "description": "'class', 'function', or 'module'",
                                "enum": ["class", "function", "module"],
                            },
                            "repository": {
                                "type": "string",
                                "description": (
                                    "Optional: scope to a single repo. "
                                    "Omit to search all."
                                ),
                            },
                        },
                        "required": ["component"],
                    },
                ),
                Tool(
                    name="find_usage",
                    description=(
                        "Find test files and usage examples for a component "
                        "across indexed codebases." + repos_hint
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "component": {
                                "type": "string",
                                "description": "Component to find examples for",
                            },
                            "repository": {
                                "type": "string",
                                "description": "Optional: scope to a single repo.",
                            },
                        },
                        "required": ["component"],
                    },
                ),
                Tool(
                    name="search_papers",
                    description=(
                        "Search indexed research papers (PDFs) for concepts, "
                        "techniques, algorithms, or related work."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Search query (e.g., 'multi-agent coordination', "
                                    "'diffusion-based copulas')"
                                ),
                            },
                            "top_k": {
                                "type": "number",
                                "description": "Number of results (default 10)",
                                "default": 10,
                            },
                            "cluster": {
                                "type": "string",
                                "description": (
                                    "Optional: filter by research cluster/topic "
                                    "(substring match on file path)"
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[TextContent]:
            # Resolve back-compat aliases
            canonical = TOOL_ALIASES.get(name, name)

            if canonical == "search_codebase":
                return await self._search_codebase(arguments)
            elif canonical == "query_implementation":
                return await self._query_implementation(arguments)
            elif canonical == "find_usage":
                return await self._find_usage(arguments)
            elif canonical == "search_papers":
                return await self._search_papers(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    @staticmethod
    def _build_where(conditions: dict[str, Any]) -> dict[str, Any] | None:
        """
        Build a Chroma `where` filter from a flat dict of conditions.

        Single condition: passed directly. Multiple: combined with $and.
        Empty: returns None.
        """
        active = {k: v for k, v in conditions.items() if v is not None}
        if not active:
            return None
        if len(active) == 1:
            return active
        return {"$and": [{k: v} for k, v in active.items()]}

    async def _search_codebase(
        self, args: dict[str, Any]
    ) -> list[TextContent]:
        query = args["query"]
        top_k = args.get("top_k", 10)

        where = self._build_where(
            {
                "repository": args.get("repository"),
                "file_path": args.get("file_filter"),
            }
        )

        results = self.vector_store.query(query, n_results=top_k, filters=where)
        return [
            TextContent(
                type="text",
                text=self._format_results_for_kaia(results, query),
            )
        ]

    async def _query_implementation(
        self, args: dict[str, Any]
    ) -> list[TextContent]:
        component = args["component"]
        component_type = args.get("component_type")
        repository = args.get("repository")

        conditions: dict[str, Any] = {"repository": repository}
        if component_type == "class":
            conditions["class_name"] = component
        elif component_type == "function":
            conditions["function_name"] = component
        elif component_type == "module":
            conditions["chunk_type"] = "module"

        where = self._build_where(conditions)
        results = self.vector_store.query(component, n_results=5, filters=where)

        if not results:
            return [
                TextContent(
                    type="text",
                    text=f"No implementation found for component: {component}",
                )
            ]

        return [
            TextContent(
                type="text",
                text=self._format_results_for_kaia(
                    results, f"Implementation of {component}"
                ),
            )
        ]

    async def _find_usage(self, args: dict[str, Any]) -> list[TextContent]:
        component = args["component"]
        repository = args.get("repository")

        where = self._build_where(
            {"repository": repository, "chunk_type": "test_example"}
        )
        results = self.vector_store.query(
            f"{component} test example usage", n_results=5, filters=where
        )

        if not results:
            # Fallback: any chunk in test files
            where = self._build_where({"repository": repository})
            all_results = self.vector_store.query(
                component, n_results=10, filters=where
            )
            results = [
                r
                for r in all_results
                if "test" in r["metadata"].get("file_path", "")
            ][:5]

        if not results:
            return [
                TextContent(
                    type="text",
                    text=f"No usage examples found for: {component}",
                )
            ]

        return [
            TextContent(
                type="text",
                text=self._format_results_for_kaia(
                    results, f"Usage examples for {component}"
                ),
            )
        ]

    async def _search_papers(self, args: dict[str, Any]) -> list[TextContent]:
        query = args["query"]
        top_k = args.get("top_k", 10)
        cluster = args.get("cluster")

        where = self._build_where(
            {"chunk_type": "pdf_section", "repository": "papers"}
        )
        results = self.vector_store.query(query, n_results=top_k, filters=where)

        # Fall back to chunk_type only if repository tag isn't on older chunks
        if not results:
            results = self.vector_store.query(
                query,
                n_results=top_k,
                filters={"chunk_type": "pdf_section"},
            )

        if cluster and results:
            cluster_lower = cluster.lower()
            cluster_under = cluster_lower.replace(" ", "_")
            results = [
                r
                for r in results
                if cluster_under in r["metadata"].get("file_path", "").lower()
                or cluster_lower
                in r["metadata"]
                .get("file_path", "")
                .lower()
                .replace("_", " ")
            ]

        return [
            TextContent(
                type="text",
                text=self._format_paper_results(results, query),
            )
        ]

    MAX_RESULT_CHARS = 5000

    @staticmethod
    def _truncate_content(content: str, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content

        truncated = content[:max_chars]
        last_para = truncated.rfind("\n\n")
        if last_para > max_chars * 0.6:
            truncated = truncated[:last_para]
        else:
            last_sentence = max(
                truncated.rfind(". "),
                truncated.rfind(".\n"),
                truncated.rfind("? "),
                truncated.rfind("! "),
            )
            if last_sentence > max_chars * 0.6:
                truncated = truncated[: last_sentence + 1]

        return truncated + "\n\n[... truncated]"

    def _format_paper_results(
        self, results: list[dict[str, Any]], query: str
    ) -> str:
        formatted = f"# Research Paper Search: {query}\n\n"

        if not results:
            return formatted + "*No results found*"

        formatted += f"Found {len(results)} relevant results:\n\n"

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            relevance = 1 - result["distance"]

            formatted += f"## Result {i} (relevance: {relevance:.2f})\n\n"

            file_path = metadata.get("file_path", "Unknown")
            formatted += f"**File**: `{file_path}`\n"

            hierarchy = metadata.get("section_hierarchy")
            if hierarchy:
                if isinstance(hierarchy, list):
                    formatted += f"**Section**: {' > '.join(hierarchy)}\n"
                else:
                    formatted += f"**Section**: {hierarchy}\n"

            content = self._truncate_content(
                result["content"], self.MAX_RESULT_CHARS
            )
            formatted += f"\n{content}\n\n"
            formatted += "---\n\n"

        return formatted

    @staticmethod
    def _format_results_for_kaia(
        results: list[dict[str, Any]], query: str
    ) -> str:
        formatted = f"# Codebase Search: {query}\n\n"

        if not results:
            return formatted + "*No results found*"

        formatted += f"Found {len(results)} relevant results:\n\n"

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            relevance = 1 - result["distance"]

            formatted += f"## Result {i} (relevance: {relevance:.2f})\n\n"

            repository = metadata.get("repository")
            if repository and repository != "unknown":
                formatted += f"**Repo**: `{repository}`\n"

            file_path = metadata.get("file_path", "Unknown")
            formatted += f"**File**: `{file_path}`\n"

            chunk_type = metadata.get("chunk_type", "Unknown")
            formatted += f"**Type**: {chunk_type}\n"

            if metadata.get("class_name"):
                formatted += f"**Class**: {metadata['class_name']}\n"
            if metadata.get("function_name"):
                formatted += f"**Function**: {metadata['function_name']}\n"
            if metadata.get("line_start"):
                line_ref = f"L{metadata['line_start']}"
                if metadata.get("line_end"):
                    line_ref += f"-L{metadata['line_end']}"
                formatted += f"**Lines**: {line_ref}\n"

            if chunk_type in ("function", "class", "module", "test_example"):
                formatted += "\n```python\n"
                formatted += result["content"]
                formatted += "\n```\n\n"
            else:
                formatted += f"\n{result['content']}\n\n"
            formatted += "---\n\n"

        return formatted

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def _parse_repo_roots() -> list[Path]:
    """
    Resolve repository roots from environment.

    Priority:
      1. KAIA_REPOS — comma-separated paths
      2. MARCUS_ROOT — single path (back-compat)
      3. Default: /Users/lwgray/dev/marcus
    """
    kaia_repos = os.getenv("KAIA_REPOS")
    if kaia_repos:
        return [
            Path(p.strip()).expanduser().resolve()
            for p in kaia_repos.split(",")
            if p.strip()
        ]

    marcus_root = os.getenv("MARCUS_ROOT", "/Users/lwgray/dev/marcus")
    return [Path(marcus_root).expanduser().resolve()]


async def main() -> None:
    repo_roots = _parse_repo_roots()
    server = KaiaMCPServer(repo_roots=repo_roots)

    existing_count = server.vector_store.count()
    if existing_count == 0:
        print(
            "⚠️  Vector store is empty. Run `kaia index -r <repo> [-r <repo>...]` "
            "before serving.",
            flush=True,
        )
    else:
        print(
            f"✓ Using existing index ({existing_count} chunks). "
            f"Configured repos: {[p.name for p in repo_roots]}",
            flush=True,
        )

    print("Starting Kaia MCP server...", flush=True)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
