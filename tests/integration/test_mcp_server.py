"""
Integration tests for MCP server.

Verifies end-to-end functionality of the Kaia multi-repo MCP server,
including indexing via MarcusIndexer and querying through the renamed
tool methods.
"""

from pathlib import Path

import pytest
from mcp.types import TextContent

from kaia.indexer import MarcusIndexer
from kaia.mcp_server import KaiaMCPServer
from kaia.vector_store import VectorStore


@pytest.mark.integration
class TestKaiaMCPServer:
    """Integration tests for KaiaMCPServer."""

    @pytest.fixture
    def test_marcus_repo(self, tmp_path: Path) -> Path:
        """Create a minimal test Marcus repository."""
        marcus = tmp_path / "marcus"
        marcus.mkdir()

        src = marcus / "src" / "core"
        src.mkdir(parents=True)

        (src / "coordinator.py").write_text('''"""Task coordination module"""

class TaskCoordinator:
    """Coordinates task assignment to agents"""

    def request_next_task(self, agent_id: str) -> dict:
        """
        Request next available task for agent.

        This implements the board-mediated coordination pattern.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier

        Returns
        -------
        dict
            Task data or empty dict if none available
        """
        return self._get_task_from_board(agent_id)

    def _get_task_from_board(self, agent_id: str) -> dict:
        """Internal method to fetch task from coordination board"""
        return {}
''')

        (src / "error_framework.py").write_text('''"""Error handling framework"""

class MarcusBaseError(Exception):
    """Base class for all Marcus errors"""
    pass

class TaskAssignmentError(MarcusBaseError):
    """Error during task assignment"""
    pass

def with_retry(config):
    """Decorator for retry logic"""
    def decorator(func):
        return func
    return decorator
''')

        return marcus

    @pytest.fixture
    async def server(
        self, test_marcus_repo: Path, tmp_path: Path
    ) -> KaiaMCPServer:
        """Create MCP server with a freshly indexed test repository."""
        # Build a vector store scoped to this test
        store = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_marcus",
        )

        # Index via the real indexer (the only supported path now)
        indexer = MarcusIndexer(repo_roots=[test_marcus_repo])
        indexer.vector_store = store
        indexer.index_all()

        server = KaiaMCPServer(repo_roots=[test_marcus_repo])
        server.vector_store = store
        return server

    @pytest.mark.asyncio
    async def test_search_codebase_tool(self, server: KaiaMCPServer) -> None:
        """search_codebase returns formatted results."""
        results = await server._search_codebase(
            {"query": "task coordination", "top_k": 5}
        )

        assert len(results) == 1
        assert isinstance(results[0], TextContent)
        content = results[0].text

        assert "Codebase Search" in content
        assert "task coordination" in content.lower()

    @pytest.mark.asyncio
    async def test_search_codebase_with_repository_filter(
        self, server: KaiaMCPServer
    ) -> None:
        """repository filter scopes results to one repo."""
        results = await server._search_codebase(
            {"query": "coordination", "top_k": 5, "repository": "marcus"}
        )

        assert len(results) == 1
        content = results[0].text
        assert "Result" in content
        # Indexed repo name is the directory name "marcus"
        assert "marcus" in content.lower()

    @pytest.mark.asyncio
    async def test_search_codebase_with_unknown_repo_returns_nothing(
        self, server: KaiaMCPServer
    ) -> None:
        """Filtering on a non-indexed repo returns empty."""
        results = await server._search_codebase(
            {"query": "coordination", "top_k": 5, "repository": "cato"}
        )

        content = results[0].text
        assert "No results found" in content

    @pytest.mark.asyncio
    async def test_query_implementation_class(
        self, server: KaiaMCPServer
    ) -> None:
        """query_implementation finds a class by name."""
        results = await server._query_implementation(
            {"component": "TaskCoordinator", "component_type": "class"}
        )

        assert len(results) == 1
        content = results[0].text
        assert "TaskCoordinator" in content
        assert "class" in content.lower()

    @pytest.mark.asyncio
    async def test_query_implementation_function(
        self, server: KaiaMCPServer
    ) -> None:
        """query_implementation finds a function by name."""
        results = await server._query_implementation(
            {"component": "request_next_task", "component_type": "function"}
        )

        assert len(results) == 1
        content = results[0].text
        assert "request_next_task" in content

    @pytest.mark.asyncio
    async def test_query_nonexistent_component(
        self, server: KaiaMCPServer
    ) -> None:
        """Nonexistent component returns the not-found message."""
        results = await server._query_implementation(
            {"component": "NonExistentClass", "component_type": "class"}
        )

        assert len(results) == 1
        assert "No implementation found" in results[0].text

    @pytest.mark.asyncio
    async def test_find_usage(self, server: KaiaMCPServer) -> None:
        """find_usage returns a single TextContent result."""
        results = await server._find_usage({"component": "TaskCoordinator"})

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_format_results_for_kaia(self) -> None:
        """Result formatting includes repo, file, type, code block."""
        results = [
            {
                "id": "test-1",
                "content": "def foo(): pass",
                "metadata": {
                    "repository": "marcus",
                    "file_path": "src/core/test.py",
                    "chunk_type": "function",
                    "function_name": "foo",
                    "line_start": 10,
                    "line_end": 12,
                },
                "distance": 0.2,
            }
        ]

        formatted = KaiaMCPServer._format_results_for_kaia(results, "test query")

        assert "# Codebase Search" in formatted
        assert "test query" in formatted
        assert "Result 1" in formatted
        assert "relevance:" in formatted
        assert "marcus" in formatted
        assert "src/core/test.py" in formatted
        assert "function" in formatted
        assert "foo" in formatted
        assert "```python" in formatted

    @pytest.mark.asyncio
    async def test_server_has_tools_registered(
        self, test_marcus_repo: Path
    ) -> None:
        """Server initializes with the kaia MCP server name."""
        server = KaiaMCPServer(repo_roots=[test_marcus_repo])
        assert server.server is not None
        assert server.server.name == "kaia"
        assert server.known_repos == ["marcus"]

    def test_build_where_single_condition(self) -> None:
        """Single condition is passed through directly."""
        where = KaiaMCPServer._build_where({"repository": "marcus"})
        assert where == {"repository": "marcus"}

    def test_build_where_multiple_conditions(self) -> None:
        """Multiple conditions are combined under $and."""
        where = KaiaMCPServer._build_where(
            {"repository": "marcus", "chunk_type": "function"}
        )
        assert where == {
            "$and": [
                {"repository": "marcus"},
                {"chunk_type": "function"},
            ]
        }

    def test_build_where_drops_none(self) -> None:
        """None values are filtered out."""
        where = KaiaMCPServer._build_where(
            {"repository": "marcus", "file_path": None}
        )
        assert where == {"repository": "marcus"}

    def test_build_where_empty(self) -> None:
        """All-None / empty input returns None (no filter)."""
        assert KaiaMCPServer._build_where({}) is None
        assert KaiaMCPServer._build_where({"repository": None}) is None

    def test_alias_resolution(self) -> None:
        """Old marcus-only tool names map to new canonical names."""
        from kaia.mcp_server import TOOL_ALIASES

        assert TOOL_ALIASES["search_marcus_architecture"] == "search_codebase"
        assert (
            TOOL_ALIASES["query_implementation_details"]
            == "query_implementation"
        )
        assert TOOL_ALIASES["find_usage_examples"] == "find_usage"
        assert TOOL_ALIASES["search_research_papers"] == "search_papers"
