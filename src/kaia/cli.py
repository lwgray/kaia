"""
CLI for managing Kaia indexing and queries.

Provides commands to index the Marcus codebase, search for content,
and start the MCP server.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kaia.indexer import MarcusIndexer
from kaia.vector_store import VectorStore

app = typer.Typer(help="Kaia - RAG system for Marcus codebase")
console = Console()


@app.command()
def index(
    repo_paths: list[str] = typer.Option(
        [],
        "--repo-path",
        "-r",
        help="Path to a repository to index (can be specified multiple times)",
    ),
    pdf_paths: list[str] = typer.Option(
        [],
        "--pdf-path",
        "-p",
        help="Path to a directory of PDFs to index (can be specified multiple times)",
    ),
) -> None:
    """Index code repositories and/or PDF research papers.

    Examples:

        # Single repo
        kaia index -r ~/dev/marcus

        # Multiple repos
        kaia index -r ~/dev/marcus -r ~/dev/cato

        # Repo + specific PDF cluster
        kaia index -r ~/dev/marcus -p ~/dev/marcus_research/clusters/01_Multi-Agent_Coordination_Communication

        # Multiple PDF clusters
        kaia index -p ~/dev/marcus_research/clusters/01_Multi-Agent_Coordination_Communication -p ~/dev/marcus_research/clusters/07_Agentic_Architectures_Orchestration

        # Everything
        kaia index -r ~/dev/marcus -r ~/dev/cato -p ~/dev/marcus_research/clusters
    """
    if not repo_paths and not pdf_paths:
        console.print("[red]Error: Specify at least one --repo-path or --pdf-path[/red]")
        raise typer.Exit(1)

    resolved_repos = [Path(p).expanduser().resolve() for p in repo_paths]
    resolved_pdfs = [Path(p).expanduser().resolve() for p in pdf_paths]

    console.print("[bold]Kaia Indexer[/bold]\n")
    for rp in resolved_repos:
        console.print(f"  Repo: {rp}")
    for pp in resolved_pdfs:
        console.print(f"  PDFs: {pp}")
    console.print()

    indexer = MarcusIndexer(repo_roots=resolved_repos, pdf_paths=resolved_pdfs)
    stats = indexer.index_all()

    console.print("\n[green]✓ Indexing complete![/green]\n")

    # Create stats table
    table = Table(title="Indexing Statistics")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Chunks", justify="right", style="magenta")
    table.add_column("Files", justify="right", style="yellow")

    if resolved_repos:
        # Per-repo breakdown
        for repo_name, repo_stats in stats["repos"].items():
            table.add_row(
                f"[dim]{repo_name}[/dim] Python Code",
                str(repo_stats["code"]["chunks"]),
                str(repo_stats["code"]["files"]),
            )
            table.add_row(
                f"[dim]{repo_name}[/dim] Documentation",
                str(repo_stats["docs"]["chunks"]),
                str(repo_stats["docs"]["files"]),
            )
            table.add_row(
                f"[dim]{repo_name}[/dim] Git History",
                str(repo_stats["git"]["chunks"]),
                "",
            )

    if stats["pdfs"]["files"] > 0:
        table.add_row(
            "PDF Papers",
            str(stats["pdfs"]["chunks"]),
            str(stats["pdfs"]["files"]),
        )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{stats['total_chunks']}[/bold]",
        "",
    )

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    file_filter: str = typer.Option(None, "--file", "-f", help="Filter by file path"),
) -> None:
    """Search the indexed codebase"""
    vector_store = VectorStore()

    # Build filters
    filters = {}
    if file_filter:
        filters["file_path"] = file_filter

    results = vector_store.query(
        query, n_results=top_k, filters=filters if filters else None
    )

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"\n[bold]Search results for:[/bold] {query}\n")

    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        relevance = 1 - result["distance"]

        console.print(f"[bold cyan]Result {i}[/bold cyan] (relevance: {relevance:.2f})")

        # Show repository if available
        if metadata.get("repository"):
            console.print(f"Repository: [green]{metadata['repository']}[/green]")

        console.print(f"File: {metadata.get('file_path', 'Unknown')}")

        if metadata.get("chunk_type"):
            console.print(f"Type: {metadata['chunk_type']}")
        if metadata.get("class_name"):
            console.print(f"Class: {metadata['class_name']}")
        if metadata.get("function_name"):
            console.print(f"Function: {metadata['function_name']}")

        console.print("\n[dim]" + result["content"][:200] + "...[/dim]\n")
        console.print("─" * 80 + "\n")


@app.command()
def serve(
    repo_paths: list[str] = typer.Option(
        [],
        "--repo",
        "-r",
        help=(
            "Repository root (repeatable). Used for display only — actual "
            "data lives in the vector store. Defaults to KAIA_REPOS env "
            "or MARCUS_ROOT."
        ),
    ),
) -> None:
    """Start the MCP server.

    Examples:

        # Use env var (KAIA_REPOS or MARCUS_ROOT)
        kaia serve

        # Explicitly pass roots (sets KAIA_REPOS for the subprocess)
        kaia serve -r ~/dev/marcus -r ~/dev/cato -r ~/dev/marcus-mini -r ~/dev/posidonius
    """
    import os as _os

    from kaia.mcp_server import main

    if repo_paths:
        resolved = [str(Path(p).expanduser().resolve()) for p in repo_paths]
        _os.environ["KAIA_REPOS"] = ",".join(resolved)

    console.print("[bold]Starting Kaia MCP server...[/bold]")
    repos_env = _os.environ.get("KAIA_REPOS") or _os.environ.get(
        "MARCUS_ROOT", "/Users/lwgray/dev/marcus"
    )
    console.print(f"Repos: {repos_env}\n")

    asyncio.run(main())


@app.command()
def stats() -> None:
    """Show statistics about the indexed database"""
    vector_store = VectorStore()
    count = vector_store.count()

    console.print("\n[bold]Kaia Database Statistics[/bold]\n")
    console.print(f"Total chunks indexed: [cyan]{count}[/cyan]")

    if count == 0:
        console.print("\n[yellow]No data indexed yet. Run 'kaia index' first.[/yellow]")


@app.command()
def clear(
    repo: str = typer.Option(
        None,
        "--repo",
        help=(
            "Optional: only clear chunks for this repository "
            "(e.g., 'marcus', 'cato', 'papers'). Without this flag, "
            "the entire collection is dropped."
        ),
    ),
) -> None:
    """Clear the indexed database (whole collection, or one repo)."""
    vector_store = VectorStore()

    if repo:
        confirm = typer.confirm(
            f"Delete all chunks for repository '{repo}'?"
        )
        if not confirm:
            console.print("Cancelled")
            return
        deleted = vector_store.delete_by_repository(repo)
        console.print(
            f"[green]✓ Deleted {deleted} chunks for repo '{repo}'[/green]"
        )
        return

    confirm = typer.confirm("Are you sure you want to clear the database?")
    if confirm:
        vector_store.delete_collection()
        console.print("[green]✓ Database cleared[/green]")
    else:
        console.print("Cancelled")


def _read_package_data(*parts: str) -> str:
    """Read a bundled data file shipped inside the kaia package."""
    from importlib.resources import files

    resource = files("kaia") / "data"
    for part in parts:
        resource = resource / part
    return resource.read_text(encoding="utf-8")


def _sync_section(target: Path, section: str) -> str:
    """Insert or replace the KAIA section in target. Returns action taken."""
    begin, end = "<!-- KAIA:BEGIN -->", "<!-- KAIA:END -->"
    section = section.strip() + "\n"

    if not target.exists():
        target.write_text(section, encoding="utf-8")
        return "created"

    text = target.read_text(encoding="utf-8")
    if begin in text and end in text:
        before = text[: text.index(begin)]
        after = text[text.index(end) + len(end) :]
        target.write_text(before + section.rstrip("\n") + after, encoding="utf-8")
        return "updated"

    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    target.write_text(text + sep + "\n" + section, encoding="utf-8")
    return "appended"


@app.command()
def init(
    directory: str = typer.Option(
        ".", "--dir", "-d", help="Project directory to set up (default: current)"
    ),
) -> None:
    """Install the Kaia Claude skill and wire it into CLAUDE.md / AGENTS.md.

    Copies the bundled /kaia skill into .claude/skills/kaia/ and inserts the
    Kaia AI-architect section into the project's CLAUDE.md and AGENTS.md.
    Safe to re-run: the section is replaced in place via marker comments.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        console.print(f"[red]Error: {root} is not a directory[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Installing Kaia into[/bold] {root}\n")

    # 1. Install the skill
    skill_dir = root / ".claude" / "skills" / "kaia"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        _read_package_data("skill", "SKILL.md"), encoding="utf-8"
    )
    console.print(f"[green]✓[/green] Skill installed → {skill_dir / 'SKILL.md'}")

    # 2. Wire into CLAUDE.md and AGENTS.md
    section = _read_package_data("kaia_claude_section.md")
    for name in ("CLAUDE.md", "AGENTS.md"):
        action = _sync_section(root / name, section)
        console.print(f"[green]✓[/green] {name} {action}")

    console.print(
        "\n[bold green]Done.[/bold green] Use [cyan]/kaia[/cyan] in Claude Code."
    )


if __name__ == "__main__":
    app()
