# Kaia — a codebase RAG system + AI architect for Claude Code

**Kaia** gives Claude Code a semantic memory of your codebase and a senior
AI-architect persona to reason with it.

It has two halves that work together:

| Half | What it is | Name you'll see |
|------|------------|-----------------|
| **The retrieval engine** | A RAG system that indexes your repos (code, docs, git history, PDFs) into a local vector store and exposes it over the Model Context Protocol (MCP). | the **`chen`** MCP server |
| **The persona** | Dr. Kaia Chen — an AI architect, mentor, and reviewer skill for Claude Code. She grounds her advice in *your actual code* by calling the `chen` MCP tools. | the **`/kaia`** skill |

You can run either half on its own, but they are best together: `kaia init`
installs the persona, `kaia serve` powers it with retrieval.

> **Scope note.** The **retrieval engine is repo-agnostic** — `kaia index` and
> the `chen` MCP server work on any codebase. **Dr. Kaia Chen the persona is
> not**: the bundled `/kaia` skill and the `CLAUDE.md`/`AGENTS.md` section are
> tailored to the [Marcus](https://github.com/lwgray/marcus) multi-agent
> coordination platform — her vocabulary, judgment, and examples assume that
> domain. Using her well on another project means tailoring the skill. See
> [Roadmap](#roadmap).

## Features

- 🔍 **Semantic Search** — find code by meaning, not just keywords
- 📦 **Local Embeddings** — no API keys, uses `sentence-transformers`
- 🚀 **Fast** — ~100–200 ms query time
- 🔒 **Private** — all data stays on your machine
- 📊 **Multi-Granularity** — indexes functions, classes, modules, docs, git history, and PDFs
- 🧑‍🏫 **AI Architect persona** — the `/kaia` skill, installed into any project with one command
- 🔌 **MCP Integration** — works with Claude Code out of the box

## What Gets Indexed

- **Python code** — `.py` files (functions, classes, modules)
- **Documentation** — `.md` files
- **Git history** — recent commits with messages and metadata
- **PDFs** — research papers and documents

## Installation

Requires **Python 3.11+**.

```bash
# From GitHub (recommended until a PyPI release is published)
pip install git+https://github.com/lwgray/kaia.git

# Or from a local clone, in editable mode for development
git clone https://github.com/lwgray/kaia.git
cd kaia
pip install -e ".[dev]"
```

This installs the `kaia` command-line tool.

> **First run:** `sentence-transformers` downloads a ~80 MB embedding model
> automatically.

---

## Part 1 — Use Kaia as a codebase search engine

### 1. Index your repositories

```bash
# Index a single repo
kaia index -r ~/path/to/your/repo

# Index multiple repos at once
kaia index -r ~/path/to/repo-a -r ~/path/to/repo-b

# Also index a folder of PDFs
kaia index -r ~/path/to/repo -p ~/path/to/papers
```

### 2. Search from the CLI

```bash
kaia search "how is authentication handled"
kaia stats          # show how many chunks are indexed
kaia clear          # wipe the index before re-indexing
kaia clear --repo myrepo   # wipe just one repo's chunks
```

### 3. Register the `chen` MCP server with Claude Code

```bash
# Activate the environment kaia is installed into first, so $(which python) resolves correctly
claude mcp add chen \
  --transport stdio \
  --env KAIA_REPOS="$HOME/path/to/repo-a,$HOME/path/to/repo-b" \
  -- "$(which python)" -m kaia.mcp_server
```

**Notes**
- Use the **absolute path** to the Python interpreter kaia is installed in.
  `$(which python)` works if that environment is active.
- `KAIA_REPOS` is comma-separated, **no spaces**. Each path's last segment
  becomes the `repository` filter value used by the MCP tools.
- `KAIA_REPOS` only drives display in tool descriptions — the searchable data
  lives in the ChromaDB created by `kaia index`.
- Restart Claude Code, then check the MCP panel: **`chen`** should show as
  *Running*.

#### MCP tools exposed

| Tool | Use it for |
|------|-----------|
| `search_marcus_architecture` | General "how does X work" questions, finding patterns |
| `query_implementation_details` | The specific code of a named class/function/module |
| `find_usage_examples` | Tests and usage examples for a component |

---

## Part 2 — Activate Dr. Kaia Chen in your project

`kaia init` installs the **`/kaia` skill** into a project and wires it into the
project's instruction files so Claude Code knows when to summon her.

> The bundled skill is **Marcus-tuned** (see [Scope note](#kaia--a-codebase-rag-system--ai-architect-for-claude-code)).
> After `kaia init`, the skill lives at `.claude/skills/kaia/SKILL.md` and the
> inserted block lives in your `CLAUDE.md` / `AGENTS.md` — **all three are plain
> files you can edit** to re-tailor her for your own project.

```bash
cd ~/path/to/your/project
kaia init
```

This does three things, all **idempotent** (safe to re-run):

1. Copies the `/kaia` skill to `.claude/skills/kaia/SKILL.md`
2. Inserts the "AI Architect Partner" section into `CLAUDE.md`
3. Inserts the same section into `AGENTS.md`

(The inserted block is fenced with `<!-- KAIA:BEGIN -->` / `<!-- KAIA:END -->`
markers, so re-running `kaia init` updates it in place instead of duplicating.)

```bash
# Set up a different directory
kaia init --dir ~/path/to/other/project
```

### Using her

Once a project has been `kaia init`-ed, invoke her in Claude Code:

```
/kaia how should I structure the retry logic here?
/kaia --review
/kaia --research vector database trade-offs
/kaia --reflect
/kaia --chat
```

Or just mention her by name — "ask Kaia", "what would Dr. Chen think?" — and
Claude Code will invoke the skill.

If the `chen` MCP server (Part 1) is also running, Kaia grounds her answers in
your indexed codebase instead of guessing. **Persona + retrieval is the
intended setup.**

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Claude Code                                 │
│   /kaia skill  ──►  Dr. Kaia Chen persona     │
└───────────────┬───────────────────────────────┘
                │ MCP Protocol (stdio)
                ▼
┌─────────────────────────────────────────────┐
│  chen MCP server                              │
│   search_marcus_architecture()                │
│   query_implementation_details()              │
│   find_usage_examples()                       │
└───────────────┬───────────────────────────────┘
                ▼
┌─────────────────────────────────────────────┐
│  ChromaDB vector store                        │
│   Indexed: code, docs, git history, PDFs      │
│   Embeddings: sentence-transformers           │
└─────────────────────────────────────────────┘
```

### Chunking strategy

- **Functions** — entire function body
- **Classes** — signature + docstring + method signatures
- **Modules** — module docstring + class/function list
- **Docs** — section-by-section (heading-based)
- **Git** — per commit

### Embedding model

- `all-MiniLM-L6-v2` (sentence-transformers), 384 dimensions
- ~90–95% of OpenAI quality for code search, free, ~100–200 ms per query

### Storage

- ChromaDB, local. Default location `./data/chroma`, override with
  `CHROMA_PERSIST_DIR`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `kaia index -r <repo> [-p <pdfs>]` | Index repos and/or PDF folders |
| `kaia search "<query>"` | Search the index from the terminal |
| `kaia serve [-r <repo>]` | Start the MCP server (also runnable via `python -m kaia.mcp_server`) |
| `kaia stats` | Show indexed chunk count |
| `kaia clear [--repo <name>]` | Clear the whole index, or one repo |
| `kaia init [--dir <path>]` | Install the `/kaia` skill + wire `CLAUDE.md`/`AGENTS.md` |

## Configuration

| Env var | Purpose |
|---------|---------|
| `KAIA_REPOS` | Comma-separated repo roots (display only — data lives in the vector store) |
| `MARCUS_ROOT` | Single-repo fallback when `KAIA_REPOS` is unset |
| `CHROMA_PERSIST_DIR` | Vector database location (default `./data/chroma`) |

## Security Considerations

⚠️ **Do not share the vector database.** Indexed chunks are stored as
plaintext. If your code or docs contain hardcoded secrets, API keys, or
credentials, those end up in the database.

**Skipped automatically:** `.venv`, `__pycache__`, `.git`, `node_modules`,
`.pytest_cache`, `.mypy_cache`, `build`, `dist`.

**Not skipped:** secrets *inside* tracked source files — review before indexing
shared or sensitive repos.

## Troubleshooting

**`chen` MCP server shows "Failed".** Run it by hand to see the error:

```bash
KAIA_REPOS="$HOME/path/to/repo" python -m kaia.mcp_server
```

Common causes: wrong Python path in `claude mcp add`; kaia not installed in
that interpreter; vector store empty (run `kaia index` first).

**`Collection expecting embedding with dimension of 1536, got 384`.** An old
OpenAI-based database exists — `kaia clear` then `kaia index` again.

**No results found.** Re-index: `kaia clear` then `kaia index -r <repo>`.

**`SyntaxWarning: invalid escape sequence` / `BertModel LOAD REPORT` warnings.**
Harmless, from dependencies. Ignore.

## Development

```bash
pytest                 # all tests
pytest tests/unit/     # unit tests only
mypy src/kaia/         # type checking
black src/kaia/ && isort src/kaia/   # formatting
```

## Project Structure

```
kaia/
├── src/kaia/
│   ├── models.py           # Data models (chunks, metadata)
│   ├── vector_store.py     # ChromaDB integration + batching
│   ├── mcp_server.py       # MCP server (stdio transport)
│   ├── indexer.py          # Indexing orchestrator
│   ├── cli.py              # CLI, including `kaia init`
│   ├── data/               # Bundled /kaia skill + CLAUDE.md section
│   └── extractors/         # Code / doc / git / PDF parsers
└── tests/
    ├── unit/
    └── integration/
```

## Roadmap

The retrieval engine is already general-purpose. The persona is not — making
Dr. Chen reusable beyond Marcus is the main planned work:

- **Templated persona.** Turn the bundled `SKILL.md` and the
  `CLAUDE.md`/`AGENTS.md` block into editable templates, so `kaia init` can
  generate a persona tailored to the target project instead of shipping the
  Marcus-specific one.
- **Per-project persona config.** Let a project declare its domain, stack, and
  conventions so the generated skill reflects them.
- **Persona presets.** Ship more than one specialist (Marcus multi-agent today;
  others later) and let `kaia init` pick.

Until then, the practical path for a non-Marcus project is to run `kaia init`
and then hand-edit `.claude/skills/kaia/SKILL.md` and the inserted
`CLAUDE.md`/`AGENTS.md` block.

## License

MIT License.

## Credits

Created by [@lwgray](https://github.com/lwgray). The `/kaia` persona was
developed alongside the [Marcus](https://github.com/lwgray/marcus) multi-agent
coordination platform.
