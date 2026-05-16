# Kaia

**A codebase RAG system and AI-architect persona for Claude Code and Codex CLI.**

Kaia gives your coding agent a semantic memory of a codebase, plus a senior
AI-architect persona — Dr. Kaia Chen — to reason with it.

It has two halves:

| Half | What it is | The name you'll see |
|------|------------|---------------------|
| **Retrieval engine** | A RAG system that indexes repos (code, docs, git history, PDFs) into a local vector store and serves it over the Model Context Protocol (MCP). | the **`chen`** MCP server |
| **Persona** | Dr. Kaia Chen — an AI architect, mentor, and reviewer, packaged as a skill for Claude Code and Codex CLI. She grounds her advice in *your actual code* through the `chen` MCP tools. | the **`/kaia`** skill |

Either half runs alone, but they are designed to work together — the persona
asks the questions, the engine answers them from real code.

> **Scope.** The **retrieval engine is repo-agnostic** — it indexes any
> codebase. **Dr. Kaia Chen the persona is not**: the bundled skill is tuned to
> the [Marcus](https://github.com/lwgray/marcus) multi-agent coordination
> platform — her vocabulary, judgment, and examples assume that domain. Using
> her on another project means tailoring the skill (the files are plain text;
> see [Roadmap](#roadmap)).

---

## Features

- 🔍 **Semantic search** — find code by meaning, not keywords
- 📦 **Local embeddings** — no API keys; uses `sentence-transformers`
- 🚀 **Fast** — ~100–200 ms per query
- 🔒 **Private** — all data stays on your machine
- 📊 **Multi-granularity** — indexes functions, classes, modules, docs, git history, and PDFs
- 🧑‍🏫 **AI-architect persona** — installed into any project with one command
- 🔌 **Works with Claude Code and Codex CLI** — same skill, both agents

## Requirements

- Python 3.11+

## Installation

```bash
# From GitHub (recommended until a PyPI release is published)
pip install git+https://github.com/lwgray/kaia.git

# Or, from a local clone for development
git clone https://github.com/lwgray/kaia.git
cd kaia
pip install -e ".[dev]"
```

This installs the `kaia` command-line tool.

> **First run:** `sentence-transformers` downloads a ~80 MB embedding model.

---

## Part 1 — Index and search a codebase

### 1. Index your repositories

```bash
# A single repo
kaia index -r ~/path/to/repo

# Multiple repos at once
kaia index -r ~/path/to/repo-a -r ~/path/to/repo-b

# Repos plus a folder of PDFs
kaia index -r ~/path/to/repo -p ~/path/to/papers
```

### 2. Search from the terminal

```bash
kaia search "how is authentication handled"
kaia stats                 # how many chunks are indexed
kaia clear                 # wipe the index before re-indexing
kaia clear --repo myrepo   # wipe just one repo's chunks
```

### 3. Register the `chen` MCP server

This exposes the index to your coding agent.

```bash
# Activate the environment kaia is installed into first,
# so $(which python) resolves to the right interpreter.
claude mcp add chen \
  --transport stdio \
  --env KAIA_REPOS="$HOME/path/to/repo-a,$HOME/path/to/repo-b" \
  -- "$(which python)" -m kaia.mcp_server
```

**Notes**
- Use the **absolute path** to the Python interpreter kaia is installed in.
- `KAIA_REPOS` is comma-separated, **no spaces**. Each path's last segment
  becomes the `repository` filter value used by the MCP tools.
- `KAIA_REPOS` only drives display in tool descriptions — the searchable data
  lives in the vector store built by `kaia index`.
- Restart the agent, then confirm the **`chen`** server shows as *Running*.

#### MCP tools exposed

| Tool | Use it for |
|------|-----------|
| `search_marcus_architecture` | "How does X work" questions, finding patterns |
| `query_implementation_details` | The specific code of a named class/function/module |
| `find_usage_examples` | Tests and usage examples for a component |

---

## Part 2 — Activate Dr. Kaia Chen

`kaia init` installs the **`/kaia` skill** into a project and wires it into the
agent's instruction files.

```bash
cd ~/path/to/your/project
kaia init                       # current directory
kaia init --dir ~/other/project # somewhere else
```

It sets up **both Claude Code and Codex CLI** — they share the same `SKILL.md`
format, only the directory differs. Every step is **idempotent** (safe to
re-run):

| Step | Path | Agent |
|------|------|-------|
| Install skill | `.claude/skills/kaia/SKILL.md` | Claude Code |
| Install skill | `.agents/skills/kaia/SKILL.md` | Codex CLI |
| Insert "AI Architect Partner" block | `CLAUDE.md` | Claude Code |
| Insert the same block | `AGENTS.md` | Codex CLI |

The inserted block is fenced with `<!-- KAIA:BEGIN -->` / `<!-- KAIA:END -->`
markers, so re-running `kaia init` updates it in place instead of duplicating.

> All four targets are **plain editable files**. Edit
> `.claude/skills/kaia/SKILL.md` (or the `.agents` copy) and the inserted
> `CLAUDE.md` / `AGENTS.md` block to re-tailor the persona for your project.

### Using her

**Claude Code** — `/kaia` slash command:

```
/kaia how should I structure the retry logic here?
/kaia --review
/kaia --research vector database trade-offs
```

**Codex CLI** — mention the skill with `$kaia`, or pick it from `/skills`:

```
$kaia how should I structure the retry logic here?
```

In either tool you can also just mention her by name — "ask Kaia", "what would
Dr. Chen think?" — and the agent invokes the skill implicitly, matching the
skill's `description`.

If the `chen` MCP server (Part 1) is also running, Kaia grounds her answers in
your indexed code instead of guessing. **Persona + retrieval is the intended
setup.**

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Claude Code  /  Codex CLI                    │
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

**Chunking strategy**

- **Functions** — entire function body
- **Classes** — signature + docstring + method signatures
- **Modules** — module docstring + class/function list
- **Docs** — section-by-section (heading-based)
- **Git** — per commit

**Embedding model** — `all-MiniLM-L6-v2` (sentence-transformers), 384
dimensions, ~90–95% of OpenAI quality for code search, free, ~100–200 ms per
query.

**Storage** — ChromaDB, local. Default location `./data/chroma`; override with
`CHROMA_PERSIST_DIR`.

## CLI reference

| Command | Description |
|---------|-------------|
| `kaia index -r <repo> [-p <pdfs>]` | Index repos and/or PDF folders |
| `kaia search "<query>"` | Search the index from the terminal |
| `kaia serve [-r <repo>]` | Start the MCP server (same as `python -m kaia.mcp_server`) |
| `kaia stats` | Show indexed chunk count |
| `kaia clear [--repo <name>]` | Clear the whole index, or one repo |
| `kaia init [--dir <path>]` | Install the `/kaia` skill for Claude Code + Codex |

## Configuration

| Env var | Purpose |
|---------|---------|
| `KAIA_REPOS` | Comma-separated repo roots (display only — data lives in the vector store) |
| `MARCUS_ROOT` | Single-repo fallback when `KAIA_REPOS` is unset |
| `CHROMA_PERSIST_DIR` | Vector database location (default `./data/chroma`) |

## Security

⚠️ **Do not share the vector database.** Indexed chunks are stored as
plaintext. If your code or docs contain hardcoded secrets, API keys, or
credentials, those end up in the database.

Skipped automatically: `.venv`, `__pycache__`, `.git`, `node_modules`,
`.pytest_cache`, `.mypy_cache`, `build`, `dist`. **Secrets inside tracked
source files are *not* skipped** — review before indexing sensitive repos.

## Troubleshooting

**`chen` MCP server shows "Failed".** Run it by hand to see the error:

```bash
KAIA_REPOS="$HOME/path/to/repo" python -m kaia.mcp_server
```

Common causes: wrong Python path in `claude mcp add`; kaia not installed in
that interpreter; empty vector store (run `kaia index` first).

**`Collection expecting embedding with dimension of 1536, got 384`.** An old
OpenAI-based database exists — run `kaia clear`, then `kaia index` again.

**No results found.** Re-index: `kaia clear` then `kaia index -r <repo>`.

**`SyntaxWarning: invalid escape sequence` / `BertModel LOAD REPORT` warnings.**
Harmless, from dependencies. Ignore.

## Development

```bash
pytest                               # all tests
pytest tests/unit/                    # unit tests only
mypy src/kaia/                        # type checking
black src/kaia/ && isort src/kaia/    # formatting
```

## Project structure

```
kaia/
├── src/kaia/
│   ├── models.py           # Data models (chunks, metadata)
│   ├── vector_store.py     # ChromaDB integration + batching
│   ├── mcp_server.py       # MCP server (stdio transport)
│   ├── indexer.py          # Indexing orchestrator
│   ├── cli.py              # CLI, including `kaia init`
│   ├── data/               # Bundled skill + CLAUDE.md/AGENTS.md section
│   └── extractors/         # Code / doc / git / PDF parsers
└── tests/
    ├── unit/
    └── integration/
```

## Roadmap

The retrieval engine is already general-purpose. Making the *persona* reusable
beyond Marcus is the main planned work:

- **Templated persona** — turn the bundled `SKILL.md` and the
  `CLAUDE.md` / `AGENTS.md` block into templates, so `kaia init` generates a
  persona tailored to the target project.
- **Per-project persona config** — let a project declare its domain, stack, and
  conventions so the generated skill reflects them.
- **Persona presets** — ship more than one specialist and let `kaia init` pick.

Until then, the path for a non-Marcus project is to run `kaia init` and
hand-edit the skill and the inserted instruction block.

## License

MIT License.

## Credits

Created by [@lwgray](https://github.com/lwgray). The Dr. Kaia Chen persona was
developed alongside the [Marcus](https://github.com/lwgray/marcus) multi-agent
coordination platform.
