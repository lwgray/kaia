# Kaia - RAG System for Marcus via MCP

**Kaia** is a Retrieval Augmented Generation (RAG) system that provides semantic search over the Marcus codebase through the Model Context Protocol (MCP). It enables Dr. Kaia Chen (your AI architect) to instantly look up implementation details, architectural patterns, and design decisions.

## Features

- 🔍 **Semantic Search** - Find code by meaning, not just keywords
- 📦 **Local Embeddings** - No API keys needed, uses sentence-transformers
- 🚀 **Fast** - ~100-200ms query time with local embeddings
- 🔒 **Private** - All data stays on your machine
- 📊 **Multi-Granularity** - Indexes functions, classes, modules, docs, and git history
- 🔌 **MCP Integration** - Works seamlessly with Claude Code

## What Gets Indexed

- **Python Code**: All `.py` files (functions, classes, modules)
- **Documentation**: All `.md` files (CLAUDE.md, README.md, etc.)
- **Git History**: Last 100 commits with messages and metadata

## Installation

### Prerequisites

- Python 3.11+
- Conda or virtualenv

### Setup

```bash
# Clone/navigate to kaia directory
cd ~/dev/kaia

# Create and activate conda environment
conda create -n kaia python=3.11
conda activate kaia

# Install Kaia in editable mode
pip install -e "."
```

**First run**: sentence-transformers will download the ~80MB embedding model automatically.

## Quick Start

### 1. Index the Marcus Codebase

```bash
# Clear old database if upgrading from OpenAI embeddings
python -m kaia.cli clear

# Index Marcus
python -m kaia.cli index --repo-path ~/dev/marcus
```

**Expected output:**
```
Indexing Python code... ━━━━━━━━━━━━━━━━━━━━━━ 538/538 0:02:15
Indexing documentation... ━━━━━━━━━━━━━━━━━━━━ 45/45   0:00:12
Indexing git history...    ━━━━━━━━━━━━━━━━━━━ 1/1     0:00:05

✓ Indexing complete!

Category        Chunks    Files
───────────────────────────────
Python Code     6,235     538
Documentation   348       45
Git History     100       100 commits
Total           6,683
```

### 2. Test Search

```bash
python -m kaia.cli search "task coordination"
```

### 3. Add to Claude Code as MCP Server

# Add it with the correct configuration (multi-repo: marcus, cato, marcus-mini, posidonius)
claude mcp add chen \
  --transport stdio \
  --env KAIA_REPOS=/Users/lwgray/dev/marcus,/Users/lwgray/dev/cato,/Users/lwgray/dev/marcus-mini,/Users/lwgray/dev/posidonius \
  -- /Users/lwgray/opt/anaconda3/envs/kaia/bin/python -m kaia.mcp_server
```

**Notes**:
- Use the full absolute path to your kaia environment's Python.
- `KAIA_REPOS` is comma-separated, no spaces. Each path is a repository root whose name (last path segment) becomes the `repository` filter value used by the MCP tools.
- For backward compatibility, `MARCUS_ROOT=<single-path>` still works as a fallback if `KAIA_REPOS` is unset.
- The env var only drives display in tool descriptions — actual searchable data lives in the persisted ChromaDB created by `kaia index`.

### 4. Restart Claude Code

Completely close and reopen Claude Code. The chen MCP server will auto-start.

### 5. Verify It's Working

Check the MCP servers panel in Claude Code. "chen" should show as **Running**.

### 6. Use It!

In Claude Code, ask questions like:

```
Chen, how does task coordination work in Marcus?

Show me the TaskCoordinator implementation

Find examples of error handling patterns in Marcus
```

## MCP Tools Available

Kaia exposes three MCP tools:

### 1. `search_marcus_architecture`
Semantic search across the entire codebase.

**Parameters:**
- `query` (required): Search query
- `top_k` (optional, default 10): Number of results
- `file_filter` (optional): Filter by file path

**Use for:** General architecture questions, finding patterns

### 2. `query_implementation_details`
Get specific implementation of a class, function, or module.

**Parameters:**
- `component` (required): Component name
- `component_type` (optional): "class", "function", or "module"

**Use for:** Looking up specific code

### 3. `find_usage_examples`
Find test files and usage examples.

**Parameters:**
- `component` (required): Component to find examples for

**Use for:** Understanding how to use a class/function

## CLI Commands

```bash
# Index codebase
python -m kaia.cli index --repo-path ~/dev/marcus

# Search indexed content
python -m kaia.cli search "query text"

# Show database statistics
python -m kaia.cli stats

# Clear database (before re-indexing)
python -m kaia.cli clear
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────┐
│  Claude Code with Dr. Kaia Chen        │
│  Ask: "How does X work in Marcus?"      │
└─────────────────┬───────────────────────┘
                  │ MCP Protocol (stdio)
                  ▼
┌─────────────────────────────────────────┐
│    Kaia MCP Server                      │
│  Tools: search_marcus_architecture()    │
│         query_implementation_details()  │
│         find_usage_examples()           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  ChromaDB Vector Store                  │
│  Indexed: Code, Docs, Git History       │
│  Embeddings: sentence-transformers      │
└─────────────────────────────────────────┘
```

### Chunking Strategy

- **Functions**: Entire function body (semantic chunking)
- **Classes**: Class signature + docstring + method signatures
- **Modules**: Module docstring + list of classes/functions
- **Docs**: Section-by-section (heading-based)
- **Git**: Per commit

### Embedding Model

- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions**: 384
- **Quality**: 90-95% as good as OpenAI for code search
- **Speed**: ~100-200ms for batch embeddings
- **Cost**: Free (no API calls)

### Storage

- **Database**: ChromaDB (local vector database)
- **Location**: `~/dev/kaia/data/chroma`
- **Format**: Plaintext chunks + embeddings
- **Size**: ~50-100MB for Marcus codebase

## Security Considerations

⚠️ **Do NOT share the vector database!**

The database contains:
- **Original code chunks** (plaintext, not encrypted)
- **All indexed content** from your codebase

If you have hardcoded API keys, passwords, or secrets in code/docs, they will be in the database.

**What's indexed:**
- ✅ All `.py` files (Python code)
- ✅ All `.md` files (Markdown docs)
- ✅ Git commit messages

**What's skipped:**
- `.venv`, `__pycache__`, `.git`, `node_modules`
- `.pytest_cache`, `.mypy_cache`, `build`, `dist`

**What's NOT skipped (potential risk):**
- Hardcoded secrets in Python files
- API keys in code comments
- Credentials in markdown docs

## Troubleshooting

### MCP Server Shows "Failed" Status

**Test manually:**
```bash
conda activate kaia
export KAIA_REPOS=/Users/lwgray/dev/marcus,/Users/lwgray/dev/cato,/Users/lwgray/dev/marcus-mini,/Users/lwgray/dev/posidonius
python -m kaia.mcp_server
```

Should print:
```
✓ Using existing index (6683 chunks). Configured repos: ['marcus', 'cato', 'marcus-mini', 'posidonius']
Starting Kaia MCP server...
```

**Common issues:**
1. Wrong Python path in `claude mcp add` command
2. Kaia not installed (`pip install -e "."`)
3. Neither `KAIA_REPOS` nor `MARCUS_ROOT` set
4. Vector store empty — run `kaia index -r <repo> [-r <repo>...]` first

### "Collection expecting embedding with dimension of 1536, got 384"

**Cause**: Old OpenAI-based database still exists.

**Fix:**
```bash
python -m kaia.cli clear
python -m kaia.cli index --repo-path ~/dev/marcus
```

### "Batch size exceeds max batch size"

**Cause**: ChromaDB batch size limit (already fixed in latest version).

**Fix:** Pull latest code with batching support.

### No Results Found

**Re-index:**
```bash
python -m kaia.cli clear
python -m kaia.cli index --repo-path ~/dev/marcus
```

### SyntaxWarning: "\d" is an invalid escape sequence

**Harmless warning** from a dependency (markdown/pygments). Can be ignored.

### BertModel LOAD REPORT: "embeddings.position_ids | UNEXPECTED"

**Harmless warning** from sentence-transformers model loading. Can be ignored.

## Configuration

### Environment Variables

- `KAIA_REPOS`: Comma-separated list of repository roots (e.g., `/Users/lwgray/dev/marcus,/Users/lwgray/dev/cato`). Used by the MCP server for display only — real data lives in the vector store.
- `MARCUS_ROOT`: Single-repo fallback when `KAIA_REPOS` is unset (back-compat).
- `CHROMA_PERSIST_DIR`: Database location (default: `./data/chroma`)

### Advanced: Custom Database Location

```bash
export CHROMA_PERSIST_DIR=/custom/path/to/chroma
python -m kaia.cli index --repo-path ~/dev/marcus
```

## Performance

**Indexing (one-time):**
- 538 Python files: ~2-3 minutes
- 45 Markdown files: ~10-15 seconds
- 100 Git commits: ~5 seconds
- **Total**: ~2-5 minutes

**Querying (real-time):**
- Embedding generation: ~50-100ms
- Vector search: ~20-50ms
- **Total**: ~100-200ms per query

**Database size:**
- Marcus codebase: ~50-100MB

## Updating the Index

When Marcus code changes:

```bash
# Clear and re-index
python -m kaia.cli clear
python -m kaia.cli index --repo-path ~/dev/marcus

# Restart Claude Code to pick up changes
```

## Development

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/
```

### Type Checking

```bash
mypy src/kaia/
```

### Code Formatting

```bash
black src/kaia/
isort src/kaia/
```

## Project Structure

```
kaia/
├── src/kaia/
│   ├── models.py           # Data models (chunks, metadata)
│   ├── vector_store.py     # ChromaDB integration + batching
│   ├── mcp_server.py       # MCP server with stdio transport
│   ├── indexer.py          # Indexing orchestrator
│   ├── cli.py              # CLI interface
│   └── extractors/
│       ├── code_extractor.py    # Python AST parsing
│       ├── doc_extractor.py     # Markdown parsing
│       └── git_extractor.py     # Git history parsing
├── tests/
│   ├── unit/               # Fast, isolated tests
│   └── integration/        # End-to-end tests
└── data/
    └── chroma/            # Vector database storage
```

## Technical Details

### Why Local Embeddings?

**Pros:**
- ✅ Free (no API costs)
- ✅ Fast (~100-200ms)
- ✅ Private (no data sent to cloud)
- ✅ No rate limits

**Cons:**
- 90-95% quality vs OpenAI (good enough for code search)
- Requires ~80MB model download (one-time)

### Why Multi-Granularity Chunking?

Different queries need different levels of detail:
- "How does X work?" → Need full function implementation
- "What does class Y do?" → Need class overview, not every method
- "What's in module Z?" → Need high-level summary

### Why ChromaDB?

- Fast vector search
- Local-first (no cloud dependency)
- Simple Python API
- Built-in persistence

## Future Enhancements

Potential improvements:
- [ ] Max chunk size limits (prevent 1000+ line chunks)
- [ ] Overlapping chunks for better retrieval
- [ ] Content filtering for secrets/API keys
- [ ] Incremental indexing (only changed files)
- [ ] Support for other languages (JavaScript, TypeScript)
- [ ] Hybrid search (semantic + keyword)

## License

MIT License - See Marcus project for details.

## Credits

Built for the [Marcus](https://github.com/lwgray/marcus) multi-agent coordination platform.
