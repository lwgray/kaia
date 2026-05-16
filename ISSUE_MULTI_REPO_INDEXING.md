# Multi-Directory Indexing Support

## Problem

Kaia currently indexes one repository at a time. You can run `index` multiple times with different `--repo-path` values, but each run is isolated — the extractors have no visibility into files from other repos. This means:

- Cross-repo imports (e.g., `from cato.utils import X` inside marcus) are not captured as relationships
- There's no unified view of how multiple codebases interact
- Re-indexing requires a full `clear` + re-index of all repos (no way to clear just one)
- Duplicate chunks accumulate if you re-index without clearing

## Proposal

Update the indexer to accept multiple directories in a single indexing pass, so all files are visible to the extractors at once.

### CLI Interface

```bash
python -m kaia.cli index \
  --repo ~/dev/marcus \
  --repo ~/dev/cato
```

Or via config file:

```yaml
repositories:
  - path: ~/dev/marcus
    name: marcus
  - path: ~/dev/cato
    name: cato
```

### Metadata Requirements

Each chunk must carry metadata identifying its origin:

- **`repository`** — repo name (e.g., `marcus`, `cato`)
- **`file_path`** — path relative to the repo root, prefixed with repo name (e.g., `marcus/src/coordinator.py`)
- **`references_repos`** — (optional) list of other repos this chunk references via imports

### What Changes

- **`MarcusIndexer`** — Accept a list of `(path, name)` pairs instead of a single `repo_root`. Walk all directories in one pass.
- **Code extractor** — Tag each chunk with its source repo. Optionally detect cross-repo imports and record them in metadata.
- **Doc extractor** — Tag each chunk with its source repo.
- **Git extractor** — Still per-repo (each has its own `.git`), but tag commits with the repo name.
- **CLI** — Support `--repo` flag (repeatable) or a config file. Deprecate `--repo-path`.
- **Vector store** — Support filtering by `repository` metadata so you can scope searches to a single repo or search across all.
- **MCP tools** — Add optional `repository` filter parameter to `search_marcus_architecture` and other tools.

### Out of Scope (for now)

- Dependency graph construction across repos
- Incremental indexing (only changed files)
- Automatic repo discovery from a parent directory

## Why This Matters

When multiple codebases are related (e.g., marcus and cato share interfaces or one imports from the other), indexing them separately loses the relationships between them. A single indexing pass with proper metadata gives semantic search the full picture while still letting users filter results by repo.
