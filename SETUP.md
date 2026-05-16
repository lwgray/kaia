# Setting Up Kaia with Claude Code

## Step 1: Install Kaia

```bash
cd ~/dev/kaia
pip install -e "."
```

The first time you run it, sentence-transformers will download the ~80MB embedding model.

## Step 2: Index Marcus Codebase

```bash
# Index the Marcus repository
python -m kaia.cli index --marcus-root ~/dev/marcus
```

This will:
- Extract all Python code (functions, classes, modules)
- Parse documentation (CLAUDE.md, README.md)
- Index git history (last 100 commits)
- Store everything in `~/dev/kaia/data/chroma`

Expected output:
```
✓ Indexing complete!

Category        Chunks    Files
───────────────────────────────
Python Code     1,234     89
Documentation   156       12
Git History     100       100 commits
Total           1,490
```

## Step 3: Test the Search

```bash
# Quick test
python -m kaia.cli search "task coordination"
```

You should see relevant results from Marcus code!

## Step 4: Configure MCP Server for Claude Code

Add Kaia to your Claude Code MCP configuration:

**Location**: `~/.config/mcp/config.json`

```json
{
  "mcpServers": {
    "kaia": {
      "command": "python",
      "args": ["-m", "kaia.mcp_server"],
      "env": {
        "MARCUS_ROOT": "/Users/lwgray/dev/marcus"
      }
    }
  }
}
```

**Important**: Use full absolute paths!

## Step 5: Restart Claude Code

After updating the MCP config:
1. Close Claude Code completely
2. Reopen it
3. Kaia MCP server will start automatically

## Step 6: Verify It's Working

In Claude Code, you can now ask:

```
"Kaia, how does task assignment work in Marcus?"

"Show me the implementation of the TaskCoordinator class"

"Find examples of error handling patterns in Marcus"
```

Behind the scenes, Claude will use these MCP tools:
- `search_marcus_architecture` - Semantic search
- `query_implementation_details` - Get specific components
- `find_usage_examples` - Find test examples

## Troubleshooting

### "Connection refused" or MCP server won't start

Check the MCP server manually:
```bash
cd ~/dev/kaia
python -m kaia.mcp_server
```

Should see:
```
Indexing Marcus codebase at /Users/lwgray/dev/marcus...
✓ Indexed 1,490 chunks from 89 files
Starting Kaia MCP server...
```

If it hangs on "Indexing...", make sure you've already indexed (Step 2).

### "No results found"

Re-index:
```bash
python -m kaia.cli clear  # Clear old index
python -m kaia.cli index --marcus-root ~/dev/marcus  # Re-index
```

### Check index stats

```bash
python -m kaia.cli stats
```

Should show:
```
Total chunks indexed: 1,490
```

If 0, you need to index first (Step 2).

## Usage in Claude Code

Once set up, just talk naturally to Kaia:

**Example 1: Architecture questions**
```
You: "Kaia, explain the board-mediated coordination pattern"
→ Uses search_marcus_architecture("board-mediated coordination")
→ Returns relevant code + docs
```

**Example 2: Implementation lookup**
```
You: "Show me how TaskCoordinator.request_next_task works"
→ Uses query_implementation_details("request_next_task", "function")
→ Returns the function code + docstring + context
```

**Example 3: Usage examples**
```
You: "How do I use the retry decorator?"
→ Uses find_usage_examples("retry decorator")
→ Returns test files showing usage
```

## Advanced Configuration

### Change Marcus root location

Edit `~/.config/mcp/config.json`:
```json
"env": {
  "MARCUS_ROOT": "/path/to/your/marcus"
}
```

### Use different database location

Edit `~/.config/mcp/config.json`:
```json
"env": {
  "MARCUS_ROOT": "/Users/lwgray/dev/marcus",
  "CHROMA_PERSIST_DIR": "/custom/path/to/chroma"
}
```

### Re-index on server start

The MCP server automatically indexes on startup. To skip (use existing index):
```json
"args": ["-m", "kaia.mcp_server", "--skip-index"]
```

## Updating the Index

When Marcus code changes:
```bash
# Clear and re-index
python -m kaia.cli clear
python -m kaia.cli index --marcus-root ~/dev/marcus

# Then restart Claude Code to pick up changes
```

---

**That's it! Kaia is now integrated with Claude Code via MCP.** 🎉

You can now ask architecture questions, look up implementations, and get instant access to Marcus internals—all without leaving your conversation with Claude!
