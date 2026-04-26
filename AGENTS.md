# AGENTS.md — Installing Aether

This file is written for an AI agent (Claude Code, Cursor, Cline, Continue,
Goose, Zed, LM Studio, etc.) that the user has asked to install `aether-core`.

Read top to bottom. Run each step. Verify each step before moving to the next.
Every step is idempotent.

If you are a human reader: you can follow this file by hand, but the README's
"Plug into Claude Code" section is shorter and aimed at you.

## What you are installing

`aether-core` is a Python package that ships an MCP server. After installation
the user's AI client will have five new tools available: `aether_remember`,
`aether_search`, `aether_sanction`, `aether_fidelity`, `aether_context`. State
persists across sessions in `~/.aether/mcp_state.json` by default.

## Step 1. Install the Python package

Run, picking the first command that exists in the user's environment:

```
pip install "aether-core[mcp,graph]"
```

If `pip` is missing, try:

```
python -m pip install "aether-core[mcp,graph]"
```

or:

```
python3 -m pip install "aether-core[mcp,graph]"
```

Required Python version is 3.10 or higher. If install fails with
"could not find version that satisfies the requirement", check
`python --version` and report the version to the user.

## Step 2. Verify the package imports

Run:

```
python -c "import aether; print(aether.__version__)"
```

Expected output: a version like `0.4.0` or higher.

If this prints `ModuleNotFoundError`, the install failed silently. Re-run
Step 1 with verbose output (`pip install -v ...`) and report the error.

## Step 3. Verify the MCP server starts

Run, with a short timeout (the server waits on stdin and would otherwise
block forever):

On Unix-like systems:

```
timeout 3 python -m aether.mcp || true
```

On Windows PowerShell:

```
$proc = Start-Process python -ArgumentList "-m","aether.mcp" -PassThru -NoNewWindow
Start-Sleep 2
Stop-Process -Id $proc.Id -Force
```

Expected: the process starts without an `ImportError`. A clean exit (or
forced kill after the timeout) is success. If you see
`ModuleNotFoundError: No module named 'mcp'`, the optional `[mcp]` extra
did not install. Re-run Step 1.

## Step 4. Configure the AI client

You are running inside one of these clients. Pick the matching subsection.
If you are not sure which client, ask the user before editing files.

### Claude Code

Open `.claude/settings.json` in the project root, or `~/.claude/settings.json`
for user-global access. Merge the following block with any existing
`mcpServers`:

```json
{
  "mcpServers": {
    "aether": {
      "command": "python",
      "args": ["-m", "aether.mcp"]
    }
  }
}
```

If the file does not exist, create it with exactly the block above.

After writing, the user must start a new Claude Code session for the new
server to load. (Existing sessions do not pick up MCP changes hot.)

### Cursor

Edit `~/.cursor/mcp.json`. Same JSON shape as the Claude Code block above.

### Cline (VS Code extension)

Open VS Code settings, search for "Cline MCP Servers", and add a new
server with command `python` and args `["-m", "aether.mcp"]`.

### Continue.dev

Edit `~/.continue/config.json`. Add an `mcpServers` entry with the same
command/args.

### Goose, Zed, LM Studio, other MCP clients

Look for an `mcpServers` configuration block in the client's settings.
Use command `python` and args `["-m", "aether.mcp"]`.

## Step 5. Confirm the tools loaded

After the client restarts (or in the next message after settings reload),
verify by listing or calling one of the new tools.

If you can call tools directly: call `aether_context` with no arguments.
Expected output: a JSON object with `memory_count`, `edge_count`,
`belnap_states`, and `state_path`.

If you cannot directly verify, ask the user to confirm that the new
tools appear in their tool listing.

## Step 6. Brief the user

Tell the user, in plain language:

> Aether is installed. State persists in `~/.aether/mcp_state.json` (or
> `$AETHER_STATE_PATH` if you set it). Five tools are now available:
> `aether_remember`, `aether_search`, `aether_sanction`, `aether_fidelity`,
> `aether_context`. Try saying "Remember that I prefer Python over Rust"
> to test it, then "What do you remember about my language preferences?"
> in a later session.

## Optional: change the state path

If the user wants state stored somewhere other than `~/.aether/mcp_state.json`,
add an `env` block to the MCP server config:

```json
{
  "mcpServers": {
    "aether": {
      "command": "python",
      "args": ["-m", "aether.mcp"],
      "env": {
        "AETHER_STATE_PATH": "/path/to/state.json"
      }
    }
  }
}
```

The directory must be writable. If it does not exist, the server will
create it on first write.

## Hooks: not yet supported

Some AI clients (Claude Code in particular) support lifecycle hooks that
fire automatically on every tool call. A hook integration would let the
substrate sanction-check actions even when the model does not think to
call `aether_sanction` itself.

This is intentionally not part of `v0.4.0`. A working hook story requires
careful design about which actions warrant pre-flight gating, and a
wrapper that translates tool inputs into something the governance layer
can meaningfully evaluate. Shipping a half-baked hook would either block
the user on noise or pass everything through unhelpfully.

Hooks are planned for `v0.5.0`. Track progress in
[ROADMAP.md](ROADMAP.md). For now, the MCP tools are model-discretion: the
LLM decides when to call them, and a sensible system prompt nudges that.

## Troubleshooting

**`ModuleNotFoundError: No module named 'mcp'` when running `python -m aether.mcp`.**
The `[mcp]` extra was not installed. Re-run:
`pip install "aether-core[mcp,graph]"`.

**`pip install` fails with "could not find version that satisfies the requirement aether-core".**
Python version too old. `aether-core` requires 3.10 or higher. Check
`python --version`.

**Tools do not appear in Claude Code after editing settings.**
Settings are loaded at session start. Start a new session.

**Tools appear but `aether_context` errors.**
Likely a state-file permissions issue. Check that `~/.aether/` is
writable, or set `AETHER_STATE_PATH` to a writable location (Step 6
optional block).

**`aether-core` installs but the version is below 0.4.0.**
The `[mcp]` extra only exists from 0.4.0 onward. Force-upgrade with
`pip install --upgrade "aether-core>=0.4.0"`.

## Reference

- Repo: https://github.com/blockhead22/aether-core
- PyPI: https://pypi.org/project/aether-core/
- README (human-facing): [README.md](README.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
