---
name: aether-init
description: Scaffold a .aether/ directory in the current project
---

Run the shell command `aether init` in the user's current working directory.

After it completes, tell the user:
1. The `.aether/` directory has been created at the project root with
   an empty `state.json`, a `README.md`, and a `.gitignore`.
2. They should `git add .aether/` and commit so their team inherits the
   substrate.
3. From now on, any MCP-aware client running inside this project tree
   will discover the repo-level substrate automatically. Memories
   written via `aether_remember` go to the project, not the user-global
   `~/.aether/`.

If the command fails because `aether-core` isn't installed, tell the user
to run `pip install "aether-core[mcp,graph,ml]"` first.
