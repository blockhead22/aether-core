# Git hooks for Aether

Drop-in scripts that wire `aether check` into your git workflow. None
are required — they're integrations that show what the substrate is
worth at the actual surface where humans write words about code.

## `pre-commit`

Reads the commit message + staged diff, runs the substrate-grounded
fidelity check, and blocks the commit when the speech/belief gap is
CRITICAL. Runs in well under a second.

### Install

```bash
cp examples/git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### What gets blocked

Any commit message that confidently asserts something the substrate
contradicts at trust ≥ 0.7. Example:

  Commit message: "Migration is fully backwards-compatible."
  Substrate has: "We removed the legacy_id column in v2" (trust 0.9)

  → CRITICAL gap, commit blocked.

### Bypass

For genuine cases where the substrate is wrong:

```bash
git commit --no-verify -m "..."
```

Then either correct the substrate (`aether_correct`) or recognize that
you've discovered a contradiction worth recording (`aether_remember`).

### Tuning

To block at lower severity:

```bash
aether check --message-file ... --fail-severity ELEVATED
```

Edit the script's `--fail-severity` flag. The default `CRITICAL` is
intentionally permissive — most commit messages won't trigger it.

## GitHub Action

See `examples/github-actions/aether-check.yml` for a workflow-level
version that runs on every PR and posts a comment with the grounding
report.
