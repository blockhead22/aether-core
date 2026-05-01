# Handoff: M2 Mac → Windows (2026-05-01, late)

Companion to `HANDOFF_2026-05-01_to_windows.md` (the morning handoff). Added late on 2026-05-01 after a smoke test on the M2 found a CLI regression that needs to be fixed on the Windows side — the M2 repo is otherwise being kept read-only during the machine move.

## State of the M2 right now

- **Repo:** `~/Documents/ai_round2/aether-core`, branch `master`, HEAD `cc7074a` ("bench: end-to-end smoke probe for v0.13.1"). Clean working tree, even with `origin/master`. No unpushed commits. Tags through `v0.13.1` are on origin.
- **Venv:** `~/.aether-venv/` editable-installed from the local repo. After `pip install -e .` today, both `pip show` metadata and `aether.__version__` report `0.13.1`. Doctor: 7 ok / 0 warn / 0 fail.
- **Plugin cache:** `/plugin update aether-core@aether` + `/reload-plugins` pulled `0.13.1/` into `~/.claude/plugins/cache/aether/aether-core/`. The two older versions (`0.12.16/`, `0.12.18/`) are still there and still registered as Stop hooks + MCP servers per doctor — the cache doesn't auto-prune. Manual cleanup is safe if you want it.
- **PyPI:** still at 0.12.17 per prior session memory; not verified today. `pip install aether-core` on a fresh machine still pulls the pre-fix code.

## The reason for this handoff: a CLI silent-output regression

`aether status`, `aether check`, and `aether contradictions` all exit 0 with **zero bytes** on stdout/stderr on HEAD `cc7074a`. `aether doctor` / `warmup` / `init` / `backfill-edges` are fine.

### Root cause (not M2-specific — affects every platform)

`aether/_lazy_encoder.py:189–194`:

```python
with contextlib.redirect_stdout(stdout_buf), \
     contextlib.redirect_stderr(stderr_buf):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(self.model_name)
```

This block lives inside `_LazyEncoder._load`, which `start_warmup()` runs on a **background thread**. `contextlib.redirect_stdout` mutates `sys.stdout` *process-globally* (it's an attribute swap, not thread-local), so during the model-load window (~few seconds, longer cold) the **main thread's `print()` calls land in that StringIO and are silently discarded**.

`StateStore.__init__` calls `start_warmup()` unconditionally. Bisection confirmed: with `enable_embeddings=False` the CLI prints fine; with embeddings on, prints emitted within ~3–8s of `StateStore()` construction are eaten. Adding `time.sleep(8)` after `StateStore()` and before `print()` makes the prints reappear (the redirect block exits when the load finishes).

`doctor` etc. work because they call `wait_until_ready()` synchronously before printing — by the time they hit stdout, the redirect's been released. `cmd_status` / `cmd_check` / `cmd_contradictions` print immediately after `_make_store()` and lose the race.

### When this was introduced

v0.12.6 / v0.12.7 ("F#8 fix" — `a3ef5d1`, `de630f9`). Latent in many tags; surfaces on commands that print quickly.

### Suggested fix (cleanest)

Suppress at source rather than redirecting. Set these before `from sentence_transformers import …`:

```python
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
```

Then drop the `redirect_stdout`/`redirect_stderr` block. If you still want a belt-and-suspenders capture on the MCP path specifically, gate it on something like `if os.environ.get("AETHER_MCP_SERVER"):` so CLI invocations never hit it.

Alternative (worse): make `cmd_status` / `cmd_check` / `cmd_contradictions` call `wait_until_ready()` before printing — slow, surrenders the cold-start optimization for no real win.

### Regression test to add

`aether status` on a machine with the encoder cached should produce non-empty stdout within 1 second of invocation. Same for `aether check --message "test"` and `aether contradictions`.

## M2-specific things the Windows side does NOT need

- Pyexpat dlopen fix (Homebrew Python 3.12.13_2 ships `pyexpat.cpython-312-darwin.so` linked to `/usr/lib/libexpat.1.dylib`; needs `install_name_tool -change` + `codesign --force --sign -`). Windows Python doesn't have this issue.
- Venv created to dodge PEP 668 lockout. Windows Python isn't externally-managed by default.

## Other open items as of today (carried over from prior session — verify still relevant)

1. Bench README under-discloses warm-mode requirement (cold-mode policy_violation is 50% miss; user running the README's force-push sanity test in cold mode would see `aether_sanction` approve `--no-verify`).
2. The "88% slot vs 40% LLM-as-judge" claim in README is unsourced — no linked dataset/script.
3. PyPI not on 0.12.20+ yet; `dist/aether_core-0.12.20.{whl,tar.gz}` was built and validated on M2 but upload was deferred. v0.13.0 / v0.13.1 wheels probably need building too.
4. Polarity-blind grounding in `compute_grounding` (CRT bench 2026-05-01): seeded "Never delete production data without verifying a recent backup" returned APPROVE for a paraphrased deletion action, with the policy as a *supporting* belief. Same architectural shape as the `_is_policy_contradiction` fix in v0.12.19, different code path. v0.12.21 added polarity-aware contradiction detection but not necessarily on the support classifier — verify.
5. `git status` (and likely all read-only git verbs) gets REJECTed by `aether_sanction` because the substring `git` matches the `push --force` policy. Sanction needs verb-stem extraction or stricter slot-key match.
6. Quantitative-shape detector misses natural paraphrases (template-aligned only); README claim should be hedged or detector extended.
7. Categorical slot-value detector requires explicit `slot=value` syntax — DECISIONS.md-style decision drift slips past.
8. `_seed_default_beliefs` order-sensitivity is latent — masked by the v0.12.19 co-policy guard but no order-independence test was added.
