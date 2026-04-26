# Contributing to Aether

Thank you for considering a contribution. Aether is a small, focused library — the bar for changes is "does this make the substrate more correct, more measurable, or more usable?"

## Dev setup

```bash
git clone https://github.com/blockhead22/aether-core.git
cd aether-core
pip install -e .[dev,all]
pytest -q
```

89 tests should pass. If they don't on a clean checkout, that's a bug worth opening an issue about.

## What good contributions look like

**Yes:**
- Bugfixes with a regression test
- New immune agents that enforce a stated law (open an issue first to discuss the law)
- More principled fact-slot extractors (regex or otherwise)
- Performance improvements with benchmarks
- Documentation improvements, especially examples
- Tighter math: bounds, theorems, counterexamples

**Probably not:**
- Adding LLM calls inside the core library — Aether is structural by design
- Adding required heavy dependencies (sentence-transformers stays optional)
- Hooking into a specific LLM vendor in the core library (vendor adapters live in `aether.adapters`, planned)
- Renaming public APIs without a deprecation cycle

## Tests

Every public function needs a test. The bar:

- Tests run without network and without GPU
- Tests run without any optional dependency installed (skip gracefully if `networkx` / `sentence-transformers` aren't present)
- Determinism: no flaky tests; seed your randomness

```bash
pytest -q                                # all tests
pytest tests/test_governance.py -v       # one module
pytest -k "test_held_contradiction" -v   # one keyword
```

## Style

- Python 3.10+ syntax (use `|` unions, `match` statements where they help)
- Type hints on public APIs
- Docstrings: short and useful; the README is the marketing surface, the docstring is for the IDE hover
- One concept per module; no god-files
- No comments explaining what well-named identifiers already say

## Discussing changes

For non-trivial changes, open an issue first. The architecture has a few load-bearing decisions (belief/speech separation, contradiction-as-signal, structural-over-semantic) and I'd rather discuss the shape than re-litigate it in PR review.

## License

By contributing you agree your contributions are licensed under MIT.
