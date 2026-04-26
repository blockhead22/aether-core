# Examples

Two runnable scripts, each ~30 lines of meaningful code, that demonstrate the core integration shape.

| Script | What it shows |
|---|---|
| `01_quickstart.py` | The belief/speech gap fires when an LLM speaks with more confidence than its belief state supports. ~60 seconds. |
| `02_full_pipeline.py` | The full BEFORE/LLM/AFTER integration pattern: fact extraction → tension measurement → LLM call → response governance. Uses a fake LLM so it runs offline. |

## Run

```bash
pip install aether-core
python examples/01_quickstart.py
python examples/02_full_pipeline.py
```

No API keys required. No network calls. Both scripts run in seconds.

## Then what

Once you've seen the shape, the README has:

- The 6 Laws and which agent enforces each
- The four pillars (governance, contradiction, epistemics, memory)
- A comparison table vs. memory layers (Mem0/Letta/Zep) and runtime policy governance (Microsoft Agent Governance Toolkit)
