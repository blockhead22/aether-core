# Plan: Fill crt-core Placeholder Namespaces + Production Integration Test

## Goal
Port proven modules from AI_round2 into crt-core's three empty namespaces, then test the package in a real agent loop.

---

## Phase 1: Port Modules (4 files → 3 namespaces)

### 1a. `crt.contradiction` — Structural Tension Meter
**Source:** `D:/AI_round2/personal_agent/structural_tension.py`
**Port difficulty:** Easy

Files to create:
- `crt/contradiction/tension.py` — StructuralTensionMeter, TensionResult, TensionRelationship, TensionAction, SlotOverlap
- `crt/contradiction/__init__.py` — re-export public API

Changes needed:
- Replace `from .embeddings import get_encoder` with constructor injection: `__init__(self, encoder=None)` where encoder is any `callable(text) -> np.ndarray`
- Replace `from .fact_slots import extract_fact_slots_contextual` with `from crt.memory import extract_fact_slots_contextual` (internal cross-reference within crt-core)
- Strip docstring references to Aether-specific integration points
- Keep numpy as only external dep (already in pyproject.toml)

### 1b. `crt.epistemics` — Belief Backpropagation
**Source:** `D:/AI_round2/papers/belief_backpropagation/backprop_engine.py`
**Port difficulty:** Trivial (zero deps)

Files to create:
- `crt/epistemics/backprop.py` — EpistemicLoss, CorrectionEvent, BackpropResult, DomainVolatility, compute_backward_gradients(), apply_trust_adjustments(), flat_demotion()
- `crt/epistemics/__init__.py` — re-export public API

Changes needed:
- None. Pure stdlib. Copy and clean up.

### 1c. `crt.memory` — Fact Slots + Memory Graph
**Source:** `D:/AI_round2/personal_agent/fact_slots.py` + `D:/AI_round2/personal_agent/memory_graph.py`
**Port difficulty:** Easy for fact_slots, Medium for memory_graph

Files to create:
- `crt/memory/slots.py` — ExtractedFact, TemporalStatus, extract_fact_slots(), extract_fact_slots_contextual(), detect_correction_type(), name helpers
- `crt/memory/graph.py` — MemoryNode, MemoryGraph, ContradictionEdge, BeliefDependencyGraph, CascadeResult, enums (MemoryType, BelnapState, EdgeType, Disposition)
- `crt/memory/__init__.py` — re-export public API

Changes needed for fact_slots:
- None. Pure stdlib (re, dataclasses). Copy and clean.

Changes needed for memory_graph:
- **EXCLUDE LiveBDG and get_live_bdg()** — these are Aether-specific (tight coupling to SQLite memory system)
- Keep MemoryGraph (standalone, JSON persistence) and BeliefDependencyGraph (pure graph, optional networkx)
- Strip `_live_bdg_instance` singleton pattern
- Keep `build_test_graph()` as a convenience for users

### 1d. Update pyproject.toml
- Version bump to 0.2.0
- Add `[project.optional-dependencies]` already covers networkx and sentence-transformers
- Update description and module table in README

---

## Phase 2: Tests

### New test files:
- `tests/test_tension.py` — measure_pair on known contradictions, known compatibles, slot overlap detection, encoder injection
- `tests/test_backprop.py` — EpistemicLoss computation, gradient propagation on test graph, trust adjustment bounds, domain volatility
- `tests/test_slots.py` — fact extraction from sample sentences, temporal status detection, correction detection, name equivalence
- `tests/test_graph.py` — MemoryGraph add/query/persist, BeliefDependencyGraph cascade propagation, contradiction density, Belnap states

### Run existing tests to confirm no regression:
- `tests/test_governance.py` — must still pass
- `tests/test_public_surface.py` — update to verify new namespaces export real classes

---

## Phase 3: Production Integration Test

### Target: Wrap an existing agent loop with crt-core

Pick ONE of these integration targets (in order of simplicity):

**Option A: Raw OpenAI SDK script** (simplest)
Create `examples/openai_integration.py`:
```python
from openai import OpenAI
from crt.governance import GovernanceLayer
from crt.contradiction import StructuralTensionMeter
from crt.memory import extract_fact_slots, MemoryGraph
from crt.epistemics import EpistemicLoss, CorrectionEvent

client = OpenAI()
gov = GovernanceLayer()
meter = StructuralTensionMeter()
graph = MemoryGraph()

# Conversation loop:
# 1. Extract facts from user message → store in graph
# 2. Check tension against existing memories
# 3. Call LLM with context
# 4. Govern response before showing to user
# 5. Extract facts from response → store with lower trust
# 6. If user corrects → backprop trust adjustment
```

**Option B: LangChain callback** (broader reach)
Create `examples/langchain_integration.py` with a CRT callback handler

**Option C: MCP server** (already partially exists)
Adapt crt_mcp_server.py to use crt-core imports instead of personal_agent imports

### Success criteria:
- `pip install -e .` from crt-core works
- Example script runs a 10-turn conversation with governance active
- At least one contradiction detected and held
- At least one trust adjustment via backprop
- All tests pass

---

## Phase 4: README Update

Update README.md to show all three namespaces as shipped:
- Quick start examples for each namespace
- Integration pattern: before/after/store
- The 3-line pitch: "Your agent already has memory. crt-core teaches it what to believe."

---

## File inventory (what gets created/modified):

### New files:
- crt/contradiction/tension.py
- crt/epistemics/backprop.py
- crt/memory/slots.py
- crt/memory/graph.py
- tests/test_tension.py
- tests/test_backprop.py
- tests/test_slots.py
- tests/test_graph.py
- examples/openai_integration.py (or langchain, or both)

### Modified files:
- crt/__init__.py (version bump)
- crt/contradiction/__init__.py (real exports)
- crt/epistemics/__init__.py (real exports)
- crt/memory/__init__.py (real exports)
- pyproject.toml (version, deps)
- README.md (full rewrite of module table + examples)
- tests/test_public_surface.py (verify new exports)

### NOT ported (stays in AI_round2):
- LiveBDG / get_live_bdg() (Aether-specific DB coupling)
- slot_name_discovery.py (explorative, 39% precision)
- crt_memory.py (product-layer, too opinionated)
- All trained models, prompts, frontend, labs
