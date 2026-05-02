"""v0.14 substrate primitive — write/read/persist/migrate."""

import json
import tempfile

import pytest

from aether.substrate import (
    SubstrateGraph,
    Namespace,
    SlotState,
    Observation,
    migrate_from_memory_graph,
)


def test_observe_creates_slot_and_state():
    sub = SubstrateGraph()
    s = sub.observe("user", "location", "Seattle", source_text="I live in Seattle")
    assert "user:location" in sub.slots
    assert s.value == "Seattle"
    assert s.normalized == "seattle"
    assert sub.current_state("user", "location").value == "Seattle"


def test_supersession_on_value_change():
    sub = SubstrateGraph()
    sub.observe("user", "location", "Seattle")
    sub.observe("user", "location", "Milwaukee")
    history = sub.history("user", "location")
    assert len(history) == 2
    assert history[0].superseded_by == history[1].state_id
    assert sub.current_state("user", "location").value == "Milwaukee"


def test_same_value_re_observation_does_not_supersede():
    sub = SubstrateGraph()
    sub.observe("user", "favorite_color", "orange")
    sub.observe("user", "favorite_color", "orange")
    history = sub.history("user", "favorite_color")
    assert len(history) == 2
    assert history[0].superseded_by is None  # same normalized value


def test_one_observation_emits_multiple_states():
    sub = SubstrateGraph()
    obs = Observation.new(source_text="I moved to Milwaukee and started at Microsoft",
                          source_type="manual")
    sub.observations[obs.observation_id] = obs
    sub.observe("user", "location", "Milwaukee", observation_id=obs.observation_id)
    sub.observe("user", "employer", "Microsoft", observation_id=obs.observation_id)
    refreshed = sub.observations[obs.observation_id]
    assert len(refreshed.emitted_state_ids) == 2


def test_persistence_roundtrip():
    sub = SubstrateGraph()
    sub.observe("user", "location", "Seattle")
    sub.observe("user", "location", "Portland")
    sub.observe("code", "my_proj:func_x:return_type", "Result[T]")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    sub.save(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "slots" in data and "states" in data
    sub2 = SubstrateGraph()
    sub2.load(path)
    assert sub2.current_state("user", "location").value == "Portland"
    assert sub2.current_state("code", "my_proj:func_x:return_type").value == "Result[T]"
    assert len(sub2.history("user", "location")) == 2


def test_namespace_isolation():
    sub = SubstrateGraph()
    sub.observe("user", "name", "Nick")
    sub.observe("session", "name", "interactive_chat")
    user_slots = sub.slots_in_namespace("user")
    session_slots = sub.slots_in_namespace("session")
    assert len(user_slots) == 1 and user_slots[0].slot_name == "name"
    assert len(session_slots) == 1 and session_slots[0].slot_name == "name"


def test_decay_rate_drops_effective_trust():
    sub = SubstrateGraph()
    s = sub.observe("session", "focus_level", "scattered",
                    trust=0.9, decay_rate=0.01)
    # decay_rate is per-second; 0.9 - 0.01 * 100 = -0.1 → clamped to 0.0
    assert s.effective_trust(now=s.observed_at + 100) == pytest.approx(0.0, abs=1e-6)
    assert s.effective_trust(now=s.observed_at + 10) == pytest.approx(0.8, abs=1e-6)


def test_find_contradictions_fallback_when_nli_disabled(monkeypatch):
    monkeypatch.delenv("AETHER_NLI_CONTRADICTION", raising=False)
    sub = SubstrateGraph()
    sub.observe("user", "location", "Seattle")
    sub.observe("user", "location", "Milwaukee")
    contras = sub.find_contradictions(namespace="user", use_nli=False)
    assert len(contras) == 1
    a, b, score = contras[0]
    assert score == 1.0


def test_migration_from_legacy_memory_graph_user_pattern():
    from aether.memory import MemoryGraph
    from aether.memory.graph import MemoryNode
    import time

    mg = MemoryGraph()
    nodes = [
        MemoryNode(memory_id="m1", text="user location: Seattle", trust=0.9, created_at=time.time()),
        MemoryNode(memory_id="m2", text="user location: Portland", trust=0.85, created_at=time.time()),
        MemoryNode(memory_id="m3", text="user favorite_color: orange (observed 5x in production)", trust=0.95, created_at=time.time()),
    ]
    for n in nodes:
        mg.add_memory(n)

    sub = migrate_from_memory_graph(mg)
    assert sub.stats_migration["user_pattern_slots"] == 3
    cs = sub.current_state("user", "favorite_color")
    assert cs is not None
    assert cs.value.lower() == "orange"


def test_migration_archives_unslottable_memories():
    from aether.memory import MemoryGraph
    from aether.memory.graph import MemoryNode
    import time

    mg = MemoryGraph()
    mg.add_memory(MemoryNode(
        memory_id="m1",
        text="The encoder warmup hangs in subprocess context due to HF connectivity check",
        trust=0.8,
        created_at=time.time(),
    ))
    sub = migrate_from_memory_graph(mg)
    assert sub.stats_migration["archived_memories"] >= 1
    archived = sub.history("meta", "archived_memory")
    assert len(archived) == 1
