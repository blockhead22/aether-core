"""Slot-first substrate primitive.

Three concepts:

* **SlotNode** -- the typed channel. ``(namespace, slot_name)`` uniquely
  identifies one. Examples: ``user:location``, ``code:my_proj:func_x``,
  ``session:focus_level``.
* **SlotState** -- one observed value of a slot at a point in time.
  Carries trust, source attribution, temporal status, decay rate.
* **Observation** -- the source event. One observation can emit
  multiple slot states (e.g. one turn of conversation says "I moved
  to Milwaukee and started at Microsoft" -> emits two slot states).
  The observation is the audit-trail node; v0.13's memory is now an
  observation in this model.

Edges in the slot graph are typed:

* ``SUPPORTS`` -- evidence relation, like v0.13's SUPPORTS but between
  slots not memories.
* ``DEPENDS_ON`` -- one slot's interpretation depends on another
  (e.g. ``code:foo:return_type`` depends on ``code:foo:source_path``).
* ``CONTRADICTS_WITH`` -- two slots are mutually exclusive on the same
  entity (most common: same slot, different values across time).
* ``SUPERSEDES`` -- one observation supersedes a prior one for the
  same slot.

The graph persists at ``~/.aether/substrate.json`` (separate from the
v0.13 ``mcp_state.json`` so both can coexist during migration).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class Namespace(str, Enum):
    """The first-class slot namespaces. Add new namespaces by extending."""
    USER = "user"          # personal facts (location, employer, ...)
    CODE = "code"          # codebase observations
    SESSION = "session"    # behavioral / inferred state
    PROJECT = "project"    # decisions on the current project
    META = "meta"          # substrate-watching-itself observations


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


class SlotEdgeType(str, Enum):
    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"
    CONTRADICTS_WITH = "contradicts_with"
    SUPERSEDES = "supersedes"


# ---------------------------------------------------------------------------
# Temporal + disposition (mirror v0.13 vocabulary so the auto-ingest
# pipeline can pass existing values through unchanged)
# ---------------------------------------------------------------------------


class TemporalStatus(str, Enum):
    ACTIVE = "active"
    PAST = "past"
    FUTURE = "future"
    POTENTIAL = "potential"


class Disposition(str, Enum):
    RESOLVABLE = "resolvable"
    HELD = "held"
    EVOLVING = "evolving"
    CONTEXTUAL = "contextual"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SlotNode:
    """A typed observation channel. (namespace, slot_name) is the unique key."""
    namespace: str
    slot_name: str
    slot_id: str = ""
    created_at: float = 0.0

    def __post_init__(self):
        if not self.slot_id:
            self.slot_id = f"{self.namespace}:{self.slot_name}"
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class SlotState:
    """One observed value of a slot at one point in time."""
    state_id: str
    slot_id: str
    value: str                   # raw value as observed
    normalized: str              # lowercase / canonicalized for comparison
    trust: float
    observed_at: float
    observation_id: str          # references Observation.observation_id
    temporal_status: str = TemporalStatus.ACTIVE.value
    decay_rate: float = 0.0      # per-second decay; 0 = never decays
    source: str = "unknown"      # 'auto_ingest' | 'manual' | 'migration' | ...
    superseded_by: Optional[str] = None  # state_id of the state that replaced this

    @classmethod
    def new(cls, slot_id: str, value: str, observation_id: str, **kw) -> 'SlotState':
        norm = (kw.pop("normalized", None) or value).strip().lower()
        return cls(
            state_id=f"st_{uuid.uuid4().hex[:12]}",
            slot_id=slot_id,
            value=value,
            normalized=norm,
            trust=kw.pop("trust", 0.7),
            observed_at=kw.pop("observed_at", time.time()),
            observation_id=observation_id,
            temporal_status=kw.pop("temporal_status", TemporalStatus.ACTIVE.value),
            decay_rate=kw.pop("decay_rate", 0.0),
            source=kw.pop("source", "unknown"),
            superseded_by=kw.pop("superseded_by", None),
        )

    def effective_trust(self, now: Optional[float] = None) -> float:
        """Trust after decay applied to current time."""
        if self.decay_rate <= 0:
            return self.trust
        elapsed = (now or time.time()) - self.observed_at
        return max(0.0, self.trust - self.decay_rate * elapsed)


@dataclass
class Observation:
    """The audit-trail entry. Records the source event that emitted slot states."""
    observation_id: str
    source_text: str
    observed_at: float
    source_type: str = "unknown"  # 'auto_ingest' | 'manual' | 'migration' | 'llm' | 'inference'
    emitted_state_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, source_text: str, source_type: str = "unknown", **kw) -> 'Observation':
        return cls(
            observation_id=f"obs_{uuid.uuid4().hex[:12]}",
            source_text=source_text,
            observed_at=kw.pop("observed_at", time.time()),
            source_type=source_type,
            emitted_state_ids=list(kw.pop("emitted_state_ids", [])),
            metadata=dict(kw.pop("metadata", {})),
        )


@dataclass
class SlotEdge:
    src_slot_id: str
    dst_slot_id: str
    edge_type: str
    weight: float = 1.0
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SubstrateGraph
# ---------------------------------------------------------------------------


class SubstrateGraph:
    """Slot-first substrate: nodes are slots, edges are typed slot relations.

    Use ``observe()`` to record a new slot state. Use ``current_state()`` to
    fetch the most recent observed value for a slot. Use ``history()`` to
    walk all states for a slot.
    """

    def __init__(self):
        self.slots: Dict[str, SlotNode] = {}                # slot_id -> SlotNode
        self.states: Dict[str, SlotState] = {}              # state_id -> SlotState
        self.observations: Dict[str, Observation] = {}      # observation_id -> Observation
        self.edges: List[SlotEdge] = []
        self._states_by_slot: Dict[str, List[str]] = {}     # slot_id -> [state_id, ...]
        self.persist_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Slot registration
    # ------------------------------------------------------------------

    def get_or_create_slot(self, namespace: str, slot_name: str) -> SlotNode:
        """Return an existing slot or create one. (namespace, slot_name) is unique."""
        slot_id = f"{namespace}:{slot_name}"
        if slot_id in self.slots:
            return self.slots[slot_id]
        node = SlotNode(namespace=namespace, slot_name=slot_name, slot_id=slot_id)
        self.slots[slot_id] = node
        self._states_by_slot.setdefault(slot_id, [])
        return node

    # ------------------------------------------------------------------
    # Core write API
    # ------------------------------------------------------------------

    def observe(
        self,
        namespace: str,
        slot_name: str,
        value: str,
        *,
        source_text: str = "",
        source_type: str = "unknown",
        trust: float = 0.7,
        temporal_status: str = TemporalStatus.ACTIVE.value,
        decay_rate: float = 0.0,
        observation_id: Optional[str] = None,
        normalized: Optional[str] = None,
        source: Optional[str] = None,
    ) -> SlotState:
        """Record a new observation: an observed value of (namespace, slot_name).

        If ``observation_id`` is None, a new Observation is created with
        ``source_text``. Multiple ``observe()`` calls can share an
        observation_id when one source event emits multiple slot values.

        The new SlotState is appended to the slot's history. If a prior
        state existed with a *different* normalized value, the prior
        state's ``superseded_by`` is set to the new state. Same-value
        re-observations are recorded as a fresh state (history) but do
        not mark a supersession (continued affirmation).
        """
        slot = self.get_or_create_slot(namespace, slot_name)

        if observation_id is None:
            obs = Observation.new(source_text=source_text or value, source_type=source_type)
            self.observations[obs.observation_id] = obs
            observation_id = obs.observation_id

        state = SlotState.new(
            slot_id=slot.slot_id,
            value=value,
            normalized=normalized,
            observation_id=observation_id,
            trust=trust,
            temporal_status=temporal_status,
            decay_rate=decay_rate,
            source=source or source_type,
        )
        self.states[state.state_id] = state
        self._states_by_slot.setdefault(slot.slot_id, []).append(state.state_id)

        # Wire the observation back to this state
        if observation_id in self.observations:
            self.observations[observation_id].emitted_state_ids.append(state.state_id)

        # Mark prior state superseded if value changed
        history = self._states_by_slot[slot.slot_id]
        if len(history) >= 2:
            prior = self.states[history[-2]]
            if prior.normalized != state.normalized and prior.superseded_by is None:
                prior.superseded_by = state.state_id
                self.add_edge(slot.slot_id, slot.slot_id,
                              SlotEdgeType.SUPERSEDES.value,
                              metadata={"from_state": prior.state_id, "to_state": state.state_id})

        return state

    def add_edge(
        self,
        src_slot_id: str,
        dst_slot_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SlotEdge:
        edge = SlotEdge(
            src_slot_id=src_slot_id,
            dst_slot_id=dst_slot_id,
            edge_type=edge_type,
            weight=weight,
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self.edges.append(edge)
        return edge

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def current_state(self, namespace: str, slot_name: str) -> Optional[SlotState]:
        """Return the most recent (non-superseded) state for a slot, or None."""
        slot_id = f"{namespace}:{slot_name}"
        history = self._states_by_slot.get(slot_id, [])
        for state_id in reversed(history):
            s = self.states[state_id]
            if s.superseded_by is None:
                return s
        return None

    def history(self, namespace: str, slot_name: str) -> List[SlotState]:
        """All states for a slot in chronological order."""
        slot_id = f"{namespace}:{slot_name}"
        return [self.states[sid] for sid in self._states_by_slot.get(slot_id, [])]

    def slots_in_namespace(self, namespace: str) -> List[SlotNode]:
        return [s for s in self.slots.values() if s.namespace == namespace]

    def all_slot_ids(self) -> List[str]:
        return list(self.slots.keys())

    def edges_for(self, slot_id: str, edge_type: Optional[str] = None) -> List[SlotEdge]:
        return [
            e for e in self.edges
            if (e.src_slot_id == slot_id or e.dst_slot_id == slot_id)
            and (edge_type is None or e.edge_type == edge_type)
        ]

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def find_contradictions(
        self,
        namespace: Optional[str] = None,
        use_nli: bool = True,
        threshold: float = 0.5,
    ) -> List[Tuple[SlotState, SlotState, float]]:
        """Find (state_a, state_b, contradiction_score) tuples within each slot.

        For each slot, compares the most recent state against earlier
        states (or pairs of distinct values). Uses NLI if available
        (opt-in via AETHER_NLI_CONTRADICTION=1); otherwise falls back
        to normalized-value mismatch with score 1.0 for any mismatch.

        Returns pairs ordered by contradiction score descending.
        """
        try:
            from aether.contradiction.nli import score_pairs, is_enabled as nli_enabled
        except ImportError:
            score_pairs = None
            nli_enabled = lambda: False  # noqa: E731

        results = []
        for slot in self.slots.values():
            if namespace and slot.namespace != namespace:
                continue
            history = self.history(slot.namespace, slot.slot_name)
            if len(history) < 2:
                continue
            # Build distinct-normalized-value pairs
            seen_norms: Dict[str, SlotState] = {}
            for s in history:
                if s.normalized not in seen_norms:
                    seen_norms[s.normalized] = s
            distinct = list(seen_norms.values())
            if len(distinct) < 2:
                continue

            pairs = [(distinct[i], distinct[j])
                     for i in range(len(distinct))
                     for j in range(i + 1, len(distinct))]
            if use_nli and score_pairs and nli_enabled():
                text_pairs = [(a.value, b.value) for a, b in pairs]
                scores, status = score_pairs(text_pairs)
                if status == "ok":
                    for (a, b), s in zip(pairs, scores):
                        if s.contradiction_prob > threshold:
                            results.append((a, b, float(s.contradiction_prob)))
                    continue  # skip fallback
            # Fallback: any value mismatch is contradiction with score 1.0
            for a, b in pairs:
                results.append((a, b, 1.0))

        results.sort(key=lambda t: -t[2])
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": "0.14.0-substrate",
            "slots": {sid: asdict(s) for sid, s in self.slots.items()},
            "states": {sid: asdict(s) for sid, s in self.states.items()},
            "observations": {oid: asdict(o) for oid, o in self.observations.items()},
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SubstrateGraph':
        sub = cls()
        for sid, sdict in (data.get("slots") or {}).items():
            sub.slots[sid] = SlotNode(**sdict)
            sub._states_by_slot.setdefault(sid, [])
        for sid, sdict in (data.get("states") or {}).items():
            state = SlotState(**sdict)
            sub.states[sid] = state
            sub._states_by_slot.setdefault(state.slot_id, []).append(state.state_id)
        # Sort histories chronologically
        for sid, hist in sub._states_by_slot.items():
            hist.sort(key=lambda s_id: sub.states[s_id].observed_at)
        for oid, odict in (data.get("observations") or {}).items():
            sub.observations[oid] = Observation(**odict)
        for edict in (data.get("edges") or []):
            sub.edges.append(SlotEdge(**edict))
        return sub

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.persist_path
        if not path:
            path = str(Path.home() / ".aether" / "substrate.json")
        self.persist_path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        os.replace(tmp, path)
        return path

    def load(self, path: Optional[str] = None) -> 'SubstrateGraph':
        path = path or self.persist_path
        if not path:
            path = str(Path.home() / ".aether" / "substrate.json")
        self.persist_path = path
        if not Path(path).exists():
            return self
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        loaded = SubstrateGraph.from_dict(data)
        self.slots = loaded.slots
        self.states = loaded.states
        self.observations = loaded.observations
        self.edges = loaded.edges
        self._states_by_slot = loaded._states_by_slot
        return self

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        n_namespaces = len(set(s.namespace for s in self.slots.values()))
        return {
            "slots": len(self.slots),
            "states": len(self.states),
            "observations": len(self.observations),
            "edges": len(self.edges),
            "namespaces": n_namespaces,
            "namespace_breakdown": {
                ns: sum(1 for s in self.slots.values() if s.namespace == ns)
                for ns in set(s.namespace for s in self.slots.values())
            },
        }

    def __len__(self) -> int:
        return len(self.slots)
