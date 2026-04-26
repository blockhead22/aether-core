"""State store for the Aether MCP server.

Wraps MemoryGraph + GovernanceLayer + StructuralTensionMeter and
handles JSON persistence. Single in-process instance for the
server's lifetime.
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path
from typing import Optional

from aether.governance import GovernanceLayer
from aether.contradiction import StructuralTensionMeter
from aether.memory import MemoryGraph, MemoryNode


def _default_state_path() -> str:
    override = os.environ.get("AETHER_STATE_PATH")
    if override:
        return override
    home = Path.home()
    return str(home / ".aether" / "mcp_state.json")


class StateStore:
    """In-process state held across MCP tool calls."""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or _default_state_path()
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)

        self.graph = MemoryGraph(persist_path=self.state_path)
        self.gov = GovernanceLayer()
        self.meter = StructuralTensionMeter()
        # Monotonic counter so rapid-fire writes don't collide on ms timestamps.
        self._id_counter = itertools.count(1)

    def save(self) -> None:
        """Persist the graph to disk. Call after every mutation."""
        self.graph.save(self.state_path)

    def add_memory(
        self,
        text: str,
        trust: float = 0.7,
        source: str = "user",
        slots: Optional[dict] = None,
    ) -> str:
        memory_id = f"m{int(time.time() * 1000)}_{next(self._id_counter)}"
        tags: list[str] = [f"source:{source}"]
        if slots:
            tags.extend(f"slot:{k}={v}" for k, v in slots.items())

        node = MemoryNode(
            memory_id=memory_id,
            text=text,
            created_at=time.time(),
            trust=trust,
            tags=tags,
        )
        self.graph.add_memory(node)
        self.save()
        return memory_id

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Simple substring + slot-overlap search.

        Pure structural retrieval; no embeddings. Sufficient for v0.4
        when the belief state is small. Returns memories ranked by
        rough relevance score.
        """
        q_lower = query.lower()
        scored: list[tuple[float, MemoryNode]] = []

        for node in self.graph.all_memories():
            text_lower = node.text.lower()
            score = 0.0
            if q_lower in text_lower:
                score += 1.0
            # token overlap as a tiebreaker
            q_tokens = set(q_lower.split())
            t_tokens = set(text_lower.split())
            overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
            score += overlap * 0.5
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, n in scored[:limit]:
            source = next(
                (t.split(":", 1)[1] for t in n.tags if t.startswith("source:")),
                "unknown",
            )
            results.append({
                "memory_id": n.memory_id,
                "text": n.text,
                "trust": n.trust,
                "source": source,
                "created_at": n.created_at,
                "score": round(score, 3),
            })
        return results

    def stats(self) -> dict:
        s = self.graph.stats()
        return {
            "memory_count": s.get("nodes", 0),
            "edge_count": s.get("edges", 0),
            "state_path": self.state_path,
            "belnap_states": s.get("belnap_states", {}),
        }
