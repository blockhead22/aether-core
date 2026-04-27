"""State store for the Aether MCP server.

Wraps MemoryGraph + GovernanceLayer + StructuralTensionMeter and
handles JSON persistence. Single in-process instance for the
server's lifetime.

v0.5.0 additions:
  - Lazy-loaded embedding encoder (sentence-transformers, all-MiniLM-L6-v2)
  - Embedding-backed search with substring fallback
  - Contradiction detection on every write (StructuralTensionMeter)
  - Substrate-grounded belief_confidence computation
  - Trust-history log for belief_history / volatility
  - BDG-aware correction with cascade
  - Lineage / cascade-preview / contradictions / resolve / session-diff
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aether.governance import GovernanceLayer
from aether.contradiction import (
    StructuralTensionMeter,
    TensionRelationship,
    TensionAction,
    detect_mutex_conflict,
)
from aether.memory import (
    MemoryGraph,
    MemoryNode,
    EdgeType,
    BelnapState,
    Disposition,
    ContradictionEdge,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Process-wide model cache. Shared across all StateStore instances so
# multiple substrates (multi-tenant scenarios, test suites, scripts
# that build several stores) only pay the cold-load cost once. The
# lock serializes load attempts to avoid duplicate torch imports.
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = None  # initialized lazily — see _get_cache_lock


def _get_cache_lock():
    """Lazy-init a threading.Lock without forcing import at module load."""
    global _MODEL_CACHE_LOCK
    if _MODEL_CACHE_LOCK is None:
        import threading
        _MODEL_CACHE_LOCK = threading.Lock()
    return _MODEL_CACHE_LOCK

# Substrate-grounded fidelity threshold:
# When the caller passes the sentinel default (0.5), we compute a real
# belief_confidence by searching the substrate. The grounding score is the
# trust-weighted average of the top-K most relevant memories.
GROUNDING_TOP_K = 5
GROUNDING_MIN_SCORE = 0.15  # Below this similarity, the memory isn't relevant
SENTINEL_BELIEF_CONF = 0.5  # Treat 0.5 as "caller didn't supply real value"

# Auto-contradiction-detection threshold on write:
# Run the StructuralTensionMeter against this many top-similar existing
# memories. Anything above the threshold gets a CONTRADICTS edge.
TENSION_TOP_K = 8
TENSION_CONFLICT_THRESHOLD = 0.55

# Policy contradiction detection (for sanction-time gating).
# StructuralTensionMeter is fact-vs-fact and misses command-vs-prohibition,
# so we add a lightweight imperative/prohibition cross-check.
PROHIBITION_CUES = (
    "never", "don't", "do not", "must not", "should not",
    "forbidden", "prohibited", "blocked", "not allowed",
    "no force push", "no force-push", "do not force",
    "shall not",
)
IMPERATIVE_CUES = (
    "force push", "force-push", "rm -rf", "delete all",
    "drop database", "drop table", "truncate", "deploy to prod",
    "push to main", "push to master", "merge to main",
    "skip tests", "disable tests", "--no-verify",
)
POLICY_CONTRA_MIN_SIMILARITY = 0.45
POLICY_CONTRA_MIN_TRUST = 0.7


def _looks_like_prohibition(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in PROHIBITION_CUES)


def _looks_like_imperative(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in IMPERATIVE_CUES)


# Generic negation cues for asymmetric-negation contradiction detection.
# When two highly-similar texts differ in negation polarity, that's a
# strong contradiction signal even when no slot conflict is present.
_NEGATION_CUES = (
    " not ", "n't ", " never ", " no ", " none ",
    " neither ", " nor ", " without ", " un",  # un-prefix verbs
    "not.", "not,", "n't.", "n't,",
)


def _has_negation(text: str) -> bool:
    t = " " + text.lower() + " "
    return any(cue in t for cue in _NEGATION_CUES)


def _is_asymmetric_negation_contradict(
    text_a: str,
    text_b: str,
    similarity: float,
    min_similarity: float = POLICY_CONTRA_MIN_SIMILARITY,
) -> bool:
    """Two highly-similar texts where exactly one contains negation cues.

    Catches cases the StructuralTensionMeter misses:
        "We use pnpm not npm" vs "We use npm"
        "Main is protected, never force push" vs "Force push to main"
        "Main is protected, never force push" vs "Force push to main"

    Not perfect (won't catch double-negation, semantic flips like
    'increased' vs 'decreased', or rephrased prohibitions), but cheap
    and high-precision for the common case.
    """
    if similarity < min_similarity:
        return False
    a_neg = _has_negation(text_a)
    b_neg = _has_negation(text_b)
    return a_neg != b_neg


def _find_repo_state(start_dir: Optional[str] = None) -> Optional[str]:
    """Walk up from `start_dir` looking for a project-level .aether/state.json.

    Stops at the first `.aether/` directory found or at the filesystem
    root. Returns None if nothing is found.

    A repo-level substrate is identified by a `.aether/state.json` file
    sitting at any ancestor directory of `start_dir`. This convention
    matches how dotfiles like .gitignore, .editorconfig, .prettierrc work.
    """
    cur = Path(start_dir or os.getcwd()).resolve()
    while True:
        candidate = cur / ".aether" / "state.json"
        if candidate.exists():
            return str(candidate)
        # Stop at filesystem root
        if cur.parent == cur:
            return None
        cur = cur.parent


def _default_state_path() -> str:
    """Resolve the substrate state path with this priority:

    1. $AETHER_STATE_PATH if set (explicit override)
    2. .aether/state.json in the current project tree (repo-scoped)
    3. ~/.aether/mcp_state.json (user-global default)
    """
    override = os.environ.get("AETHER_STATE_PATH")
    if override:
        return override

    # Skip repo discovery when AETHER_NO_REPO_DISCOVERY=1 is set
    if not os.environ.get("AETHER_NO_REPO_DISCOVERY"):
        repo = _find_repo_state()
        if repo:
            return repo

    home = Path.home()
    return str(home / ".aether" / "mcp_state.json")


def _trust_history_path(state_path: str) -> str:
    """Side-car file: append-only trust history per memory."""
    return state_path.replace(".json", "_trust_history.json")


# ---------------------------------------------------------------------------
# Lightweight encoder wrapper
# ---------------------------------------------------------------------------

class _LazyEncoder:
    """Wraps sentence-transformers with a graceful fallback.

    Loading SentenceTransformer is slow (torch import, possibly model
    download, device probing). Doing it synchronously inside an MCP
    tool call blocks the entire turn — on Windows with a cold cache,
    that can be 30s to several minutes.

    This class never blocks a tool call:

    - `start_warmup()` kicks off the load on a background thread.
    - `encode()` returns None immediately if the model isn't loaded yet.
      The caller (search, grounding) falls back to substring matching.
    - Once the background thread finishes, subsequent `encode()` calls
      use the model.

    Behavior summary:
        encoder = _LazyEncoder()
        encoder.start_warmup()      # returns instantly, model loads behind
        encoder.encode("hello")     # may return None if not warm yet
        # ... 30 seconds later ...
        encoder.encode("hello")     # now returns a numpy vector

    If the [ml] extra isn't installed, `start_warmup()` immediately
    flags the encoder as unavailable. Same fallback as before.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None
        self._unavailable = False
        self._warmup_thread = None
        self._warmup_started = False

    def _load(self):
        """Synchronous load. Use start_warmup() for non-blocking init.

        Uses a process-wide cache so multiple StateStore instances in
        the same process share one model load. The lock serializes
        first-load attempts to avoid duplicate torch imports.
        """
        if self._model is not None or self._unavailable:
            return
        # Fast path: model already cached for this name
        cached = _MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached
            return
        # Slow path: load under lock
        lock = _get_cache_lock()
        with lock:
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is not None:
                self._model = cached
                return
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = model
                self._model = model
            except Exception:
                self._unavailable = True

    def start_warmup(self) -> None:
        """Kick off background model load. Idempotent; returns instantly."""
        if self._warmup_started or self._model is not None or self._unavailable:
            return
        self._warmup_started = True
        try:
            import threading
            t = threading.Thread(
                target=self._load,
                name="aether-encoder-warmup",
                daemon=True,
            )
            t.start()
            self._warmup_thread = t
        except Exception:
            # If threading fails (rare), fall back to marking unavailable.
            # We never block a tool call to recover.
            self._unavailable = True

    @property
    def available(self) -> bool:
        """Force-load the model and return whether it's usable.

        WARNING: this triggers a *synchronous* SentenceTransformer load
        on first access — can take tens of seconds to minutes on a cold
        cache. Almost no caller should use this. Use `is_loaded` for
        observation, or call `start_warmup()` then poll `is_loaded`.
        """
        if self._model is None and not self._unavailable:
            self._load()
        return not self._unavailable and self._model is not None

    @property
    def is_loaded(self) -> bool:
        """Whether the encoder is already loaded. Never triggers loading."""
        return self._model is not None

    @property
    def is_unavailable(self) -> bool:
        """Whether a previous load attempt failed. Never triggers loading."""
        return self._unavailable

    @property
    def is_warming(self) -> bool:
        """Whether a background warmup is currently in flight."""
        return (
            self._warmup_started
            and not self.is_loaded
            and not self._unavailable
        )

    def encode(self, text: str):
        """Encode text. Returns None if the model isn't ready yet.

        Crucially, this NEVER blocks waiting for the model. If warmup
        hasn't completed, the caller gets None and is expected to fall
        back to substring / structural matching. Once the model is
        warm, subsequent calls return real vectors.
        """
        if not self.is_loaded:
            return None
        try:
            import numpy as np
            vec = self._model.encode([text], convert_to_numpy=True)[0]
            n = float(np.linalg.norm(vec))
            return vec / n if n > 0 else vec
        except Exception:
            return None

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until the encoder is loaded or unavailable.

        Use ONLY in tests or one-off scripts where blocking is acceptable.
        MCP tool handlers must never call this — they should let
        encode() return None and fall back to substring matching.

        Returns True if the encoder is ready (loaded), False on timeout
        or if it failed to load (unavailable).
        """
        # If warmup hasn't been started, kick it off so we don't wait
        # forever on a thread that doesn't exist.
        if not self._warmup_started and not self._unavailable:
            self.start_warmup()
        if self._warmup_thread is not None:
            self._warmup_thread.join(timeout=timeout)
        return self.is_loaded


# ---------------------------------------------------------------------------
# State store
# ---------------------------------------------------------------------------

class StateStore:
    """In-process state held across MCP tool calls."""

    def __init__(
        self,
        state_path: Optional[str] = None,
        encoder: Optional[Any] = None,
        enable_embeddings: bool = True,
    ):
        self.state_path = state_path or _default_state_path()
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)

        self.graph = MemoryGraph(persist_path=self.state_path)
        self.gov = GovernanceLayer()

        self._enable_embeddings = enable_embeddings
        self._encoder = encoder if encoder is not None else (
            _LazyEncoder() if enable_embeddings else None
        )
        # Kick off model loading in the background so the first
        # search-style tool call doesn't block on cold-start.
        # Tools that try to encode before this completes get None
        # back and fall through to substring matching.
        if self._encoder is not None and hasattr(self._encoder, "start_warmup"):
            try:
                self._encoder.start_warmup()
            except Exception:
                pass
        # Tension meter shares the encoder so it can use real similarity.
        self.meter = StructuralTensionMeter(encoder=self._encoder)

        self._id_counter = itertools.count(1)
        self._trust_history: Dict[str, List[Dict[str, Any]]] = {}
        self._load_trust_history()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the graph + trust history to disk."""
        self.graph.save(self.state_path)
        self._save_trust_history()

    def _save_trust_history(self) -> None:
        path = _trust_history_path(self.state_path)
        try:
            with open(path, "w") as f:
                json.dump(self._trust_history, f, indent=2)
        except OSError:
            pass

    def _load_trust_history(self) -> None:
        path = _trust_history_path(self.state_path)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                self._trust_history = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._trust_history = {}

    def _record_trust(
        self,
        memory_id: str,
        new_trust: float,
        old_trust: Optional[float],
        source: str,
        reason: str = "",
    ) -> None:
        entry = {
            "ts": time.time(),
            "trust": new_trust,
            "old_trust": old_trust,
            "delta": (new_trust - old_trust) if old_trust is not None else None,
            "source": source,
            "reason": reason,
        }
        self._trust_history.setdefault(memory_id, []).append(entry)

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str):
        if self._encoder is None:
            return None
        try:
            return self._encoder.encode(text)
        except Exception:
            return None

    def _cosine(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        try:
            import numpy as np
            an = float(np.linalg.norm(a))
            bn = float(np.linalg.norm(b))
            if an == 0 or bn == 0:
                return 0.0
            return float(np.dot(a, b) / (an * bn))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Memory writes (with contradiction detection)
    # ------------------------------------------------------------------

    def add_memory(
        self,
        text: str,
        trust: float = 0.7,
        source: str = "user",
        slots: Optional[dict] = None,
        detect_contradictions: bool = True,
    ) -> Dict[str, Any]:
        """Add a memory, optionally running contradiction detection.

        Returns a dict with `memory_id`, `extracted_slots`, and a
        `tension_findings` list describing CONTRADICTS edges added.
        """
        memory_id = f"m{int(time.time() * 1000)}_{next(self._id_counter)}"
        tags: list[str] = [f"source:{source}"]
        if slots:
            tags.extend(f"slot:{k}={v}" for k, v in slots.items())

        embedding = self._encode(text) if self._encoder else None

        node = MemoryNode(
            memory_id=memory_id,
            text=text,
            created_at=time.time(),
            trust=trust,
            tags=tags,
        )
        self.graph.add_memory(node, embedding=embedding)
        self._record_trust(memory_id, trust, None, source, "initial assertion")

        findings: List[Dict[str, Any]] = []
        if detect_contradictions:
            findings = self._detect_and_record_tensions(memory_id, text, trust, source)

        self.save()
        return {
            "memory_id": memory_id,
            "trust": trust,
            "source": source,
            "extracted_slots": slots or {},
            "tension_findings": findings,
        }

    def _detect_and_record_tensions(
        self,
        memory_id: str,
        text: str,
        trust: float,
        source: str,
    ) -> List[Dict[str, Any]]:
        """Run tension meter against top-K most similar existing memories."""
        new_emb = self.graph.get_embedding(memory_id)
        candidates: List[Tuple[float, MemoryNode]] = []

        for other in self.graph.all_memories():
            if other.memory_id == memory_id:
                continue
            other_emb = self.graph.get_embedding(other.memory_id)
            if new_emb is not None and other_emb is not None:
                sim = self._cosine(new_emb, other_emb)
            else:
                # Fallback: token overlap as similarity proxy
                a = set(text.lower().split())
                b = set(other.text.lower().split())
                sim = len(a & b) / max(len(a | b), 1)
            candidates.append((sim, other))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:TENSION_TOP_K]

        findings: List[Dict[str, Any]] = []
        for sim, other in top:
            if sim < 0.2:  # Don't bother — clearly unrelated
                continue
            try:
                result = self.meter.measure(
                    text_a=other.text,
                    text_b=text,
                    trust_a=other.trust,
                    trust_b=trust,
                    source_a=_extract_source(other.tags),
                    source_b=source,
                    timestamp_a=other.created_at,
                    timestamp_b=time.time(),
                )
            except Exception:
                continue

            # Four independent contradiction signals — any one triggers an edge.
            slot_conflict = (
                result.tension_score >= TENSION_CONFLICT_THRESHOLD
                and result.relationship in (
                    TensionRelationship.CONFLICT,
                    TensionRelationship.TENSION,
                )
            )
            sim_for_check = result.supporting_signals.get(
                "embedding_similarity", sim
            )
            asymm_neg = _is_asymmetric_negation_contradict(
                other.text, text, sim_for_check,
            )
            policy = (
                sim_for_check >= POLICY_CONTRA_MIN_SIMILARITY
                and other.trust >= POLICY_CONTRA_MIN_TRUST
                and (
                    (_looks_like_prohibition(other.text)
                     and _looks_like_imperative(text)
                     and not _looks_like_prohibition(text))
                    or (_looks_like_imperative(other.text)
                        and _looks_like_prohibition(text)
                        and not _looks_like_imperative(text))
                )
            )
            mutex_hit = detect_mutex_conflict(other.text, text)

            if not (slot_conflict or asymm_neg or policy or mutex_hit):
                continue

            disposition = self._tension_to_disposition(result)
            # If only mutex fired, classify as RESOLVABLE (one of the
            # canonical values is wrong or stale)
            if mutex_hit and not slot_conflict:
                disposition = Disposition.RESOLVABLE
            # Build the rule trace so callers can see *why* this fired.
            trace = [result.action.value]
            kind = "structural"
            if mutex_hit:
                trace.append(
                    f"mutex:{mutex_hit.class_name}"
                    f"={mutex_hit.value_a}<>{mutex_hit.value_b}"
                )
                kind = "mutex"
            if asymm_neg:
                trace.append("asymmetric_negation")
                kind = "negation_asymmetry" if kind == "structural" else kind
            if policy:
                trace.append("policy_contradiction")
                kind = "policy" if kind == "structural" else kind

            edge = ContradictionEdge(
                disposition=disposition.value,
                nli_score=max(
                    result.tension_score,
                    0.85 if mutex_hit else 0.0,
                ),
                overlap_integral=result.supporting_signals.get(
                    "embedding_similarity", sim
                ),
                detected_at=time.time(),
                classification_confidence=(
                    mutex_hit.confidence if mutex_hit else result.confidence
                ),
                rule_trace=trace,
            )
            self.graph.add_contradiction(other.memory_id, memory_id, edge)
            finding = {
                "with_memory_id": other.memory_id,
                "with_text": other.text[:140],
                "relationship": result.relationship.value,
                "disposition": disposition.value,
                "tension_score": round(
                    max(result.tension_score, 0.85 if mutex_hit else 0.0),
                    3,
                ),
                "recommended_action": result.action.value,
                "kind": kind,
                "trace": trace,
            }
            if mutex_hit:
                finding["mutex"] = {
                    "class": mutex_hit.class_name,
                    "value_a": mutex_hit.value_a,
                    "value_b": mutex_hit.value_b,
                    "cue": mutex_hit.cue_used,
                }
            findings.append(finding)

        return findings

    @staticmethod
    def _tension_to_disposition(result) -> Disposition:
        # Conservative mapping: outright conflict + correction = resolvable;
        # ambiguous tension on the same topic = held; everything else evolving.
        if result.relationship == TensionRelationship.CONFLICT:
            return Disposition.RESOLVABLE
        if result.action == TensionAction.KEEP_BOTH:
            return Disposition.HELD
        if result.action == TensionAction.FLAG_FOR_REVIEW:
            return Disposition.HELD
        return Disposition.EVOLVING

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid search: embedding cosine when available, substring fallback.

        Returns memories ranked by combined score. Each result includes a
        `score`, `similarity` (cosine if embeddings used), and the
        familiar substring/token overlap score.
        """
        q_lower = query.lower()
        q_emb = self._encode(query) if self._encoder else None
        scored: list[tuple[float, dict]] = []

        for node in self.graph.all_memories():
            text_lower = node.text.lower()
            substring_score = 0.0
            if q_lower in text_lower:
                substring_score += 1.0
            q_tokens = set(q_lower.split())
            t_tokens = set(text_lower.split())
            overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
            substring_score += overlap * 0.5

            sim = 0.0
            if q_emb is not None:
                node_emb = self.graph.get_embedding(node.memory_id)
                if node_emb is not None:
                    sim = self._cosine(q_emb, node_emb)

            # Combined: weight embedding similarity higher when available,
            # but always include substring as a tiebreaker / safety net.
            if q_emb is not None:
                combined = 0.7 * sim + 0.3 * min(substring_score, 1.5)
            else:
                combined = substring_score

            if combined <= 0:
                continue

            source = _extract_source(node.tags)
            belnap = getattr(node, "belnap_state", "T")
            warnings: List[str] = []
            if belnap == "Both":
                warnings.append("contested: held contradiction on this memory")
            elif belnap == "F":
                warnings.append("deprecated: superseded or resolved away")
            elif belnap == "Neither":
                warnings.append("uncertain: insufficient evidence")
            scored.append((
                combined,
                {
                    "memory_id": node.memory_id,
                    "text": node.text,
                    "trust": node.trust,
                    "source": source,
                    "created_at": node.created_at,
                    "score": round(combined, 3),
                    "similarity": round(sim, 3) if q_emb is not None else None,
                    "substring_score": round(substring_score, 3),
                    "belnap_state": belnap,
                    "warnings": warnings,
                },
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    # ------------------------------------------------------------------
    # Substrate-grounded belief_confidence
    # ------------------------------------------------------------------

    def compute_grounding(
        self,
        text: str,
        top_k: int = GROUNDING_TOP_K,
    ) -> Dict[str, Any]:
        """Compute belief_confidence by searching substrate for grounding.

        Returns:
            {
              "belief_confidence": float in [0, 1],
              "support": [memories that ground the claim],
              "contradict": [memories that conflict with the claim],
              "method": "embedding" | "substring" | "empty",
            }

        Heuristic:
          1. Find top-K memories by similarity.
          2. Drop ones below GROUNDING_MIN_SCORE.
          3. Use the StructuralTensionMeter to classify each: support,
             contradict, or unrelated.
          4. belief_confidence = trust-weighted mean of support, minus a
             contradiction penalty.
        """
        results = self.search(text, limit=top_k * 2)
        if not results:
            return {
                "belief_confidence": 0.3,  # neutral-low when substrate empty
                "support": [],
                "contradict": [],
                "method": "empty",
            }

        method = "embedding" if results[0].get("similarity") is not None else "substring"

        kept = [r for r in results if r["score"] >= GROUNDING_MIN_SCORE][:top_k]
        if not kept:
            return {
                "belief_confidence": 0.3,
                "support": [],
                "contradict": [],
                "method": method,
            }

        support: List[Dict[str, Any]] = []
        contradict: List[Dict[str, Any]] = []
        weights = []

        for hit in kept:
            try:
                tr = self.meter.measure(
                    text_a=hit["text"],
                    text_b=text,
                    trust_a=hit["trust"],
                    trust_b=0.7,
                    source_a=hit.get("source", "user"),
                    source_b="query",
                )
            except Exception:
                continue

            # Standard fact-vs-fact tension classification
            is_factual_contradict = (
                tr.relationship == TensionRelationship.CONFLICT or (
                    tr.relationship == TensionRelationship.TENSION
                    and tr.tension_score >= TENSION_CONFLICT_THRESHOLD
                )
            )

            # Policy contradiction (command vs prohibition).
            # The tension meter misses this case because there are no
            # shared slots between an imperative and a policy statement.
            sim = tr.supporting_signals.get("embedding_similarity", 0.0)
            mem_prohibits = _looks_like_prohibition(hit["text"])
            mem_imperative = _looks_like_imperative(hit["text"])
            new_imperative = _looks_like_imperative(text)
            new_prohibits = _looks_like_prohibition(text)
            high_overlap = (
                sim >= POLICY_CONTRA_MIN_SIMILARITY
                and hit["trust"] >= POLICY_CONTRA_MIN_TRUST
            )
            is_policy_contradict = high_overlap and (
                (mem_prohibits and new_imperative and not new_prohibits)
                or (mem_imperative and new_prohibits and not new_imperative)
            )

            # Asymmetric-negation: "We use pnpm not npm" vs "We use npm"
            is_asymm_neg = _is_asymmetric_negation_contradict(
                hit["text"], text, sim,
            ) and hit["trust"] >= POLICY_CONTRA_MIN_TRUST

            # Mutual-exclusion: "we use AWS" vs "we use GCP"
            mutex_hit = detect_mutex_conflict(hit["text"], text)

            if is_factual_contradict or is_policy_contradict or is_asymm_neg or mutex_hit:
                kind = "factual"
                if is_asymm_neg and not is_factual_contradict:
                    kind = "negation_asymmetry"
                if is_policy_contradict and not is_factual_contradict:
                    kind = "policy"
                if mutex_hit and not is_factual_contradict:
                    kind = "mutex"
                entry = {
                    **hit,
                    "tension_score": round(
                        max(tr.tension_score, 0.85 if mutex_hit else 0.0),
                        3,
                    ),
                    "kind": kind,
                }
                if mutex_hit:
                    entry["mutex"] = {
                        "class": mutex_hit.class_name,
                        "value_a": mutex_hit.value_a,
                        "value_b": mutex_hit.value_b,
                    }
                contradict.append(entry)
            elif tr.relationship in (
                TensionRelationship.DUPLICATE,
                TensionRelationship.REFINEMENT,
                TensionRelationship.COMPATIBLE,
            ):
                support.append({**hit, "tension_score": round(tr.tension_score, 3)})
                weights.append(hit["trust"])

        if not support and not contradict:
            return {
                "belief_confidence": 0.4,
                "support": [],
                "contradict": [],
                "method": method,
            }

        if support:
            base = sum(weights) / len(weights)
        else:
            base = 0.3

        # Penalty: each high-confidence contradiction drops belief
        contradiction_penalty = sum(
            min(0.4, c["trust"] * c.get("tension_score", 0.5))
            for c in contradict
        )
        belief = max(0.0, min(1.0, base - contradiction_penalty))

        return {
            "belief_confidence": round(belief, 3),
            "support": support,
            "contradict": contradict,
            "method": method,
        }

    # ------------------------------------------------------------------
    # Correction with cascade
    # ------------------------------------------------------------------

    def correct(
        self,
        memory_id: str,
        new_trust: Optional[float] = None,
        replacement_text: Optional[str] = None,
        reason: str = "",
        source: str = "user",
    ) -> Dict[str, Any]:
        """Correct a memory: update trust, optionally replace text, cascade
        the trust drop to dependent memories via SUPPORTS / DERIVED_FROM
        edges.

        Returns the cascade result + adjusted nodes.
        """
        node = self.graph.get_memory(memory_id)
        if node is None:
            return {"error": f"unknown memory_id: {memory_id}"}

        old_trust = node.trust
        if new_trust is None:
            # Default: halve the trust as a soft demotion
            new_trust = max(0.0, old_trust * 0.5)

        # Apply the correction
        self.graph.graph.nodes[memory_id]["trust"] = new_trust
        if replacement_text is not None:
            self.graph.graph.nodes[memory_id]["text"] = replacement_text
            new_emb = self._encode(replacement_text)
            if new_emb is not None:
                self.graph._embeddings[memory_id] = new_emb

        self._record_trust(memory_id, new_trust, old_trust, source, reason)

        # Cascade to supporters via in-edges (NOT contradicts)
        cascade = self._cascade_correction(memory_id, old_trust - new_trust)

        self.save()
        return {
            "memory_id": memory_id,
            "old_trust": old_trust,
            "new_trust": new_trust,
            "delta": round(new_trust - old_trust, 3),
            "reason": reason,
            "cascade": cascade,
        }

    def _cascade_correction(
        self,
        source_id: str,
        loss: float,
        damping: float = 0.6,
        min_impact: float = 0.02,
        max_depth: int = 6,
    ) -> Dict[str, Any]:
        """BFS backward over SUPPORTS / DERIVED_FROM edges, demoting trust."""
        from collections import deque
        affected: Dict[str, float] = {}
        depth_map: Dict[str, int] = {source_id: 0}
        queue = deque([(source_id, loss, 0)])
        skip = {EdgeType.CONTRADICTS.value, "CONTRADICTS"}

        while queue:
            node_id, impact, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for pred, _, edata in self.graph.graph.in_edges(node_id, data=True):
                if pred == source_id:
                    continue
                etype = edata.get("edge_type", "supports")
                if etype in skip:
                    continue
                w = edata.get("weight", 0.5)
                propagated = impact * w * damping
                if propagated < min_impact:
                    continue
                if pred in affected and propagated <= affected[pred]:
                    continue
                affected[pred] = propagated
                depth_map[pred] = depth + 1
                queue.append((pred, propagated, depth + 1))

        # Apply demotions
        for nid, impact in affected.items():
            old = self.graph.graph.nodes[nid].get("trust", 0.5)
            new = max(0.0, old - impact)
            self.graph.graph.nodes[nid]["trust"] = new
            self._record_trust(nid, new, old, "cascade",
                               f"cascade from {source_id}")

        return {
            "source": source_id,
            "loss": round(loss, 3),
            "affected_nodes": [
                {
                    "memory_id": nid,
                    "depth": depth_map.get(nid, -1),
                    "trust_delta": round(-impact, 3),
                }
                for nid, impact in sorted(affected.items(), key=lambda x: -x[1])
            ],
            "total_affected": len(affected),
        }

    def cascade_preview(
        self,
        memory_id: str,
        proposed_delta: float = -0.4,
    ) -> Dict[str, Any]:
        """Dry-run a correction. Same engine as `correct()` but no commit."""
        node = self.graph.get_memory(memory_id)
        if node is None:
            return {"error": f"unknown memory_id: {memory_id}"}

        # Snapshot trust values
        snapshot: Dict[str, float] = {
            nid: data.get("trust", 0.5)
            for nid, data in self.graph.graph.nodes(data=True)
        }

        loss = abs(proposed_delta)
        cascade = self._cascade_correction(
            memory_id, loss,
        )

        # Restore (the cascade already mutated; revert)
        for nid, t in snapshot.items():
            if nid in self.graph.graph:
                self.graph.graph.nodes[nid]["trust"] = t
        # Also strip the cascade trust history entries we just created
        for nid in list(self._trust_history.keys()):
            self._trust_history[nid] = [
                e for e in self._trust_history[nid] if e.get("source") != "cascade"
                or e.get("ts", 0) < (time.time() - 0.5)
            ]

        return {
            "preview": True,
            "source": memory_id,
            "proposed_delta": proposed_delta,
            **cascade,
        }

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def lineage(
        self,
        memory_id: str,
        hops: int = 3,
    ) -> Dict[str, Any]:
        """Walk SUPPORTS / DERIVED_FROM edges back to source memories."""
        node = self.graph.get_memory(memory_id)
        if node is None:
            return {"error": f"unknown memory_id: {memory_id}"}

        from collections import deque
        visited = {memory_id}
        ancestors: List[Dict[str, Any]] = []
        queue = deque([(memory_id, 0)])
        skip = {EdgeType.CONTRADICTS.value, "CONTRADICTS"}

        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for pred, _, edata in self.graph.graph.in_edges(current, data=True):
                etype = edata.get("edge_type", "supports")
                if etype in skip:
                    continue
                if pred in visited:
                    continue
                visited.add(pred)
                pred_node = self.graph.get_memory(pred)
                if pred_node is None:
                    continue
                ancestors.append({
                    "memory_id": pred,
                    "text": pred_node.text,
                    "trust": pred_node.trust,
                    "depth": depth + 1,
                    "edge_type": etype,
                    "weight": edata.get("weight", 0.5),
                    "from": current,
                })
                queue.append((pred, depth + 1))

        return {
            "memory_id": memory_id,
            "text": node.text,
            "trust": node.trust,
            "ancestors": ancestors,
            "ancestor_count": len(ancestors),
        }

    # ------------------------------------------------------------------
    # Contradictions and resolution
    # ------------------------------------------------------------------

    def list_contradictions(
        self,
        disposition: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List CONTRADICTS edges, optionally filtered by disposition."""
        results = []
        seen = set()
        for u, v, data in self.graph.graph.edges(data=True):
            if data.get("edge_type") != EdgeType.CONTRADICTS.value:
                continue
            disp = data.get("disposition", "unknown")
            if disposition and disp != disposition:
                continue
            pair = tuple(sorted([u, v]))
            if pair in seen:
                continue
            seen.add(pair)
            a = self.graph.get_memory(u)
            b = self.graph.get_memory(v)
            if a is None or b is None:
                continue
            results.append({
                "memory_a": {
                    "memory_id": a.memory_id, "text": a.text, "trust": a.trust,
                    "belnap_state": getattr(a, "belnap_state", "T"),
                },
                "memory_b": {
                    "memory_id": b.memory_id, "text": b.text, "trust": b.trust,
                    "belnap_state": getattr(b, "belnap_state", "T"),
                },
                "disposition": disp,
                "tension_score": data.get("nli_score", 0.0),
                "detected_at": data.get("detected_at", 0.0),
            })
            if len(results) >= limit:
                break
        return results

    def resolve_contradiction(
        self,
        memory_id_a: str,
        memory_id_b: str,
        keep: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Resolve a contradiction by deprecating one side or marking HELD.

        keep:
            "a"     - deprecate b, supersede with a
            "b"     - deprecate a, supersede with b
            "both"  - mark both Belnap=BOTH (held)
            "drop"  - drop both
        """
        a = self.graph.get_memory(memory_id_a)
        b = self.graph.get_memory(memory_id_b)
        if a is None or b is None:
            return {"error": "unknown memory_id"}

        if keep == "a":
            self.graph.deprecate(memory_id_b, memory_id_a, reason=reason)
            self._record_trust(memory_id_b, 0.0, b.trust, "resolution",
                               f"superseded by {memory_id_a}: {reason}")
            outcome = "deprecated_b"
        elif keep == "b":
            self.graph.deprecate(memory_id_a, memory_id_b, reason=reason)
            self._record_trust(memory_id_a, 0.0, a.trust, "resolution",
                               f"superseded by {memory_id_b}: {reason}")
            outcome = "deprecated_a"
        elif keep == "both":
            self.graph.update_belnap(memory_id_a, BelnapState.BOTH)
            self.graph.update_belnap(memory_id_b, BelnapState.BOTH)
            outcome = "held_both"
        elif keep == "drop":
            self.graph.update_belnap(memory_id_a, BelnapState.FALSE)
            self.graph.update_belnap(memory_id_b, BelnapState.FALSE)
            outcome = "dropped_both"
        else:
            return {"error": f"unknown keep value: {keep}"}

        self.save()
        return {
            "outcome": outcome,
            "memory_a": memory_id_a,
            "memory_b": memory_id_b,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Belief history
    # ------------------------------------------------------------------

    def belief_history(self, memory_id: str) -> Dict[str, Any]:
        node = self.graph.get_memory(memory_id)
        if node is None:
            return {"error": f"unknown memory_id: {memory_id}"}
        history = self._trust_history.get(memory_id, [])
        return {
            "memory_id": memory_id,
            "text": node.text,
            "current_trust": node.trust,
            "history": history,
            "n_changes": len(history),
        }

    # ------------------------------------------------------------------
    # Memory detail
    # ------------------------------------------------------------------

    def memory_detail(self, memory_id: str) -> Dict[str, Any]:
        node = self.graph.get_memory(memory_id)
        if node is None:
            return {"error": f"unknown memory_id: {memory_id}"}

        contradictions = []
        for _, target, data in self.graph.graph.edges(memory_id, data=True):
            if data.get("edge_type") == EdgeType.CONTRADICTS.value:
                contradictions.append({
                    "with": target,
                    "disposition": data.get("disposition"),
                    "tension_score": data.get("nli_score"),
                })

        in_edges = []
        for pred, _, data in self.graph.graph.in_edges(memory_id, data=True):
            in_edges.append({
                "from": pred,
                "edge_type": data.get("edge_type"),
                "weight": data.get("weight"),
            })

        out_edges = []
        for _, target, data in self.graph.graph.edges(memory_id, data=True):
            if data.get("edge_type") == EdgeType.CONTRADICTS.value:
                continue
            out_edges.append({
                "to": target,
                "edge_type": data.get("edge_type"),
                "weight": data.get("weight"),
            })

        return {
            "memory_id": memory_id,
            "text": node.text,
            "trust": node.trust,
            "memory_type": node.memory_type,
            "belnap_state": node.belnap_state,
            "created_at": node.created_at,
            "tags": node.tags,
            "contradictions": contradictions,
            "in_edges": in_edges,
            "out_edges": out_edges,
            "trust_history_length": len(self._trust_history.get(memory_id, [])),
        }

    # ------------------------------------------------------------------
    # Session diff
    # ------------------------------------------------------------------

    def session_diff(self, since: float) -> Dict[str, Any]:
        """What changed in the substrate since the given timestamp."""
        new_memories = []
        recent_corrections = []
        new_contradictions = []

        for node in self.graph.all_memories():
            if node.created_at >= since:
                new_memories.append({
                    "memory_id": node.memory_id,
                    "text": node.text[:200],
                    "trust": node.trust,
                    "created_at": node.created_at,
                })

        for memory_id, history in self._trust_history.items():
            recent = [e for e in history if e.get("ts", 0) >= since
                      and e.get("source") in ("user", "resolution", "cascade")]
            if recent:
                recent_corrections.append({
                    "memory_id": memory_id,
                    "changes": recent,
                })

        for u, v, data in self.graph.graph.edges(data=True):
            if (data.get("edge_type") == EdgeType.CONTRADICTS.value
                    and data.get("detected_at", 0) >= since):
                new_contradictions.append({
                    "between": [u, v],
                    "disposition": data.get("disposition"),
                    "tension_score": data.get("nli_score"),
                    "detected_at": data.get("detected_at"),
                })

        return {
            "since": since,
            "now": time.time(),
            "new_memories": new_memories,
            "recent_corrections": recent_corrections,
            "new_contradictions": new_contradictions,
            "summary": {
                "memories_added": len(new_memories),
                "trust_changes": sum(len(c["changes"]) for c in recent_corrections),
                "contradictions_added": len(new_contradictions),
            },
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Cheap dashboard snapshot. Must NOT trigger encoder load.

        Three encoder fields:
            embeddings_available  encoder is wired up (vs disabled)
            embeddings_loaded     model is warm in memory
            embeddings_warming    background warmup is in flight

        None of these force a load — that's reserved for the synchronous
        `available` property which no MCP tool should ever access.
        """
        s = self.graph.stats()
        encoder_configured = (
            self._encoder is not None
            and not getattr(self._encoder, "is_unavailable", False)
        )
        encoder_loaded = (
            self._encoder is not None
            and getattr(self._encoder, "is_loaded", False)
        )
        encoder_warming = (
            self._encoder is not None
            and getattr(self._encoder, "is_warming", False)
        )
        return {
            "memory_count": s.get("nodes", 0),
            "edge_count": s.get("edges", 0),
            "state_path": self.state_path,
            "belnap_states": s.get("belnap_states", {}),
            "edge_types": s.get("edge_types", {}),
            "held_contradictions": s.get("held_contradictions", 0),
            "evolving_contradictions": s.get("evolving_contradictions", 0),
            "embeddings_available": encoder_configured,
            "embeddings_loaded": encoder_loaded,
            "embeddings_warming": encoder_warming,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_source(tags: List[str]) -> str:
    for t in tags:
        if t.startswith("source:"):
            return t.split(":", 1)[1]
    return "unknown"
