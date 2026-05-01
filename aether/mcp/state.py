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

import functools
import itertools
import json
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from aether._lazy_encoder import (
    LazyEncoder,
    DEFAULT_EMBED_MODEL,
    _MODEL_CACHE,
    _get_cache_lock,
)
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

# DEFAULT_EMBED_MODEL, _MODEL_CACHE, _get_cache_lock are now imported from
# aether._lazy_encoder so the MCP state store, governance immune agents, and
# any other caller share one process-wide encoder cache.

# Substrate-grounded fidelity threshold:
# When the caller passes the sentinel default (0.5), we compute a real
# belief_confidence by searching the substrate. The grounding score is the
# trust-weighted average of the top-K most relevant memories.
GROUNDING_TOP_K = 5
GROUNDING_MIN_SCORE = 0.15  # Below this similarity, the memory isn't relevant
# v0.9.5: substring scores are typically lower than embedding scores (the
# combined score in embedding mode = 0.7*cosine + 0.3*substring; in cold
# mode it's just substring). Use a lower floor when running cold so
# clearly-related memories at substring score 0.10-0.15 still surface.
GROUNDING_MIN_SCORE_SUBSTRING = 0.10
SENTINEL_BELIEF_CONF = 0.5  # Treat 0.5 as "caller didn't supply real value"

# F#10: trust as a permanent term in search ranking.
# Without this, cosine alone ranks demoted entries (e.g. corrected-down
# user.favorite_color: red trust=0.67) above current truths (trust=0.95)
# whenever the demoted text happens to be shorter / less annotated and
# thus closer to the bare query embedding. Multiplicative weight with a
# floor: trust=0 still surfaces as a *0.3 penalty (relevant low-trust
# matches don't disappear), trust=1 leaves the score unchanged.
SEARCH_TRUST_WEIGHT = 0.7

# Auto-contradiction-detection threshold on write:
# Run the StructuralTensionMeter against this many top-similar existing
# memories. Anything above the threshold gets a CONTRADICTS edge.
TENSION_TOP_K = 8
TENSION_CONFLICT_THRESHOLD = 0.55

# Auto-link threshold for RELATED_TO edges (v0.9.1 fix, refined v0.9.5).
# In the same top-K candidate scan that catches contradictions, any
# candidate above this similarity that did NOT trigger a CONTRADICTS
# edge gets a bidirectional RELATED_TO edge. Without this, the MCP
# write surface produces orphan nodes and aether_path returns only
# the target — that's the bug v0.9.0 shipped.
#
# v0.9.5: ADAPTIVE THRESHOLD. The original 0.7 was tuned for embedding
# cosine similarity (sentence-transformers all-MiniLM-L6-v2). When the
# encoder is cold (first ~30s after MCP server start), `link_sim` is
# Jaccard token-overlap instead — and Jaccard rarely hits 0.7 even on
# clearly-related text. The agent's v0.9.4 re-run caught this:
# `aether_remember` writes produced orphan nodes in production despite
# v0.9.1's auto-link existing, because the threshold was unreachable
# in substring mode. Use a separate, lower threshold when the link_sim
# came from Jaccard rather than embedding cosine.
AUTO_LINK_THRESHOLD = float(os.environ.get("AETHER_AUTO_LINK_THRESHOLD", "0.7"))
AUTO_LINK_THRESHOLD_SUBSTRING = float(
    os.environ.get("AETHER_AUTO_LINK_THRESHOLD_SUBSTRING", "0.4")
)


def _link_threshold(mode: str) -> float:
    """Adaptive auto-link threshold by similarity mode.

    Both `add_memory` (write-time) and `backfill_edges` (retroactive)
    pick the same threshold based on whether the similarity score
    came from embedding cosine or Jaccard token-overlap. The two
    sites used to duplicate this decision; this helper pins the
    formula in one place.
    """
    return AUTO_LINK_THRESHOLD if mode == "embedding" else AUTO_LINK_THRESHOLD_SUBSTRING

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
    # v0.12.2: real CLI forms. The original list missed `--force`,
    # `-f origin`, etc. which are how force-push actually shows up in
    # commands. F#4 e2e finding: a high-trust "never force-push" belief
    # didn't catch `git push --force origin main` because the cue match
    # required substrings like "force push" (with space).
    "--force", " -f ", "-f origin", "-f main", "-f master",
    "push --force", "push -f",
)
POLICY_CONTRA_MIN_SIMILARITY = 0.45
POLICY_CONTRA_MIN_TRUST = 0.7
# v0.12.2: when belief trust is very high AND cues align on both sides,
# the sim-gate is bypassed. The gate exists to suppress false positives
# on co-topical memories; a high-trust prohibition with a clear
# imperative match is strong enough on its own. Closes the cold-encoder
# blind spot where Jaccard token overlap between a belief and a CLI
# action falls below 0.45 even when the conflict is unambiguous.
POLICY_CONTRA_STRONG_TRUST = 0.85


# Methodological overclaim detection (v0.9.3, Layer 2).
#
# Fidelity catches FACTUAL contradictions ("Seattle vs Portland") via the
# StructuralTensionMeter. It misses METHODOLOGICAL overclaims —
# "v3 was worse than v1, so CALIC is bad" when memory says
# "this conclusion is unsupported because v1-vs-v3 confounded predictor
# and strategy map." The slot extractions don't conflict; the meter sees
# them as unrelated.
#
# Layer 2 fix: a separate channel that fires when the draft makes an
# inference AND a topically-similar memory contains methodological-warning
# language (or carries the `source:methodological_gap` tag). Surfaces in
# compute_grounding's output as `methodological_concerns`. Also reduces
# belief_confidence so gap_score / severity reflect the concern.

INFERENCE_MARKERS = (
    " so ", " therefore ", " thus ", " hence ",
    " means that ", " proves ", " shows that ",
    " implies ", " demonstrates that ",
    " concludes ", " conclusion is ",
    " because ", " since ",
    "=>",  # informal logical connective sometimes used in code comments
)

METHODOLOGICAL_GAP_SIGNALS = (
    "unsupported", "not supported",
    "doesn't follow", "does not follow",
    "missing cell", "confounded", "confounding",
    "non-causal", "non causal",
    "lazy reading", "lazy read",
    "incorrect inference", "this conclusion is",
    "the conclusion that", "methodological gap",
    "methodological concern", "two variables changed",
    "experimental matrix", "design flaw",
)

METHODOLOGICAL_GAP_SOURCE = "methodological_gap"


def _has_inference_marker(text: str) -> bool:
    """True if the text makes an inference (claims X entails Y).

    Conservative: leading whitespace required to avoid matching substrings
    inside larger words (e.g. "boson" should not match " so ").
    """
    padded = " " + text.lower() + " "
    return any(cue in padded for cue in INFERENCE_MARKERS)


def _has_methodological_signal(text: str, source: Optional[str] = None) -> bool:
    """True if the memory carries methodological-warning language.

    Two pathways:
      1. `source == "methodological_gap"` — explicit tag set at write time.
      2. Memory text contains any of METHODOLOGICAL_GAP_SIGNALS.
    """
    if source and METHODOLOGICAL_GAP_SOURCE in source.lower():
        return True
    t = text.lower()
    return any(sig in t for sig in METHODOLOGICAL_GAP_SIGNALS)


def _looks_like_prohibition(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in PROHIBITION_CUES)


def _looks_like_imperative(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in IMPERATIVE_CUES)


def _sync_first(method: Callable) -> Callable:
    """Decorator: call `_sync_from_disk_if_stale()` before the wrapped
    StateStore method runs.

    F#7 fix. The MCP server holds graph state in memory; the Stop hook
    writes directly to the same JSON file. Without this sync, server
    writes can clobber hook-written memories. Decorating each public
    StateStore method that backs an MCP tool ensures every tool call
    starts from disk-fresh state.

    Cheap when nothing has changed externally (single os.stat); pays
    a json.load cost only when the hook has written something new.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._sync_from_disk_if_stale()
        return method(self, *args, **kwargs)
    return wrapper


def _is_policy_contradiction(
    mem_text: str,
    new_text: str,
    sim: float,
    mem_trust: float,
) -> bool:
    """Catches command-vs-prohibition contradictions.

    The structural tension meter misses these because there are no
    shared slots between an imperative and a policy statement, so we
    add this lightweight cue cross-check on top.

    Two ways to fire (both require trust >= POLICY_CONTRA_MIN_TRUST and
    one side prohibits while the other commands):

    1. Standard:  sim  >= POLICY_CONTRA_MIN_SIMILARITY (0.45)
                  trust >= POLICY_CONTRA_MIN_TRUST (0.7)
    2. Strong-trust: trust >= POLICY_CONTRA_STRONG_TRUST (0.85)
                  --- sim gate bypassed ---

    The strong-trust path closes F#4: cold-encoder Jaccard overlap
    between a high-trust prohibition belief and a real CLI action falls
    below 0.45, but the cue alignment is strong enough on its own when
    the user has explicitly stored a high-trust policy.

    The two call sites (write-time tension scan + read-time grounding)
    used to inline this logic; pulling it into one helper means a future
    threshold tweak hits both paths together.
    """
    if mem_trust < POLICY_CONTRA_MIN_TRUST:
        return False

    mem_prohibits = _looks_like_prohibition(mem_text)
    mem_imperative = _looks_like_imperative(mem_text)
    new_prohibits = _looks_like_prohibition(new_text)
    new_imperative = _looks_like_imperative(new_text)

    # Two prohibitions don't contradict each other — they're co-policies.
    # A well-written prohibition that quotes the forbidden command (e.g.
    # "Never force-push to main", "Never bypass commit hooks with --no-verify")
    # will trip IMPERATIVE_CUES, but quoting a command isn't issuing one.
    # Without this guard the asymmetric cue check below fires on
    # prohibition-vs-prohibition pairs and seeds fake HELD contradictions.
    if mem_prohibits and new_prohibits:
        return False

    cues_align = (
        (mem_prohibits and new_imperative and not new_prohibits)
        or (mem_imperative and new_prohibits and not new_imperative)
    )
    if not cues_align:
        return False

    if sim >= POLICY_CONTRA_MIN_SIMILARITY:
        return True
    if mem_trust >= POLICY_CONTRA_STRONG_TRUST:
        return True
    return False


# Generic negation cues for asymmetric-negation contradiction detection.
# When two highly-similar texts differ in negation polarity, that's a
# strong contradiction signal even when no slot conflict is present.
#
# v0.12.21: dropped bare " un" — it was matching " until", " unique",
# " unrelated", " under", etc., none of which are negations. The 2026-05-01
# DECISIONS.md bench showed " un" (matching " until") was the dominant
# false-positive driver in `_is_asymmetric_negation_contradict`. Replaced
# with explicit un-prefix negation words.
_NEGATION_CUES = (
    " not ", "n't ", " never ", " no ", " none ",
    " neither ", " nor ", " without ",
    "not.", "not,", "n't.", "n't,",
    # Explicit un-prefix negations (replacing the over-broad bare " un").
    " unable", " unaware", " uncertain", " unwilling",
    " unsure", " unclear", " unconvinced", " unlikely",
)


def _has_negation(text: str) -> bool:
    t = " " + text.lower() + " "
    return any(cue in t for cue in _NEGATION_CUES)


# v0.12.21: selection-rejection cues. When BOTH sides of a pair contain
# language indicating "this option was deferred / not chosen / set aside,"
# they are *co-policies* describing complementary parts of one decision —
# not contradicting facts. ("CRT did not pick Pinecone" + "CRT did not pick
# FastAPI" are both real, both true, both about non-chosen alternatives.)
# This is the same architectural shape as the v0.12.19 prohibition
# co-policy guard in `_is_policy_contradiction`: matched-polarity pairs
# don't contradict each other.
_SELECTION_REJECTION_CUES = (
    "did not pick", "did not choose", "did not select",
    "not chosen", "not selected",
    # Verb-form variants — "rejected" / "rejects" / "rejecting" all signal
    # the same selection-rejection class.
    "rejected", "rejects", "rejecting",
    "deferred", "defers", "deferring",
    "disabled", "disables", "disabling",
    "set aside", "passed on", "passes on", "passing on",
)


def _expresses_selection_rejection(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in _SELECTION_REJECTION_CUES)


# v0.12.21: stop-word set for content-token overlap. Used by the
# polarity-flip guard in compute_grounding to require domain alignment
# before flipping a "compatible" classification to a contradiction.
# Includes function words and the negation cues themselves (we want to
# compare *content*, not co-presence of prohibition language).
_CONTENT_OVERLAP_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "at", "by", "for", "with", "from", "as", "and", "or",
    "but", "this", "that", "these", "those", "it", "its",
    "i", "we", "you", "they", "their", "his", "her",
    "do", "does", "did", "don", "doesn",
    "never", "no", "not", "never.", "no.", "not.",
})


def _content_token_overlap(text_a: str, text_b: str) -> float:
    """Fraction of text_a's content tokens that appear in text_b.

    Used to gate the polarity-flip guard in compute_grounding: the
    guard should only fire when the action and the prohibition belief
    share enough domain content. "git status" vs "Never use git push
    --force" overlap only on `git` (1/8 ≈ 0.13) — too thin. "delete
    X without backing up" vs "Never delete prod data without backup"
    overlaps on `delete`/`without`/`backup` substrings — strong enough.
    """
    import re
    def tokens(s):
        return {
            t for t in re.findall(r"\b[a-z][a-z0-9_-]+\b", s.lower())
            if t not in _CONTENT_OVERLAP_STOPWORDS and len(t) >= 3
        }
    a, b = tokens(text_a), tokens(text_b)
    if not a:
        return 0.0
    # Stem-tolerant overlap: a token in `a` matches `b` if any token in
    # `b` shares a 4+ char prefix. Keeps "backup" / "backing" aligned,
    # "tables" / "table" aligned, without pulling in heavy stemming deps.
    matched = 0
    for ta in a:
        for tb in b:
            if ta == tb or (len(ta) >= 4 and len(tb) >= 4 and (
                ta[:4] == tb[:4] and (ta.startswith(tb) or tb.startswith(ta))
            )):
                matched += 1
                break
    return matched / len(a)


# v0.12.21: asymm_neg gets its own (higher) similarity threshold.
# At the policy threshold (0.45) co-topical-but-different-fact pairs slip
# through because their negation polarity happens to differ. Real
# asymmetric negation ("We use pnpm" vs "We use npm not pnpm") has
# similarity > 0.85 because the texts are near-paraphrases. The 2026-05-01
# DECISIONS.md bench showed that cross-domain pairs ("uses Flask" vs
# "did not pick Pinecone") fire below 0.75 — so we draw the line there.
ASYMM_NEG_MIN_SIMILARITY = 0.75


def _is_asymmetric_negation_contradict(
    text_a: str,
    text_b: str,
    similarity: float,
    min_similarity: float = ASYMM_NEG_MIN_SIMILARITY,
) -> bool:
    """Two highly-similar texts where exactly one contains negation cues.

    Catches cases the StructuralTensionMeter misses:
        "We use pnpm not npm" vs "We use npm"
        "Main is protected, never force push" vs "Force push to main"

    v0.12.21: returns False when both sides express selection-rejection
    (deferred/rejected/disabled/not chosen). Two co-policies about
    different non-chosen alternatives are not in conflict — they are
    parts of the same compound decision. The 2026-05-01 reflexive bench
    on a real DECISIONS.md showed this case dominated the false positives.

    v0.12.21: similarity threshold raised to 0.75 (was 0.45 inherited
    from POLICY_CONTRA_MIN_SIMILARITY). Real asymmetric negation requires
    near-paraphrase similarity; co-topical-but-different-fact pairs were
    firing below 0.75 in the bench.

    Still not perfect (won't catch double-negation, semantic flips like
    'increased' vs 'decreased', or the "uses A" vs "did not pick B" case
    where A and B are co-options in a closed set — that needs slot-key
    canonicalization, scheduled for v0.13).
    """
    if similarity < min_similarity:
        return False
    a_neg = _has_negation(text_a)
    b_neg = _has_negation(text_b)
    if a_neg == b_neg:
        return False
    # Co-rejection guard: if both sides describe a non-chosen / deferred
    # alternative, they are co-policies, not contradictions.
    if _expresses_selection_rejection(text_a) and _expresses_selection_rejection(text_b):
        return False
    return True


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


def _receipts_path(state_path: str) -> str:
    """Side-car file: action receipts (v0.10).

    Closes the governance loop: aether_sanction is the gate, receipts are
    the audit trail of what actually executed and what its outcome was.
    """
    return state_path.replace(".json", "_receipts.json")


# ---------------------------------------------------------------------------
# Action receipts (v0.10) — ported from personal_agent/action_receipts.py
# ---------------------------------------------------------------------------

@dataclass
class ActionReceipt:
    """A single action's audit record.

    Created at sanction time with the verdict, then updated with the
    actual outcome via aether_receipt once the caller has executed
    (or skipped) the action. Persisted to a side-car JSON file at
    `<state_path>_receipts.json`.

    Fields ported from main repo's ActionReceipt; personal_agent-specific
    fields (thread_id, agent_name, orchestration_id, run_step_id,
    expectation_keywords) are dropped because they don't apply to OSS
    single-substrate use. model_attribution is kept because it's useful
    for the cross-vendor validation work coming up.
    """

    receipt_id: str                          # uuid4 string
    timestamp: float                         # creation time (set at sanction)
    action: str                              # human description of the proposed action
    sanction_verdict: str                    # APPROVE / HOLD / REJECT
    tool_name: Optional[str] = None          # "shell", "file_write", "git", etc.
    target: Optional[str] = None             # path / URL / command / memory_id
    result: Optional[str] = None             # success / error / partial / skipped
    reversible: Optional[bool] = None        # can the caller undo it?
    reverse_action: Optional[str] = None     # how to undo (if reversible)
    details: Dict[str, Any] = field(default_factory=dict)
    verification_passed: Optional[bool] = None
    verification_reason: Optional[str] = None
    model_attribution: Optional[str] = None  # which LLM produced the action
    completed_at: Optional[float] = None     # set when aether_receipt is called
    sanction_memory_ids: List[str] = field(default_factory=list)
    # ^ supporting + contradicting memory_ids that informed the sanction


# ---------------------------------------------------------------------------
# Lightweight encoder wrapper
# ---------------------------------------------------------------------------

# Backward-compatible alias. The real implementation lives in
# aether._lazy_encoder so the MCP state store, governance immune agents,
# and any other caller can share the same process-wide cache.
_LazyEncoder = LazyEncoder


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
        # v0.10: action receipts (the audit trail half of the governance loop)
        self._receipts: Dict[str, Dict[str, Any]] = {}
        self._load_receipts()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the graph + trust history + receipts to disk."""
        self.graph.save(self.state_path)
        self._save_trust_history()
        self._save_receipts()

    def _sync_from_disk_if_stale(self) -> None:
        """Reload graph from disk if external writers (e.g. the Stop
        hook) have modified the state file since our last load/save.

        F#7 fix. The MCP server holds graph state in memory and saves
        on its own writes; the Stop hook writes directly to the same
        file. Without this sync, the server's saves silently clobber
        hook-written memories.

        Wrapped via the `@_sync_first` decorator on every public
        StateStore method that backs an MCP tool. Cheap when nothing
        has changed (single os.stat), copies the file's contents into
        memory when it has.

        Trust history and receipts are also reloaded so the full state
        view stays consistent.
        """
        if not self.graph.is_stale_on_disk():
            return
        self.graph.load(self.state_path)
        # Both loaders mutate self in place. Calling them keeps the
        # auxiliary stores consistent with the freshly-loaded graph.
        self._load_trust_history()
        self._load_receipts()

    def _save_receipts(self) -> None:
        path = _receipts_path(self.state_path)
        try:
            with open(path, "w") as f:
                json.dump(self._receipts, f, indent=2)
        except OSError:
            pass

    def _load_receipts(self) -> None:
        path = _receipts_path(self.state_path)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                self._receipts = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._receipts = {}

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

    @_sync_first
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

        v0.12: when `slots` is None, automatically run extract_fact_slots
        on the text to populate slot tags. Without this, memories written
        through normal channels (no explicit `slots=` parameter) carry no
        slot tags and the slot-equality detector has nothing to compare.
        Production substrate organically discovered 7 user-facing slots
        through this path; OSS now mirrors that.
        """
        memory_id = f"m{int(time.time() * 1000)}_{next(self._id_counter)}"
        tags: list[str] = [f"source:{source}"]

        # v0.12: auto-extract slots from text when none provided
        auto_extracted: Dict[str, str] = {}
        if slots is None:
            try:
                from aether.memory import extract_fact_slots
                facts = extract_fact_slots(text)
                auto_extracted = {k: v.normalized for k, v in facts.items()}
            except Exception:
                pass

        effective_slots = slots if slots is not None else auto_extracted
        if effective_slots:
            tags.extend(f"slot:{k}={v}" for k, v in effective_slots.items())

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
            findings = self._detect_and_record_tensions(
                memory_id, text, trust, source, tags=tags,
            )

        self.save()
        return {
            "memory_id": memory_id,
            "trust": trust,
            "source": source,
            "extracted_slots": effective_slots,
            "tension_findings": findings,
        }

    def _detect_and_record_tensions(
        self,
        memory_id: str,
        text: str,
        trust: float,
        source: str,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run tension meter against top-K most similar existing memories.

        v0.12: `tags` is the new memory's tag list (including any
        slot:KEY=VALUE entries from slot extraction). Used by
        slot-equality contradiction detection.
        """
        new_emb = self.graph.get_embedding(memory_id)
        candidates: List[Tuple[float, MemoryNode]] = []

        # v0.12: extract slot keys from the new memory's tags so we can
        # pre-screen for slot-shared candidates regardless of textual
        # similarity. Production data has cases like "remember my name
        # is Nick" vs "your name is Aether" — only ~0.14 Jaccard, below
        # the 0.2 gate, but the slot:user.name key matches and the
        # values disagree. Without this pre-screen the slot-equality
        # detector never sees the pair.
        new_slot_keys: set[str] = set()
        for t in (tags or []):
            if t.startswith("slot:") and "=" in t[len("slot:"):]:
                new_slot_keys.add(t[len("slot:"):].split("=", 1)[0])

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

        # v0.12: augment top-K with any candidates that share a slot key
        # with the new memory but didn't make the textual top-K. Bounded
        # so we never blow past 2x TENSION_TOP_K total candidates.
        if new_slot_keys:
            seen = {o.memory_id for _, o in top}
            slot_extras: List[Tuple[float, MemoryNode]] = []
            for sim, other in candidates:
                if other.memory_id in seen:
                    continue
                other_slot_keys = set()
                for t in (other.tags or []):
                    if t.startswith("slot:") and "=" in t[len("slot:"):]:
                        other_slot_keys.add(t[len("slot:"):].split("=", 1)[0])
                if new_slot_keys & other_slot_keys:
                    slot_extras.append((sim, other))
                    seen.add(other.memory_id)
                    if len(slot_extras) >= TENSION_TOP_K:
                        break
            top = list(top) + slot_extras

        findings: List[Dict[str, Any]] = []
        for sim, other in top:
            # v0.12: if the candidate shares a slot key with the new
            # memory, skip the sim gate — slot-equality is a categorical
            # signal independent of textual similarity.
            shares_slot = False
            if new_slot_keys:
                for t in (other.tags or []):
                    if (t.startswith("slot:")
                            and "=" in t[len("slot:"):]
                            and t[len("slot:"):].split("=", 1)[0] in new_slot_keys):
                        shares_slot = True
                        break
            if sim < 0.2 and not shares_slot:  # Don't bother — clearly unrelated
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
            policy = _is_policy_contradiction(
                mem_text=other.text,
                new_text=text,
                sim=sim_for_check,
                mem_trust=other.trust,
            )
            mutex_hit = detect_mutex_conflict(other.text, text)

            # v0.11: shape contradiction (typed-pattern conflicts).
            # v0.12: shape now uses local-context gating to suppress
            # false positives on co-topical memories with unrelated
            # numerics.
            from aether.patterns import shape as _shape_match
            from aether.patterns import slot_equality as _slot_equality_match
            shape_result = _shape_match(other.text, text)
            shape_hit = (
                shape_result.score >= 1.0
                and shape_result.evidence.get("conflicts")
            )

            # v0.12: slot-equality contradiction. Closes Lab A v2 finding —
            # production has 42 real categorical contradictions on slots
            # (user.name, user.favorite_color, user.location, etc.) that
            # the previous detection layer was 0/42 on.
            slot_eq_result = _slot_equality_match(
                other.tags or [],
                tags or [],
            )
            slot_eq_hit = (
                slot_eq_result.score >= 1.0
                and slot_eq_result.evidence.get("conflicts")
            )

            if not (slot_conflict or asymm_neg or policy or mutex_hit
                    or shape_hit or slot_eq_hit):
                # No contradiction. v0.9.1: if this candidate is similar
                # enough, wire a RELATED_TO edge so aether_path has
                # something to walk. Bidirectional because RELATED_TO is
                # semantically symmetric (Dijkstra walks in_edges).
                #
                # v0.9.5: ADAPTIVE THRESHOLD. When the encoder is cold,
                # `sim` is Jaccard token-overlap (typically 0.0-0.5) and
                # the meter populates embedding_similarity=0.0 (not None,
                # so `is not None` checks don't distinguish modes). Use
                # the encoder's is_loaded state to pick the right mode.
                encoder_ready = (
                    self._encoder is not None and self._encoder.is_loaded
                )
                if encoder_ready:
                    link_sim = result.supporting_signals.get(
                        "embedding_similarity", sim,
                    )
                    mode = "embedding"
                else:
                    link_sim = sim  # Jaccard token overlap fallback
                    mode = "substring"
                threshold = _link_threshold(mode)
                if link_sim >= threshold:
                    metadata = {
                        "similarity": float(round(link_sim, 4)),
                        "auto": True,
                        "mode": mode,
                    }
                    self.graph.add_edge(
                        other.memory_id, memory_id,
                        EdgeType.RELATED_TO, metadata=metadata,
                    )
                    self.graph.add_edge(
                        memory_id, other.memory_id,
                        EdgeType.RELATED_TO, metadata=metadata,
                    )
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
            if shape_hit:
                # v0.11: typed-shape conflict (numeric/version/date).
                # Closes v0.9.4 known_gap_quantitative cases.
                conflict_summary = ", ".join(
                    f"{c['shape']}:{c['a']}<>{c['b']}"
                    for c in shape_result.evidence.get("conflicts", [])[:2]
                )
                trace.append(f"shape:{conflict_summary}")
                if kind == "structural":
                    kind = "quantitative"
                # Shape conflict is decisive — bump tension score.
                if not slot_conflict:
                    disposition = Disposition.RESOLVABLE
            if slot_eq_hit:
                # v0.12: slot-equality conflict. Categorical conflict on
                # a known slot (e.g. user.name=Nick vs user.name=Aether).
                slot_summary = ", ".join(
                    f"{c['slot']}:{c['a']}<>{c['b']}"
                    for c in slot_eq_result.evidence.get("conflicts", [])[:2]
                )
                trace.append(f"slot_value_conflict:{slot_summary}")
                if kind == "structural":
                    kind = "slot_value_conflict"
                if not slot_conflict and not shape_hit:
                    disposition = Disposition.RESOLVABLE

            # v0.12: slot-equality contributes 0.9 to nli_score (categorical
            # conflict on a known slot is high-confidence — exactly the
            # 0.9 weight Lab A v2 recommended for the new detector).
            edge = ContradictionEdge(
                disposition=disposition.value,
                nli_score=max(
                    result.tension_score,
                    0.85 if mutex_hit else 0.0,
                    0.9 if slot_eq_hit else 0.0,
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
                    max(
                        result.tension_score,
                        0.85 if mutex_hit else 0.0,
                        0.9 if slot_eq_hit else 0.0,
                    ),
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
    # Explicit edge creation (v0.9.1)
    # ------------------------------------------------------------------

    _MANUAL_EDGE_TYPES = (
        EdgeType.SUPPORTS,
        EdgeType.DERIVED_FROM,
        EdgeType.RELATED_TO,
    )

    @_sync_first
    def add_link(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "supports",
        weight: float = 0.7,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Add a typed edge between two existing memories.

        edge_type is validated against EdgeType. Only SUPPORTS,
        DERIVED_FROM, and RELATED_TO are user-creatable here —
        CONTRADICTS is added by the contradiction detection path on
        write, and SUPERSEDES is added by aether_resolve.

        SUPPORTS / DERIVED_FROM are directional (source → target).
        RELATED_TO is symmetric and added in both directions so
        Dijkstra walks either way.
        """
        if source_id == target_id:
            raise ValueError("source_id and target_id must differ")
        try:
            et = EdgeType(edge_type.lower())
        except ValueError:
            valid = [e.value for e in self._MANUAL_EDGE_TYPES]
            raise ValueError(
                f"invalid edge_type {edge_type!r}; must be one of {valid}"
            )
        if et not in self._MANUAL_EDGE_TYPES:
            raise ValueError(
                f"{et.value!r} edges are managed automatically; use "
                f"aether_remember (contradicts) or aether_resolve (supersedes)"
            )
        if self.graph.get_memory(source_id) is None:
            raise KeyError(f"unknown source memory_id: {source_id}")
        if self.graph.get_memory(target_id) is None:
            raise KeyError(f"unknown target memory_id: {target_id}")

        metadata = {"weight": float(weight), "auto": False}
        if reason:
            metadata["reason"] = reason

        self.graph.add_edge(source_id, target_id, et, metadata=metadata)
        bidirectional = et == EdgeType.RELATED_TO
        if bidirectional:
            self.graph.add_edge(target_id, source_id, et, metadata=metadata)
        self.save()

        return {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": et.value,
            "weight": weight,
            "reason": reason,
            "bidirectional": bidirectional,
        }

    @_sync_first
    def backfill_edges(
        self,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retroactively wire RELATED_TO edges into pre-v0.9.1 substrates.

        For each pair of memories with similarity >= threshold AND no
        existing edge between them in either direction, add a
        bidirectional RELATED_TO edge. Pairs that already have any
        edge (CONTRADICTS, SUPPORTS, etc.) are skipped — backfill only
        wires orphans.

        Args:
            threshold: similarity threshold. When None (default), the
                threshold is mode-adaptive (v0.9.5): AUTO_LINK_THRESHOLD
                (0.7) for embedding cosine, AUTO_LINK_THRESHOLD_SUBSTRING
                (0.4) for Jaccard fallback. Pass an explicit value to
                override.
        """
        explicit_threshold = threshold
        memories = list(self.graph.all_memories())
        added = 0
        skipped_existing_edge = 0
        skipped_low_sim = 0
        compared = 0
        for i, a in enumerate(memories):
            a_emb = self.graph.get_embedding(a.memory_id)
            for b in memories[i + 1:]:
                compared += 1
                if (self.graph.graph.has_edge(a.memory_id, b.memory_id) or
                        self.graph.graph.has_edge(b.memory_id, a.memory_id)):
                    skipped_existing_edge += 1
                    continue
                b_emb = self.graph.get_embedding(b.memory_id)
                if a_emb is not None and b_emb is not None:
                    sim = self._cosine(a_emb, b_emb)
                    mode = "embedding"
                else:
                    ta = set(a.text.lower().split())
                    tb = set(b.text.lower().split())
                    sim = len(ta & tb) / max(len(ta | tb), 1)
                    mode = "substring"
                threshold = (
                    explicit_threshold
                    if explicit_threshold is not None
                    else _link_threshold(mode)
                )
                if sim < threshold:
                    skipped_low_sim += 1
                    continue
                metadata = {
                    "similarity": float(round(sim, 4)),
                    "auto": True,
                    # v0.10.1: was "source": "backfill" -- renamed to
                    # avoid collision with the JSON edge schema's endpoint
                    # key, which corrupted save/load round-trips and
                    # produced stub "backfill" nodes that crashed
                    # aether_path's graph walk.
                    "origin": "backfill",
                    "mode": mode,
                }
                self.graph.add_edge(
                    a.memory_id, b.memory_id,
                    EdgeType.RELATED_TO, metadata=metadata,
                )
                self.graph.add_edge(
                    b.memory_id, a.memory_id,
                    EdgeType.RELATED_TO, metadata=metadata,
                )
                added += 1
        if added:
            self.save()
        return {
            "added": added,
            "skipped_existing_edge": skipped_existing_edge,
            "skipped_low_sim": skipped_low_sim,
            "compared_pairs": compared,
            "threshold": threshold,
            "total_memories": len(memories),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @_sync_first
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

            # F#10: weight by trust so demoted entries don't outrank the
            # current truth when their cosine is similar. Multiplicative
            # with floor so low-trust relevant matches still surface.
            trust_factor = (
                1.0 - SEARCH_TRUST_WEIGHT
                + SEARCH_TRUST_WEIGHT * float(getattr(node, "trust", 0.0) or 0.0)
            )
            combined *= trust_factor

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
                    # v0.12: include tags so downstream slot-equality
                    # detection works on the read path
                    "tags": list(node.tags or []),
                },
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    # ------------------------------------------------------------------
    # Substrate-grounded belief_confidence
    # ------------------------------------------------------------------

    @_sync_first
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
                "methodological_concerns": [],
                "method": "empty",
            }

        method = "embedding" if results[0].get("similarity") is not None else "substring"

        # v0.9.5: adaptive minimum score. Substring scores are lower; use
        # GROUNDING_MIN_SCORE_SUBSTRING (0.10) in cold mode, GROUNDING_MIN_SCORE
        # (0.15) in embedding mode.
        min_score = (
            GROUNDING_MIN_SCORE
            if method == "embedding"
            else GROUNDING_MIN_SCORE_SUBSTRING
        )
        kept = [r for r in results if r["score"] >= min_score][:top_k]
        if not kept:
            return {
                "belief_confidence": 0.3,
                "support": [],
                "contradict": [],
                "methodological_concerns": [],
                "method": method,
            }

        # v0.9.3 (Layer 2): if the draft makes an inference, we'll check each
        # candidate memory for methodological-warning language. This is the
        # channel that catches "v3 was worse than v1, so CALIC is bad" against
        # a memory that says "the v1-vs-v3 conclusion is unsupported."
        draft_has_inference = _has_inference_marker(text)

        support: List[Dict[str, Any]] = []
        contradict: List[Dict[str, Any]] = []
        methodological: List[Dict[str, Any]] = []
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
            is_policy_contradict = _is_policy_contradiction(
                mem_text=hit["text"],
                new_text=text,
                sim=sim,
                mem_trust=hit["trust"],
            )

            # Asymmetric-negation: "We use pnpm not npm" vs "We use npm"
            is_asymm_neg = _is_asymmetric_negation_contradict(
                hit["text"], text, sim,
            ) and hit["trust"] >= POLICY_CONTRA_MIN_TRUST

            # Mutual-exclusion: "we use AWS" vs "we use GCP"
            mutex_hit = detect_mutex_conflict(hit["text"], text)

            # v0.11/v0.12: shape contradiction with local-context gate.
            from aether.patterns import shape as _shape_match
            from aether.patterns import slot_equality as _slot_equality_match
            _shape_grounding = _shape_match(hit["text"], text)
            shape_hit = (
                _shape_grounding.score >= 1.0
                and _shape_grounding.evidence.get("conflicts")
            )

            # v0.12: slot-equality conflict on the read side. The draft
            # text gets slot-extracted on the fly; if any extracted slot
            # disagrees with the candidate memory's slot tags, that's a
            # categorical contradiction the user should see.
            from aether.memory import extract_fact_slots
            _draft_slots = extract_fact_slots(text)
            _draft_tags = [
                f"slot:{k}={v.normalized}" for k, v in _draft_slots.items()
            ]
            _slot_eq_grounding = _slot_equality_match(
                hit.get("tags") or [], _draft_tags,
            )
            slot_eq_hit = (
                _slot_eq_grounding.score >= 1.0
                and _slot_eq_grounding.evidence.get("conflicts")
            )

            # v0.9.3 (Layer 2): methodological overclaim check. Independent
            # of slot-conflict detection. Fires when:
            #   1. The draft makes an inference (`so X`, `therefore Y`, etc.)
            #   2. The memory carries methodological-warning language OR is
            #      tagged source:methodological_gap.
            #   3. Topical similarity threshold passed (already filtered by
            #      GROUNDING_MIN_SCORE above).
            #
            # Checked BEFORE factual contradiction because a methodological
            # memory often also looks like a negation-asymmetry contradiction
            # (it contains words like "unsupported" / "doesn't follow"). The
            # methodological framing is more informative — it tells the user
            # WHY the inference is flawed, not just THAT a memory disagrees.
            # When both fire, methodological wins.
            is_methodological = (
                draft_has_inference and
                _has_methodological_signal(hit["text"], hit.get("source"))
            )

            if is_methodological:
                methodological.append({
                    **hit,
                    "kind": "methodological",
                    "concern": (
                        "Draft makes an inference; memory cautions that "
                        "this conclusion is unsupported / confounded / "
                        "premature."
                    ),
                    "tension_score": round(tr.tension_score, 3),
                })
            elif is_factual_contradict or is_policy_contradict or is_asymm_neg or mutex_hit or shape_hit or slot_eq_hit:
                kind = "factual"
                if is_asymm_neg and not is_factual_contradict:
                    kind = "negation_asymmetry"
                if is_policy_contradict and not is_factual_contradict:
                    kind = "policy"
                if mutex_hit and not is_factual_contradict:
                    kind = "mutex"
                if shape_hit and not is_factual_contradict:
                    kind = "quantitative"
                if slot_eq_hit and not is_factual_contradict:
                    kind = "slot_value_conflict"
                entry = {
                    **hit,
                    "tension_score": round(
                        max(
                            tr.tension_score,
                            0.85 if mutex_hit else 0.0,
                            0.85 if shape_hit else 0.0,
                            0.9 if slot_eq_hit else 0.0,
                        ),
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
                if slot_eq_hit:
                    entry["slot_conflicts"] = (
                        _slot_eq_grounding.evidence.get("conflicts", [])[:3]
                    )
                if shape_hit:
                    entry["shape_conflicts"] = (
                        _shape_grounding.evidence.get("conflicts", [])[:3]
                    )
                contradict.append(entry)
            elif tr.relationship in (
                TensionRelationship.DUPLICATE,
                TensionRelationship.REFINEMENT,
                TensionRelationship.COMPATIBLE,
            ):
                # v0.12.21: polarity-flip guard. The tension meter's
                # relationship classifier is polarity-blind on high-token-
                # overlap pairs. When the candidate is a high-trust
                # prohibition belief and the query lacks prohibition language
                # AND the action shares enough content with the prohibition
                # to be in the same domain, "compatible" is the wrong call —
                # the belief forbids the kind of action the query proposes.
                # Reclassify as contradict.
                #
                # Catches the 2026-05-01 sanction bench case:
                #   memory: "Never delete production data without verifying
                #            a recent backup." (trust 0.92)
                #   action: "delete secure_memory/faiss.index without backing
                #            it up"
                # The meter saw shared tokens (delete, without, backup) and
                # called them compatible. They are not.
                #
                # The 0.25 content-overlap floor avoids false-flipping cases
                # where the action only shares an incidental prefix word
                # with the prohibition (e.g. "git status" vs "Never use git
                # push --force" overlaps only on `git`, ratio 0.13).
                POLARITY_FLIP_OVERLAP_MIN = 0.25
                if (
                    hit["trust"] >= POLICY_CONTRA_MIN_TRUST
                    and _looks_like_prohibition(hit["text"])
                    and not _looks_like_prohibition(text)
                    and _content_token_overlap(hit["text"], text) >= POLARITY_FLIP_OVERLAP_MIN
                ):
                    contradict.append({
                        **hit,
                        "tension_score": round(tr.tension_score, 3),
                        "kind": "polarity_flip",
                    })
                    continue
                support.append({**hit, "tension_score": round(tr.tension_score, 3)})
                weights.append(hit["trust"])

        if not support and not contradict and not methodological:
            return {
                "belief_confidence": 0.4,
                "support": [],
                "contradict": [],
                "methodological_concerns": [],
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
        # v0.9.3: methodological concerns reduce belief_confidence the
        # same way factual contradictions do, so gap_score / severity
        # naturally reflect the concern. The signals are still surfaced
        # in their own channel for transparency.
        methodological_penalty = sum(
            min(0.4, m["trust"] * 0.7)
            for m in methodological
        )
        belief = max(0.0, min(1.0, base - contradiction_penalty - methodological_penalty))

        return {
            "belief_confidence": round(belief, 3),
            "support": support,
            "contradict": contradict,
            "methodological_concerns": methodological,
            "method": method,
        }

    # ------------------------------------------------------------------
    # Correction with cascade
    # ------------------------------------------------------------------

    @_sync_first
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

    @_sync_first
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

    @_sync_first
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
    # Shortest-path retrieval (Dijkstra over BDG)
    # ------------------------------------------------------------------

    @_sync_first
    def compute_path(
        self,
        query: str,
        max_tokens: int = 2000,
        max_hops: int = 8,
    ) -> Dict[str, Any]:
        """Dijkstra shortest-path retrieval — the "park map" idea.

        Find the dependency chain that grounds a query at the cheapest
        token cost. Walks the BDG backward from the top-1 cosine match
        through SUPPORTS / DERIVED_FROM / RELATED_TO edges, weighted
        by `(1 - trust) * token_estimate(memory.text)` so high-trust
        memories are "cheap" and low-trust ones are "expensive."
        Skips CONTRADICTS edges entirely (held contradictions = closed
        paths, exactly like a closed ride in RCT).

        Returns the chain of memories that fits in `max_tokens`,
        ordered by Dijkstra distance (closest to target first).

        Args:
            query: The text to ground. Top-1 cosine match becomes the
                target node.
            max_tokens: Token budget for the returned path. Memories
                are added in distance order until the budget would be
                exceeded.
            max_hops: Safety limit on how far back to walk. Default 8.

        Returns:
            {
                "query": str,
                "target": {memory_id, text, trust, similarity},
                "path": [{memory_id, text, trust, distance, depth}, ...],
                "token_cost": int,
                "token_budget": int,
                "method": "dijkstra" | "no_target" | "no_substrate",
                "closed_paths": int,  # CONTRADICTS edges encountered
            }
        """
        # 1. Find target via cosine top-1
        results = self.search(query, limit=1)
        if not results:
            return {
                "query": query,
                "target": None,
                "path": [],
                "token_cost": 0,
                "token_budget": max_tokens,
                "method": "no_substrate",
                "closed_paths": 0,
            }

        target_hit = results[0]
        target_id = target_hit["memory_id"]
        target_node = self.graph.get_memory(target_id)
        if target_node is None:
            return {
                "query": query,
                "target": None,
                "path": [],
                "token_cost": 0,
                "token_budget": max_tokens,
                "method": "no_target",
                "closed_paths": 0,
            }

        # 2. Dijkstra backward from target
        import heapq
        skip_edge_types = {EdgeType.CONTRADICTS.value, "CONTRADICTS"}
        distances: Dict[str, float] = {target_id: 0.0}
        depth_map: Dict[str, int] = {target_id: 0}
        # heap entries: (cumulative_cost, depth, memory_id)
        heap: List[Tuple[float, int, str]] = [(0.0, 0, target_id)]
        visited: set = set()
        closed_paths = 0

        while heap:
            cost, depth, node_id = heapq.heappop(heap)
            if node_id in visited:
                continue
            visited.add(node_id)
            if depth >= max_hops:
                continue
            for pred, _, edata in self.graph.graph.in_edges(node_id, data=True):
                etype = edata.get("edge_type", "supports")
                if etype in skip_edge_types:
                    closed_paths += 1
                    continue
                pred_node = self.graph.get_memory(pred)
                if pred_node is None:
                    continue
                trust = pred_node.trust
                tokens = _estimate_tokens(pred_node.text)
                # Edge weight: low trust + long text = expensive
                edge_weight = (1.0 - trust) * tokens + 1.0
                new_cost = cost + edge_weight
                if new_cost < distances.get(pred, float("inf")):
                    distances[pred] = new_cost
                    depth_map[pred] = depth + 1
                    heapq.heappush(heap, (new_cost, depth + 1, pred))

        # 3. Build budget-fitting path. Always include target.
        # Order ancestors by Dijkstra distance (closest to target first).
        # F#3 guard: treat corrupt nodes as trust=0 in the sort so they
        # don't AttributeError on `.trust` access. The downstream loop
        # already skips them.
        def _sort_key(mid):
            node = self.graph.get_memory(mid)
            return (distances[mid], -(node.trust if node is not None else 0.0))
        ordered = sorted(distances.keys(), key=_sort_key)
        path: List[Dict[str, Any]] = []
        used_tokens = 0
        for mid in ordered:
            node = self.graph.get_memory(mid)
            if node is None:
                continue
            tokens = _estimate_tokens(node.text)
            if path and used_tokens + tokens > max_tokens:
                # Budget exhausted — stop adding (but keep target which
                # was added first)
                continue
            path.append({
                "memory_id": mid,
                "text": node.text,
                "trust": node.trust,
                "belnap_state": getattr(node, "belnap_state", "T"),
                "distance": round(distances[mid], 3),
                "depth": depth_map.get(mid, 0),
                "token_cost": tokens,
                "is_target": mid == target_id,
            })
            used_tokens += tokens

        return {
            "query": query,
            "target": {
                "memory_id": target_id,
                "text": target_node.text,
                "trust": target_node.trust,
                "similarity": target_hit.get("similarity"),
            },
            "path": path,
            "token_cost": used_tokens,
            "token_budget": max_tokens,
            "method": "dijkstra",
            "closed_paths": closed_paths,
            "path_length": len(path),
        }

    # ------------------------------------------------------------------
    # Contradictions and resolution
    # ------------------------------------------------------------------

    @_sync_first
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

    @_sync_first
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

    @_sync_first
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

    @_sync_first
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

    @_sync_first
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
    # Action receipts (v0.10) — the audit half of the governance loop
    # ------------------------------------------------------------------

    @_sync_first
    def open_receipt(
        self,
        action: str,
        sanction_verdict: str,
        sanction_memory_ids: Optional[List[str]] = None,
    ) -> str:
        """Create a new receipt at sanction time. Returns the receipt_id.

        Called from aether_sanction. The receipt starts in 'open' state
        with no result yet; the caller updates it with aether_receipt
        once the action has executed (or been skipped).
        """
        rid = str(uuid.uuid4())
        receipt = ActionReceipt(
            receipt_id=rid,
            timestamp=time.time(),
            action=action,
            sanction_verdict=sanction_verdict,
            sanction_memory_ids=list(sanction_memory_ids or []),
        )
        from dataclasses import asdict
        self._receipts[rid] = asdict(receipt)
        self._save_receipts()
        return rid

    @_sync_first
    def record_receipt(
        self,
        receipt_id: str,
        result: str,
        tool_name: Optional[str] = None,
        target: Optional[str] = None,
        reversible: Optional[bool] = None,
        reverse_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        verification_passed: Optional[bool] = None,
        verification_reason: Optional[str] = None,
        model_attribution: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing receipt with the actual outcome.

        Called by the agent after it has executed (or skipped) a
        sanctioned action. Preserves the original sanction verdict and
        sanction_memory_ids; fills in tool_name, target, result, details.
        """
        if receipt_id not in self._receipts:
            raise KeyError(f"unknown receipt_id: {receipt_id}")
        r = self._receipts[receipt_id]
        r["result"] = result
        r["completed_at"] = time.time()
        if tool_name is not None:
            r["tool_name"] = tool_name
        if target is not None:
            r["target"] = target
        if reversible is not None:
            r["reversible"] = reversible
        if reverse_action is not None:
            r["reverse_action"] = reverse_action
        if details is not None:
            r["details"] = details
        if verification_passed is not None:
            r["verification_passed"] = verification_passed
        if verification_reason is not None:
            r["verification_reason"] = verification_reason
        if model_attribution is not None:
            r["model_attribution"] = model_attribution
        self._save_receipts()
        return dict(r)

    @_sync_first
    def get_receipt(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """Return one receipt's full record, or None if unknown."""
        r = self._receipts.get(receipt_id)
        return dict(r) if r is not None else None

    @_sync_first
    def list_receipts(
        self,
        limit: int = 20,
        result_filter: Optional[str] = None,
        verdict_filter: Optional[str] = None,
        only_open: bool = False,
    ) -> List[Dict[str, Any]]:
        """List receipts, newest first.

        Filters:
            result_filter   only receipts with matching `result` field
            verdict_filter  only receipts with matching sanction_verdict
            only_open       only receipts with no result yet (sanctioned but
                            outcome not recorded)
        """
        rs = list(self._receipts.values())
        rs.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        out: List[Dict[str, Any]] = []
        for r in rs:
            if result_filter is not None and r.get("result") != result_filter:
                continue
            if verdict_filter is not None and r.get("sanction_verdict") != verdict_filter:
                continue
            if only_open and r.get("result") is not None:
                continue
            out.append(dict(r))
            if len(out) >= limit:
                break
        return out

    @_sync_first
    def receipt_summary(self) -> Dict[str, Any]:
        """Aggregate stats over all receipts: counts, verdict breakdown,
        outcome breakdown, verification pass rate.

        The number to watch: open receipts (sanctioned actions that
        never had their outcome recorded). High counts of those mean
        the agent isn't closing the loop.
        """
        rs = list(self._receipts.values())
        total = len(rs)
        verdicts: Dict[str, int] = {}
        results: Dict[str, int] = {}
        verified = 0
        passed = 0
        failed = 0
        open_count = 0
        for r in rs:
            v = r.get("sanction_verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1
            res = r.get("result")
            if res is None:
                open_count += 1
            else:
                results[res] = results.get(res, 0) + 1
            vp = r.get("verification_passed")
            if vp is not None:
                verified += 1
                if vp:
                    passed += 1
                else:
                    failed += 1
        return {
            "total_receipts": total,
            "verdicts": verdicts,
            "results": results,
            "open_receipts": open_count,
            "verification": {
                "total_verified": verified,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / verified, 3) if verified else None,
            },
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @_sync_first
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


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style tokenizer
    rule-of-thumb). Avoids pulling tiktoken as a hard dependency.
    Returns at least 1 so empty edge costs don't collapse to zero.
    """
    if not text:
        return 1
    # Char-based estimate is within ~15% of tiktoken for English prose.
    return max(1, len(text) // 4)
