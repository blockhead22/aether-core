"""
ContinuityAuditor -- Immune Agent for CRT Law 6

CRT Law 6:
    "Confidence must not exceed continuity."

Pre-generation gate that checks whether the system has answered a similar
question before.  If prior responses exist, it injects them as context so
the LLM can maintain stance consistency.  If prior responses contradict
each other, it escalates to HEDGE -- the system should surface uncertainty
rather than confidently picking one prior stance.

Threat model:
  - The user asks the same question across sessions
  - The system has no awareness it answered before
  - Without continuity context, it generates a fresh answer that may
    contradict its prior stance
  - The user receives conflicting advice from the same system identity
  - Over time, this erodes trust and creates continuity drift

This agent makes continuity visible, measurable, and actionable.

Integration:
  - Runs BEFORE generation (pre-generation hook in crt_rag.py)
  - Queries the belief_speech table for prior responses on same topic
  - Injects prior stance as reasoning context so the LLM can be consistent
  - Feeds results to GapAuditor (Law 5) for post-generation gap check
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class ContinuityAction(str, Enum):
    PASS = "PASS"          # No prior responses found, proceed normally
    INJECT = "INJECT"      # Prior responses found, inject as context
    HEDGE = "HEDGE"        # Prior responses contradict each other


# ---------------------------------------------------------------------------
# Input / output data structures
# ---------------------------------------------------------------------------

@dataclass
class ContinuityCheck:
    """Input to the ContinuityAuditor: what we know before generating."""
    query: str
    query_embedding: Optional[np.ndarray] = None   # 384-dim, pre-computed
    thread_id: Optional[str] = None                 # exclude current thread


@dataclass
class PriorResponse:
    """A single prior response on a similar topic."""
    entry_id: int
    query: str
    response: str
    timestamp: float
    is_belief: bool
    trust_avg: Optional[float]
    similarity: float           # cosine sim between current query and this prior query

    @property
    def age_label(self) -> str:
        """Human-readable age of this response."""
        age_s = time.time() - self.timestamp
        if age_s < 3600:
            return f"{int(age_s / 60)}m ago"
        elif age_s < 86400:
            return f"{int(age_s / 3600)}h ago"
        else:
            return f"{int(age_s / 86400)}d ago"


@dataclass
class ContinuityVerdict:
    """Result of a continuity audit."""
    has_prior: bool
    prior_count: int
    max_similarity: float
    prior_responses: List[PriorResponse] = field(default_factory=list)
    continuity_context: Optional[str] = None
    internal_consistency: Optional[float] = None   # similarity among prior responses
    action: ContinuityAction = ContinuityAction.PASS
    law: str = "Law 6: Confidence must not exceed continuity"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.75      # min cosine sim to consider a query "the same topic"
CONTRADICTION_THRESHOLD = 0.6    # if prior response embeddings < this, they disagree
TOP_K = 5                        # max prior responses to inject
MAX_CONTEXT_CHARS = 2000         # cap injected context length


# ---------------------------------------------------------------------------
# ContinuityAuditor
# ---------------------------------------------------------------------------

class ContinuityAuditor:
    """
    Pre-generation gate that checks for prior responses on the same topic.

    Usage:
        auditor = ContinuityAuditor(db_path="crt_memory.db")
        verdict = auditor.check(ContinuityCheck(
            query="What's your opinion on tabs vs spaces?",
            query_embedding=embed("What's your opinion on tabs vs spaces?"),
        ))
        # verdict.action == ContinuityAction.INJECT
        # verdict.continuity_context == "You have previously responded..."
    """

    def __init__(
        self,
        db_path: str,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        contradiction_threshold: float = CONTRADICTION_THRESHOLD,
        top_k: int = TOP_K,
        encode_fn=None,
    ):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.top_k = top_k
        self._encode_fn = encode_fn

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _encode(self, text: str):
        """Encode text to embedding vector. Uses injected fn or shared LazyEncoder.

        v0.9.2: never blocks. Returns None when the encoder is still
        warming up. Callers must handle None — never block waiting.
        """
        if self._encode_fn is not None:
            return self._encode_fn(text)
        if not hasattr(self, '_lazy_encoder') or self._lazy_encoder is None:
            from aether._lazy_encoder import LazyEncoder
            self._lazy_encoder = LazyEncoder()
            self._lazy_encoder.start_warmup()
        if not self._lazy_encoder.is_loaded:
            return None
        try:
            vec = self._lazy_encoder.model.encode(text)
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            log.warning("ContinuityAuditor: embedding failed: %s", e)
            return None

    # -------------------------------------------------------------------
    # Core check
    # -------------------------------------------------------------------

    def check(self, audit: ContinuityCheck) -> ContinuityVerdict:
        """
        Check if we have prior responses on the same topic.

        Args:
            audit: The incoming query and optional embedding.

        Returns:
            ContinuityVerdict with action, prior responses, and context to inject.
        """
        # Step 1: Get or compute query embedding
        query_emb = audit.query_embedding
        if query_emb is None:
            try:
                query_emb = self._encode(audit.query)
            except Exception as e:
                log.warning("ContinuityAuditor: embedding failed: %s", e)
                return ContinuityVerdict(
                    has_prior=False, prior_count=0, max_similarity=0.0,
                    action=ContinuityAction.PASS,
                )
            # v0.9.2: encoder warming up — pass through without continuity check.
            if query_emb is None:
                return ContinuityVerdict(
                    has_prior=False, prior_count=0, max_similarity=0.0,
                    action=ContinuityAction.PASS,
                )

        # Step 2: Fetch all prior entries with embeddings
        priors = self._fetch_prior_entries(query_emb, audit.thread_id)

        if not priors:
            return ContinuityVerdict(
                has_prior=False, prior_count=0, max_similarity=0.0,
                action=ContinuityAction.PASS,
            )

        # Step 3: Check internal consistency among prior responses
        internal_consistency = self._compute_internal_consistency(priors)

        # Step 4: Determine action
        if internal_consistency is not None and internal_consistency < self.contradiction_threshold:
            action = ContinuityAction.HEDGE
        else:
            action = ContinuityAction.INJECT

        # Step 5: Build context string
        context = self._build_context(priors, action, internal_consistency)

        return ContinuityVerdict(
            has_prior=True,
            prior_count=len(priors),
            max_similarity=priors[0].similarity,
            prior_responses=priors,
            continuity_context=context,
            internal_consistency=internal_consistency,
            action=action,
        )

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    def _fetch_prior_entries(
        self,
        query_emb: np.ndarray,
        exclude_thread_id: Optional[str] = None,
    ) -> List[PriorResponse]:
        """Fetch and rank prior belief_speech entries by embedding similarity."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """SELECT entry_id, query, response, timestamp, is_belief,
                          trust_avg, query_embedding, response_embedding
                   FROM belief_speech
                   WHERE query_embedding IS NOT NULL"""
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []
        query_unit = query_emb / query_norm

        candidates: List[PriorResponse] = []
        for row in rows:
            entry_id, query, response, ts, is_belief, trust_avg, qe_blob, re_blob = row

            if qe_blob is None:
                continue

            prior_emb = np.frombuffer(qe_blob, dtype=np.float32)
            prior_norm = np.linalg.norm(prior_emb)
            if prior_norm == 0:
                continue

            sim = float(np.dot(query_unit, prior_emb / prior_norm))

            if sim >= self.similarity_threshold:
                candidates.append(PriorResponse(
                    entry_id=entry_id,
                    query=query,
                    response=response[:500],   # truncate long responses
                    timestamp=ts,
                    is_belief=bool(is_belief),
                    trust_avg=trust_avg,
                    similarity=round(sim, 4),
                ))

        # Sort by similarity descending, then by recency
        candidates.sort(key=lambda p: (-p.similarity, -p.timestamp))

        return candidates[:self.top_k]

    # -------------------------------------------------------------------
    # Internal consistency
    # -------------------------------------------------------------------

    def _compute_internal_consistency(self, priors: List[PriorResponse]) -> Optional[float]:
        """
        Measure how consistent prior responses are with each other.

        Returns mean pairwise cosine similarity of response embeddings.
        If we can't compute it (no response embeddings), returns None.
        """
        if len(priors) < 2:
            return 1.0  # single prior is consistent with itself

        # Fetch response embeddings for these entries
        conn = self._get_connection()
        try:
            placeholders = ",".join("?" * len(priors))
            rows = conn.execute(
                f"""SELECT entry_id, response_embedding
                    FROM belief_speech
                    WHERE entry_id IN ({placeholders})
                    AND response_embedding IS NOT NULL""",
                [p.entry_id for p in priors],
            ).fetchall()
        finally:
            conn.close()

        embeddings = {}
        for entry_id, re_blob in rows:
            if re_blob:
                emb = np.frombuffer(re_blob, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embeddings[entry_id] = emb / norm

        if len(embeddings) < 2:
            return None

        # Pairwise cosine similarities
        ids = list(embeddings.keys())
        sims = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = float(np.dot(embeddings[ids[i]], embeddings[ids[j]]))
                sims.append(sim)

        return round(float(np.mean(sims)), 4) if sims else None

    # -------------------------------------------------------------------
    # Context building
    # -------------------------------------------------------------------

    def _build_context(
        self,
        priors: List[PriorResponse],
        action: ContinuityAction,
        internal_consistency: Optional[float],
    ) -> str:
        """Build the context string to inject into reasoning context."""
        lines = []

        if action == ContinuityAction.HEDGE:
            lines.append(
                "[CONTINUITY WARNING] You have given INCONSISTENT responses "
                "to similar questions before. Do NOT confidently pick one stance. "
                "Acknowledge the inconsistency or explain what changed."
            )
        else:
            lines.append(
                "[CONTINUITY CONTEXT] You have responded to similar questions before. "
                "Maintain consistency with your prior stance unless the user has "
                "provided new information that warrants a change."
            )

        lines.append("")

        for i, prior in enumerate(priors[:3], 1):  # inject at most 3 to stay concise
            ts_str = datetime.fromtimestamp(prior.timestamp).strftime("%Y-%m-%d %H:%M")
            belief_tag = "belief" if prior.is_belief else "speech"
            trust_str = f" trust={prior.trust_avg:.2f}" if prior.trust_avg else ""
            lines.append(
                f"Prior {i} ({ts_str}, {prior.age_label}, {belief_tag}{trust_str}, "
                f"sim={prior.similarity:.2f}):"
            )
            # Truncate response to fit context budget
            resp_preview = prior.response[:400]
            if len(prior.response) > 400:
                resp_preview += "..."
            lines.append(f"  Q: {prior.query[:200]}")
            lines.append(f"  A: {resp_preview}")
            lines.append("")

        if internal_consistency is not None:
            lines.append(
                f"Internal consistency of prior responses: {internal_consistency:.2f}"
            )

        context = "\n".join(lines)

        # Hard cap
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS - 3] + "..."

        return context


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("ContinuityAuditor -- Law 6 self-test")
    print("=" * 60)

    # Create a temp DB with belief_speech table
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS belief_speech (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            is_belief INTEGER NOT NULL,
            memory_ids_json TEXT,
            trust_avg REAL,
            source TEXT,
            query_embedding BLOB,
            response_embedding BLOB,
            topic_id INTEGER
        )
    """)

    # Helper: fake embedding (deterministic from text hash)
    def fake_embed(text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        v = rng.randn(384).astype(np.float32)
        return v / np.linalg.norm(v)

    # Insert some prior entries
    now = time.time()

    # Prior 1: "What's your favorite programming language?"
    q1 = "What's your favorite programming language?"
    r1 = "I think Python is excellent for most tasks due to its readability."
    q1_emb = fake_embed(q1)
    r1_emb = fake_embed(r1)
    conn.execute(
        "INSERT INTO belief_speech (timestamp, query, response, is_belief, trust_avg, query_embedding, response_embedding) VALUES (?,?,?,?,?,?,?)",
        (now - 86400, q1, r1, 1, 0.8, q1_emb.tobytes(), r1_emb.tobytes()),
    )

    # Prior 2: same topic, consistent response
    q2 = "Which programming language do you prefer?"
    r2 = "Python remains my recommendation for its clarity and ecosystem."
    q2_emb = fake_embed(q2)
    r2_emb = fake_embed(r2)
    conn.execute(
        "INSERT INTO belief_speech (timestamp, query, response, is_belief, trust_avg, query_embedding, response_embedding) VALUES (?,?,?,?,?,?,?)",
        (now - 43200, q2, r2, 1, 0.85, q2_emb.tobytes(), r2_emb.tobytes()),
    )

    # Prior 3: different topic entirely
    q3 = "What's the weather like today?"
    r3 = "I don't have access to real-time weather data."
    q3_emb = fake_embed(q3)
    r3_emb = fake_embed(r3)
    conn.execute(
        "INSERT INTO belief_speech (timestamp, query, response, is_belief, trust_avg, query_embedding, response_embedding) VALUES (?,?,?,?,?,?,?)",
        (now - 3600, q3, r3, 0, 0.5, q3_emb.tobytes(), r3_emb.tobytes()),
    )

    conn.commit()
    conn.close()

    # Create auditor with fake embedding function
    auditor = ContinuityAuditor(db_path=db_path, encode_fn=fake_embed)
    passed = 0
    total = 0

    # Test 1: Query with no similar priors -> PASS
    total += 1
    verdict = auditor.check(ContinuityCheck(query="Tell me about quantum physics"))
    if verdict.action == ContinuityAction.PASS and not verdict.has_prior:
        print(f"  [PASS] Test 1: No prior -> PASS (prior_count={verdict.prior_count})")
        passed += 1
    else:
        print(f"  [FAIL] Test 1: Expected PASS, got {verdict.action} (prior_count={verdict.prior_count})")

    # Test 2: Query similar to priors -> should find matches
    # (with fake embeddings, same text = identical embedding = sim=1.0)
    total += 1
    verdict = auditor.check(ContinuityCheck(query=q1))
    if verdict.has_prior and verdict.action in (ContinuityAction.INJECT, ContinuityAction.HEDGE):
        print(f"  [PASS] Test 2: Same query -> {verdict.action} (prior_count={verdict.prior_count}, sim={verdict.max_similarity})")
        passed += 1
    else:
        print(f"  [FAIL] Test 2: Expected INJECT/HEDGE, got {verdict.action} (has_prior={verdict.has_prior})")

    # Test 3: Context string is generated when priors exist
    total += 1
    if verdict.continuity_context and len(verdict.continuity_context) > 0:
        print(f"  [PASS] Test 3: Context generated ({len(verdict.continuity_context)} chars)")
        passed += 1
    else:
        print(f"  [FAIL] Test 3: No context generated")

    # Test 4: Exact same query should give sim=1.0
    total += 1
    if verdict.has_prior and verdict.max_similarity >= 0.99:
        print(f"  [PASS] Test 4: Self-similarity = {verdict.max_similarity}")
        passed += 1
    else:
        print(f"  [FAIL] Test 4: Expected sim >= 0.99, got {verdict.max_similarity}")

    # Test 5: PriorResponse has correct fields
    total += 1
    if verdict.prior_responses and verdict.prior_responses[0].query == q1:
        pr = verdict.prior_responses[0]
        print(f"  [PASS] Test 5: Prior fields correct (age={pr.age_label}, belief={pr.is_belief})")
        passed += 1
    else:
        print(f"  [FAIL] Test 5: Prior response fields missing or wrong")

    # Test 6: Internal consistency computed
    total += 1
    if verdict.internal_consistency is not None:
        print(f"  [PASS] Test 6: Internal consistency = {verdict.internal_consistency}")
        passed += 1
    else:
        print(f"  [FAIL] Test 6: Internal consistency not computed")

    # Test 7: Empty DB -> PASS
    total += 1
    tmp2 = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp2.close()
    conn2 = sqlite3.connect(tmp2.name)
    conn2.execute("""
        CREATE TABLE IF NOT EXISTS belief_speech (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, query TEXT NOT NULL, response TEXT NOT NULL,
            is_belief INTEGER NOT NULL, memory_ids_json TEXT, trust_avg REAL,
            source TEXT, query_embedding BLOB, response_embedding BLOB, topic_id INTEGER
        )
    """)
    conn2.commit()
    conn2.close()
    empty_auditor = ContinuityAuditor(db_path=tmp2.name, encode_fn=fake_embed)
    v = empty_auditor.check(ContinuityCheck(query="anything"))
    if v.action == ContinuityAction.PASS and v.prior_count == 0:
        print(f"  [PASS] Test 7: Empty DB -> PASS")
        passed += 1
    else:
        print(f"  [FAIL] Test 7: Expected PASS on empty DB, got {v.action}")

    # Test 8: Threshold boundary -- slightly below threshold should not match
    total += 1
    strict_auditor = ContinuityAuditor(
        db_path=db_path,
        similarity_threshold=1.01,  # impossible threshold
        encode_fn=fake_embed,
    )
    v = strict_auditor.check(ContinuityCheck(query=q1))
    if v.action == ContinuityAction.PASS:
        print(f"  [PASS] Test 8: Threshold boundary -- impossible threshold -> PASS")
        passed += 1
    else:
        print(f"  [FAIL] Test 8: Expected PASS with impossible threshold, got {v.action}")

    # Cleanup
    os.unlink(db_path)
    os.unlink(tmp2.name)

    print()
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests passed!")
    else:
        print(f"FAILURES: {total - passed}")
