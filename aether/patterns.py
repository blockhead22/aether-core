"""Composable pattern-matching primitives (v0.11).

Up through v0.10.x, every detector in aether-core was hand-coded regex:
the methodological inference markers, the policy prohibition cues, the
asymmetric-negation gate, the mutex registry. That worked when the
patterns were narrow and stable, but it left several known gaps:

    - known_gap_quantitative (Python 3.10 vs 3.8, dates, counts)
      never caught because slot extraction is regex-based and doesn't
      recognize numeric/version/date as comparable types
    - cold-mode policy_violation 0% because the gate is
      embedding_similarity >= 0.45 and embeddings aren't ready yet
    - cold-mode negation_asymmetry 0% same root cause
    - the mutex registry is a hardcoded ~10-class enum

This module adds four composable, regex-free primitives that
SUPPLEMENT (not replace) the existing detectors:

    1. token_overlap   -- Jaccard similarity on token sets
    2. shape           -- typed-pattern detection with comparison
                          functions: categorical_mutex, numeric_tuple,
                          chronological, magnitude
    3. substring_window -- multiple substrings within N tokens
    4. ncd             -- normalized compression distance, no model

Each primitive returns a real-valued score in [0, 1] plus structured
evidence. They compose: a detector can run several and combine via
weighted sum, max, or threshold gating.

Design constraints:
    - Zero external dependencies beyond stdlib (gzip is stdlib)
    - No model loads, no embeddings required
    - Cold-mode-safe: every primitive works without an encoder
    - Inspectable: each match returns its evidence, not just a score
    - Composable: detectors are recipes over the primitives, not
      monolithic functions

Future v0.11.x: extend with substrate-resident pattern definitions
(prototypes mined from memory tags), cascade-aware confidence,
receipts-driven weight tuning. This file is the primitive layer.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Match result shape
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """The output shape every primitive returns.

    score: real-valued in [0, 1]; 0 = no match, 1 = exact match
    evidence: structured detail about WHY the score landed where it did
    primitive: which primitive produced this result
    """
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    primitive: str = ""


# ---------------------------------------------------------------------------
# Primitive 1 — token_overlap (Jaccard)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Cheap tokenization: lowercase, split on whitespace and punctuation,
    drop empty tokens. Good enough for substring/Jaccard purposes."""
    # Split on any non-word character; lowercase
    tokens = re.split(r"\W+", text.lower())
    return {t for t in tokens if t}


def token_overlap(text_a: str, text_b: str) -> MatchResult:
    """Jaccard similarity on token sets.

    Symmetric. Bounded [0, 1]. Cold-mode safe (no embeddings).
    Returns 1.0 for identical token sets, 0.0 for disjoint.
    """
    a = _tokenize(text_a)
    b = _tokenize(text_b)
    if not a and not b:
        return MatchResult(0.0, {"reason": "both_empty"}, "token_overlap")
    intersection = a & b
    union = a | b
    score = len(intersection) / max(len(union), 1)
    return MatchResult(
        score=score,
        evidence={
            "intersection_size": len(intersection),
            "union_size": len(union),
            "shared_tokens": sorted(intersection)[:8],
            "a_size": len(a),
            "b_size": len(b),
        },
        primitive="token_overlap",
    )


# ---------------------------------------------------------------------------
# Primitive 2 — shape (typed-pattern detection with comparison)
# ---------------------------------------------------------------------------

# Patterns for typed shapes. These are the types the substrate cares about
# for comparison. Adding a new type means: regex + parser + comparator.

_SHAPE_PATTERNS = {
    # Semantic version: 1.2.3, 0.9.5, 3.10
    "version": (
        re.compile(r"\b(\d+(?:\.\d+){1,3})\b"),
        lambda m: tuple(int(x) for x in m.group(1).split(".")),
    ),
    # ISO date: 2026-04-27, 2025-01-15
    "date": (
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
        lambda m: m.group(1),  # ISO-8601 strings compare lexically as dates
    ),
    # Plain integer: 222, 99
    "integer": (
        re.compile(r"\b(\d+)\b"),
        lambda m: int(m.group(1)),
    ),
    # Float: 3.14, 0.95
    "float": (
        re.compile(r"\b(\d+\.\d+)\b"),
        lambda m: float(m.group(1)),
    ),
}


def extract_shapes(text: str) -> List[Tuple[str, Any, str, Tuple[int, int]]]:
    """Find every typed value in text. Returns list of
    (shape_type, parsed_value, raw_match_str, span).

    v0.12: added span (char-offset tuple) so callers can compute LOCAL
    context around each typed value. Without local-context comparison,
    the shape primitive false-fires on co-topical memories whose only
    commonality is unrelated numeric tokens (e.g. "v0.11" vs "v0.12"
    in two release notes — same shape, different things being counted).

    Note: a single text can have multiple shapes (e.g. "version 1.2.3
    released 2024-01-15" yields a version AND a date). Float matches
    will also match as integer for the integer pattern; we resolve by
    preferring more-specific shapes (version > date > float > integer).
    """
    found: List[Tuple[str, Any, str, Tuple[int, int]]] = []
    consumed_spans: List[Tuple[int, int]] = []

    # Order matters: most-specific first
    for shape_type in ("version", "date", "float", "integer"):
        pattern, parser = _SHAPE_PATTERNS[shape_type]
        for m in pattern.finditer(text):
            span = m.span()
            # Skip if this match overlaps a more-specific earlier match
            if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1]
                   for s in consumed_spans):
                continue
            try:
                value = parser(m)
            except (ValueError, AttributeError):
                continue
            found.append((shape_type, value, m.group(0), span))
            consumed_spans.append(span)

    return found


# v0.12: local context size for shape conflict gating
LOCAL_CONTEXT_TOKENS = 3
LOCAL_CONTEXT_MIN_OVERLAP = 0.30


def _local_context(text: str, span: Tuple[int, int],
                   n_tokens: int = LOCAL_CONTEXT_TOKENS) -> set:
    """Return the n tokens before and after `span` as a set.

    Used by shape() to gate conflict detection: two typed values that
    differ are only a real conflict when their immediate surroundings
    overlap above LOCAL_CONTEXT_MIN_OVERLAP. "Python 3.10" vs "Python
    3.8" share "python" in local context -> real conflict. "v0.12 add
    slot detector" vs "v0.11 detection layer" don't share local
    context around the version -> false positive, suppressed.
    """
    before_chars = text[:span[0]]
    after_chars = text[span[1]:]
    # Get last n tokens before, first n tokens after
    before_tokens = re.split(r"\W+", before_chars.lower())
    before_tokens = [t for t in before_tokens if t][-n_tokens:]
    after_tokens = re.split(r"\W+", after_chars.lower())
    after_tokens = [t for t in after_tokens if t][:n_tokens]
    return set(before_tokens + after_tokens)


# Comparison functions per shape type. Returns:
#   "equal"   if values match
#   "differ"  if values are mutually exclusive (mutex-shaped)
#   None      if not comparable (different types, ranges, etc.)

def _compare_categorical(a: Any, b: Any) -> Optional[str]:
    """Plain string equality for categorical values."""
    if a == b:
        return "equal"
    return "differ"


def _compare_numeric_tuple(a: tuple, b: tuple) -> Optional[str]:
    """Compare version tuples. (3, 10) != (3, 8) → differ."""
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        return None
    if a == b:
        return "equal"
    return "differ"


def _compare_chronological(a: str, b: str) -> Optional[str]:
    """ISO date strings compare lexically."""
    if a == b:
        return "equal"
    return "differ"


def _compare_magnitude(a: float, b: float) -> Optional[str]:
    """Numeric magnitude. Equal if within 0.5% relative tolerance."""
    if a == b:
        return "equal"
    # Allow small relative tolerance for floats
    denom = max(abs(a), abs(b), 1e-9)
    if abs(a - b) / denom < 0.005:
        return "equal"
    return "differ"


_COMPARATORS = {
    "version": _compare_numeric_tuple,
    "date": _compare_chronological,
    "integer": _compare_magnitude,
    "float": _compare_magnitude,
    "categorical": _compare_categorical,
}


def shape(text_a: str, text_b: str) -> MatchResult:
    """Detect typed-shape conflicts between two texts.

    Score:
        1.0  -- a typed shape conflict was detected (texts disagree on
                a numeric/version/date/etc. value with the same shape)
                AND the local context around each value matches
        0.5  -- shared shape, equal values (texts agree quantitatively)
        0.0  -- no shared shape detected, or shapes match but local
                context differs (suppresses false positives)

    This primitive is what closes the v0.9.4 known_gap_quantitative
    cases (Python 3.10 vs 3.8, 222 vs 99 tests, 2026-04-27 vs
    2025-01-15) without needing embeddings.

    v0.12: added LOCAL CONTEXT GATE. Without it, the primitive false-
    fires on co-topical memories that share unrelated numeric tokens
    (e.g. "v0.11 fixes bug" vs "v0.12 adds detector" — both have
    version-shaped tokens but they refer to different releases, not
    contradicting facts). The gate requires the immediate surrounding
    tokens (LOCAL_CONTEXT_TOKENS=3) to overlap above
    LOCAL_CONTEXT_MIN_OVERLAP (0.3) before treating differing values
    as a conflict.
    """
    shapes_a = extract_shapes(text_a)
    shapes_b = extract_shapes(text_b)

    if not shapes_a or not shapes_b:
        return MatchResult(
            0.0,
            {"reason": "no_shapes_detected",
             "a_shapes": [s[0] for s in shapes_a],
             "b_shapes": [s[0] for s in shapes_b]},
            "shape",
        )

    # For each shape type that appears in both, run the comparator
    conflicts: List[Dict[str, Any]] = []
    agreements: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []  # v0.12: low-context-overlap conflicts

    for type_a, val_a, raw_a, span_a in shapes_a:
        comparator = _COMPARATORS.get(type_a)
        if comparator is None:
            continue
        for type_b, val_b, raw_b, span_b in shapes_b:
            if type_a != type_b:
                continue
            verdict = comparator(val_a, val_b)
            entry = {
                "shape": type_a,
                "a": raw_a,
                "b": raw_b,
                "verdict": verdict,
            }
            if verdict == "differ":
                # v0.12: require local context overlap. Without this,
                # any two memories that mention different version
                # numbers register as conflicting even when discussing
                # different topics.
                ctx_a = _local_context(text_a, span_a)
                ctx_b = _local_context(text_b, span_b)
                if ctx_a and ctx_b:
                    local_intersection = ctx_a & ctx_b
                    local_union = ctx_a | ctx_b
                    local_overlap = len(local_intersection) / max(len(local_union), 1)
                else:
                    local_overlap = 0.0
                entry["local_overlap"] = round(local_overlap, 3)
                if local_overlap >= LOCAL_CONTEXT_MIN_OVERLAP:
                    entry["local_context"] = sorted(local_intersection)[:5]
                    conflicts.append(entry)
                else:
                    entry["reason"] = "low_local_context_overlap"
                    suppressed.append(entry)
            elif verdict == "equal":
                agreements.append(entry)

    if conflicts:
        return MatchResult(
            score=1.0,
            evidence={
                "conflicts": conflicts,
                "agreements": agreements,
                "suppressed": suppressed,  # v0.12: visible for debugging
            },
            primitive="shape",
        )
    if agreements:
        return MatchResult(
            score=0.5,
            evidence={
                "conflicts": [],
                "agreements": agreements,
                "suppressed": suppressed,
            },
            primitive="shape",
        )
    return MatchResult(
        0.0,
        {"reason": "no_shared_shapes_or_uncompared",
         "a_shapes": [s[0] for s in shapes_a],
         "b_shapes": [s[0] for s in shapes_b],
         "suppressed": suppressed},
        "shape",
    )


# ---------------------------------------------------------------------------
# Primitive 3 — substring_window (multi-substring co-occurrence)
# ---------------------------------------------------------------------------

def substring_window(
    text: str,
    targets: List[str],
    window: int = 10,
) -> MatchResult:
    """All target substrings appear within `window` tokens of each other.

    Useful for catching multi-clause patterns the methodological detector
    cares about, e.g. an inference marker AND a methodological signal
    in the same sentence: ["so", "unsupported"] within 10 tokens.

    Score:
        1.0  -- all targets found AND span <= window
        0.5  -- all targets found but span > window
        0.0  -- one or more targets missing

    Cold-mode safe (pure substring search, no embeddings).
    """
    if not targets:
        return MatchResult(0.0, {"reason": "no_targets"}, "substring_window")

    text_lower = text.lower()
    tokens = re.split(r"\s+", text_lower)

    # Find each target's first token-position in the text
    positions: Dict[str, int] = {}
    for target in targets:
        target_lower = target.lower()
        # Substring search across token boundaries
        # First find char-index, then convert to token-index
        char_idx = text_lower.find(target_lower)
        if char_idx == -1:
            return MatchResult(
                0.0,
                {"reason": "target_missing", "missing": target,
                 "found": list(positions.keys())},
                "substring_window",
            )
        # Convert char index to token index (rough)
        prefix_tokens = len(re.split(r"\s+", text_lower[:char_idx]))
        positions[target] = prefix_tokens

    span = max(positions.values()) - min(positions.values())
    score = 1.0 if span <= window else 0.5
    return MatchResult(
        score=score,
        evidence={
            "positions": positions,
            "span_tokens": span,
            "window": window,
            "all_targets_found": True,
        },
        primitive="substring_window",
    )


# ---------------------------------------------------------------------------
# Primitive 3.5 — slot_equality (v0.12)
# ---------------------------------------------------------------------------

def _extract_slot_tags(tags: List[str]) -> Dict[str, str]:
    """Pull `slot:KEY=VALUE` entries out of a tag list into a dict.

    Returns {KEY: VALUE} mapping. Tags with non-slot prefixes are
    ignored. Multiple values for the same slot in one tag list are
    fine — the dict keeps the last one (callers shouldn't have multi-
    valued slots per memory anyway).
    """
    out: Dict[str, str] = {}
    for t in tags or []:
        if not t.startswith("slot:"):
            continue
        body = t[len("slot:"):]
        if "=" not in body:
            continue
        key, _, value = body.partition("=")
        if key:
            out[key] = value
    return out


def slot_equality(tags_a: List[str], tags_b: List[str]) -> MatchResult:
    """Detect categorical conflicts on shared slot tags.

    The simplest possible contradiction class: two memories tag the
    same slot (e.g. `slot:user.name=Nick` and `slot:user.name=Aether`)
    with different values. v0.12 adds this because Lab A v2 found the
    production substrate has 42 real contradictions on slots like
    user.name / user.favorite_color / user.location that the v0.11
    detection layer was completely blind to (0/42 caught).

    Score:
        1.0  -- one or more shared slots have differing values
        0.5  -- shared slots all agree
        0.0  -- no shared slots

    Cold-mode safe (pure tag comparison, no embeddings, no regex on
    the memory text). The slot taxonomy comes from the substrate's
    accumulated slot extractions, not from a hardcoded registry —
    every new slot the extractor recognizes becomes a contradiction
    detection point automatically.
    """
    slots_a = _extract_slot_tags(tags_a)
    slots_b = _extract_slot_tags(tags_b)
    shared = set(slots_a.keys()) & set(slots_b.keys())

    if not shared:
        return MatchResult(
            0.0,
            {"reason": "no_shared_slots",
             "a_slots": list(slots_a.keys()),
             "b_slots": list(slots_b.keys())},
            "slot_equality",
        )

    conflicts: List[Dict[str, Any]] = []
    agreements: List[Dict[str, Any]] = []
    for key in sorted(shared):
        va, vb = slots_a[key], slots_b[key]
        # Normalize for case-insensitive categorical comparison
        if va.strip().lower() == vb.strip().lower():
            agreements.append({"slot": key, "value": va})
        else:
            conflicts.append({"slot": key, "a": va, "b": vb})

    if conflicts:
        return MatchResult(
            score=1.0,
            evidence={"conflicts": conflicts, "agreements": agreements},
            primitive="slot_equality",
        )
    return MatchResult(
        score=0.5,
        evidence={"conflicts": [], "agreements": agreements},
        primitive="slot_equality",
    )


# ---------------------------------------------------------------------------
# Primitive 4 — NCD (normalized compression distance)
# ---------------------------------------------------------------------------

def _gzip_size(data: bytes) -> int:
    """Compressed size in bytes via gzip. Used as the C(x) in NCD."""
    return len(gzip.compress(data, compresslevel=6))


def ncd(text_a: str, text_b: str) -> MatchResult:
    """Normalized Compression Distance via gzip.

        NCD(a, b) = (C(ab) - min(C(a), C(b))) / max(C(a), C(b))

    Bounded roughly in [0, 1]. Lower = more similar (more shared
    structure). Compute SIMILARITY as 1 - NCD so it's directionally
    consistent with token_overlap (high = similar).

    No model. No training. Just a compressor as the learner.
    Cold-mode safe by construction.
    """
    if not text_a or not text_b:
        return MatchResult(0.0, {"reason": "empty_input"}, "ncd")

    a = text_a.encode("utf-8")
    b = text_b.encode("utf-8")
    ca = _gzip_size(a)
    cb = _gzip_size(b)
    cab = _gzip_size(a + b"\n" + b)
    if max(ca, cb) == 0:
        return MatchResult(0.0, {"reason": "zero_size"}, "ncd")

    distance = (cab - min(ca, cb)) / max(ca, cb)
    # Clamp to [0, 1]
    distance = max(0.0, min(1.0, distance))
    similarity = 1.0 - distance
    return MatchResult(
        score=similarity,
        evidence={
            "distance": round(distance, 4),
            "C_a": ca,
            "C_b": cb,
            "C_ab": cab,
            "similarity": round(similarity, 4),
        },
        primitive="ncd",
    )


# ---------------------------------------------------------------------------
# Composable matcher
# ---------------------------------------------------------------------------

def combined_score(
    results: List[MatchResult],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Combine multiple primitive results into one score.

    Default behavior: weighted average across primitives present in
    `results`. Pass `weights` to override defaults per primitive.

    Returns (combined_score, evidence_dict).
    """
    if not results:
        return 0.0, {"reason": "no_results"}

    default_weights = {
        "token_overlap": 0.25,
        "shape": 0.35,
        "substring_window": 0.20,
        "ncd": 0.20,
    }
    w = weights if weights is not None else default_weights

    total_weight = 0.0
    total_score = 0.0
    breakdown: Dict[str, Any] = {}
    for r in results:
        weight = w.get(r.primitive, 0.0)
        total_weight += weight
        total_score += weight * r.score
        breakdown[r.primitive] = {"score": r.score, "weight": weight}

    combined = total_score / max(total_weight, 1e-9)
    return combined, {"breakdown": breakdown, "total_weight": total_weight}
