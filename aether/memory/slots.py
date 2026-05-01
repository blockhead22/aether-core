"""Lightweight fact-slot extraction for crt-core.

Goal: reduce false contradiction triggers from generic semantic similarity by
comparing only facts that refer to the same attribute ("slot").

This is intentionally heuristic (no ML) and is tuned to the kinds of personal
profile facts used in CRT stress tests.

Phase 2.0 Updates:
- Extended ExtractedFact with temporal_status, period_text, and domains
- Added temporal pattern extraction for past/active/future status
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


# ============================================================================
# TEMPORAL STATUS CONSTANTS
# ============================================================================

class TemporalStatus:
    """Temporal status constants for facts."""
    PAST = "past"           # No longer true (e.g., "I used to work at X")
    ACTIVE = "active"       # Currently true (default)
    FUTURE = "future"       # Will be true (e.g., "I'm starting at X next month")
    POTENTIAL = "potential" # Might be true (e.g., "I might take the job")


# ============================================================================
# TEMPORAL PATTERN EXTRACTION
# ============================================================================

# Patterns that indicate temporal status with their corresponding status
TEMPORAL_PATTERNS: List[Tuple[re.Pattern, str, Optional[str]]] = [
    # Past indicators
    (re.compile(r"(?:i\s+)?(?:used to|formerly|previously)\s+", re.IGNORECASE), TemporalStatus.PAST, None),
    (re.compile(r"(?:i\s+)?(?:no longer|don't|do not|stopped|quit|left)\s+", re.IGNORECASE), TemporalStatus.PAST, None),
    (re.compile(r"\b(?:back when|when i was|in the past)\b", re.IGNORECASE), TemporalStatus.PAST, None),
    (re.compile(r"\bformer\s+", re.IGNORECASE), TemporalStatus.PAST, None),
    (re.compile(r"\bex-", re.IGNORECASE), TemporalStatus.PAST, None),
    (re.compile(r"\banymore\b", re.IGNORECASE), TemporalStatus.PAST, None),

    # Active indicators
    (re.compile(r"(?:i\s+)?(?:currently|now|presently|still)\s+", re.IGNORECASE), TemporalStatus.ACTIVE, None),
    (re.compile(r"\b(?:i am|i'm)\s+(?:currently|still)\s+", re.IGNORECASE), TemporalStatus.ACTIVE, None),
    (re.compile(r"\bthese days\b", re.IGNORECASE), TemporalStatus.ACTIVE, None),

    # Future indicators
    (re.compile(r"(?:i\s+)?(?:will|plan to|going to|about to)\s+", re.IGNORECASE), TemporalStatus.FUTURE, None),
    (re.compile(r"\b(?:starting|beginning|joining)\s+(?:next|soon|in)\s+", re.IGNORECASE), TemporalStatus.FUTURE, None),
    (re.compile(r"\bnext (?:week|month|year)\b", re.IGNORECASE), TemporalStatus.FUTURE, None),

    # Potential indicators
    (re.compile(r"(?:i\s+)?(?:might|may|could|considering)\s+", re.IGNORECASE), TemporalStatus.POTENTIAL, None),
    (re.compile(r"\bthinking about\b", re.IGNORECASE), TemporalStatus.POTENTIAL, None),
]

# Period extraction patterns (capture date ranges)
PERIOD_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "from 2020 to 2024" or "from 2020-2024"
    (re.compile(r"from\s+(\d{4})\s*(?:to|-)\s*(\d{4}|present)", re.IGNORECASE), "period"),
    # "2020-2024" or "2020 - 2024"
    (re.compile(r"\b(\d{4})\s*-\s*(\d{4}|present)\b", re.IGNORECASE), "period"),
    # "since 2020" or "since last year"
    (re.compile(r"since\s+(\d{4}|\w+\s+(?:year|month))", re.IGNORECASE), "since"),
    # "until 2024" or "till 2024"
    (re.compile(r"(?:until|till)\s+(\d{4})", re.IGNORECASE), "until"),
    # "in 2020"
    (re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE), "year"),
]


# ============================================================================
# DIRECT CORRECTION PATTERNS
# ============================================================================
# Patterns that detect explicit corrections like "I'm actually 34, not 32"
# Returns: (new_value, old_value) where new_value is what's being corrected TO

DIRECT_CORRECTION_PATTERNS: List[re.Pattern] = [
    # "I'm actually X, not Y" - extracts (X, Y) - handles numbers and words
    re.compile(r"(?:i'm|i am)\s+actually\s+(\d+|\w+),?\s+not\s+(\d+|\w+)", re.IGNORECASE),
    # "Actually it's X, not Y"
    re.compile(r"actually\s+(?:it's|it is)\s+(\d+|\w+),?\s+not\s+(\d+|\w+)", re.IGNORECASE),
    # "No, I'm X not Y"
    re.compile(r"no,?\s+(?:i'm|i am)\s+(\d+|\w+)\s+not\s+(\d+|\w+)", re.IGNORECASE),
    # "Correction: X not Y"
    re.compile(r"correction:?\s+(\d+|\w+)\s+not\s+(\d+|\w+)", re.IGNORECASE),
    # "Actually X, not Y" (shorter form)
    re.compile(r"actually\s+(\d+|\w+),?\s+not\s+(\d+|\w+)", re.IGNORECASE),
    # "Wait, it's X, not Y"
    re.compile(r"wait,?\s+(?:it's|it is)\s+(\d+|\w+),?\s+not\s+(\d+|\w+)", re.IGNORECASE),
    # "Wait, I'm actually X" (without explicit "not Y")
    re.compile(r"wait,?\s+(?:i'm|i am)\s+actually\s+(\d+)", re.IGNORECASE),
    # "Actually, I work at Amazon, not Microsoft"
    re.compile(r"actually,?\s+i\s+work\s+(?:at|for)\s+([A-Za-z0-9&\-. ]+?),?\s+not\s+([A-Za-z0-9&\-. ]+)", re.IGNORECASE),
]


# ============================================================================
# HEDGED CORRECTION PATTERNS
# ============================================================================
# Patterns that detect soft corrections like "I said 10 years but it's closer to 12"
# Returns: (old_value, new_value) where old_value is what was previously said

HEDGED_CORRECTION_PATTERNS: List[re.Pattern] = [
    # "I think I said X [years of programming] but it's closer to Y" - handles numbers with multiple words after
    re.compile(r"(?:i think\s+)?i\s+said\s+(\d+)(?:\s+\w+)*?\s+but\s+(?:it's|it is)\s+(?:closer to\s+)?(\d+)", re.IGNORECASE),
    # "I may have said X, but actually Y"
    re.compile(r"i\s+(?:may have|might have)\s+said\s+(\d+|\w+),?\s+but\s+(?:actually\s+)?(\d+|\w+)", re.IGNORECASE),
    # "Earlier I mentioned X, it's really Y"
    re.compile(r"earlier\s+i\s+(?:mentioned|said)\s+(\d+|\w+),?\s+(?:it's|it is)\s+really\s+(\d+|\w+)", re.IGNORECASE),
    # "I said X but it's more like Y"
    re.compile(r"i\s+said\s+(\d+)(?:\s+\w+)*?\s+but\s+(?:it's|it is)\s+more\s+like\s+(\d+)", re.IGNORECASE),
    # "I mentioned X earlier but Y is more accurate"
    re.compile(r"i\s+(?:mentioned|said)\s+(\d+|\w+)\s+(?:earlier\s+)?but\s+(\d+|\w+)\s+is\s+(?:more\s+)?accurate", re.IGNORECASE),
    # "Actually closer to X" (simple hedged with number) - fallback pattern
    re.compile(r"(?:it's|it is)\s+(?:actually\s+)?(?:closer to|more like)\s+(\d+)", re.IGNORECASE),
]


def extract_temporal_status(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract temporal status and period from text.

    Args:
        text: The text to analyze

    Returns:
        Tuple of (temporal_status, period_text).
        - temporal_status: "past", "active", "future", or "potential"
        - period_text: Human-readable period like "2020-2024" or None
    """
    if not text:
        return (TemporalStatus.ACTIVE, None)

    text_lower = text.lower()
    detected_status = TemporalStatus.ACTIVE  # Default
    period_text = None

    # Check temporal status patterns
    for pattern, status, _ in TEMPORAL_PATTERNS:
        if pattern.search(text):
            detected_status = status
            break

    # Check period patterns
    for pattern, pattern_type in PERIOD_PATTERNS:
        match = pattern.search(text)
        if match:
            if pattern_type == "period":
                start = match.group(1)
                end = match.group(2) if match.lastindex >= 2 else "present"
                period_text = f"{start}-{end}"
            elif pattern_type == "since":
                period_text = f"{match.group(1)}-present"
            elif pattern_type == "until":
                period_text = f"?-{match.group(1)}"
            elif pattern_type == "year":
                period_text = match.group(1)
            break

    return (detected_status, period_text)


def extract_direct_correction(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract a direct correction pattern from text.

    Detects patterns like:
    - "I'm actually 34, not 32" -> (corrected_to="34", corrected_from="32")
    - "Actually it's Google, not Microsoft" -> (corrected_to="Google", corrected_from="Microsoft")
    - "No, I'm Sarah not Susan" -> (corrected_to="Sarah", corrected_from="Susan")

    Args:
        text: The text to analyze

    Returns:
        Tuple of (new_value, old_value) if correction detected, None otherwise.
        Note: new_value is what's being corrected TO, old_value is what's being corrected FROM.
    """
    if not text:
        return None

    for pattern in DIRECT_CORRECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            new_value = match.group(1).strip()
            # Some patterns only capture the new value (e.g., "Wait, I'm actually X")
            if match.lastindex >= 2:
                old_value = match.group(2).strip()
            else:
                old_value = None  # Old value not specified in this pattern
            return (new_value, old_value)

    return None


def extract_hedged_correction(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract a hedged/soft correction pattern from text.

    Detects patterns like:
    - "I said 10 years but it's closer to 12" -> (old="10", new="12")
    - "I may have said Microsoft, but actually Google" -> (old="Microsoft", new="Google")
    - "Earlier I mentioned Python, it's really JavaScript" -> (old="Python", new="JavaScript")

    Args:
        text: The text to analyze

    Returns:
        Tuple of (old_value, new_value) if correction detected, None otherwise.
        Note: Returns (old, new) - what was said vs what is correct.
    """
    if not text:
        return None

    for pattern in HEDGED_CORRECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            # Some patterns only capture new value (e.g., "closer to X")
            if match.lastindex == 1:
                # Only one group - this is the new value, old value unknown
                new_value = match.group(1).strip()
                return (None, new_value)  # Return None for old_value
            else:
                old_value = match.group(1).strip()
                new_value = match.group(2).strip()
                return (old_value, new_value)
            return (old_value, new_value)

    return None


def detect_correction_type(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Detect any correction pattern (direct or hedged) in text.

    This is the main entry point for correction detection. It checks both
    direct corrections ("I'm X, not Y") and hedged corrections ("I said X but Y").

    Args:
        text: The text to analyze

    Returns:
        Tuple of (correction_type, old_value, new_value) or None
        - correction_type: "direct_correction" or "hedged_correction"
        - old_value: The value being corrected FROM
        - new_value: The value being corrected TO
    """
    # Check for direct correction first (more explicit)
    direct = extract_direct_correction(text)
    if direct:
        new_val, old_val = direct  # direct returns (new, old)
        return ("direct_correction", old_val, new_val)

    # Check for hedged correction
    hedged = extract_hedged_correction(text)
    if hedged:
        old_val, new_val = hedged  # hedged returns (old, new)
        return ("hedged_correction", old_val, new_val)

    return None


@dataclass(frozen=True)
class ExtractedFact:
    """
    A single extracted fact with temporal and domain metadata.

    Phase 2.0 Extension: Added temporal_status, period_text, and domains
    to support context-aware memory and smarter contradiction detection.
    """
    slot: str
    value: Any
    normalized: str

    # Phase 2.0: Temporal metadata
    temporal_status: str = TemporalStatus.ACTIVE  # past | active | future | potential
    period_text: Optional[str] = None             # Human-readable: "2020-2024"

    # Phase 2.0: Domain context (tuple for frozen dataclass)
    domains: Tuple[str, ...] = ()                 # e.g., ("print_shop", "freelance")

    # Source tracking
    confidence: float = 0.9


def create_simple_fact(value: Any, temporal_status: str = TemporalStatus.ACTIVE,
                       domains: Tuple[str, ...] = ()) -> ExtractedFact:
    """
    Create a simple ExtractedFact from a value.

    Useful for converting LLM-extracted tuples to ExtractedFact format
    for compatibility with existing code.

    Phase 2.0: Now supports temporal_status and domains parameters.

    Args:
        value: The fact value
        temporal_status: Temporal status (past/active/future/potential)
        domains: Tuple of domain names for context

    Returns:
        ExtractedFact with slot="", value=value, normalized=str(value).lower()
    """
    return ExtractedFact(
        slot="",
        value=value,
        normalized=str(value).lower().strip(),
        temporal_status=temporal_status,
        domains=domains
    )


_WS_RE = re.compile(r"\s+")


# ============================================================================
# NICKNAME MAPPINGS - Used to recognize name relationships
# ============================================================================

NICKNAME_MAPPINGS = {
    # Common English nicknames
    "alex": {"alexander", "alexandra", "alexis", "alejandro", "alessandra"},
    "bob": {"robert", "bobby", "rob", "robbie"},
    "bill": {"william", "billy", "will", "willy"},
    "mike": {"michael", "mickey", "mick"},
    "nick": {"nicholas", "nicolas", "nicky", "nico"},
    "kate": {"katherine", "catherine", "kathryn", "kathy", "katie"},
    "liz": {"elizabeth", "elisabeth", "beth", "betty", "eliza"},
    "tom": {"thomas", "tommy"},
    "jim": {"james", "jimmy", "jamie"},
    "joe": {"joseph", "joey"},
    "dan": {"daniel", "danny"},
    "sam": {"samuel", "samantha", "sammy"},
    "chris": {"christopher", "christine", "christina", "christian"},
    "matt": {"matthew", "matty"},
    "dave": {"david", "davy"},
    "steve": {"steven", "stephen"},
    "ben": {"benjamin", "benny"},
    "jen": {"jennifer", "jenny", "jenna"},
    "meg": {"megan", "margaret", "maggie"},
    "ed": {"edward", "eddie", "ted", "teddy"},
    "rick": {"richard", "ricky", "dick"},
    "tony": {"anthony"},
    "andy": {"andrew", "drew"},
    "pat": {"patrick", "patricia", "patty"},
}


def names_are_related(name1: str, name2: str) -> bool:
    """
    Check if two names could refer to the same person.

    Examples that return True:
    - "Alex" vs "Alexandra" (nickname)
    - "Alex Chen" vs "Alexandra Chen" (full name with nickname)
    - "Bob" vs "Robert" (nickname mapping)
    """
    n1 = str(name1).lower().strip()
    n2 = str(name2).lower().strip()

    # Exact match
    if n1 == n2:
        return True

    # One is substring of other (Alex Chen vs Alexandra Chen)
    if n1 in n2 or n2 in n1:
        return True

    # Extract first names for comparison
    n1_first = n1.split()[0] if n1 else ""
    n2_first = n2.split()[0] if n2 else ""

    # Check nickname mappings
    for nickname, full_names in NICKNAME_MAPPINGS.items():
        all_names = {nickname} | full_names
        n1_match = n1_first in all_names or any(name in n1_first for name in all_names)
        n2_match = n2_first in all_names or any(name in n2_first for name in all_names)
        if n1_match and n2_match:
            return True

    return False


# Names that belong to the assistant, not the user. Never store these as user.name.
_ASSISTANT_IDENTITY_NAMES = {
    "aether", "groundcheck", "crt",
}

_NAME_STOPWORDS = {
    # Common non-name tokens that appear after "I'm ..." in normal sentences.
    # --- articles / pronouns / misc ---
    "a",
    "an",
    "the",
    "ai",
    "to",
    "just",
    "not",
    "also",
    "now",
    "currently",
    "presently",
    "still",
    "really",
    "very",
    "so",
    "quite",
    "pretty",
    "kinda",
    # --- common state / activity words ---
    "back",
    "building",
    "build",
    "busy",
    "done",
    "here",
    "help",
    "home",
    "new",
    "ready",
    "trying",
    "working",
    "going",
    "looking",
    "thinking",
    "wondering",
    "learning",
    "running",
    "leaving",
    "staying",
    "moving",
    "starting",
    "waiting",
    "telling",
    "asking",
    "saying",
    "getting",
    "having",
    "making",
    "coming",
    "taking",
    "doing",
    "using",
    "feeling",
    "talking",
    "writing",
    "reading",
    "playing",
    "testing",
    # --- informal activity words ---
    "gonna",
    "gotta",
    "wanna",
    "hafta",
    "tryna",
    "run",
    "grab",
    "quick",
    "head",
    "hop",
    "pop",
    "step",
    "swing",
    "jump",
    "dash",
    "rush",
    "hurry",
    "high",
    "low",
    "off",
    "out",
    # --- emotional / sentiment adjectives (the big fix) ---
    "annoyed",
    "angry",
    "furious",
    "mad",
    "upset",
    "frustrated",
    "irritated",
    "pissed",
    "happy",
    "glad",
    "pleased",
    "thrilled",
    "excited",
    "ecstatic",
    "delighted",
    "cheerful",
    "sad",
    "depressed",
    "miserable",
    "unhappy",
    "heartbroken",
    "worried",
    "anxious",
    "nervous",
    "stressed",
    "scared",
    "afraid",
    "terrified",
    "confused",
    "puzzled",
    "lost",
    "bored",
    "lonely",
    "jealous",
    "embarrassed",
    "ashamed",
    "guilty",
    "proud",
    "grateful",
    "thankful",
    "hopeful",
    "optimistic",
    "pessimistic",
    "curious",
    "surprised",
    "shocked",
    "amazed",
    "disgusted",
    "overwhelmed",
    "exhausted",
    "disappointed",
    "content",
    # --- common adjectives / states ---
    "fine",
    "good",
    "great",
    "okay",
    "ok",
    "alright",
    "sorry",
    "sure",
    "tired",
    "sick",
    "hungry",
    "cold",
    "hot",
    "warm",
    "sleepy",
    "awake",
    "alive",
    "well",
    "better",
    "worse",
    "interested",
    "impressed",
    "concerned",
    "convinced",
    "aware",
    "certain",
    "available",
    "unable",
    "able",
}

_NAME_LEADING_FILLERS = {
    "actually",
    "definitely",
    "indeed",
    "just",
    "literally",
    "okay",
    "ok",
    "really",
    "still",
    "well",
    "yeah",
    "yep",
    "yes",
}


def _norm_text(value: str) -> str:
    value = _WS_RE.sub(" ", value.strip())
    return value.lower()


def _normalize_name_parts(value: str) -> List[str]:
    parts = [p for p in re.split(r"\s+", _norm_text(str(value or ""))) if p]
    while parts and parts[0] in _NAME_LEADING_FILLERS:
        parts.pop(0)
    return parts


def names_look_equivalent(left: str, right: str) -> bool:
    """Return True when two name strings look like the same identity.

    This is intentionally conservative and covers common refinement cases like:
    - "Nick" vs "Nick Block"
    - "Nick Block" vs "Nick B"
    - repeated exact matches with different spacing/case
    """
    left_parts = _normalize_name_parts(left)
    right_parts = _normalize_name_parts(right)
    left_norm = " ".join(left_parts)
    right_norm = " ".join(right_parts)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm.startswith(right_norm) or right_norm.startswith(left_norm):
        return True

    if not left_parts or not right_parts:
        return False

    if left_parts[0] == right_parts[0]:
        return True

    if len(left_parts) >= 2 and len(right_parts) >= 2:
        left_first, left_last = left_parts[0], left_parts[-1]
        right_first, right_last = right_parts[0], right_parts[-1]
        if left_last == right_last and (
            left_first.startswith(right_first) or right_first.startswith(left_first)
        ):
            return True

    return False


def is_explicit_name_declaration_text(text: str) -> bool:
    """Return True for texts that are clearly intended to declare a user's name."""
    raw = str(text or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if re.search(r"^\s*(?:fact|pref):\s*name\s*=", raw, flags=re.IGNORECASE):
        return True
    if "my name is" in lowered or "call me" in lowered:
        return True
    if re.search(
        r"\bi(?:'m| am)\s+[A-Z][A-Za-z'-]{1,40}(?:\s+[A-Z][A-Za-z'-]{1,40}){0,2}(?:[,.!?]|$)",
        raw,
    ):
        return True
    return False


def is_question(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if "?" in text:
        return True
    lowered = text.lower()
    return lowered.startswith((
        "what ", "where ", "when ", "why ", "how ", "who ", "which ",
        "do ", "does ", "did ", "can ", "could ", "should ", "would ",
        "is ", "are ", "am ", "was ", "were ", "tell me ",
    ))


def extract_fact_slots(text: str) -> Dict[str, ExtractedFact]:
    """
    Extract a small set of personal-profile fact slots from free text.

    Note: Regex parsing is relatively expensive. Consider caching results
    at the call site if the same text is processed multiple times.
    """
    facts: Dict[str, ExtractedFact] = {}

    if not text or not text.strip():
        return facts

    # Structured facts/preferences (useful for onboarding and explicit corrections).
    # Examples:
    # - "FACT: name = Nick"
    # - "PREF: communication_style = concise"
    # - "FACT: favorite_snack = popcorn" (dynamic category)
    structured = re.search(
        r"\b(?:FACT|PREF):\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if structured:
        slot = structured.group(1).strip().lower()
        value_raw = structured.group(2).strip()

        # Core slots (always allowed)
        core_slots = {
            "name",
            "employer",
            "title",
            "location",
            "pronouns",
            "communication_style",
            "goals",
            "favorite_color",
        }

        # Dynamic fact support: Allow any slot name that starts with "favorite_"
        # or matches common preference patterns, enabling unlimited fact categories
        is_core_slot = slot in core_slots
        is_favorite = slot.startswith("favorite_")
        is_preference = slot.endswith("_preference") or slot.startswith("pref_")
        # More specific dynamic patterns to avoid false positives
        is_my_prefix = slot.startswith("my_") and len(slot) > 3
        is_name_suffix = slot.endswith("_name") and len(slot) > 5
        is_type_suffix = slot.endswith("_type") and len(slot) > 5
        is_status_suffix = slot.endswith("_status") and len(slot) > 7
        is_count_suffix = slot.endswith("_count") and len(slot) > 6

        # Accept if it's a core slot OR a recognized dynamic pattern
        if (is_core_slot or is_favorite or is_preference or
            is_my_prefix or is_name_suffix or is_type_suffix or
            is_status_suffix or is_count_suffix) and value_raw:
            # Clean name values — strip trailing stopwords like "remember", "please"
            if slot == "name":
                _name_stop = {"remember", "recall", "please", "thanks", "okay", "ok",
                              "btw", "lol", "haha", "right", "though", "actually"}
                _name_tokens = value_raw.split()
                while len(_name_tokens) > 1 and _name_tokens[-1].lower() in _name_stop:
                    _name_tokens.pop()
                value_raw = " ".join(_name_tokens)
            facts[slot] = ExtractedFact(slot, value_raw, _norm_text(value_raw))
            return facts

    # ──────────────────────────────────────────────────────────────────────
    # NAME EXTRACTION DISABLED (Session 6, March 2026)
    # Names should be set explicitly via Settings → Profile, not inferred
    # from conversation. Auto-extraction produced garbage like "Nick remember".
    # The auth.display_name field is the canonical source of truth.
    # ──────────────────────────────────────────────────────────────────────
    # Legacy patterns kept but gated behind a flag for future reference.
    _EXTRACT_NAMES_FROM_CONVERSATION = False

    name_pat = r"([A-Za-z][A-Za-z'-]{1,40}(?:\s+[A-Za-z][A-Za-z'-]{1,40}){0,2})"
    name_pat_title = r"([A-Z][A-Za-z'-]{1,40}(?:\s+[A-Z][A-Za-z'-]{1,40}){0,2})"

    # Conjunctions/words that should NOT be part of a name when followed by pronouns/verbs
    _NAME_BOUNDARY_WORDS = {"but", "and", "or", "so", "yet", "for", "nor", "said", "says", "told", "you", "i", "he", "she", "they", "we", "it",
                             "remember", "recall", "right", "though", "please", "already", "actually", "btw", "okay", "ok", "lol", "haha",
                             "thanks", "thank", "hey", "hi", "hello", "from", "here", "there", "now", "then", "also", "too", "just"}

    def _clean_name_value(raw_name: str) -> str:
        """Remove trailing conjunctions and pronouns from name matches."""
        tokens = raw_name.split()
        while tokens and tokens[0].lower() in _NAME_LEADING_FILLERS:
            tokens.pop(0)
        # Walk backwards and remove boundary words
        while len(tokens) > 1 and tokens[-1].lower() in _NAME_BOUNDARY_WORDS:
            tokens.pop()
        # Also check for "X but Y" or "X and Y" patterns in the middle
        cleaned = []
        for i, tok in enumerate(tokens):
            if tok.lower() in _NAME_BOUNDARY_WORDS and i > 0:
                # Stop here - everything before is the name
                break
            cleaned.append(tok)
        return " ".join(cleaned) if cleaned else raw_name

    # Very explicit "call me" pattern.
    m = re.search(r"\bcall me\s+" + name_pat + r"\b", text, flags=re.IGNORECASE)
    if m:
        name = _clean_name_value(m.group(1).strip())
        tokens = [t for t in re.split(r"\s+", name) if t]
        token_lowers = [t.lower() for t in tokens]
        if tokens and not any(t in _NAME_STOPWORDS for t in token_lowers) and name.lower() not in _ASSISTANT_IDENTITY_NAMES:
            facts["name"] = ExtractedFact("name", name, _norm_text(name))

    # Short correction pattern: "Nick not Ben".
    if "name" not in facts:
        m = re.match(
            r"^\s*([A-Z][A-Za-z'-]{1,40})\s+not\s+([A-Z][A-Za-z'-]{1,40})\s*[\.!?]?\s*$",
            text,
        )
        if m:
            cand = m.group(1).strip()
            if cand and cand.lower() not in _NAME_STOPWORDS and cand.lower() not in _ASSISTANT_IDENTITY_NAMES:
                facts["name"] = ExtractedFact("name", cand, _norm_text(cand))

    # "my name is X" pattern - apply _clean_name_value to handle "my name is nick but you..."
    # Also handles correction-style: "my real name is X", "my actual name is X"
    m = re.search(r"\bmy\s+(?:real|actual|true|full)?\s*name is\s+" + name_pat + r"\b", text, flags=re.IGNORECASE)
    # Guard: skip extraction when preceded by negation context (gaslighting pattern)
    if m:
        prefix = text[:m.start()].lower().strip()
        _negation_prefixes = (
            "you think", "why do you think", "you said", "you believe",
            "you told me", "you claim", "you assumed", "i don't know why you think",
            "i never said", "who told you",
        )
        if any(prefix.endswith(neg) for neg in _negation_prefixes):
            m = None  # suppress extraction — likely gaslighting
    if m:
        name = _clean_name_value(m.group(1).strip())
        tokens = [t for t in re.split(r"\s+", name) if t]
        token_lowers = [t.lower() for t in tokens]
        if tokens and not any(t in _NAME_STOPWORDS for t in token_lowers) and name.lower() not in _ASSISTANT_IDENTITY_NAMES:
            facts["name"] = ExtractedFact("name", name, _norm_text(name))

    if "name" not in facts:
        # Prefer TitleCase names for the generic "I'm X" pattern.
        # Match various apostrophe types: ' (straight), curly quotes (U+2018, U+2019)
        # Pattern: i + optional-whitespace + apostrophe + m + whitespace + Name
        # OR: i + apostrophe + m (no space between i and apostrophe)
        apostrophe_pat = r"[\u0027\u2018\u2019]"  # straight, left curly, right curly
        m = re.search(r"\bi" + apostrophe_pat + r"m\s+" + name_pat_title, text, flags=re.IGNORECASE)
        if not m:
            # Also try with whitespace before apostrophe: "I 'm"
            m = re.search(r"\bi\s+" + apostrophe_pat + r"m\s+" + name_pat_title, text, flags=re.IGNORECASE)
        if not m:
            # Also try "I am" pattern
            m = re.search(r"\bi\s+am\s+" + name_pat_title, text, flags=re.IGNORECASE)
        if not m:
            # Allow a single-token lowercase name, but only when it appears as a direct
            # name declaration (no extra trailing content).
            m = re.search(r"^\s*i" + apostrophe_pat + r"m\s+([a-z][a-z'-]{1,40})\s*[\.!?]?\s*$", text, flags=re.IGNORECASE)
    if m:
        name = _clean_name_value(m.group(1).strip())
        tokens = [t for t in re.split(r"\s+", name) if t]
        token_lowers = [t.lower() for t in tokens]

        # Filter obvious non-name phrases like "I'm trying to build ...".
        trailing = (text[m.end():] or "").lstrip().lower()
        looks_like_infinitive = trailing.startswith("to ")
        has_stopword = any(t in _NAME_STOPWORDS for t in token_lowers)

        # Structural guard: if the extracted "name" is followed by a preposition,
        # it's almost certainly an adjective/state, not a name.
        # E.g. "I'm annoyed with you", "I'm angry at this", "I'm confused about it"
        _PREPOSITIONS = {"with", "at", "about", "of", "for", "by", "in", "on",
                         "over", "from", "into", "that", "because", "right"}
        looks_like_adjective = any(
            trailing.startswith(prep + " ") or trailing == prep
            for prep in _PREPOSITIONS
        )

        # Also reject words ending in common adjective suffixes (ed, ing, ous, ful, etc.)
        first_token = token_lowers[0] if token_lowers else ""
        _ADJ_SUFFIXES = ("ed", "ing", "ous", "ful", "ive", "ish", "ent", "ant", "ble", "ious", "ical")
        looks_like_suffix = (
            len(first_token) > 4 and any(first_token.endswith(s) for s in _ADJ_SUFFIXES)
        )

        # Article guard: "I'm currently a freelance web developer" should not parse "currently" as name.
        # Names are not normally followed by "a/an/the ..." in first-person declarations.
        looks_like_role_phrase = trailing.startswith("a ") or trailing.startswith("an ") or trailing.startswith("the ")

        # For name_pat_title patterns: re.IGNORECASE makes [A-Z] match lowercase,
        # so "gonna quick run" can match. Guard: first char must actually be uppercase
        # (or the match came from the explicit lowercase single-token fallback pattern).
        first_char_is_upper = bool(name) and name[0].isupper()
        came_from_lowercase_pattern = (
            m.re.pattern.startswith(r"^\s*i")  # the explicit lowercase-only fallback
        )
        looks_like_case_false_positive = not first_char_is_upper and not came_from_lowercase_pattern

        # Reject common non-name tokens, infinitive phrases, adjective+preposition, suffix patterns, and role phrases.
        if (
            tokens
            and not has_stopword
            and not looks_like_infinitive
            and not looks_like_adjective
            and not looks_like_suffix
            and not looks_like_role_phrase
            and not looks_like_case_false_positive
            and name.lower() not in _ASSISTANT_IDENTITY_NAMES
        ):
            facts["name"] = ExtractedFact("name", name, _norm_text(name))

    # Compound introduction: "I am a Web Developer from Milwaukee Wisconsin"
    compound_intro = re.search(
        r"\bI (?:am|'m) (?:a |an )?(?P<occupation>[^,]+?)\s+(?:from|in)\s+(?P<location>.+?)(?:\.|$|,)",
        text,
        re.IGNORECASE
    )
    if compound_intro:
        occ = compound_intro.group("occupation").strip()
        loc = compound_intro.group("location").strip()
        # Only extract if occupation looks like a job title (not a state of being)
        if occ and len(occ) > 2 and not any(word in occ.lower() for word in ["going", "coming", "person", "student", "happy", "sad"]):
            facts["occupation"] = ExtractedFact("occupation", occ, occ.lower())
        if loc and len(loc) > 2:
            facts["location"] = ExtractedFact("location", loc, loc.lower())

    # Assistant Name
    # Examples:
    # - "Let's call you Aether"
    # - "I'll call you Claude"
    # - "Your name is GPT"
    # - "Call yourself Aria"
    # - "You should be called Nova"
    # Match name pattern with title case (1-3 tokens)
    asst_name_pat = r"([A-Z][A-Za-z'-]{1,40}(?:\s+[A-Z][A-Za-z'-]{1,40}){0,2})"

    def _is_valid_asst_name(name: str) -> bool:
        """Name must start with an actual uppercase letter (guards against re.IGNORECASE
        causing the [A-Z] anchor in asst_name_pat to match lowercase words like 'starting')."""
        return (
            bool(name)
            and name[0].isupper()
            and name.lower() not in _NAME_STOPWORDS
        )

    # Pattern 1: "call you X" or "I'll call you X"
    m = re.search(r"\b(?:I'll|I will|let's|lets)\s+call you\s+" + asst_name_pat, text, flags=re.IGNORECASE)
    if m:
        asst_name = m.group(1).strip()
        if _is_valid_asst_name(asst_name):
            facts["assistant_name"] = ExtractedFact("assistant_name", asst_name, _norm_text(asst_name))

    # Pattern 2: "your name is X" or "you're X" or "you are X"
    if "assistant_name" not in facts:
        m = re.search(r"\byour name is\s+" + asst_name_pat, text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\byou(?:'re| are)\s+" + asst_name_pat + r"(?:\s|[,\.!?]|$)", text, flags=re.IGNORECASE)
        if m:
            asst_name = m.group(1).strip()
            if _is_valid_asst_name(asst_name) and asst_name.lower() not in {"working", "great", "awesome", "helpful", "right", "correct", "wrong"}:
                facts["assistant_name"] = ExtractedFact("assistant_name", asst_name, _norm_text(asst_name))

    # Pattern 3: "call yourself X" or "you should be called X"
    if "assistant_name" not in facts:
        m = re.search(r"\bcall yourself\s+" + asst_name_pat, text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\byou should be called\s+" + asst_name_pat, text, flags=re.IGNORECASE)
        if m:
            asst_name = m.group(1).strip()
            if _is_valid_asst_name(asst_name):
                facts["assistant_name"] = ExtractedFact("assistant_name", asst_name, _norm_text(asst_name))

    # Employer
    # Examples:
    # - "I work at Microsoft as a senior developer."
    # - "I work as a data scientist at Vertex Analytics."
    # - "I work at Amazon, not Microsoft."
    # - "I run a sticker shop called The Printing Lair"
    # - "I work for myself" / "I'm self-employed"
    # - "I don't work at Google anymore" (negation/correction)
    # - "I'm still at Microsoft" (confirmation)

    # Check for employer negations first ("I don't work at X anymore")
    # Also look for "I left X" patterns
    m = re.search(
        r"\bi (?:don't|do not|no longer) work (?:at|for)\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+anymore|\s+now)?(?:\s*[,\.;]|\s*$)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r"\bi left\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+last|\s+this|\s+a|\s*[,\.;]|\s*$)",
            text,
            flags=re.IGNORECASE,
        )
    if m:
        old_employer = m.group(1).strip()
        # Store in employer slot with "LEFT:" prefix - this allows contradiction detection
        # against previous employer values
        facts["employer"] = ExtractedFact("employer", f"LEFT:{old_employer}", f"left {_norm_text(old_employer)}")

    # Check for employer confirmations ("I'm still at X")
    m = re.search(
        r"\bi(?:'m| am) still (?:at|with)\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s*[,\.;]|\s*$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        employer = m.group(1).strip()
        facts["employer"] = ExtractedFact("employer", employer, _norm_text(employer))

    # Check for self-employment first
    if re.search(r"\b(?:i work for myself|i'm self[- ]?employed|i am self[- ]?employed)", text, flags=re.IGNORECASE):
        facts["employer"] = ExtractedFact("employer", "self-employed", "self-employed")

    # Check for "I run [business]" pattern
    m = re.search(r"\bi run (?:a |an )?([^\n\r\.;,]+?)(?:\s+(?:called|and|but|,|\.|;)|\s*$)", text, flags=re.IGNORECASE)
    if m and "employer" not in facts:
        business = m.group(1).strip()
        # Extract business name if "called X" follows
        m2 = re.search(r"called\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+(?:and|but|,|\.|;\()|\s*$)", text)
        if m2:
            business = m2.group(1).strip()
        if business:
            facts["employer"] = ExtractedFact("employer", f"self-employed ({business})", _norm_text(business))

    # Try "I work at/for X" pattern
    if "employer" not in facts:
        has_education_cue = bool(
            re.search(
                r"\b(phd|doctorate|doctoral|master'?s|bachelor'?s|degree|graduat(?:ed|ion)|university|college|school)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        m = re.search(
            r"\b(?:i work at|i work for)\s+([^\n\r\.;,]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not m and not has_education_cue:
            # Fallback: look for "at [company]" anywhere (for "I work as X at Y" patterns)
            m = re.search(r"\bat\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+(?:as|and|but|in|on|for|with|where|,|\.|;)|\s*$)", text)
        if m:
            employer_raw = m.group(1)
            # Trim at common continuations
            employer_raw = re.split(r"\b(?:as|and|but|though|however|,|\.|;|\(|\))\b", employer_raw, maxsplit=1, flags=re.IGNORECASE)[0]
            employer_raw = employer_raw.strip()
            if employer_raw:
                facts["employer"] = ExtractedFact("employer", employer_raw, _norm_text(employer_raw))

    # Job title / role / occupation
    m = re.search(r"\bmy (?:role|job title|title) is\s+([^\n\r\.;,]+)", text, flags=re.IGNORECASE)
    if not m:
        # Match "I am a [title]" or "[title] by degree/trade/profession"
        m = re.search(r"\b(?:i am a|i'm a)\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:by|at|for|and)|\s*$)", text)
        if not m:
            m = re.search(r"\b([A-Z][A-Za-z\s]+?)\s+by\s+(?:degree|trade|profession)", text)
    if m:
        title_raw = m.group(1).strip()
        title_raw = re.split(r"\b(?:at|for|in|by)\b", title_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if title_raw and len(title_raw.split()) <= 4:  # Keep it reasonable (1-4 words)
            facts["title"] = ExtractedFact("title", title_raw, _norm_text(title_raw))

    # Location
    # Examples:
    # - "I live in Seattle, Washington."
    # - "I live in the Seattle metro area, specifically in Bellevue."
    # - "I moved to Denver last month"
    m = re.search(r"\bi live in\s+([^\n\r\.;]+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bi moved to\s+([A-Z][a-zA-Z .'-]{2,60})", text, flags=re.IGNORECASE)
    if m:
        loc_raw = m.group(1).strip()
        # Prefer the last "in X" if present ("specifically in Bellevue")
        m2 = re.search(r"\bin\s+([A-Z][a-zA-Z .'-]{2,60})\b", loc_raw)
        if m2:
            loc_value = m2.group(1).strip()
        else:
            # Split on temporal markers or punctuation
            loc_value = re.split(r"\s+(?:last|this|in|on|during)\s+|\.|,", loc_raw, maxsplit=1)[0].strip()
        if loc_value:
            facts["location"] = ExtractedFact("location", loc_value, _norm_text(loc_value))

    # Years programming experience
    # Examples:
    # - "I've been programming for 10 years"
    # - "it's closer to 12 years" (correction)
    # - "actually 12 years of experience" (correction)
    # - "more like 12 years" (correction)
    m = re.search(
        r"\b(?:i'?ve been programming for|i have been programming for)\s+(\d{1,3})\s+years\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        # Try correction patterns
        m = re.search(
            r"(?:it'?s\s+)?(?:closer to|actually|more like|really)\s+(\d{1,3})(?:\s+years)?(?:\s+(?:of\s+)?(?:experience|programming))?",
            text,
            flags=re.IGNORECASE,
        )
    if m:
        years = int(m.group(1))
        facts["programming_years"] = ExtractedFact("programming_years", years, str(years))

    # Age
    # Examples:
    # - "I am 25 years old"
    # - "I'm 30 years old"
    # - "I'm 28"
    # - "I just turned 29 today"
    # - "I am twenty-five years old"
    # - "I'm actually 34, not 32" (correction)
    # - "Wait, I'm actually 34" (correction)
    # - "My age is actually 34" (correction)

    # Try correction patterns first (they're more specific)
    m = re.search(
        r"\b(?:i'?m|i am)\s+actually\s+(\d{1,3})(?:\s*,?\s*not\s+\d+)?(?:\s+years old)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r"\b(?:wait|actually)[,\s]+(?:i'?m|i am)\s+(?:actually\s+)?(\d{1,3})(?:\s+years old)?\b",
            text,
            flags=re.IGNORECASE,
        )
    if not m:
        m = re.search(
            r"\bmy age is (?:actually\s+)?(\d{1,3})\b",
            text,
            flags=re.IGNORECASE,
        )
    if not m:
        # Standard patterns
        m = re.search(
            r"\bi(?:'m| am)\s+(\d{1,3})(?:\s+years old)?\b",
            text,
            flags=re.IGNORECASE,
        )
    if not m:
        m = re.search(
            r"\bi (?:just )?turned\s+(\d{1,3})\b",
            text,
            flags=re.IGNORECASE,
        )
    if m:
        age = int(m.group(1))
        # Sanity check: age should be between 1 and 120
        if 1 <= age <= 120:
            facts["age"] = ExtractedFact("age", age, str(age))

    # First programming language
    # Examples:
    # - "I've been programming for 8 years, starting with Python."
    # - "I started with Python."
    # - "My first programming language was Python."
    m = re.search(
        r"\b(?:starting with|started with|my first (?:programming )?language was)\s+([A-Z][A-Za-z0-9+_.#-]{1,40})\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        lang = m.group(1).strip()
        facts["first_language"] = ExtractedFact("first_language", lang, _norm_text(lang))

    # Team size
    m = re.search(r"\bteam of\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bteam is\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
    if m:
        size = int(m.group(1))
        facts["team_size"] = ExtractedFact("team_size", size, str(size))

    # Remote vs office preference
    pref: Optional[bool] = None
    lowered = text.lower()
    if "prefer working remotely" in lowered or "prefer remote" in lowered:
        pref = True
    elif "hate working remotely" in lowered or "prefer being in the office" in lowered or "prefer the office" in lowered:
        pref = False

    if pref is not None:
        facts["remote_preference"] = ExtractedFact("remote_preference", pref, "remote" if pref else "office")

    # Favorite color
    # Examples:
    # - "My favorite color is orange."
    # - "My favourite colour is light blue."
    # - "My favortie color is orange." (common typos)
    m = re.search(
        r"\bmy\s+fav(?:ou?rite|ortie|orite|ourite|rite)\s+colou?r\s+is\s+([^\n\r;,!\?]{2,60})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        color_raw = m.group(1).strip().rstrip(" .")
        # Trim at common continuations.
        color_raw = re.split(r"\b(?:and|but|though|however)\b", color_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        _known_color_words = {
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
            "white", "brown", "gray", "grey", "gold", "silver", "teal", "cyan",
            "magenta", "violet", "indigo", "turquoise", "maroon", "navy", "olive",
            "coral", "salmon", "crimson",
        }
        color_tokens = [tok for tok in re.findall(r"[a-z]+", color_raw.lower()) if tok in _known_color_words]
        if color_raw and len(color_tokens) <= 1:
            facts["favorite_color"] = ExtractedFact("favorite_color", color_raw, _norm_text(color_raw))

    # Skip if category is too generic or already handled
    # Common words that don't make good fact categories
    # Can be extended as needed for specific use cases
    _SKIP_FAVORITE_CATEGORIES = {"thing", "one", "part", "time", "way", "place"}

    # Generic favorite X pattern (dynamic fact categories)
    # Examples:
    # - "My favorite snack is popcorn"
    # - "My favorite movie is The Matrix"
    # - "My favourite book is 1984"
    if "favorite_color" not in facts:  # Don't override specific patterns
        m = re.search(
            r"\bmy\s+fav(?:ou?rite|ortie|orite|ourite|rite)\s+([a-z_]+)\s+is\s+([^\n\r;,!\?]{2,60})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            category = m.group(1).strip().lower()
            value_raw = m.group(2).strip().rstrip(" .")
            # Trim at common continuations
            value_raw = re.split(r"\b(?:and|but|though|however)\b", value_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            if category not in _SKIP_FAVORITE_CATEGORIES and value_raw:
                slot_name = f"favorite_{category}"
                facts[slot_name] = ExtractedFact(slot_name, value_raw, _norm_text(value_raw))

    # Education (very rough; enough for Stanford vs MIT undergrad contradictions)
    # Combined pattern: "both my undergrad and Master's were from MIT"
    m = re.search(
        r"\bboth\s+my\s+(?:undergrad|undergraduate)(?:\s+degree)?\s+and\s+(?:my\s+)?master'?s(?:\s+degree)?\s+(?:were|was)?\s*(?:from|at)\s+([A-Z][A-Za-z .'-]{2,60})\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        school = m.group(1).strip()
        if school:
            facts["undergrad_school"] = ExtractedFact("undergrad_school", school, _norm_text(school))
            facts["masters_school"] = ExtractedFact("masters_school", school, _norm_text(school))

    m = re.search(r"\bundergraduate (?:degree )?was from\s+([A-Z][A-Za-z .'-]{2,60})\b", text, flags=re.IGNORECASE)
    if m:
        school = m.group(1).strip()
        facts["undergrad_school"] = ExtractedFact("undergrad_school", school, _norm_text(school))

    m = re.search(r"\bmaster'?s (?:degree )?.*?from\s+([A-Z][A-Za-z .'-]{2,60})\b", text, flags=re.IGNORECASE)
    if m:
        school = m.group(1).strip()
        facts["masters_school"] = ExtractedFact("masters_school", school, _norm_text(school))

    # Siblings
    # Examples:
    # - "I have two siblings"
    # - "I have 3 siblings"
    m = re.search(r"\bi have\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sibling", text, flags=re.IGNORECASE)
    if m:
        count_str = m.group(1).strip()
        # Convert words to numbers
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                       "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        count_normalized = word_to_num.get(count_str.lower(), count_str)
        facts["siblings"] = ExtractedFact("siblings", count_normalized, count_normalized)

    # Languages spoken
    # Examples:
    # - "I speak three languages"
    # - "I speak 5 languages"
    m = re.search(r"\bi speak\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+language", text, flags=re.IGNORECASE)
    if m:
        count_str = m.group(1).strip()
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                       "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        count_normalized = word_to_num.get(count_str.lower(), count_str)
        facts["languages_spoken"] = ExtractedFact("languages_spoken", count_normalized, count_normalized)

    # Graduation year
    # Examples:
    # - "I graduated in 2020"
    # - "I graduated from Stanford in 2018"
    m = re.search(r"\bi graduated\s+(?:in|from.*in)\s+(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if m:
        year = m.group(1).strip()
        facts["graduation_year"] = ExtractedFact("graduation_year", year, year)

    # Degree type (PhD vs Master's vs Bachelor's)
    # Examples:
    # - "I have a PhD"
    # - "I have a Master's degree"
    # - "I have a PhD in Machine Learning"
    # - "I never said I had a PhD. I have a Master's degree."
    # - "I do have a PhD"
    degree_patterns = [
        (r"\bi have a\s+(phd|ph\.d\.?|doctorate)\b", "PhD"),
        (r"\bi have a\s+(master'?s?\s*(?:degree)?)", "Masters"),
        (r"\bi have a\s+(bachelor'?s?\s*(?:degree)?)", "Bachelors"),
        (r"\bi do have a\s+(phd|ph\.d\.?|doctorate)\b", "PhD"),
        (r"\bi do have a\s+(master'?s?\s*(?:degree)?)", "Masters"),
        (r"\bi do have a\s+(bachelor'?s?\s*(?:degree)?)", "Bachelors"),
        (r"\bcompleted my\s+(doctorate|phd|ph\.d\.?)", "PhD"),
        (r"\bcompleted my\s+(master'?s)", "Masters"),
    ]
    for pattern, degree_type in degree_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            facts["degree_type"] = ExtractedFact("degree_type", degree_type, degree_type.lower())
            break

    # School from degree context: "I have a PhD ... from Stanford"
    # This ensures "I have a PhD in Machine Learning from Stanford" also extracts school,
    # not just degree_type. Without this, contradiction "I went to MIT, not Stanford"
    # (which extracts school=MIT) would never match.
    if "school" not in facts:
        m = re.search(
            r"\b(?:i (?:have|got|earned|received)|completed my)\s+(?:a\s+)?(?:phd|ph\.d\.?|doctorate|master'?s?|bachelor'?s?)"
            r"(?:\s+(?:degree\s+)?in\s+[A-Za-z\s]+?)?\s+from\s+([A-Z][A-Za-z\s.'-]{1,50}?)(?:\.|,|;|\s*$)",
            text, flags=re.IGNORECASE
        )
        if m:
            school = m.group(1).strip()
            facts["school"] = ExtractedFact("school", school, _norm_text(school))

    # Project name/description
    # Examples:
    # - "My project is called CRT"
    # - "My current project is building a recommendation engine"
    # - "My project focus has shifted to real-time anomaly detection"
    m = re.search(r"\bmy (?:current )?project\s+(?:is\s+called|'?s\s+name\s+is|name\s+is|is\s+building)\s+(?:a\s+)?([A-Za-z][A-Za-z0-9+_.#\s-]{1,60}?)(?:\.|,|;|\s+for|\s+that|\s+to|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bmy project focus\s+(?:has\s+)?shifted to\s+([A-Za-z][A-Za-z0-9+_.#\s-]{1,60}?)(?:\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        project = m.group(1).strip()
        facts["project"] = ExtractedFact("project", project, _norm_text(project))

    # School (standalone "graduated from X" without year)
    # Examples:
    # - "I graduated from MIT"
    # - "I graduated from Stanford in 2018" (captured by graduation_year above, also here)
    m = re.search(r"\bi graduated from\s+([A-Z][A-Za-z\s.'-]{1,50}?)(?:\s+in\s+\d{4}|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        school = m.group(1).strip()
        facts["school"] = ExtractedFact("school", school, _norm_text(school))

    # Favorite programming language
    # Examples:
    # - "My favorite programming language is Rust"
    # - "Python is my favorite language"
    # - "Python is actually my favorite language now"
    # - "I prefer Python" (only if Python is a known programming language)
    #
    # Known programming languages to avoid false positives like "I prefer working"
    _KNOWN_PROG_LANGS = {
        "python", "javascript", "typescript", "java", "rust", "go", "golang",
        "c", "cpp", "c++", "csharp", "c#", "ruby", "php", "swift", "kotlin",
        "scala", "haskell", "perl", "r", "matlab", "julia", "elixir", "erlang",
        "clojure", "lua", "dart", "fortran", "cobol", "assembly", "sql", "bash",
        "powershell", "groovy", "f#", "ocaml", "lisp", "scheme", "prolog",
        "objective-c", "objectivec", "zig", "nim", "crystal", "elm"
    }

    m = re.search(r"\bmy favorite (?:programming )?language is\s+([A-Z][A-Za-z0-9+#]{1,20})\b", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b([A-Z][A-Za-z0-9+#]{1,20})\s+is (?:actually )?my favorite (?:programming )?language", text, flags=re.IGNORECASE)
    if not m:
        # "I prefer X" pattern - only match if X is a known programming language
        m = re.search(r"\bi prefer\s+([A-Z][A-Za-z0-9+#]{1,20})\b", text, flags=re.IGNORECASE)
        if m and m.group(1).lower().strip() not in _KNOWN_PROG_LANGS:
            m = None  # Not a known programming language, skip
    if m:
        lang = m.group(1).strip()
        facts["programming_language"] = ExtractedFact("programming_language", lang, _norm_text(lang))

    # Pet (type and name)
    # Examples:
    # - "I have a golden retriever named Murphy"
    # - "My dog is a labrador"
    # - "Murphy is a labrador, not a golden retriever"
    _PET_NAME_STOPWORDS = {"my", "job", "role", "work", "name", "pet", "title", "career",
                           "it", "this", "that", "he", "she", "there", "here", "what",
                           "nick", "the", "his", "her", "our", "their"}
    # Words that should NOT be pet types (job/occupation/description words)
    _PET_TYPE_STOPWORDS = {"freelance", "freelancer", "developer", "engineer", "manager",
                           "designer", "analyst", "consultant", "teacher", "professor",
                           "doctor", "lawyer", "student", "intern", "director", "writer",
                           "artist", "musician", "chef", "nurse", "pilot", "driver",
                           "great", "good", "nice", "cool", "interesting", "really",
                           "very", "pretty", "big", "small", "new", "old"}
    m = re.search(r"\bi have a\s+([a-z]+(?:\s+[a-z]+)?)\s+named\s+([A-Z][a-z]+)", text, flags=re.IGNORECASE)
    if m:
        pet_type = m.group(1).strip()
        pet_name = m.group(2).strip()
        if pet_type.lower().split()[0] not in _PET_TYPE_STOPWORDS:
            facts["pet"] = ExtractedFact("pet", pet_type, _norm_text(pet_type))
            facts["pet_name"] = ExtractedFact("pet_name", pet_name, _norm_text(pet_name))
    else:
        # Try just pet type
        m = re.search(r"\bmy (?:dog|cat|pet) is a\s+([a-z]+(?:\s+[a-z]+)?)", text, flags=re.IGNORECASE)
        if not m:
            # Try "[name] is a [breed]" pattern
            # Use case-sensitive match to avoid "my job is a ..." false positives.
            m = re.search(r"\b([A-Z][a-z]+)\s+is a\s+([a-z]+(?:\s+[a-z]+)?)", text)
            if m:
                pet_name = m.group(1).strip()
                pet_type = m.group(2).strip()
                if pet_name.lower() in _PET_NAME_STOPWORDS:
                    m = None
                elif pet_type.split()[0].lower() in _PET_TYPE_STOPWORDS:
                    m = None  # "Nick is a freelance developer" is NOT a pet
                else:
                    facts["pet"] = ExtractedFact("pet", pet_type, _norm_text(pet_type))
                    facts["pet_name"] = ExtractedFact("pet_name", pet_name, _norm_text(pet_name))
        if m and not facts.get("pet"):
            pet_type = m.group(1).strip()
            if pet_type.split()[0].lower() not in _PET_TYPE_STOPWORDS:
                facts["pet"] = ExtractedFact("pet", pet_type, _norm_text(pet_type))

    # Coffee preference
    # Examples:
    # - "I prefer dark roast coffee"
    # - "My coffee preference is light roast"
    # - "I've switched to light roast lately"
    m = re.search(r"\bi prefer\s+(dark|light|medium)\s+roast", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bmy coffee preference is\s+(dark|light|medium)\s+roast", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bswitched to\s+(dark|light|medium)\s+roast", text, flags=re.IGNORECASE)
    if m:
        coffee = m.group(1).strip() + " roast"
        facts["coffee"] = ExtractedFact("coffee", coffee, _norm_text(coffee))

    # Hobby
    # Examples:
    # - "My weekend hobby is rock climbing"
    # - "I enjoy trail running"
    # - "I've taken up trail running instead of climbing"
    m = re.search(r"\bmy (?:weekend )?hobby is\s+([a-z][a-z\s-]{2,40}?)(?:\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bi enjoy\s+([a-z][a-z\s-]{2,40}?)(?:\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\btaken up\s+([a-z][a-z\s-]{2,40}?)(?:\s+instead|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        hobby = m.group(1).strip()
        facts["hobby"] = ExtractedFact("hobby", hobby, _norm_text(hobby))

    # Book currently reading
    # Examples:
    # - "I'm reading 'Designing Data-Intensive Applications'"
    # - "Now reading 'The Pragmatic Programmer'"
    m = re.search(r"\bi'?m reading ['\"]([^'\"]{5,80})['\"]", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bnow reading ['\"]([^'\"]{5,80})['\"]", text, flags=re.IGNORECASE)
    if m:
        book = m.group(1).strip()
        facts["book"] = ExtractedFact("book", book, _norm_text(book))

    # ==================================================================
    # Correction-aware patterns (F2 fix)
    # These handle messages like "Actually X", "For the record, X",
    # "I went to X, not Y", "my partner's name is X, not Y"
    # ==================================================================

    # School: "I went to MIT" / "For the record, I went to MIT, not Stanford"
    if "school" not in facts and "masters_school" not in facts:
        m = re.search(
            r"\bi (?:went to|attended|studied at|got my (?:degree|phd|master'?s?) (?:at|from))\s+"
            r"([A-Z][A-Za-z\s.'-]{1,50}?)(?:\s*[,.]|\s+not\b|\s*$)",
            text, flags=re.IGNORECASE
        )
        if m:
            school = m.group(1).strip().rstrip(",.")
            facts["school"] = ExtractedFact("school", school, _norm_text(school))

    # Spouse/partner: "my partner's name is Casey" / "I'm married to Casey"
    if "spouse" not in facts:
        spouse_patterns = [
            r"\bmy (?:partner|spouse|wife|husband|significant other|fiancee?|girlfriend|boyfriend)(?:'?s)?\s+(?:name\s+is|is)\s+([A-Z][A-Za-z'-]{1,40})",
            r"\bi(?:'m| am) married to\s+(?:someone (?:named|called)\s+)?([A-Z][A-Za-z'-]{1,40})",
            r"\bmy (?:partner|spouse|wife|husband)(?:'?s)?\s+(?:name\s+is|is\s+(?:actually\s+)?)\s*([A-Z][A-Za-z'-]{1,40})",
        ]
        for pat in spouse_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                spouse_name = m.group(1).strip()
                facts["spouse"] = ExtractedFact("spouse", spouse_name, _norm_text(spouse_name))
                break

    # Pet correction: "Murphy is actually a labrador, not a golden retriever"
    if "pet" not in facts:
        m = re.search(
            r"\b([A-Z][a-z]+)\s+is (?:actually )?a\s+([a-z]+(?:\s+[a-z]+)?)\s*[,.]?\s*(?:not\b|$)",
            text
        )
        if m:
            pet_name = m.group(1).strip()
            pet_type = m.group(2).strip()
            if pet_name.lower() not in _PET_NAME_STOPWORDS:
                facts["pet"] = ExtractedFact("pet", pet_type, _norm_text(pet_type))
                facts["pet_name"] = ExtractedFact("pet_name", pet_name, _norm_text(pet_name))

    # Programming language: "I've fully switched to Rust" / "switched to Rust"
    if "programming_language" not in facts:
        m = re.search(
            r"\b(?:switched|moved|transitioned|migrated)\s+to\s+([A-Z][A-Za-z0-9+#]{1,20})\b",
            text, flags=re.IGNORECASE
        )
        if m:
            lang = m.group(1).strip()
            if lang.lower() in _KNOWN_PROG_LANGS:
                facts["programming_language"] = ExtractedFact("programming_language", lang, _norm_text(lang))

    # Coffee/drink: "Tea only now" / "I've gone off coffee" / "I switched to tea"
    if "coffee" not in facts:
        # "gone off coffee" / "quit coffee" / "stopped drinking coffee"
        m = re.search(r"\b(?:gone off|quit|stopped|no more)\s+coffee\b", text, flags=re.IGNORECASE)
        if m:
            # Check if they mention what they switched TO
            m2 = re.search(r"\b(tea|matcha|water|juice|decaf)\s+(?:only|now|instead)\b", text, flags=re.IGNORECASE)
            if m2:
                drink = m2.group(1).strip()
                facts["coffee"] = ExtractedFact("coffee", drink, _norm_text(drink))
            else:
                facts["coffee"] = ExtractedFact("coffee", "none", "none")
        else:
            m = re.search(r"\bswitched to\s+(tea|matcha|water|juice|decaf)\b", text, flags=re.IGNORECASE)
            if m:
                drink = m.group(1).strip()
                facts["coffee"] = ExtractedFact("coffee", drink, _norm_text(drink))

    # Employer correction: "I work at Amazon, not Google" / "I should clarify — I work at Amazon"
    if "employer" not in facts:
        m = re.search(
            r"\bi (?:work|am working)\s+(?:at|for)\s+([A-Z][A-Za-z\s.'-]{1,50}?)(?:\s*[,.]|\s+not\b|\s*$)",
            text, flags=re.IGNORECASE
        )
        if m:
            employer = m.group(1).strip().rstrip(",.")
            facts["employer"] = ExtractedFact("employer", employer, _norm_text(employer))

    # ────────────────────────────────────────────────────────────────────
    # v0.13.0 Phase A: Third-person + project-fact slots.
    #
    # The 2026-05-01 reflexive bench surfaced that aether_fidelity could
    # not catch hallucinations because slot extraction was first-person-
    # only. A draft like "Nick is a chef in Paris" extracted no slots, so
    # compute_grounding had no slot evidence and fell back to similarity
    # alone (which classified semantically-related-but-irrelevant memories
    # as supporting at high trust).
    #
    # This block adds extractors that fire on:
    #   - third-person prose ("Nick is a chef in Paris", "Nick uses emacs")
    #   - editor / IDE preferences (no extractor existed for this)
    #   - project facts ("CRT uses FAISS", "vector dimension is 768")
    #
    # All produce slot:k=v tags compatible with the existing slot_equality
    # detector in compute_grounding.
    # ────────────────────────────────────────────────────────────────────

    # Editor / IDE preference. Fires on first-person and third-person.
    # Examples:
    #   "I prefer vim over emacs"        → editor=vim
    #   "Nick uses emacs primarily"       → editor=emacs
    #   "We use VS Code in the team"      → editor=vs_code
    if "editor" not in facts:
        _editor_pat = re.compile(
            r"\b(?:i|we|[A-Z][a-zA-Z]+(?:'s)?)\s+"
            r"(?:prefer|prefers|use|uses|using|switched to|moved to)\s+"
            r"(vim|emacs|neovim|nvim|vs ?code|sublime|atom|jetbrains|"
            r"intellij|pycharm|webstorm|goland|rider|fleet|zed|cursor|helix)"
            r"(?:\s+(?:over|instead of|primarily|exclusively|for))?",
            re.IGNORECASE,
        )
        m = _editor_pat.search(text)
        if m:
            editor_raw = m.group(1).strip().lower().replace(" ", "_")
            facts["editor"] = ExtractedFact("editor", editor_raw, editor_raw)

    # Third-person occupation: "X is a Y" or "X is the Y of Z"
    # Produces the same slot key as the first-person compound_intro
    # extractor above, so first-person and third-person prose about the
    # same fact collide on the same slot when checking contradictions.
    # Only fires when subject looks like a proper name (capitalized,
    # not "I"/"We"/"It"/"This"/"That") and the occupation is a noun phrase.
    if "occupation" not in facts:
        _third_person_occ = re.compile(
            r"\b([A-Z][A-Za-z'-]{1,40})\s+is\s+"
            r"(?:a|an|the)\s+"
            r"([a-z][a-zA-Z\s-]{2,40}?)"
            r"(?:\s+(?:of|in|at|for|on)\s+|\s*[.,;]|$)",
        )
        m = _third_person_occ.search(text)
        if m:
            subj = m.group(1).strip()
            occ = m.group(2).strip()
            # Skip if subj is a stopword that the regex anchors caught
            _subj_stopwords = {"This", "That", "These", "Those", "It"}
            if subj not in _subj_stopwords and len(occ) >= 3:
                # Drop if it's clearly not an occupation (e.g. "good", "happy")
                _occ_blocklist = {"good", "great", "happy", "sad", "right",
                                  "wrong", "here", "there", "back", "ready"}
                if occ.lower().split()[0] not in _occ_blocklist:
                    facts["occupation"] = ExtractedFact("occupation", occ, _norm_text(occ))

    # Third-person location: "X lives in Y" / "X is in Y" / "X is based in Y".
    # Excludes "works at" — that's the employer pattern, captured separately.
    if "location" not in facts:
        _third_person_loc = re.compile(
            r"\b[A-Z][A-Za-z'-]{1,40}\s+(?:lives|is|is based|resides)\s+(?:in)\s+"
            r"([A-Z][A-Za-z\s.'-]{1,40}?)(?:\s*[.,]|\s+(?:and|but|so)\s|$)",
        )
        m = _third_person_loc.search(text)
        if m:
            loc = m.group(1).strip().rstrip(",.")
            facts["location"] = ExtractedFact("location", loc, _norm_text(loc))
        else:
            # Compound form caught by occupation regex above already extracted
            # role. Try to also pull out a location from "X is a Y in Z".
            _compound = re.compile(
                r"\b[A-Z][A-Za-z'-]{1,40}\s+is\s+(?:a|an|the)\s+"
                r"[a-z][a-zA-Z\s-]{2,40}?\s+(?:in|at)\s+"
                r"([A-Z][A-Za-z\s.'-]{1,40}?)(?:\s*[.,]|$)",
            )
            m2 = _compound.search(text)
            if m2:
                loc = m2.group(1).strip().rstrip(",.")
                facts["location"] = ExtractedFact("location", loc, _norm_text(loc))

    # Third-person employer: "X works at Y"
    if "employer" not in facts:
        _third_person_emp = re.compile(
            r"\b[A-Z][A-Za-z'-]{1,40}\s+works?\s+(?:at|for)\s+"
            r"([A-Z][A-Za-z\s.'-]{1,50}?)(?:\s*[,.]|\s+on\b|\s+as\b|\s*$)",
        )
        m = _third_person_emp.search(text)
        if m:
            emp = m.group(1).strip().rstrip(",.")
            facts["employer"] = ExtractedFact("employer", emp, _norm_text(emp))

    # Project facts: "X uses Y" / "X is built with Y" — for project-shaped
    # subjects (uppercase, often acronym-ish) and known framework/tool values.
    # Examples:
    #   "CRT uses FAISS"          → project_vector_store=faiss (when value matches)
    #   "We use Postgres"          → project_database=postgres
    if "project_vector_store" not in facts:
        _vector_store_pat = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9_-]{1,30}|the (?:project|repo|codebase)|we|the team)\s+"
            r"(?:uses?|using|built with|based on|backed by)\s+"
            r"(faiss|pinecone|weaviate|chromadb?|qdrant|milvus|pgvector|elasticsearch)",
            re.IGNORECASE,
        )
        m = _vector_store_pat.search(text)
        if m:
            store = m.group(1).strip().lower()
            facts["project_vector_store"] = ExtractedFact(
                "project_vector_store", store, store
            )

    # Project embedding dimension: "vector dimension is N" / "N-dim embeddings"
    # / "embedding size of N" — captures the integer.
    if "project_embedding_dim" not in facts:
        _dim_pats = [
            re.compile(
                r"\b(?:vector dimension|embedding dim(?:ension)?|"
                r"hidden size|embedding size)\s+(?:is|of|=)\s+(\d{2,5})\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(\d{2,5})[ -]?dim(?:ensional)?\s+(?:embed|vector)",
                re.IGNORECASE,
            ),
        ]
        for pat in _dim_pats:
            m = pat.search(text)
            if m:
                dim = m.group(1)
                facts["project_embedding_dim"] = ExtractedFact(
                    "project_embedding_dim", dim, dim
                )
                break

    # Project framework / web stack: "X uses Flask" / "built with FastAPI"
    if "project_framework" not in facts:
        _framework_pat = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9_-]{1,30}|the (?:project|repo|codebase|app|api)|we|the team)\s+"
            r"(?:uses?|using|built with|based on|written in)\s+"
            r"(flask|fastapi|django|express|nextjs|next\.js|nuxt|svelte|sveltekit|"
            r"react|vue|angular|tornado|aiohttp|sanic|rails|spring|laravel)",
            re.IGNORECASE,
        )
        m = _framework_pat.search(text)
        if m:
            fw = m.group(1).strip().lower().replace(".", "").replace(" ", "_")
            facts["project_framework"] = ExtractedFact(
                "project_framework", fw, fw
            )

    # v0.13.1 Phase B: project chosen option (capital-letter option
    # labels in decision prose). Catches Test #3 Case B from the
    # 2026-05-01 reflexive bench — paraphrased decision drift across
    # natural prose.
    # Examples:
    #   "CRT picked Option A"           → project_chosen_option=A
    #   "we pivoted to Option B"        → project_chosen_option=B
    #   "the team went with Option C"   → project_chosen_option=C
    if "project_chosen_option" not in facts:
        _chosen_option_pat = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9_-]{1,30}|the (?:project|repo|codebase|team|app)|we|i)\s+"
            r"(?:picked|chose|went with|pivoted to|selected|decided on|opted for)\s+"
            r"(?:the\s+)?Option\s+([A-Z])\b",
            re.IGNORECASE,
        )
        m = _chosen_option_pat.search(text)
        if m:
            opt = m.group(1).strip().upper()
            facts["project_chosen_option"] = ExtractedFact(
                "project_chosen_option", opt, opt
            )

    # Strip name/assistant_name if auto-extraction is disabled.
    # Names should come from auth.display_name, not conversation inference.
    if not _EXTRACT_NAMES_FROM_CONVERSATION:
        facts.pop("name", None)
        facts.pop("assistant_name", None)

    return facts


def is_temporal_update(old_fact: ExtractedFact, new_fact: ExtractedFact) -> bool:
    """
    Check if a new fact represents a temporal update (not a contradiction).

    Examples of temporal updates:
    - "I work at Google" -> "I don't work at Google anymore" (status change)
    - Same value, new status = temporal update
    - "used to work at X" + "now work at Y" = both valid (different times)

    Args:
        old_fact: The existing fact
        new_fact: The incoming fact

    Returns:
        True if this is a temporal update, False if it might be a contradiction
    """
    # Different slots = not related
    if old_fact.slot != new_fact.slot:
        return False

    # Same value, different status = temporal update
    if old_fact.normalized == new_fact.normalized:
        if old_fact.temporal_status != new_fact.temporal_status:
            return True

    # New fact says old value is now "past" = temporal update
    if new_fact.temporal_status == TemporalStatus.PAST:
        # Check if new fact references the old value
        old_value_lower = str(old_fact.value).lower()
        new_value_lower = str(new_fact.value).lower()

        # "LEFT:Google" normalizes to "left google" or similar
        if old_value_lower in new_value_lower or new_value_lower.replace("left:", "").strip() == old_value_lower:
            return True

    # Both are "past" = historical, not current contradiction
    if old_fact.temporal_status == TemporalStatus.PAST and new_fact.temporal_status == TemporalStatus.PAST:
        return True

    return False


def domains_overlap(fact1: ExtractedFact, fact2: ExtractedFact) -> bool:
    """
    Check if two facts have overlapping domains.

    Used for contradiction detection: facts in different domains can coexist.
    e.g., "I'm a programmer" (tech domain) and "I'm a photographer" (creative domain)

    Args:
        fact1: First fact
        fact2: Second fact

    Returns:
        True if domains overlap (potential conflict), False if disjoint (can coexist)
    """
    domains1 = set(fact1.domains) if fact1.domains else {"general"}
    domains2 = set(fact2.domains) if fact2.domains else {"general"}

    # "general" overlaps with everything
    if domains1 == {"general"} or domains2 == {"general"}:
        return True

    # Check for actual overlap
    return bool(domains1 & domains2)
