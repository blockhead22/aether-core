"""LLM-based fact-slot extraction (local-first, vendor-optional).

Why this exists
---------------
The regex extractor in ``slots.py`` is fast, free, and tuned for short
conversational turns. On archival conversational corpora (see
``bench/slot_coverage_gpt_corpus.py``) it produces:

  * 94.92% zero-slot turns (paraphrase-blind floor on a real GPT export)
  * heavy false-positives on the 5% it does fire on (prose nouns and
    code identifiers tagged as personal facts)

This module provides an LLM-backed alternative that asks the model to
extract ``{slot, value, temporal_status, polarity}`` triples in a tight
JSON format. It is **opt-in** and **local-first**: by default it talks
to a local Ollama daemon. No external API, no API key, no telemetry.

Backends supported
------------------
* ``ollama`` (default): ``http://localhost:11434/api/generate``, native
  JSON mode via ``format="json"``. Recommended models: ``llama3.2:3b``,
  ``qwen2.5:3b``, ``phi3.5:3.8b`` — small enough to run on CPU.
* ``openai`` (any OpenAI-compatible server: LM Studio, llama.cpp's
  ``./server``, vLLM, ollama's ``/v1`` endpoint, even the real OpenAI
  API if the user explicitly opts in).
* ``none`` (default-default if ``AETHER_LLM_EXTRACT`` is not set): the
  module is importable but ``extract_fact_slots_llm`` returns ``{}``.

Discovery order for a backend (first match wins):
  1. ``AETHER_LLM_BACKEND`` env var (explicit user choice)
  2. ``AETHER_LLM_URL`` set → assume ``openai``-compatible
  3. Ollama reachable at ``http://localhost:11434`` → ``ollama``
  4. ``none``

Failure modes
-------------
Every failure path returns ``{}`` rather than raising. The auto-ingest
hot path must NEVER fail because the LLM is down — the regex extractor
is always the floor. Errors are logged to ``~/.aether/llm_extract.log``
when ``AETHER_LLM_DEBUG=1``.

Usage
-----
::

    from aether.memory.llm_extract import extract_fact_slots_llm

    facts = extract_fact_slots_llm("I moved from Seattle to Milwaukee last month")
    # {'location': ExtractedFact(slot='location', value='Milwaukee', ...)}

For the hybrid path (regex first, LLM fallback on long zero-slot turns)::

    from aether.memory.llm_extract import extract_fact_slots_hybrid

    facts = extract_fact_slots_hybrid(text, min_chars_for_fallback=100)

Status
------
**STUB** — the LLM call is wired through end-to-end against Ollama, but
the prompt vocabulary is intentionally minimal. The slot canonicalization
table needs to be expanded to match ``slots.py``'s regex coverage before
this can be the auto-ingest default. See ``ROADMAP.md`` Phase C.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aether.memory.slots import ExtractedFact, TemporalStatus

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OPENAI_MODEL = "local-model"  # LM Studio / llama.cpp don't care
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_INPUT_CHARS = 4000  # truncate long turns to keep latency bounded


# Slot vocabulary the LLM should populate. Keep tight — the LLM is far
# more accurate when the schema is small. Mirrors the high-frequency
# slots from slots.py; expand cautiously.
SLOT_VOCABULARY = [
    "name",
    "location",
    "occupation",
    "employer",
    "title",
    "first_language",
    "programming_language",
    "editor",
    "favorite_color",
    "favorite_drink",
    "favorite_food",
    "pet",
    "pet_name",
    "hobby",
    "age",
    "school",
    "remote_preference",
    "project_framework",
    "project_vector_store",
    "project_embedding_dim",
    "project_chosen_option",
]

# Temporal vocabulary mirrors slots.TemporalStatus but exposed flat for
# the prompt.
TEMPORAL_VALUES = ["active", "past", "future", "potential"]


# Single source of truth for the system prompt. Stable across calls so
# prompt caching (if a backend supports it) can take effect.
_SYSTEM_PROMPT_TEMPLATE = """You extract structured personal facts from a user's
conversational turn. Output JSON only, no prose, no markdown fences.

Rules:
1. Only extract facts the USER is stating about THEMSELVES. Never about
   other people. Never about hypothetical or fictional characters.
2. Skip code blocks, code comments, and quoted speech.
3. Skip questions ("what is my favorite drink?" — extract nothing).
4. Skip third-person facts ("my friend works at Google" — skip).
5. Output canonical short values. "Microsoft" not "the Microsoft
   corporation". "blue" not "the color blue". Lowercase strings.
6. If the user states they NO LONGER hold a fact ("I don't drink coffee
   anymore"), use polarity="deny" and temporal_status="past".
7. If the user is correcting a prior fact ("actually it's water now"),
   use polarity="affirm" and add a "supersedes_prior": true field.
8. If nothing applies, return an object with an empty "facts" list. Do NOT invent.

Slot vocabulary (use these exact slot names, or omit the fact):
__VOCABULARY__

Output schema:
{
  "facts": [
    {
      "slot": "<one of the vocabulary>",
      "value": "<canonical short string>",
      "temporal_status": "<active|past|future|potential>",
      "polarity": "<affirm|deny>",
      "supersedes_prior": false
    }
  ]
}
"""


def _system_prompt() -> str:
    """Build the system prompt with the slot vocabulary substituted in."""
    return _SYSTEM_PROMPT_TEMPLATE.replace("__VOCABULARY__", ", ".join(SLOT_VOCABULARY))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_logger = logging.getLogger("aether.llm_extract")


def _log_debug(msg: str) -> None:
    """Append a line to the debug log if AETHER_LLM_DEBUG=1."""
    if not os.environ.get("AETHER_LLM_DEBUG"):
        return
    try:
        log_dir = Path.home() / ".aether"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "llm_extract.log", "a", encoding="utf-8") as fh:
            fh.write(msg.rstrip() + "\n")
    except Exception:
        pass  # never let logging crash the hot path


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------


def _ollama_reachable(url: str = DEFAULT_OLLAMA_URL, timeout_s: float = 1.0) -> bool:
    """Cheap probe — does an Ollama daemon answer at this URL?"""
    try:
        host = url.split("://", 1)[-1].split("/", 1)[0]
        if ":" in host:
            host_only, port_str = host.rsplit(":", 1)
            port = int(port_str)
        else:
            host_only, port = host, 11434
        with socket.create_connection((host_only, port), timeout=timeout_s):
            return True
    except (OSError, ValueError):
        return False


def _resolve_backend() -> Tuple[str, Dict[str, str]]:
    """
    Pick a backend and return (name, config_dict).

    Returns ('none', {}) if no backend is configured AND no daemon is
    reachable. Caller should then short-circuit to {}.
    """
    explicit = os.environ.get("AETHER_LLM_BACKEND", "").strip().lower()
    url = os.environ.get("AETHER_LLM_URL", "").strip()
    model = os.environ.get("AETHER_LLM_MODEL", "").strip()

    if explicit == "ollama":
        return "ollama", {
            "url": url or DEFAULT_OLLAMA_URL,
            "model": model or DEFAULT_OLLAMA_MODEL,
        }
    if explicit == "openai":
        return "openai", {
            "url": url or "http://localhost:1234/v1",
            "model": model or DEFAULT_OPENAI_MODEL,
            "api_key": os.environ.get("AETHER_LLM_API_KEY", "not-needed"),
        }
    if explicit == "none":
        return "none", {}

    # Implicit discovery
    if url:
        return "openai", {
            "url": url,
            "model": model or DEFAULT_OPENAI_MODEL,
            "api_key": os.environ.get("AETHER_LLM_API_KEY", "not-needed"),
        }
    if _ollama_reachable():
        return "ollama", {
            "url": DEFAULT_OLLAMA_URL,
            "model": model or DEFAULT_OLLAMA_MODEL,
        }
    return "none", {}


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _call_ollama(text: str, *, url: str, model: str, timeout_s: float) -> Optional[str]:
    """
    POST to Ollama's /api/generate with format=json. Returns the raw JSON
    string the model produced, or None on failure.

    Ollama's JSON mode constrains decoding to valid JSON which dramatically
    improves robustness for small models.
    """
    body = {
        "model": model,
        "system": _system_prompt(),
        "prompt": text[:DEFAULT_MAX_INPUT_CHARS],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
        },
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        _log_debug(f"ollama_failure: {type(e).__name__} {e}")
        return None


def _call_openai_compatible(
    text: str,
    *,
    url: str,
    model: str,
    api_key: str,
    timeout_s: float,
) -> Optional[str]:
    """
    POST to /v1/chat/completions on an OpenAI-compatible endpoint
    (LM Studio, llama.cpp ./server, vLLM, ollama's /v1 shim, or the
    real OpenAI API if the user explicitly opts in).

    Uses ``response_format={"type": "json_object"}`` when supported.
    Returns the raw JSON string of the assistant message, or None.
    """
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(),
            },
            {"role": "user", "content": text[:DEFAULT_MAX_INPUT_CHARS]},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        _log_debug(f"openai_compat_failure: {type(e).__name__} {e}")
        return None


# ---------------------------------------------------------------------------
# JSON → ExtractedFact translation
# ---------------------------------------------------------------------------


def _coerce_temporal_status(raw: Optional[str]) -> str:
    """Map an LLM-emitted temporal value onto the canonical TemporalStatus."""
    if not raw:
        return TemporalStatus.ACTIVE
    raw = raw.strip().lower()
    return raw if raw in TEMPORAL_VALUES else TemporalStatus.ACTIVE


def _parse_facts_payload(raw_json: str) -> List[Dict[str, Any]]:
    """
    Tolerantly parse whatever JSON the model produced.

    Accepts either ``{"facts": [...]}`` (the prompt-specified shape) or a
    bare list ``[...]`` (small models often drop the wrapper). Returns
    [] on any parse error.
    """
    if not raw_json or not raw_json.strip():
        return []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # Try to recover from common small-model patterns: trailing prose
        # after the JSON object. Find the first { and the last matching }.
        start = raw_json.find("{")
        end = raw_json.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw_json[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    if isinstance(data, dict):
        facts = data.get("facts")
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict)]
        # Dict with slots-as-keys shape
        if all(isinstance(v, (str, int, float, bool)) for v in data.values()):
            return [{"slot": k, "value": v} for k, v in data.items()]
        return []
    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict)]
    return []


def _to_extracted_fact(payload: Dict[str, Any]) -> Optional[Tuple[str, ExtractedFact]]:
    """Translate one LLM-emitted dict into (slot_name, ExtractedFact)."""
    slot = (payload.get("slot") or "").strip().lower()
    if not slot or slot not in SLOT_VOCABULARY:
        return None
    value = payload.get("value")
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        value = str(value)
    polarity = (payload.get("polarity") or "affirm").strip().lower()
    if polarity == "deny":
        return None  # the regex extractor doesn't model deny-affirmations
                     # at the slot level; skip rather than corrupt the slot.
    temporal_status = _coerce_temporal_status(payload.get("temporal_status"))
    normalized = value.strip().lower()
    fact = ExtractedFact(
        slot=slot,
        value=value.strip(),
        normalized=normalized,
        temporal_status=temporal_status,
        period_text=None,
        domains=(),
        confidence=0.85,  # slightly below the regex's 0.9 default until
                          # we have a calibration study showing parity.
    )
    return slot, fact


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def extract_fact_slots_llm(
    text: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, ExtractedFact]:
    """
    Extract personal-fact slots from a conversational turn using a local LLM.

    Returns a dict shaped like ``extract_fact_slots`` (slot_name ->
    ExtractedFact) so it is drop-in compatible. Returns ``{}`` on any
    backend failure, missing daemon, malformed output, etc.

    The auto-ingest hot path can call this safely; the worst case is an
    empty result, never an exception.
    """
    if not text or not text.strip():
        return {}
    if not os.environ.get("AETHER_LLM_EXTRACT"):
        return {}

    backend, cfg = _resolve_backend()
    if backend == "none":
        _log_debug("no backend resolved; skipping")
        return {}

    if backend == "ollama":
        raw = _call_ollama(text, url=cfg["url"], model=cfg["model"], timeout_s=timeout_s)
    elif backend == "openai":
        raw = _call_openai_compatible(
            text,
            url=cfg["url"],
            model=cfg["model"],
            api_key=cfg.get("api_key", ""),
            timeout_s=timeout_s,
        )
    else:
        raw = None

    if raw is None:
        return {}

    payloads = _parse_facts_payload(raw)
    out: Dict[str, ExtractedFact] = {}
    for p in payloads:
        translated = _to_extracted_fact(p)
        if translated is None:
            continue
        slot, fact = translated
        # Last write wins per slot, mirroring extract_fact_slots semantics.
        out[slot] = fact

    if not out:
        _log_debug(f"empty result from backend={backend}; raw[:200]={raw[:200]!r}")

    return out


def extract_fact_slots_hybrid(
    text: str,
    *,
    min_chars_for_fallback: int = 80,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, ExtractedFact]:
    """
    Run the regex extractor first; fall through to the LLM if (a) the
    regex returned no slots AND (b) the turn is at least
    ``min_chars_for_fallback`` long.

    The threshold exists because most short turns genuinely have no
    facts ("ok", "thanks", "yeah") and there's no point spending an
    LLM call on them. Tune via the kwarg or env override
    ``AETHER_LLM_FALLBACK_MIN_CHARS``.
    """
    from aether.memory.slots import extract_fact_slots  # avoid circular import at module load

    facts = extract_fact_slots(text) or {}
    if facts:
        return facts

    threshold = int(os.environ.get("AETHER_LLM_FALLBACK_MIN_CHARS", min_chars_for_fallback))
    if len(text) < threshold:
        return facts

    return extract_fact_slots_llm(text, timeout_s=timeout_s)


def diagnostics() -> Dict[str, Any]:
    """
    Report what the module can see right now. Useful for `aether doctor`
    integration and for the user to debug their local setup without
    digging through env vars.
    """
    backend, cfg = _resolve_backend()
    enabled = bool(os.environ.get("AETHER_LLM_EXTRACT"))
    return {
        "enabled": enabled,
        "backend": backend,
        "config": {k: v for k, v in cfg.items() if k != "api_key"},
        "ollama_reachable": _ollama_reachable(),
        "env": {
            "AETHER_LLM_EXTRACT": os.environ.get("AETHER_LLM_EXTRACT", ""),
            "AETHER_LLM_BACKEND": os.environ.get("AETHER_LLM_BACKEND", ""),
            "AETHER_LLM_URL": os.environ.get("AETHER_LLM_URL", ""),
            "AETHER_LLM_MODEL": os.environ.get("AETHER_LLM_MODEL", ""),
        },
    }


__all__ = [
    "extract_fact_slots_llm",
    "extract_fact_slots_hybrid",
    "diagnostics",
    "SLOT_VOCABULARY",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
]
