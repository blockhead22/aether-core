"""Mode resolution for integration adapters.

Each integration source uses the same four-mode model:

    read | write | consult | full

Mode is set per-source via ``AETHER_<SOURCE>_INTEGRATION``. Empty / unset
means the integration is off entirely. ``full`` enables all three modes.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}
_VALID_MODES = {"", "off", "read", "write", "consult", "full"}


def integration_mode(source: str) -> str:
    """Return the mode for a given integration source name.

    Reads ``AETHER_<SOURCE>_INTEGRATION``; lower-cases and trims. Returns
    "" when unset, invalid, or "off".
    """
    var = f"AETHER_{source.upper()}_INTEGRATION"
    raw = os.environ.get(var, "").strip().lower()
    if raw in _VALID_MODES and raw != "off":
        return raw
    return ""


def _is_mode_enabled(source: str, mode: str) -> bool:
    current = integration_mode(source)
    if not current:
        return False
    if current == "full":
        return True
    return current == mode


def is_read_enabled(source: str) -> bool:
    return _is_mode_enabled(source, "read")


def is_write_enabled(source: str) -> bool:
    return _is_mode_enabled(source, "write")


def is_consult_enabled(source: str) -> bool:
    return _is_mode_enabled(source, "consult")
