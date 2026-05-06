"""Aether integrations — pluggable adapters for external belief stores.

Each integration declares which of four modes it supports:

  * ``read``    — substrate reads from the external source on search/path
  * ``write``   — external source's writes are pulled into the substrate
                  (poll/sync; this mode does not require the external
                  source to know about aether)
  * ``consult`` — external source asks substrate before acting; requires
                  changes on the external source's side
  * ``full``    — read + write + consult

Modes are user-selectable via env vars. The default is no integration —
adapters import safely whether or not their target source is present.

See ``aether.integrations.crt`` for the CRT (Coherent Recall Triplet)
adapter; pattern repeats for any future external source.
"""

from .config import (
    integration_mode,
    is_read_enabled,
    is_write_enabled,
    is_consult_enabled,
)

__all__ = [
    "integration_mode",
    "is_read_enabled",
    "is_write_enabled",
    "is_consult_enabled",
]
