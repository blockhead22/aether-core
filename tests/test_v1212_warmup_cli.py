"""v0.12.12: ``aether warmup`` CLI command.

Production-readiness gap #5: when sentence-transformers cannot reach
HuggingFace Hub on first install (corporate proxy, blocked network,
HF rate limit), the encoder silently fails to warm up. The substrate
falls back to cold mode and the user does not realize they're in a
degraded state — search results just look worse than they should.

The fix: ``aether warmup`` is an explicit "load the model now and tell
me if it worked" command. Run once after ``pip install``. On success,
prints `[ OK ] encoder loaded`. On failure, surfaces the last few
lines of `encoder_warmup.log` plus a remediation cookbook (retry on
HF-reachable network, pre-cache, install [ml] extra, fall back to
cold mode).

These tests stub out LazyEncoder to verify both paths without
depending on the real model load.
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from aether.cli import cmd_warmup


def _make_args(timeout=10):
    ns = argparse.Namespace()
    ns.timeout = timeout
    return ns


class _FakeReadyEncoder:
    """Stub encoder that loads successfully."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    is_unavailable = False
    is_loaded = True

    def start_warmup(self):
        pass

    def wait_until_ready(self, timeout):
        return True


class _FakeFailedEncoder:
    """Stub encoder that flips to is_unavailable=True after warmup."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    is_unavailable = True
    is_loaded = False

    def start_warmup(self):
        pass

    def wait_until_ready(self, timeout):
        return False


class _FakeTimeoutEncoder:
    """Stub encoder that times out without flipping is_unavailable."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    is_unavailable = False
    is_loaded = False

    def start_warmup(self):
        pass

    def wait_until_ready(self, timeout):
        return False


class TestWarmupSuccess:
    def test_returns_zero_when_encoder_loads(self, capsys):
        with patch("aether._lazy_encoder.LazyEncoder", _FakeReadyEncoder):
            rc = cmd_warmup(_make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "encoder loaded" in out
        assert "MiniLM" in out


class TestWarmupFailure:
    def test_returns_nonzero_when_encoder_unavailable(self, capsys):
        with patch("aether._lazy_encoder.LazyEncoder", _FakeFailedEncoder):
            rc = cmd_warmup(_make_args())
        assert rc == 1
        captured = capsys.readouterr()
        # Failure path writes to stderr.
        assert "encoder warmup did not complete" in captured.err
        assert "remediation" in captured.err

    def test_failure_includes_hf_remediation_when_unavailable(self, capsys):
        with patch("aether._lazy_encoder.LazyEncoder", _FakeFailedEncoder):
            cmd_warmup(_make_args())
        err = capsys.readouterr().err
        # User sees the HF Hub guidance + the [ml] extra hint.
        assert "huggingface.co" in err
        assert "aether-core[ml]" in err

    def test_timeout_path_suggests_increased_timeout(self, capsys):
        with patch("aether._lazy_encoder.LazyEncoder", _FakeTimeoutEncoder):
            rc = cmd_warmup(_make_args(timeout=10))
        assert rc == 1
        err = capsys.readouterr().err
        assert "increase timeout" in err
        assert "--timeout 20" in err

    def test_failure_mentions_cold_mode_fallback(self, capsys):
        """Critical: user must know the substrate is not bricked, just
        running in cold mode without the encoder."""
        with patch("aether._lazy_encoder.LazyEncoder", _FakeFailedEncoder):
            cmd_warmup(_make_args())
        err = capsys.readouterr().err
        assert "cold mode" in err.lower()


class TestWarmupCLIWiring:
    """End-to-end: build_parser registers warmup as a subcommand."""

    def test_subcommand_registered(self):
        from aether.cli import build_parser
        parser = build_parser()
        # If parser cannot find `warmup`, this raises SystemExit.
        ns = parser.parse_args(["warmup", "--timeout", "5"])
        assert ns.cmd == "warmup"
        assert ns.timeout == 5.0

    def test_default_timeout_is_120(self):
        from aether.cli import build_parser
        parser = build_parser()
        ns = parser.parse_args(["warmup"])
        assert ns.timeout == 120.0
