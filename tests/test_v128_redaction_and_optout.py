"""v0.12.9: secret redaction + AETHER_DISABLE_AUTOINGEST opt-out.

Production-readiness gap surfaced 2026-04-30: the auto-ingest hook
captures every user prompt and writes high-trust facts to disk with
no opt-out, no redaction, no pause. A user debugging an OAuth flow
or pasting a one-off API key would leak the secret into the substrate
with no way to prevent it short of uninstalling the hook.

The fix is two-layer:

1. ``AETHER_DISABLE_AUTOINGEST=1`` short-circuits ``extract_facts``
   and ``ingest_turn`` immediately. Hook stays installed; nothing is
   written.

2. ``redact_secrets`` runs before pattern matching. Common API-key
   shapes, bearer tokens, ``password=...`` style key=value pairs,
   and PEM private-key blocks are replaced with ``[REDACTED]``.
   Conservative on purpose — emails and phone numbers are left
   alone because they're often legitimate context.

These tests pin both contracts.
"""

from __future__ import annotations

import os

import pytest

from aether.memory.auto_ingest import (
    extract_facts,
    redact_secrets,
)


# ==========================================================================
# Redaction
# ==========================================================================

class TestRedactSecrets:
    """``redact_secrets`` catches the common high-signal secret shapes."""

    def test_openai_anthropic_style_key(self):
        text = "my key is sk-abc123def456ghi789jkl012mno345 use it carefully"
        out = redact_secrets(text)
        assert "sk-abc123def456ghi789jkl012mno345" not in out
        assert "[REDACTED]" in out

    def test_aws_access_key_id(self):
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE for the deploy"
        out = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in out

    def test_github_pat(self):
        text = "use this token: ghp_abcdefghijklmnopqrstuvwxyz0123456789 to push"
        out = redact_secrets(text)
        assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in out

    def test_stripe_style_key(self):
        # Built at runtime — the literal must not appear in source or
        # GitHub push protection flags it as a Stripe key. Low-entropy
        # value so even reconstructed scanners do not match.
        prefix = "sk" + "_" + "live" + "_"
        fake = prefix + ("Z" * 24)
        text = f"STRIPE_KEY={fake} for billing"
        out = redact_secrets(text)
        assert fake not in out

    def test_slack_token(self):
        text = "hook url uses xoxb-123456789-abcdef-ghijkl tokens"
        out = redact_secrets(text)
        assert "xoxb-123456789-abcdef-ghijkl" not in out

    def test_bearer_header(self):
        text = "Authorization: Bearer abc123def456ghi789jkl012mno345"
        out = redact_secrets(text)
        assert "abc123def456ghi789jkl012mno345" not in out

    def test_pem_private_key_block(self):
        text = (
            "before\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0Z3VS5uBfx8\n"
            "abcdef1234567890abcdef==\n"
            "-----END RSA PRIVATE KEY-----\n"
            "after"
        )
        out = redact_secrets(text)
        assert "MIIEpAIBAAKCAQEA0Z3VS5uBfx8" not in out
        assert "before" in out and "after" in out

    def test_password_kv_form_keeps_key_redacts_value(self):
        text = "password=hunter2lol then more text"
        out = redact_secrets(text)
        assert "hunter2lol" not in out
        # Keep the keyword visible — useful when debugging false positives.
        assert "password" in out
        assert "[REDACTED]" in out

    def test_api_key_colon_form(self):
        text = "api_key: secretvalue123abc and other config"
        out = redact_secrets(text)
        assert "secretvalue123abc" not in out

    def test_empty_string_returns_unchanged(self):
        assert redact_secrets("") == ""
        assert redact_secrets(None) is None  # type: ignore[arg-type]

    def test_idempotent(self):
        """Running the redactor twice yields the same string as once."""
        text = "key sk-abc123def456ghi789jkl012mno345 and password=hunter2"
        once = redact_secrets(text)
        twice = redact_secrets(once)
        assert once == twice

    def test_does_not_redact_emails(self):
        """Emails are usually legit context — should pass through."""
        text = "ping me at nick@example.com about the deploy"
        out = redact_secrets(text)
        assert "nick@example.com" in out

    def test_does_not_redact_phone_numbers(self):
        text = "call 555-123-4567 if it breaks"
        out = redact_secrets(text)
        assert "555-123-4567" in out


# ==========================================================================
# Opt-out (AETHER_DISABLE_AUTOINGEST)
# ==========================================================================

class TestOptOut:
    """When the env var is set, ``extract_facts`` returns immediately."""

    def test_extract_facts_returns_empty_when_disabled(self, monkeypatch):
        monkeypatch.setenv("AETHER_DISABLE_AUTOINGEST", "1")
        result = extract_facts(
            user_message="I prefer Python with type hints in strict mode",
            assistant_response=None,
        )
        assert result == []

    def test_extract_facts_runs_normally_when_unset(self, monkeypatch):
        monkeypatch.delenv("AETHER_DISABLE_AUTOINGEST", raising=False)
        result = extract_facts(
            user_message="I prefer Python with type hints",
            assistant_response=None,
        )
        # Without the env var, the user_preference rule fires.
        assert len(result) >= 1
        assert any("Python" in f.text for f in result)

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_recognized_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("AETHER_DISABLE_AUTOINGEST", value)
        assert extract_facts(user_message="I prefer X") == []

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_values_do_not_disable(self, monkeypatch, value):
        monkeypatch.setenv("AETHER_DISABLE_AUTOINGEST", value)
        result = extract_facts(user_message="I prefer Python over Ruby")
        assert len(result) >= 1


# ==========================================================================
# End-to-end: redaction applies before extraction
# ==========================================================================

class TestRedactionBeforeExtraction:
    """Secrets in the input are redacted *before* the rules see them,
    so candidate fact text contains [REDACTED] instead of the secret.
    """

    def test_constraint_with_secret_value_redacts(self, monkeypatch):
        monkeypatch.delenv("AETHER_DISABLE_AUTOINGEST", raising=False)
        # The "never" prefix triggers the constraint rule. The secret
        # should be in the captured snippet but redacted by the time
        # it reaches the candidate fact text.
        result = extract_facts(
            user_message=(
                "never share the api_key=sk-abc123def456ghi789jkl with anyone"
            ),
        )
        assert len(result) >= 1
        for f in result:
            assert "sk-abc123def456ghi789jkl" not in f.text
            assert "sk-abc123def456ghi789jkl" not in f.raw_match

    def test_identity_with_aws_key_redacts(self, monkeypatch):
        monkeypatch.delenv("AETHER_DISABLE_AUTOINGEST", raising=False)
        result = extract_facts(
            user_message="my role is admin and AKIAIOSFODNN7EXAMPLE is the key",
        )
        for f in result:
            assert "AKIAIOSFODNN7EXAMPLE" not in f.text
