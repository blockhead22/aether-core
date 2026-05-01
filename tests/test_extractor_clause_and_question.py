"""Regression tests for two extract_facts quality bugs.

Bug 1: clause-boundary leak.
    "my name is nick and my favorite color is orange"
    used to produce one frankenfact ("nick and my favorite color is orange")
    because the user_identity capture group ran past the conjunction.

Bug 2: question false-positive.
    "does it work even if I don't call it."
    used to fire the constraint rule on `don't call it` even though the
    sentence is interrogative (no `?` because the user wrote a period).

Fix: _trim_at_clause_boundary trims at conjunction-introduces-new-subject
patterns (and-my, and-i, but, because, so, ...). _is_inside_question
walks back from the match to its carrying sentence and skips when the
sentence opens with an interrogative word or terminates on `?`.
"""

from aether.memory import extract_facts


# --- Bug 1: clause-boundary leak --------------------------------------------

def test_name_and_color_does_not_merge():
    cs = extract_facts(
        user_message="my name is nick and my favorite color is orange",
    )
    assert len(cs) == 1
    assert cs[0].signal == "user_identity"
    assert cs[0].text == "User: nick"


def test_name_and_role_clause_trim():
    cs = extract_facts(user_message="my name is alice and I work at Anthropic")
    assert len(cs) == 1
    assert cs[0].text == "User: alice"


def test_clause_trim_at_because():
    cs = extract_facts(user_message="we use Postgres because Redis was slow")
    assert len(cs) == 1
    assert cs[0].text == "This project: Postgres"


def test_clause_trim_at_but():
    """`but` is a clause boundary — the user_identity capture must stop
    before it. The trailing clause may itself be a real fact (the
    project_fact rule fires on `we use Claude`), which is fine — we just
    need the identity capture to not eat it."""
    cs = extract_facts(user_message="my company is Anthropic but we use Claude")
    by_signal = {c.signal: c for c in cs}
    assert "user_identity" in by_signal
    assert by_signal["user_identity"].text == "User: Anthropic"


def test_list_continuation_preserved():
    """`and X` followed by a non-subject token is a list, not a new clause."""
    cs = extract_facts(user_message="we use Postgres and Redis")
    assert len(cs) == 1
    assert cs[0].text == "This project: Postgres and Redis"


# --- Bug 2: question false-positive -----------------------------------------

def test_constraint_inside_question_period():
    """Question written with a period (no `?`) — opener-based detection."""
    cs = extract_facts(user_message="does it work even if I don't call it.")
    assert cs == []


def test_constraint_inside_question_mark():
    cs = extract_facts(user_message="Don't we use Postgres?")
    assert cs == []


def test_identity_inside_question():
    cs = extract_facts(user_message="is my name nick?")
    assert cs == []


# --- Positive controls (must still fire after the fix) ----------------------

def test_preference_still_fires():
    cs = extract_facts(user_message="I prefer pnpm over npm.")
    assert len(cs) == 1
    assert cs[0].signal == "user_preference"


def test_identity_still_fires():
    cs = extract_facts(user_message="I am a senior engineer.")
    assert len(cs) == 1
    assert cs[0].signal == "user_identity"


def test_constraint_still_fires():
    cs = extract_facts(user_message="Never force-push to main.")
    assert len(cs) == 1
    assert cs[0].signal == "constraint"


def test_constraint_with_quoted_command_still_fires():
    cs = extract_facts(
        user_message="Never bypass commit hooks with --no-verify.",
    )
    assert len(cs) == 1
    assert cs[0].signal == "constraint"
