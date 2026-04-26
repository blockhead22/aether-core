"""Tests for aether.memory.slots — fact slot extraction."""

from aether.memory.slots import (
    ExtractedFact,
    TemporalStatus,
    extract_fact_slots,
    extract_temporal_status,
    extract_direct_correction,
    extract_hedged_correction,
    detect_correction_type,
    create_simple_fact,
    names_are_related,
    names_look_equivalent,
    is_explicit_name_declaration_text,
    is_question,
)


class TestFactSlotExtraction:
    def test_extract_location(self):
        facts = extract_fact_slots("I live in Seattle, Washington.")
        assert "location" in facts
        assert "seattle" in facts["location"].normalized

    def test_extract_employer(self):
        facts = extract_fact_slots("I work at Microsoft as a senior developer.")
        assert "employer" in facts
        assert "microsoft" in facts["employer"].normalized

    def test_extract_age(self):
        facts = extract_fact_slots("I'm 31 years old")
        assert "age" in facts
        assert facts["age"].value == 31

    def test_name_extraction_disabled_by_default(self):
        # Conversation-based name extraction is intentionally disabled
        # to avoid false positives like "I'm building something".
        # Names should be set via structured facts instead.
        facts = extract_fact_slots("call me Nick")
        assert "name" not in facts  # disabled by design

    def test_name_via_structured_fact(self):
        # Structured FACT: syntax always works for names
        facts = extract_fact_slots("FACT: name = Nick Block")
        assert "name" in facts
        assert "nick block" in facts["name"].normalized

    def test_extract_programming_years(self):
        facts = extract_fact_slots("I've been programming for 10 years")
        assert "programming_years" in facts
        assert facts["programming_years"].value == 10

    def test_extract_favorite_color(self):
        facts = extract_fact_slots("My favorite color is orange.")
        assert "favorite_color" in facts
        assert "orange" in facts["favorite_color"].normalized

    def test_extract_structured_fact(self):
        facts = extract_fact_slots("FACT: name = Nick")
        assert "name" in facts
        assert "nick" in facts["name"].normalized

    def test_empty_text_returns_empty(self):
        assert extract_fact_slots("") == {}
        assert extract_fact_slots("   ") == {}

    def test_no_false_positive_on_emotion(self):
        facts = extract_fact_slots("I'm frustrated with this bug")
        assert "name" not in facts

    def test_no_false_positive_on_activity(self):
        facts = extract_fact_slots("I'm working on a project")
        assert "name" not in facts

    def test_extract_pet(self):
        facts = extract_fact_slots("I have a golden retriever named Murphy")
        assert "pet" in facts
        assert "pet_name" in facts

    def test_self_employed(self):
        facts = extract_fact_slots("I work for myself")
        assert "employer" in facts
        assert "self-employed" in facts["employer"].normalized


class TestTemporalStatus:
    def test_past_detection(self):
        status, _ = extract_temporal_status("I used to work at Google")
        assert status == TemporalStatus.PAST

    def test_active_default(self):
        status, _ = extract_temporal_status("I work at Google")
        assert status == TemporalStatus.ACTIVE

    def test_future_detection(self):
        status, _ = extract_temporal_status("I will start at Google next month")
        assert status == TemporalStatus.FUTURE

    def test_potential_detection(self):
        status, _ = extract_temporal_status("I might take the job at Google")
        assert status == TemporalStatus.POTENTIAL

    def test_period_extraction(self):
        _, period = extract_temporal_status("I worked there from 2020 to 2024")
        assert period == "2020-2024"


class TestCorrectionDetection:
    def test_direct_correction(self):
        result = extract_direct_correction("I'm actually 34, not 32")
        assert result is not None
        assert result[0] == "34"

    def test_hedged_correction(self):
        result = extract_hedged_correction("I said 10 years but it's closer to 12")
        assert result is not None

    def test_detect_correction_type_direct(self):
        result = detect_correction_type("I'm actually 34, not 32")
        assert result is not None
        assert result[0] == "direct_correction"

    def test_no_correction_in_normal_text(self):
        assert detect_correction_type("I live in Seattle") is None


class TestNameHelpers:
    def test_names_are_related_nickname(self):
        assert names_are_related("Bob", "Robert")
        assert names_are_related("Nick", "Nicholas")

    def test_names_are_related_substring(self):
        assert names_are_related("Alex", "Alex Chen")

    def test_names_not_related(self):
        assert not names_are_related("Alice", "Bob")

    def test_names_look_equivalent_case(self):
        assert names_look_equivalent("nick", "Nick")
        assert names_look_equivalent("Nick Block", "nick block")

    def test_names_look_equivalent_prefix(self):
        assert names_look_equivalent("Nick", "Nick Block")

    def test_is_explicit_name_declaration(self):
        assert is_explicit_name_declaration_text("my name is Nick")
        assert is_explicit_name_declaration_text("call me Nick")
        assert not is_explicit_name_declaration_text("the weather is nice")

    def test_is_question(self):
        assert is_question("What is your name?")
        assert is_question("How do I do this")
        assert not is_question("I live in Seattle")


class TestCreateSimpleFact:
    def test_creates_fact(self):
        fact = create_simple_fact("Seattle")
        assert fact.value == "Seattle"
        assert fact.normalized == "seattle"
        assert fact.temporal_status == TemporalStatus.ACTIVE
