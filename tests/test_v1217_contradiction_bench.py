"""v0.12.17: contradiction-detection bench (structural vs LLM judge).

The bench reproduces the README's "structure beats semantics" claim
on the existing fidelity_corpus.json. Tests cover the parts that
don't require an API key — the accuracy math, the per-category
aggregation, the disagreement extraction, and the markdown rendering
shape.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("mcp")

from bench.run_contradiction_bench import (
    SHOULD_CONTRADICT,
    aggregate,
    grade_case,
    render_markdown,
)


def _case(id_, category, claim="X", memory="Y"):
    return {
        "id": id_,
        "category": category,
        "claim": claim,
        "substrate": [{"text": memory, "trust": 0.9, "source": "user"}],
    }


class TestGroundTruthMap:
    def test_factual_contradiction_is_positive(self):
        assert SHOULD_CONTRADICT["factual_contradiction"] is True

    def test_mutex_contradiction_is_positive(self):
        assert SHOULD_CONTRADICT["mutex_contradiction"] is True

    def test_false_positive_guard_is_negative(self):
        assert SHOULD_CONTRADICT["false_positive_guard"] is False

    def test_no_issue_grounded_is_negative(self):
        assert SHOULD_CONTRADICT["no_issue_grounded"] is False

    def test_methodological_overclaim_excluded(self):
        # methodological is a different channel — None means "out of scope."
        assert SHOULD_CONTRADICT["methodological_overclaim"] is None


class TestGradeCase:
    def test_structural_correct_when_predicts_match(self):
        graded = grade_case(
            _case("f1", "factual_contradiction"),
            structural=True, llm=None,
        )
        assert graded["expected"] is True
        assert graded["structural_predicted"] is True
        assert graded["structural_correct"] is True
        assert graded["llm_correct"] is None  # skipped

    def test_structural_wrong_when_predicts_diverge(self):
        graded = grade_case(
            _case("g1", "no_issue_grounded"),
            structural=True, llm=None,
        )
        # Expected False (this should NOT contradict), but predicted True.
        assert graded["expected"] is False
        assert graded["structural_correct"] is False

    def test_disagree_flag_set_when_methods_diverge(self):
        graded = grade_case(
            _case("c1", "mutex_contradiction"),
            structural=True, llm=False,
        )
        assert graded["disagree"] is True

    def test_no_disagree_when_both_agree(self):
        graded = grade_case(
            _case("c1", "mutex_contradiction"),
            structural=True, llm=True,
        )
        assert graded["disagree"] is False

    def test_excluded_category_has_none_correctness(self):
        graded = grade_case(
            _case("m1", "methodological_overclaim"),
            structural=True, llm=True,
        )
        assert graded["expected"] is None
        assert graded["structural_correct"] is None


class TestAggregate:
    def test_in_scope_count_excludes_methodological(self):
        graded = [
            grade_case(_case("f1", "factual_contradiction"), True, None),
            grade_case(_case("f2", "factual_contradiction"), True, None),
            grade_case(_case("m1", "methodological_overclaim"), True, None),
        ]
        agg = aggregate(graded)
        assert agg["n_in_scope"] == 2
        assert agg["n_excluded"] == 1

    def test_structural_accuracy_correct(self):
        graded = [
            grade_case(_case("f1", "factual_contradiction"), True, None),   # correct
            grade_case(_case("f2", "factual_contradiction"), False, None),  # wrong
            grade_case(_case("g1", "no_issue_grounded"), False, None),      # correct
            grade_case(_case("g2", "no_issue_grounded"), True, None),       # wrong
        ]
        agg = aggregate(graded)
        assert agg["structural"]["correct"] == 2
        assert agg["structural"]["total"] == 4
        assert agg["structural"]["accuracy"] == pytest.approx(0.5)

    def test_llm_accuracy_only_counts_runs_that_completed(self):
        graded = [
            grade_case(_case("f1", "factual_contradiction"), True, True),   # llm correct
            grade_case(_case("f2", "factual_contradiction"), True, None),   # llm skipped
            grade_case(_case("g1", "no_issue_grounded"), False, True),      # llm wrong
        ]
        agg = aggregate(graded)
        # Only 2 cases had an LLM verdict; 1 was correct.
        assert agg["llm"]["total"] == 2
        assert agg["llm"]["correct"] == 1
        assert agg["llm"]["accuracy"] == pytest.approx(0.5)
        assert agg["llm"]["skipped"] == 1

    def test_disagreements_extracted(self):
        graded = [
            grade_case(_case("f1", "factual_contradiction"), True, True),    # agree
            grade_case(_case("f2", "factual_contradiction"), True, False),   # disagree
            grade_case(_case("g1", "no_issue_grounded"), False, True),       # disagree
        ]
        agg = aggregate(graded)
        assert len(agg["disagreements"]) == 2
        ids = {d["id"] for d in agg["disagreements"]}
        assert ids == {"f2", "g1"}


class TestRenderMarkdown:
    def test_skipped_llm_shows_skipped_message(self):
        graded = [grade_case(_case("f1", "factual_contradiction"), True, None)]
        agg = aggregate(graded)
        out = render_markdown(graded, agg)
        assert "ANTHROPIC_API_KEY" in out
        assert "SKIPPED" in out

    def test_full_run_shows_both_columns(self):
        graded = [
            grade_case(_case("f1", "factual_contradiction"), True, True),
            grade_case(_case("g1", "no_issue_grounded"), False, False),
        ]
        agg = aggregate(graded)
        out = render_markdown(graded, agg)
        # Both methods get a row.
        assert "Structural meter:" in out
        assert "LLM as judge:" in out
        assert "SKIPPED" not in out

    def test_disagreement_section_lists_cases(self):
        graded = [
            grade_case(_case("f1", "factual_contradiction"), True, False,),
            grade_case(_case("g1", "no_issue_grounded"), False, True),
        ]
        agg = aggregate(graded)
        out = render_markdown(graded, agg)
        assert "Cases where structural and LLM judge disagreed" in out
        assert "f1" in out
        assert "g1" in out

    def test_methodology_caveats_present(self):
        graded = [grade_case(_case("f1", "factual_contradiction"), True, None)]
        agg = aggregate(graded)
        out = render_markdown(graded, agg)
        assert "Methodology caveats" in out
        # Honest about the corpus being one the meter was tuned on.
        assert "tuned for" in out.lower() or "tuned on" in out.lower()
