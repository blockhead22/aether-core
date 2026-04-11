"""Tests for crt.epistemics — belief backpropagation engine."""

import time
from crt.epistemics import (
    EpistemicLoss,
    CorrectionEvent,
    BackpropResult,
    DomainVolatility,
    compute_backward_gradients,
    apply_trust_adjustments,
    flat_demotion,
)


class TestEpistemicLoss:
    def test_high_trust_high_loss(self):
        loss_fn = EpistemicLoss()
        event = CorrectionEvent(
            corrected_node_id="m1",
            trust_at_assertion=0.95,
            times_corrected=0,
            correction_source="user",
            time_since_assertion=0,
            domain="employer",
        )
        loss = loss_fn.compute(event)
        assert loss > 0.5, f"High-trust correction should produce high loss, got {loss}"

    def test_low_trust_low_loss(self):
        loss_fn = EpistemicLoss()
        event = CorrectionEvent(
            corrected_node_id="m1",
            trust_at_assertion=0.1,
            times_corrected=0,
            correction_source="inference",
            time_since_assertion=0,
            domain="color",
        )
        loss = loss_fn.compute(event)
        assert loss < 0.1, f"Low-trust inference correction should produce low loss, got {loss}"

    def test_repeated_corrections_increase_loss(self):
        loss_fn = EpistemicLoss()
        base = CorrectionEvent("m1", 0.7, 0, "user", 0, "name")
        repeated = CorrectionEvent("m1", 0.7, 5, "user", 0, "name")
        assert loss_fn.compute(repeated) > loss_fn.compute(base)

    def test_user_source_highest_weight(self):
        loss_fn = EpistemicLoss()
        user = CorrectionEvent("m1", 0.7, 0, "user", 100, "x")
        system = CorrectionEvent("m1", 0.7, 0, "system", 100, "x")
        inference = CorrectionEvent("m1", 0.7, 0, "inference", 100, "x")
        assert loss_fn.compute(user) > loss_fn.compute(system) > loss_fn.compute(inference)

    def test_old_errors_cost_more(self):
        loss_fn = EpistemicLoss(time_decay_halflife=3600)
        recent = CorrectionEvent("m1", 0.7, 0, "user", 100, "x")
        old = CorrectionEvent("m1", 0.7, 0, "user", 7200, "x")
        assert loss_fn.compute(old) > loss_fn.compute(recent)


class TestDomainVolatility:
    def test_no_history_zero_volatility(self):
        dv = DomainVolatility()
        assert dv.get_volatility("unknown") == 0.0

    def test_all_corrections_high_volatility(self):
        dv = DomainVolatility()
        now = time.time()
        for i in range(10):
            dv.record_correction("color", now - i * 60)
        vol = dv.get_volatility("color", now)
        assert vol > 0.9

    def test_no_corrections_low_volatility(self):
        dv = DomainVolatility()
        now = time.time()
        for i in range(10):
            dv.record_assertion("name", now - i * 60)
        vol = dv.get_volatility("name", now)
        assert vol == 0.0

    def test_learning_rate_scales_with_volatility(self):
        dv = DomainVolatility()
        now = time.time()
        # Seed high volatility
        dv.seed_history("volatile", assertions=5, corrections=5, base_time=now)
        dv.seed_history("stable", assertions=10, corrections=0, base_time=now)
        lr_volatile = dv.get_learning_rate("volatile", now=now)
        lr_stable = dv.get_learning_rate("stable", now=now)
        assert lr_volatile > lr_stable

    def test_seed_history(self):
        dv = DomainVolatility()
        now = time.time()
        dv.seed_history("test", assertions=8, corrections=2, base_time=now)
        vol = dv.get_volatility("test", now)
        assert 0.0 < vol < 1.0


class TestBackwardGradients:
    def _make_graph(self):
        try:
            import networkx as nx
        except ImportError:
            return None
        g = nx.DiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", weight=0.8, edge_type="SUPPORTS")
        g.add_edge("B", "C", weight=0.6, edge_type="SUPPORTS")
        return g

    def test_gradients_propagate_backward(self):
        g = self._make_graph()
        if g is None:
            return  # skip if no networkx
        result = compute_backward_gradients(
            g, corrected_node="C", loss=1.0,
            learning_rates={"A": 0.1, "B": 0.1, "C": 0.1},
        )
        assert "C" in result.gradients
        assert "B" in result.gradients
        assert result.gradients["B"] < result.gradients["C"]
        assert result.depth >= 1

    def test_contradicts_edges_blocked(self):
        try:
            import networkx as nx
        except ImportError:
            return
        g = nx.DiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=0.9, edge_type="CONTRADICTS")
        result = compute_backward_gradients(
            g, corrected_node="B", loss=1.0,
            learning_rates={"A": 0.1},
        )
        assert "A" not in result.trust_adjustments


class TestTrustAdjustments:
    def test_apply_adjustments(self):
        scores = {"A": 0.8, "B": 0.6, "C": 0.9}
        adjustments = {"A": -0.2, "B": -0.1}
        new = apply_trust_adjustments(scores, adjustments)
        assert abs(new["A"] - 0.6) < 0.001
        assert abs(new["B"] - 0.5) < 0.001
        assert new["C"] == 0.9  # untouched

    def test_clamps_to_bounds(self):
        scores = {"A": 0.05}
        adjustments = {"A": -0.5}
        new = apply_trust_adjustments(scores, adjustments)
        assert new["A"] == 0.0

    def test_flat_demotion(self):
        assert abs(flat_demotion(0.8) - 0.32) < 0.001
        assert abs(flat_demotion(1.0, 0.5) - 0.5) < 0.001
