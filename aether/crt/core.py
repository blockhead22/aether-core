"""
CRT Core - Cognitive-Reflective Transformer Mathematical Framework

Implements:
- Trust vs Confidence separation
- Drift detection and measurement
- Belief-weighted retrieval scoring
- SSE mode selection (Lossless/Cogni/Hybrid)
- Trust evolution equations
- Reconstruction constraints

Philosophy:
- Memory first (coherence over time)
- Honesty over performance (contradictions are signals, not bugs)
- Belief evolves slower than speech
- "The mouth must never outweigh the self"
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from difflib import SequenceMatcher
import math
import re

ANTONYMS = {
    "love": {"hate", "dislike"},
    "like": {"dislike", "hate"},
    "hate": {"love", "like"},
    "dislike": {"like", "love"},
    "prefer": {"hate", "dislike"},
    "enjoy": {"hate", "dislike"},
    "good": {"bad", "terrible", "awful"},
    "bad": {"good", "great"},
    "always": {"never"},
    "never": {"always"},
}

# Transient state cues (mood/energy/temporary condition).
_TRANSIENT_STATE_WORDS = {
    "tired", "exhausted", "fatigued", "sleepy", "burned out",
    "sad", "down", "depressed", "depression", "anxious", "anxiety",
    "stressed", "overwhelmed", "okay", "ok", "fine", "good", "bad",
    "sick", "ill", "hurt", "hurting", "in pain", "recovering",
    "lonely", "upset", "angry", "frustrated", "confused",
}

_MOOD_SLOTS = {
    "mood", "feeling", "emotion", "emotions", "status",
    "user.mood", "user.feeling", "user.emotion",
}


def _is_transient_state_value(value: Optional[str]) -> bool:
    if value is None:
        return False
    low = str(value).lower().strip()
    if not low:
        return False
    for word in _TRANSIENT_STATE_WORDS:
        needle = str(word).lower().strip()
        if not needle:
            continue
        if " " in needle:
            if re.search(rf"(^|\W){re.escape(needle)}($|\W)", low):
                return True
        else:
            if re.search(rf"\b{re.escape(needle)}\b", low):
                return True
    return False


# ---------------------------------------------------------------------------
# CRT Math Upgrade #2: Beta Distribution Trust
# ---------------------------------------------------------------------------

@dataclass
class BetaTrust:
    """Beta distribution trust representation.

    Instead of scalar trust, model trust as Beta(alpha, beta_param).
    - mean = alpha / (alpha + beta_param) = trust score
    - variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)) = uncertainty
    - confidence = alpha + beta_param (total pseudo-observations)

    New memories start at Beta(2, 2) → mean=0.5, high variance.
    Confirmed memories accumulate alpha → high mean, low variance.
    """
    alpha: float = 2.0
    beta_param: float = 2.0

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta_param
        return self.alpha / total if total > 0 else 0.5

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta_param
        total = a + b
        if total <= 0:
            return 0.25
        return (a * b) / (total * total * (total + 1.0))

    @property
    def confidence(self) -> float:
        """Total pseudo-observations. Higher = more certain."""
        return self.alpha + self.beta_param

    def update_aligned(self, weight: float = 1.0) -> None:
        """Conjugate Bayesian update for aligned evidence."""
        self.alpha += weight
        logger.info(
            "[CRT_MATH] beta_trust update_aligned: α=%.2f, β=%.2f, mean=%.3f, var=%.4f, conf=%.1f",
            self.alpha, self.beta_param, self.mean, self.variance, self.confidence,
        )

    def update_contradicted(self, weight: float = 1.0) -> None:
        """Conjugate Bayesian update for contradicting evidence."""
        self.beta_param += weight
        logger.info(
            "[CRT_MATH] beta_trust update_contradicted: α=%.2f, β=%.2f, mean=%.3f, var=%.4f, conf=%.1f",
            self.alpha, self.beta_param, self.mean, self.variance, self.confidence,
        )


class SSEMode(Enum):
    """SSE compression modes based on significance."""
    LOSSLESS = "L"   # Identity-critical, contradiction-heavy
    COGNI = "C"      # Fast sketch, "what it felt like"
    HYBRID = "H"     # Adaptive mix


class MemorySource(Enum):
    """Source of memory item."""
    USER = "user"
    SYSTEM = "system"
    FALLBACK = "fallback"
    EXTERNAL = "external"
    REFLECTION = "reflection"
    SELF_REFLECTION = "self_reflection"
    LLM_OUTPUT = "llm_output"
    MCP_CLIENT = "mcp_client"


@dataclass
class CRTConfig:
    """CRT system configuration parameters."""
    
    # Trust evolution rates
    eta_pos: float = 0.1          # Trust increase rate for aligned memories
    eta_reinforce: float = 0.05   # Reinforcement rate for validated memories
    eta_neg: float = 0.15         # Trust decrease rate for contradictions
    
    # Thresholds
    theta_align: float = 0.15     # Drift threshold for alignment
    theta_contra: float = 0.28    # Drift threshold for contradiction (tuned from stress test analysis)
    theta_min: float = 0.30       # Minimum drift for confidence-based contradiction
    theta_drop: float = 0.30      # Confidence drop threshold
    theta_fallback: float = 0.42  # Drift threshold for fallback contradictions
    
    # Reconstruction gates
    theta_intent: float = 0.5     # Intent alignment gate (lowered from 0.7 to reduce gate failures)
    theta_mem: float = 0.38       # Memory alignment gate (lowered from 0.45 to allow detailed explanatory responses)
    
    # Reflection triggers
    theta_reflect: float = 0.5    # Volatility threshold for reflection
    
    # Retrieval
    lambda_time: float = 86400.0  # Time constant (1 day in seconds)
    alpha_trust: float = 0.7      # Trust weight in retrieval (vs confidence)
    
    # Trust bounds
    tau_base: float = 0.7         # Base trust for new memories (FIX: raised to avoid uncertainty trigger)
    tau_fallback_cap: float = 0.3 # Max trust for fallback speech
    tau_train_min: float = 0.6    # Min trust for weight updates
    
    # SSE mode selection
    T_L: float = 0.7              # Lossless threshold
    T_C: float = 0.3              # Cogni threshold
    
    # SSE significance weights
    w_emotion: float = 0.2
    w_novelty: float = 0.25
    w_user_mark: float = 0.3
    w_contradiction: float = 0.15
    w_future: float = 0.1
    
    # Volatility weights
    beta_drift: float = 0.3
    beta_alignment: float = 0.25
    beta_contradiction: float = 0.3
    beta_fallback: float = 0.15

    # --- CRT Math Upgrade #1: Learnable gain/decay via domain volatility ---
    vol_beta: float = 0.6     # Gain dampening coefficient (high vol = slower trust gain)
    vol_gamma: float = 0.8    # Decay amplification coefficient (high vol = faster trust loss)

    # --- CRT Math Upgrade #3: Unified gate equation ---
    gate_theta_base: float = 0.3   # Base relevance threshold
    gate_lambda: float = 0.5       # Drift penalty on threshold
    gate_gamma: float = 0.15       # Depth attenuation factor
    
    @staticmethod
    def load_from_calibration(
        calibration_path: str = "artifacts/calibrated_thresholds.json"
    ) -> 'CRTConfig':
        """
        Load CRTConfig with calibrated thresholds from file.
        
        Falls back to default values if calibration file is missing or invalid.
        
        Args:
            calibration_path: Path to calibrated thresholds JSON
            
        Returns:
            CRTConfig with calibrated or default thresholds
        """
        import json
        import logging
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        config = CRTConfig()  # Start with defaults
        
        try:
            threshold_file = Path(calibration_path)
            if not threshold_file.exists():
                logger.info(
                    f"[CRT_CONFIG] Calibration file not found: {calibration_path}, "
                    f"using defaults"
                )
                return config
            
            with open(threshold_file) as f:
                data = json.load(f)
            
            # Update thresholds based on calibrated values
            # Map calibrated zones to CRT thresholds
            # Note: Calibrated thresholds are similarity scores (0-1), where high=similar
            # CRT drift thresholds work inversely: high drift = low similarity
            # Therefore we invert: drift = 1 - similarity
            if "green_zone" in data:
                # Use green_zone as the threshold for high-confidence alignment
                config.theta_align = 1.0 - data["green_zone"]  # High similarity → low drift
                logger.info(
                    f"[CRT_CONFIG] Loaded calibrated theta_align: {config.theta_align:.3f} "
                    f"(from green_zone: {data['green_zone']:.3f})"
                )
            
            if "red_zone" in data:
                # Use red_zone as the threshold for contradictions
                config.theta_contra = 1.0 - data["red_zone"]  # Low similarity → high drift
                logger.info(
                    f"[CRT_CONFIG] Loaded calibrated theta_contra: {config.theta_contra:.3f} "
                    f"(from red_zone: {data['red_zone']:.3f})"
                )
            
            if "yellow_zone" in data:
                # Use yellow_zone for fallback threshold
                config.theta_fallback = 1.0 - data["yellow_zone"]
                logger.info(
                    f"[CRT_CONFIG] Loaded calibrated theta_fallback: {config.theta_fallback:.3f} "
                    f"(from yellow_zone: {data['yellow_zone']:.3f})"
                )
            
            logger.info(f"[CRT_CONFIG] Successfully loaded calibrated thresholds from {calibration_path}")
            
        except Exception as e:
            logger.warning(
                f"[CRT_CONFIG] Failed to load calibration from {calibration_path}: {e}, "
                f"using defaults"
            )
        
        return config


class CRTMath:
    """
    Core CRT mathematical operations.
    
    Implements:
    1. Similarity and drift measurement
    2. Trust-weighted retrieval scoring
    3. SSE mode selection
    4. Trust evolution
    5. Reconstruction constraints
    6. Contradiction detection
    7. Reflection triggers
    """
    
    def __init__(self, config: Optional[CRTConfig] = None):
        """Initialize with configuration."""
        self.config = config or CRTConfig()
    
    # ========================================================================
    # 1. Similarity and Drift
    # ========================================================================
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.
        
        sim(a,b) = (a·b) / (||a|| ||b||)
        """
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        # Check dimension compatibility
        if len(a) != len(b):
            # Dimension mismatch - likely from old data
            return 0.0
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def novelty(self, z_new: np.ndarray, memory_vectors: List[np.ndarray]) -> float:
        """
        Novelty of new input relative to stored memory.
        
        novelty(x) = 1 - max_i sim(z_new, z_i)
        """
        if not memory_vectors:
            return 1.0
        
        max_sim = max(self.similarity(z_new, z_mem) for z_mem in memory_vectors)
        return 1.0 - max_sim
    
    def drift_meaning(self, z_new: np.ndarray, z_prior: np.ndarray) -> float:
        """
        Meaning drift between new output and prior belief.
        
        D_mean = 1 - sim(z_new, z_prior)
        """
        return 1.0 - self.similarity(z_new, z_prior)
    
    # ========================================================================
    # 2. Trust-Weighted Retrieval Scoring
    # ========================================================================
    
    def recency_weight(self, t_memory: float, t_now: float) -> float:
        """
        Recency weighting with exponential decay.
        
        ρ_i = exp(-(t_now - t_i) / λ)
        """
        delta_t = t_now - t_memory
        return math.exp(-delta_t / self.config.lambda_time)
    
    def belief_weight(self, trust: float, confidence: float) -> float:
        """
        Combined belief weight (trust + confidence).
        
        w_i = α·τ_i + (1-α)·c_i
        """
        alpha = self.config.alpha_trust
        return alpha * trust + (1 - alpha) * confidence
    
    def retrieval_score(
        self,
        similarity: float,
        recency: float,
        belief: float
    ) -> float:
        """
        Final retrieval score.
        
        R_i = s_i · ρ_i · w_i
        """
        return similarity * recency * belief
    
    def compute_retrieval_scores(
        self,
        query_vector: np.ndarray,
        memories: List[Dict[str, Any]],
        t_now: float
    ) -> List[Tuple[int, float]]:
        """
        Compute retrieval scores for all memories.
        
        Returns list of (index, score) tuples sorted by score descending.
        """
        scores = []
        
        for i, mem in enumerate(memories):
            # Similarity
            s_i = self.similarity(query_vector, mem['vector'])
            
            # Recency
            rho_i = self.recency_weight(mem['timestamp'], t_now)
            
            # Belief weight
            w_i = self.belief_weight(mem['trust'], mem['confidence'])
            
            # Final score
            R_i = self.retrieval_score(s_i, rho_i, w_i)
            
            scores.append((i, R_i))
        
        # Sort by score descending
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    # ========================================================================
    # 3. SSE Mode Selection
    # ========================================================================
    
    def compute_significance(
        self,
        emotion_intensity: float,
        novelty: float,
        user_marked: float,
        contradiction_signal: float,
        future_relevance: float
    ) -> float:
        """
        Compute significance score for SSE mode selection.
        
        S = w1·e + w2·n + w3·u + w4·k + w5·f
        """
        cfg = self.config
        return (
            cfg.w_emotion * emotion_intensity +
            cfg.w_novelty * novelty +
            cfg.w_user_mark * user_marked +
            cfg.w_contradiction * contradiction_signal +
            cfg.w_future * future_relevance
        )
    
    def select_sse_mode(self, significance: float) -> SSEMode:
        """
        Select SSE compression mode based on significance.
        
        if S ≥ T_L  → SSE-L (lossless)
        if S ≤ T_C  → SSE-C (cogni/sketch)
        else        → SSE-H (hybrid)
        """
        if significance >= self.config.T_L:
            return SSEMode.LOSSLESS
        elif significance <= self.config.T_C:
            return SSEMode.COGNI
        else:
            return SSEMode.HYBRID
    
    # ========================================================================
    # 4. Trust Evolution
    # ========================================================================
    
    def evolve_trust_aligned(
        self,
        tau_current: float,
        drift: float,
        volatility: float = 0.0,
        trust_alpha: float = 0.0,
        trust_beta: float = 0.0,
    ) -> float:
        """
        Trust evolution for aligned memories (low drift).

        Upgrade #1: gain = eta_pos * (1 - vol_beta * volatility) * (1 - drift)
        Upgrade #2: If beta params provided, conjugate update alpha += weight*(1-drift)
        """
        gain_mod = 1.0 - self.config.vol_beta * volatility
        base_delta = self.config.eta_pos * gain_mod * (1.0 - drift)

        # Upgrade #2: Beta conjugate update
        if trust_alpha > 0 and trust_beta > 0:
            weight = base_delta * 10.0  # scale to Beta pseudo-count
            new_alpha = trust_alpha + weight
            tau_new = new_alpha / (new_alpha + trust_beta)
            logger.info(
                "[CRT_MATH] evolve_aligned(beta): trust %.3f->%.3f, drift=%.3f, vol_d=%.3f, "
                "α=%.2f->%.2f, β=%.2f",
                tau_current, tau_new, drift, volatility,
                trust_alpha, new_alpha, trust_beta,
            )
            return float(np.clip(tau_new, 0.0, 1.0))

        tau_new = tau_current + base_delta
        logger.info(
            "[CRT_MATH] evolve_aligned: trust %.3f->%.3f, drift=%.3f, vol_d=%.3f, gain_mod=%.3f",
            tau_current, float(np.clip(tau_new, 0.0, 1.0)), drift, volatility, gain_mod,
        )
        return float(np.clip(tau_new, 0.0, 1.0))
    
    def evolve_trust_reinforced(
        self,
        tau_current: float,
        drift: float,
        volatility: float = 0.0,
    ) -> float:
        """
        Trust reinforcement for validated memories.

        Upgrade #1: gain = eta_reinforce * (1 - vol_beta * volatility) * (1 - drift)
        """
        gain_mod = 1.0 - self.config.vol_beta * volatility
        tau_new = tau_current + self.config.eta_reinforce * gain_mod * (1.0 - drift)
        logger.info(
            "[CRT_MATH] evolve_reinforced: trust %.3f->%.3f, drift=%.3f, vol_d=%.3f",
            tau_current, float(np.clip(tau_new, 0.0, 1.0)), drift, volatility,
        )
        return float(np.clip(tau_new, 0.0, 1.0))
    
    def evolve_trust_contradicted(
        self,
        tau_current: float,
        drift: float,
        volatility: float = 0.0,
        trust_alpha: float = 0.0,
        trust_beta: float = 0.0,
    ) -> float:
        """
        Trust degradation for contradicted memories.

        Upgrade #1: decay = eta_neg * (1 + vol_gamma * volatility) * drift
        Upgrade #2: If beta params provided, conjugate update beta += weight*drift
        """
        decay_mod = 1.0 + self.config.vol_gamma * volatility
        base_decay = self.config.eta_neg * decay_mod * drift

        # Upgrade #2: Beta conjugate update
        if trust_alpha > 0 and trust_beta > 0:
            weight = base_decay * 10.0
            new_beta = trust_beta + weight
            tau_new = trust_alpha / (trust_alpha + new_beta)
            logger.info(
                "[CRT_MATH] evolve_contradicted(beta): trust %.3f->%.3f, drift=%.3f, vol_d=%.3f, "
                "α=%.2f, β=%.2f->%.2f",
                tau_current, tau_new, drift, volatility,
                trust_alpha, trust_beta, new_beta,
            )
            return float(np.clip(tau_new, 0.0, 1.0))

        tau_new = tau_current * (1.0 - base_decay)
        logger.info(
            "[CRT_MATH] evolve_contradicted: trust %.3f->%.3f, drift=%.3f, vol_d=%.3f, decay_mod=%.3f",
            tau_current, float(np.clip(tau_new, 0.0, 1.0)), drift, volatility, decay_mod,
        )
        return float(np.clip(tau_new, 0.0, 1.0))
    
    def cap_fallback_trust(self, tau: float, source: MemorySource) -> float:
        """
        Cap trust for fallback sources.
        
        if src == fallback:
            τ = min(τ, τ_fallback_cap)
        """
        if source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT}:
            return min(tau, self.config.tau_fallback_cap)
        return tau
    
    # ========================================================================
    # 4b. Unified Gate Equation (Upgrade #3)
    # ========================================================================

    def unified_gate(
        self,
        relevance: float,
        drift: float,
        depth: int = 0,
    ) -> float:
        """Unified gate: gate(v) = I(R > θ + λ*drift) * exp(-γ*depth).

        Collapses salience gate + trust gate + marble drift check into one
        continuous-valued function. Returns 0.0-1.0 gate score.
        """
        cfg = self.config
        theta_eff = cfg.gate_theta_base + cfg.gate_lambda * drift
        passed = relevance > theta_eff
        depth_atten = float(np.exp(-cfg.gate_gamma * depth))
        gate_val = (1.0 if passed else 0.0) * depth_atten
        logger.info(
            "[CRT_MATH] unified_gate: R=%.3f, θ_eff=%.3f (base=%.2f+λ*D=%.3f), "
            "depth_atten=%.3f, gate=%.3f, pass=%s",
            relevance, theta_eff, cfg.gate_theta_base, cfg.gate_lambda * drift,
            depth_atten, gate_val, passed,
        )
        return gate_val

    # ========================================================================
    # 5. Reconstruction Constraints (Holden Gates)
    # ========================================================================
    
    def intent_alignment(
        self,
        input_intent: np.ndarray,
        output_intent: np.ndarray
    ) -> float:
        """
        Intent alignment score.
        
        A_intent = sim(I(x), I(ŷ))
        """
        return self.similarity(input_intent, output_intent)
    
    def memory_alignment(
        self,
        output_vector: np.ndarray,
        retrieved_memories: List[Dict[str, Any]],
        retrieval_scores: List[float],
        output_text: str = ""
    ) -> float:
        """
        Memory alignment score (weighted by retrieval strength).
        
        A_mem = Σ_i (softmax(R_i) · sim(E(ŷ), z_i))
        
        Special handling: If output is a short substring of any memory,
        boost alignment to reward fact extraction.
        """
        if not retrieved_memories:
            return 0.0
        
        # Check for short fact extraction (answer is substring of memory)
        if output_text and len(output_text) < 50:
            output_lower = output_text.lower().strip()
            for mem in retrieved_memories[:3]:
                mem_text = mem.get('text', '').lower() if isinstance(mem.get('text'), str) else ''
                if output_lower and mem_text and output_lower in mem_text:
                    # Answer is extracted from memory - high alignment
                    return 0.95
        
        # Softmax over retrieval scores
        scores_array = np.array(retrieval_scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))
        weights = exp_scores / np.sum(exp_scores)
        
        # Weighted similarity
        alignment = 0.0
        for i, mem in enumerate(retrieved_memories):
            sim = self.similarity(output_vector, mem['vector'])
            alignment += weights[i] * sim
        
        return alignment
    
    def check_reconstruction_gates(
        self,
        intent_align: float,
        memory_align: float,
        has_grounding_issues: bool = False,
        has_contradiction_issues: bool = False,
        has_extraction_issues: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if reconstruction passes gates (legacy v1).
        
        DEPRECATED: Use check_reconstruction_gates_v2 with response_type awareness.
        
        Accept if:
            A_intent ≥ θ_intent AND A_mem ≥ θ_mem
        
        Returns (passed, reason)
        """
        # Priority order: specific fails before general alignment fails
        if has_grounding_issues:
            return False, "grounding_fail"
        
        if has_contradiction_issues:
            return False, "contradiction_fail"
        
        if has_extraction_issues:
            return False, "extraction_fail"
        
        if intent_align < self.config.theta_intent:
            return False, f"intent_fail (align={intent_align:.3f} < {self.config.theta_intent})"
        
        if memory_align < self.config.theta_mem:
            return False, f"memory_fail (align={memory_align:.3f} < {self.config.theta_mem})"
        
        return True, "gates_passed"
    
    def check_reconstruction_gates_v2(
        self,
        intent_align: float,
        memory_align: float,
        response_type: str,
        grounding_score: float = 1.0,
        contradiction_severity: str = "none",
        blindspot_gate_boost: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Gradient gates with response-type awareness (v2).

        Key improvements over v1:
        1. Different thresholds for factual/explanatory/conversational
        2. Grounding score (0-1) instead of binary check
        3. Contradiction severity levels (blocking/note/none)
        4. Adaptive thresholds via blindspot_gate_boost from reflection loop

        Response types:
        - factual: Strict gates for factual claims (What is my X?)
        - explanatory: Relaxed gates for synthesis/explanation (How/Why questions)
        - conversational: Minimal gates for chat/acknowledgment

        Args:
            intent_align: Intent alignment score (0-1)
            memory_align: Memory alignment score (0-1)
            response_type: "factual" | "explanatory" | "conversational"
            grounding_score: How well grounded in memory (0-1)
            contradiction_severity: "blocking" | "note" | "none"
            blindspot_gate_boost: Additive threshold raise (0.0-0.15) from
                self-model behavioral directives. When the query touches a
                known blindspot domain, thresholds are raised so the system
                demands higher alignment before passing gates.

        Returns:
            (passed, reason)
        """
        # Clamp boost to safe range
        boost = max(0.0, min(blindspot_gate_boost, 0.15))
        boost_note = f" +boost={boost:.2f}" if boost > 0 else ""

        # Blocking contradictions always fail
        if contradiction_severity == "blocking":
            return False, "contradiction_fail"

        # Response-type specific thresholds (boosted by reflection awareness)
        if response_type == "factual":
            t_intent = 0.35 + boost
            t_memory = 0.35 + boost
            t_ground = 0.30 + boost
            if intent_align < t_intent:
                return False, f"factual_intent_fail (align={intent_align:.3f} < {t_intent:.2f}{boost_note})"
            if memory_align < t_memory:
                return False, f"factual_memory_fail (align={memory_align:.3f} < {t_memory:.2f}{boost_note})"
            if grounding_score < t_ground:
                return False, f"factual_grounding_fail (score={grounding_score:.3f} < {t_ground:.2f}{boost_note})"

        elif response_type == "explanatory":
            t_intent = 0.35 + boost
            t_memory = 0.18 + boost
            t_ground = 0.20 + boost
            if intent_align < t_intent:
                return False, f"explanatory_intent_fail (align={intent_align:.3f} < {t_intent:.2f}{boost_note})"
            if memory_align < t_memory:
                return False, f"explanatory_memory_fail (align={memory_align:.3f} < {t_memory:.2f}{boost_note})"
            if grounding_score < t_ground:
                return False, f"explanatory_grounding_fail (score={grounding_score:.3f} < {t_ground:.2f}{boost_note})"

        else:  # conversational
            t_intent = 0.3 + boost
            if intent_align < t_intent:
                return False, f"conversational_intent_fail (align={intent_align:.3f} < {t_intent:.2f}{boost_note})"
            # No memory/grounding requirements for conversational

        # Non-blocking contradictions pass but add metadata
        if contradiction_severity == "note":
            return True, "gates_passed_with_contradiction_note"

        return True, "gates_passed"
    
    # ========================================================================
    # 6. Contradiction Detection
    # ========================================================================
    
    def detect_contradiction(
        self,
        drift: float,
        confidence_new: float,
        confidence_prior: float,
        source: MemorySource,
        text_new: str = "",
        text_prior: str = "",
        slot: Optional[str] = None,
        value_new: Optional[str] = None,
        value_prior: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Detect if contradiction event should be triggered.

        Fast-path heuristics (order matters):
        0a. Entity swap (proper-noun slot values)
        0b. Negation (bidirectional + temporal-update filter)
        0c. Preference / boolean inversion (slot-aware)
        0d. Numerical / quantifier mismatch
        0e. Antonym polarity flip
        0f. Paraphrase tolerance (escape valve before drift rules)
        Semantic rules:
        1. D_mean > θ_contra
        2. (Δc > θ_drop AND D_mean > θ_min)
        3. (src == fallback AND D_mean > θ_fallback)

        Returns (is_contradiction, reason)
        """
        cfg = self.config

        # Rule 0a: Entity swap (same slot, different proper-noun values)
        entity_swap, entity_reason = self._detect_entity_swap(slot, value_new, value_prior, text_new, text_prior)
        if entity_swap:
            return True, entity_reason

        # Rule 0b: Negation contradiction (bidirectional + temporal filter)
        negation_detected, negation_reason = self._detect_negation_contradiction(text_new, text_prior)
        if negation_detected:
            return True, negation_reason

        # Rule 0c: Preference / boolean inversion (slot-aware)
        inversion_detected, inversion_reason = self._is_boolean_inversion(text_new, text_prior, slot)
        if inversion_detected:
            return True, inversion_reason

        # Rule 0d: Numerical / quantifier contradiction
        num_contra, num_reason = self._detect_numerical_contradiction(text_new, text_prior)
        if num_contra:
            return True, num_reason

        # Rule 0e: Antonym polarity flip
        antonym_contra, antonym_reason = self._detect_antonym_contradiction(text_new, text_prior)
        if antonym_contra:
            return True, antonym_reason

        # Rule 0f: Paraphrase tolerance (escape valve before drift rules)
        if text_new and text_prior and drift > 0.35:
            if self._is_likely_paraphrase(text_new, text_prior, drift):
                return False, f"Paraphrase detected (drift={drift:.3f}, not contradiction)"

        # Rule 1: High drift
        if drift > cfg.theta_contra:
            return True, f"High semantic drift: {drift:.3f} > {cfg.theta_contra}"

        # Rule 2: Confidence drop with moderate drift
        delta_c = confidence_prior - confidence_new
        if delta_c > cfg.theta_drop and drift > cfg.theta_min:
            return True, f"Significant confidence drop: Δc={delta_c:.3f}, drift={drift:.3f}"

        # Rule 3: Fallback source with drift
        if source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT} and drift > cfg.theta_fallback:
            return True, f"Fallback source drift: {drift:.3f} > {cfg.theta_fallback}"

        return False, "No contradiction"
    
    def _is_likely_paraphrase(self, text_new: str, text_prior: str, drift: float) -> bool:
        """Jaccard similarity on content words + key-element extraction."""
        if drift < 0.35 or drift > 0.60:
            return False

        def normalize_and_tokenize(text: str) -> set:
            cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
            tokens = cleaned.split()
            stopwords = {"a", "an", "the", "to", "in", "of", "on", "for", "with", "at",
                         "my", "your", "our", "their", "his", "her", "i", "me", "you", "we", "they"}
            return {t for t in tokens if t and t not in stopwords}

        words_new = normalize_and_tokenize(text_new)
        words_prior = normalize_and_tokenize(text_prior)

        if not words_new or not words_prior:
            return False

        intersection = len(words_new & words_prior)
        union = len(words_new | words_prior)
        jaccard = intersection / union if union > 0 else 0

        def extract_key_elements(text: str) -> set:
            numbers = set(re.findall(r'\d+(?:\.\d+)?', text))
            dates = set(re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
            proper = set(re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
            return numbers | dates | proper

        keys_new = extract_key_elements(text_new)
        keys_prior = extract_key_elements(text_prior)

        if (keys_new or keys_prior) and keys_new != keys_prior:
            return False

        key_overlap = len(keys_new & keys_prior) / max(len(keys_new | keys_prior), 1) if keys_new or keys_prior else 1.0

        return jaccard > 0.65 or key_overlap > 0.75

    def _detect_entity_swap(
        self,
        slot: Optional[str],
        value_new: Optional[str],
        value_prior: Optional[str],
        text_new: str = "",
        text_prior: str = "",
    ) -> Tuple[bool, str]:
        """
        Detect entity swap: same slot but conflicting proper-noun values.
        """
        if not slot or value_new is None or value_prior is None:
            return False, ""

        candidate_new = str(value_new).strip()
        candidate_prior = str(value_prior).strip()
        if not candidate_new or not candidate_prior:
            return False, ""

        candidate_new_norm = candidate_new.lower()
        candidate_prior_norm = candidate_prior.lower()

        # If normalized values match, not a swap
        if candidate_new_norm == candidate_prior_norm:
            return False, ""

        if not (self._looks_like_entity(candidate_new) and self._looks_like_entity(candidate_prior)):
            return False, ""

        # Avoid false positives for near-identical strings (e.g., nicknames)
        similarity = SequenceMatcher(None, candidate_new_norm, candidate_prior_norm).ratio()
        if similarity > 0.78:
            return False, ""

        return True, f"Entity swap detected for slot '{slot}': '{candidate_prior}' -> '{candidate_new}'"

    def _looks_like_entity(self, value: str) -> bool:
        """Heuristic to decide if a value is a named entity (proper noun or acronym)."""
        if not value:
            return False

        tokens = re.findall(r"[A-Za-z][\w.&'-]*", value)
        return any(token[0].isupper() or token.isupper() for token in tokens)

    def _is_boolean_inversion(self, text_new: str, text_prior: str, slot: Optional[str] = None) -> Tuple[bool, str]:
        """Detect preference/boolean inversions. Slot-aware: 'target changed' only fires for single-value slots."""
        if not text_new or not text_prior:
            return False, ""

        def extract_preferences(text: str) -> List[Tuple[str, str]]:
            patterns = [
                (r"\bprefer[s]?\s+(?P<obj>[^.;,!?:\n]+)", "prefer"),
                (r"\blike[s]?\s+(?P<obj>[^.;,!?:\n]+)", "like"),
                (r"\blove[s]?\s+(?P<obj>[^.;,!?:\n]+)", "like"),
                (r"\benjoy[s]?\s+(?P<obj>[^.;,!?:\n]+)", "like"),
                (r"\bdislike[s]?\s+(?P<obj>[^.;,!?:\n]+)", "dislike"),
                (r"\bhate[s]?\s+(?P<obj>[^.;,!?:\n]+)", "dislike"),
                (r"\bavoid[s]?\s+(?P<obj>[^.;,!?:\n]+)", "dislike"),
            ]
            preferences: List[Tuple[str, str]] = []
            lowered = text.lower()
            for pattern, label in patterns:
                for match in re.finditer(pattern, lowered):
                    obj = match.group("obj").strip()
                    obj = re.split(r"\b(but|however|though|although)\b", obj)[0].strip()
                    preferences.append((label, obj))
            return preferences

        def normalize_obj(obj: str) -> str:
            cleaned = re.sub(r"[^a-z0-9 ]+", " ", obj.lower())
            stopwords = {"a", "an", "the", "to", "in", "of", "on", "for", "with", "at", "my", "your", "our", "their", "his", "her"}
            return " ".join(w for w in cleaned.split() if w and w not in stopwords)

        def polarity(label: str) -> int:
            return 1 if label in {"prefer", "like"} else -1

        def objects_match(a: str, b: str) -> bool:
            if not a or not b:
                return False
            if a == b:
                return True
            ratio = SequenceMatcher(None, a, b).ratio()
            return ratio >= 0.75 or a in b or b in a

        prefs_new = extract_preferences(text_new)
        prefs_prior = extract_preferences(text_prior)

        if not prefs_new or not prefs_prior:
            return False, ""

        is_single_value_slot = slot and any(k in slot.lower() for k in ("favorite", "current", "primary", "best", "main"))

        for label_new, obj_new_raw in prefs_new:
            obj_new = normalize_obj(obj_new_raw)
            if not obj_new:
                continue
            pol_new = polarity(label_new)

            for label_prior, obj_prior_raw in prefs_prior:
                obj_prior = normalize_obj(obj_prior_raw)
                if not obj_prior:
                    continue
                pol_prior = polarity(label_prior)

                same_target = objects_match(obj_new, obj_prior)
                if same_target and pol_new != pol_prior:
                    return True, f"Preference polarity inversion on '{obj_new}'"

                if pol_new == pol_prior == 1 and not same_target and is_single_value_slot:
                    return True, f"Preference target changed in single-value slot '{slot}'"

        return False, ""

    def _detect_negation_contradiction(self, text_new: str, text_prior: str) -> Tuple[bool, str]:
        """Bidirectional negation detection with temporal-update suppression."""
        if not text_new or not text_prior:
            return False, ""

        t_new = text_new.lower()
        t_prior = text_prior.lower()

        negation_patterns = [
            r"(?:i\s+)?(?:don'?t|do\s+not|no\s+longer|not\s+anymore|never)\s+(\w+(?:\s+\w+){0,4})",
            r"(?:i\s+)?(?:stopped|quit|left|gave up|no longer)\s+(\w+(?:\s+\w+){0,4})",
            r"(?:i'?m|i am)\s+not\s+(\w+(?:\s+\w+){0,3})",
        ]

        temporal_update_markers = {"used to", "formerly", "previously", "before", "now", "these days"}

        def extract_negated_actions(text: str) -> set:
            actions = set()
            for pat in negation_patterns:
                for m in re.finditer(pat, text):
                    actions.add(m.group(1).strip())
            return actions

        def has_negation(text: str) -> bool:
            return any(re.search(p, text) for p in negation_patterns)

        def action_affirmed_in(action: str, text: str) -> bool:
            words = action.split()[:3]
            pattern = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
            return bool(re.search(pattern, text))

        def check_direction(negated_text: str, affirming_text: str, direction: str) -> Tuple[bool, str]:
            actions = extract_negated_actions(negated_text)
            if not actions:
                return False, ""
            for action in actions:
                if action_affirmed_in(action, affirming_text) and not has_negation(affirming_text):
                    if any(marker in t_new or marker in t_prior for marker in temporal_update_markers):
                        return False, ""
                    return True, f"Negation contradiction ({direction}): '{action}'"
            return False, ""

        found, reason = check_direction(t_new, t_prior, "new negates prior")
        if found:
            return True, reason
        return check_direction(t_prior, t_new, "prior negates new")

    def _detect_numerical_contradiction(self, text_new: str, text_prior: str) -> Tuple[bool, str]:
        """Catches differing numbers when surrounding text context overlaps.
        Limitation: may fire on log-line timestamps or similarity scores —
        callers processing structured output should pre-filter those.
        """
        nums_new = re.findall(r'\d+(?:\.\d+)?', text_new)
        nums_prior = re.findall(r'\d+(?:\.\d+)?', text_prior)
        if not nums_new or not nums_prior or nums_new == nums_prior:
            return False, ""

        words_new = set(re.sub(r"[^a-z0-9 ]", " ", text_new.lower()).split())
        words_prior = set(re.sub(r"[^a-z0-9 ]", " ", text_prior.lower()).split())
        stopwords = {"a", "an", "the", "to", "in", "of", "on", "for", "with", "at", "i", "me"}
        content_new = words_new - stopwords
        content_prior = words_prior - stopwords
        overlap = len(content_new & content_prior) / max(len(content_new | content_prior), 1)

        if overlap > 0.5:
            return True, f"Numerical contradiction: {nums_prior} vs {nums_new}"
        return False, ""

    def _detect_antonym_contradiction(self, text_new: str, text_prior: str) -> Tuple[bool, str]:
        """Catches classic polarity flips using the ANTONYMS table."""
        if not text_new or not text_prior:
            return False, ""

        t_new = text_new.lower()
        t_prior = text_prior.lower()

        for word, opposites in ANTONYMS.items():
            if word in t_new and any(opp in t_prior for opp in opposites):
                return True, f"Antonym contradiction: '{word}' vs its opposite in prior text"
            if word in t_prior and any(opp in t_new for opp in opposites):
                return True, f"Antonym contradiction: '{word}' vs its opposite in new text"

        return False, ""
    
    # ========================================================================
    # 6b. Phase 2.0 Context-Aware Contradiction Detection
    # ========================================================================
    
    def is_true_contradiction_contextual(
        self,
        slot: Optional[str],
        value_new: Optional[str],
        value_prior: Optional[str],
        temporal_status_new: str = "active",
        temporal_status_prior: str = "active",
        domains_new: Optional[list] = None,
        domains_prior: Optional[list] = None,
        drift: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Phase 2.0: Context-aware contradiction detection.
        
        This method reduces false positives by considering:
        1. Temporal status (past/active/future facts don't always conflict)
        2. Domain context (different domains can have same-slot values)
        3. Slot matching (different slots are never contradictions)
        
        Only flag contradiction if:
        - Same slot
        - Overlapping time periods (or both active)
        - Same or overlapping domains
        - Mutually exclusive values
        
        Args:
            slot: The fact slot (e.g., "employer", "name")
            value_new: The new fact value
            value_prior: The existing fact value
            temporal_status_new: Temporal status of new fact (past/active/future/potential)
            temporal_status_prior: Temporal status of prior fact
            domains_new: Domain context of new fact
            domains_prior: Domain context of prior fact
            drift: Semantic drift score between the facts
            
        Returns:
            Tuple of (is_contradiction, reason)
        """
        # No slot match = not a contradiction
        if not slot:
            return False, "no_slot_no_contradiction"
        
        # Handle None values
        if value_new is None or value_prior is None:
            return False, "missing_values"
        
        # Normalize values
        value_new_norm = str(value_new).lower().strip()
        value_prior_norm = str(value_prior).lower().strip()

        slot_lower = str(slot).lower()

        # Transient state updates are not contradictions (identity vs transient state).
        if slot_lower in _MOOD_SLOTS:
            return False, "transient_state_slot_update"
        if _is_transient_state_value(value_new_norm) or _is_transient_state_value(value_prior_norm):
            return False, "transient_state_update"
        
        # Same normalized value = not a contradiction (might be status update)
        if value_new_norm == value_prior_norm:
            return False, "same_value"
        
        # Handle "LEFT:" prefix for employer negations
        if value_new_norm.startswith("left:"):
            # User is saying they no longer work somewhere
            left_value = value_new_norm.replace("left:", "").strip()
            if left_value == value_prior_norm:
                # "LEFT:Google" vs "Google" = temporal update, not contradiction
                return False, "temporal_update_left_employer"
        
        # Both are "past" = historical facts, not current contradiction
        if temporal_status_new == "past" and temporal_status_prior == "past":
            return False, "both_past_no_conflict"
        
        # New fact says something is "past" + prior says "active" = temporal update
        if temporal_status_new == "past" and temporal_status_prior == "active":
            return False, "temporal_deprecation"
        
        # Check domain overlap
        domains_new = domains_new or ["general"]
        domains_prior = domains_prior or ["general"]
        domains_new_set = set(domains_new)
        domains_prior_set = set(domains_prior)
        
        # "general" overlaps with everything
        has_general = "general" in domains_new_set or "general" in domains_prior_set
        has_specific_overlap = bool(domains_new_set & domains_prior_set - {"general"})
        
        # No overlap and no general = different contexts, can coexist
        if not has_general and not has_specific_overlap:
            return False, "different_domains_coexist"
        
        # Both "future" or "potential" = not yet realized, don't conflict
        if temporal_status_new in ("future", "potential") and temporal_status_prior in ("future", "potential"):
            return False, "future_plans_no_conflict"
        
        # At this point: same slot, overlapping domains, overlapping time
        # Different values = TRUE CONTRADICTION
        return True, f"true_contradiction: same slot '{slot}', overlapping context, different values"
    
    def _is_numeric_contradiction(self, value_new: str, value_prior: str, threshold: float = 0.20) -> Tuple[bool, str]:
        """
        Check if two numeric values contradict (>20% difference by default).
        
        This method handles numeric drift detection for facts like:
        - "8 years" vs "12 years" programming experience
        - "25" vs "34" age
        - "5 people" vs "10 people" team size
        
        Args:
            value_new: The new fact value (may contain non-numeric text)
            value_prior: The prior fact value (may contain non-numeric text)
            threshold: Percentage difference threshold (default 20%)
            
        Returns:
            Tuple of (is_contradiction, reason)
            - is_contradiction: True if numeric difference exceeds threshold
            - reason: Description of the comparison result
        """
        try:
            # Extract numeric values from strings
            match_new = re.search(r'[\d.]+', str(value_new))
            match_prior = re.search(r'[\d.]+', str(value_prior))
            
            if not match_new or not match_prior:
                return False, "not_numeric"
            
            num_new = float(match_new.group())
            num_prior = float(match_prior.group())
            
            # Handle zero case specially
            if num_prior == 0:
                is_contra = num_new != 0
                return is_contra, "numeric_zero_comparison" if is_contra else "both_zero"
            
            # Calculate percentage difference
            diff_pct = abs(num_new - num_prior) / abs(num_prior)
            
            if diff_pct > threshold:
                return True, f"numeric_drift_{diff_pct:.0%}"
            
            return False, f"numeric_within_tolerance_{diff_pct:.0%}"
            
        except (AttributeError, ValueError, TypeError) as e:
            return False, f"not_numeric: {e}"
    
    def classify_fact_change(
        self,
        slot: str,
        value_new: str,
        value_prior: str,
        text_new: str = "",
        text_prior: str = "",
    ) -> str:
        """
        Classify the type of fact change for contradiction ledger.
        
        Returns one of:
        - "refinement": More specific information (Seattle → Bellevue)
        - "revision": Explicit correction ("actually", "I meant")
        - "temporal": Time-based progression (Senior → Principal)
        - "conflict": Mutually exclusive facts (Microsoft vs Amazon)
        
        Used by crt_ledger to set contradiction_type field.
        """
        text_new_lower = (text_new or "").lower()
        value_new_lower = str(value_new).lower()
        value_prior_lower = str(value_prior).lower()
        
        # Check for explicit revision markers
        revision_markers = [
            "actually", "correction", "i meant", "i mean", "to clarify",
            "wrong", "mistake", "not", "no longer", "left", "quit"
        ]
        if any(marker in text_new_lower for marker in revision_markers):
            return "revision"
        
        # Check for temporal progression markers
        temporal_markers = [
            "now", "currently", "recently", "promoted", "moved to",
            "started", "new", "changed to"
        ]
        if any(marker in text_new_lower for marker in temporal_markers):
            return "temporal"
        
        # Check for refinement (new value contains or extends old)
        if value_prior_lower in value_new_lower:
            return "refinement"
        
        # Check geographic refinement (Seattle → Bellevue, both valid)
        geographic_refinement_pairs = [
            ("seattle", "bellevue"), ("new york", "brooklyn"),
            ("los angeles", "santa monica"), ("san francisco", "oakland"),
        ]
        for general, specific in geographic_refinement_pairs:
            if value_prior_lower == general and specific in value_new_lower:
                return "refinement"
        
        # Default to conflict
        return "conflict"
    
    # ========================================================================
    # 7. Reflection Triggers
    # ========================================================================
    
    def compute_volatility(
        self,
        drift: float,
        memory_alignment: float,
        is_contradiction: bool,
        is_fallback: bool
    ) -> float:
        """
        Compute volatility/instability score.
        
        V = β1·D_mean + β2·(1 - A_mem) + β3·contradiction + β4·fallback
        """
        cfg = self.config
        
        contra_flag = 1.0 if is_contradiction else 0.0
        fallback_flag = 1.0 if is_fallback else 0.0
        
        return (
            cfg.beta_drift * drift +
            cfg.beta_alignment * (1.0 - memory_alignment) +
            cfg.beta_contradiction * contra_flag +
            cfg.beta_fallback * fallback_flag
        )
    
    def should_reflect(self, volatility: float) -> bool:
        """
        Check if reflection should be triggered.
        
        if V ≥ θ_reflect → trigger reflection
        """
        return volatility >= self.config.theta_reflect
    
    # ========================================================================
    # 8. Safety Boundaries
    # ========================================================================
    
    def can_train_on_memory(
        self,
        trust: float,
        has_open_contradiction: bool,
        source: MemorySource
    ) -> Tuple[bool, str]:
        """
        Check if memory can be used for training/weight updates.
        
        Requirements:
        - τ ≥ τ_train_min
        - No open contradictions
        - Not from fallback (unless verified)
        """
        if trust < self.config.tau_train_min:
            return False, f"Trust too low: {trust:.3f} < {self.config.tau_train_min}"
        
        if has_open_contradiction:
            return False, "Open contradiction exists"
        
        if source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT}:
            return False, "Fallback source not verified"
        
        return True, "Safe to train"


# ============================================================================
# Utility Functions
# ============================================================================

_SHARED_ENCODER = None


def _hash_vector(text: str) -> np.ndarray:
    """Deterministic 32-D hash-based fallback when no real encoder is available.

    Used when the [ml] extra isn't installed or the LazyEncoder hasn't
    finished warming up. Same shape contract as a real embedding (1-D
    normalized float32) so callers can mix the two without branching.

    Note on the implementation: previous versions reinterpreted SHA-256 bytes
    directly as float32, which landed in the subnormal range and produced
    all-zero normalized vectors (similarity would always return 0). We instead
    treat each of the 32 digest bytes as an unsigned int, recenter to [-1, 1],
    and then L2-normalize. That gives a real vector with meaningful spread
    while still being deterministic per input text.
    """
    import hashlib
    digest = hashlib.sha256(text.encode()).digest()  # 32 bytes
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    centered = (raw - 127.5) / 127.5
    norm = np.linalg.norm(centered)
    if norm > 0:
        centered = centered / norm
    return centered


def encode_vector(text: str, encoder=None) -> np.ndarray:
    """Encode text to a semantic vector.

    Resolution order:
        1. Caller-supplied `encoder` callable (used by tests / advanced wiring).
        2. aether-core's process-wide LazyEncoder (real sentence-transformer
           embeddings if the [ml] extra is installed and the model is warm).
        3. Deterministic SHA-256 hash fallback — keeps the function total
           even on a cold install or when warmup is still in flight.

    The hash fallback preserves shape but obviously loses semantic meaning.
    Callers that depend on real similarity should warm the encoder first
    (see `aether._lazy_encoder.LazyEncoder.start_warmup`).
    """
    if encoder is not None:
        return encoder(text)

    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        try:
            from aether._lazy_encoder import LazyEncoder
            _SHARED_ENCODER = LazyEncoder()
            _SHARED_ENCODER.start_warmup()
        except Exception:
            _SHARED_ENCODER = False  # sentinel: don't retry on every call

    if _SHARED_ENCODER:
        try:
            vec = _SHARED_ENCODER.encode(text)
            if vec is not None:
                return np.asarray(vec, dtype=np.float32)
        except Exception:
            pass

    return _hash_vector(text)


def extract_emotion_intensity(text: str) -> float:
    """
    Extract emotion intensity from text (0-1).
    
    Placeholder - integrate with emotion detection.
    """
    # Simple heuristic: exclamation marks, caps, emotion words
    emotion_words = ['love', 'hate', 'fear', 'angry', 'happy', 'sad', 'excited', 'worried']
    
    intensity = 0.0
    
    # Exclamation marks
    intensity += min(text.count('!') * 0.1, 0.3)
    
    # Caps ratio
    if len(text) > 0:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        intensity += min(caps_ratio * 0.5, 0.3)
    
    # Emotion words
    text_lower = text.lower()
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    intensity += min(emotion_count * 0.1, 0.4)
    
    return min(intensity, 1.0)


def extract_future_relevance(text: str) -> float:
    """
    Extract future relevance proxy (0-1).
    
    Detects: questions, plans, "remember", etc.
    """
    relevance = 0.0
    text_lower = text.lower()
    
    # Questions
    if '?' in text:
        relevance += 0.3
    
    # Planning words
    planning_words = ['remember', 'later', 'tomorrow', 'next', 'plan', 'will', 'going to']
    for word in planning_words:
        if word in text_lower:
            relevance += 0.2
            break
    
    # Time references
    time_words = ['when', 'where', 'how long', 'until']
    for word in time_words:
        if word in text_lower:
            relevance += 0.2
            break
    
    return min(relevance, 1.0)
