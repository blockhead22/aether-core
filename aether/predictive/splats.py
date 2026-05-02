"""Memory Splats — Phase 1 (Diagonal Covariance)

Represents memories as Gaussian splats: center + covariance + confidence.
Enables geometric contradiction detection via overlap integrals.

Phase 1: Diagonal covariance (384 extra floats per memory).
Phase 2: Low-rank covariance (Sigma = D + UU^T, k=8-16).
Phase 3: Context-dependent covariance warping.

Key operations:
  - Bhattacharyya distance between splats
  - Overlap integral (continuous contradiction measure)
  - KL divergence (asymmetric similarity)
  - Covariance update from new evidence
  - Trajectory tracking (center + covariance snapshots over time)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Memory Splat
# ---------------------------------------------------------------------------

@dataclass
class MemorySplat:
    """A memory represented as a Gaussian region in semantic space.

    Instead of a point (just mu), a splat has:
      - mu: center (where the belief lives)
      - sigma: diagonal covariance (shape of uncertainty)
      - alpha: confidence/opacity (how much it contributes to world model)
      - metadata for tracking
    """
    memory_id: str
    mu: np.ndarray                      # center, shape (d,)
    sigma: np.ndarray                   # diagonal covariance, shape (d,)
    alpha: float = 0.8                  # confidence/opacity 0-1
    text: str = ""
    memory_type: str = "belief"
    created_at: float = 0.0
    last_updated: float = 0.0
    update_count: int = 0

    # Trajectory: snapshots of (mu, sigma, alpha, timestamp)
    trajectory: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.sigma = np.asarray(self.sigma, dtype=np.float32)
        # Ensure positive covariance
        self.sigma = np.maximum(self.sigma, 1e-8)

    @property
    def dim(self) -> int:
        return len(self.mu)

    @property
    def total_uncertainty(self) -> float:
        """Scalar summary of uncertainty: trace of covariance."""
        return float(np.sum(self.sigma))

    @property
    def avg_uncertainty(self) -> float:
        """Average per-dimension uncertainty."""
        return float(np.mean(self.sigma))

    def snapshot(self):
        """Save current state to trajectory."""
        self.trajectory.append({
            'mu': self.mu.copy(),
            'sigma': self.sigma.copy(),
            'alpha': self.alpha,
            'timestamp': time.time(),
        })

    def storage_bytes(self) -> int:
        """Actual storage cost."""
        # mu (d * 4) + sigma (d * 4) + alpha (4) + overhead (~32)
        return self.dim * 4 * 2 + 4 + 32


# ---------------------------------------------------------------------------
# Splat creation
# ---------------------------------------------------------------------------

def create_splat(
    memory_id: str,
    embedding: np.ndarray,
    text: str = "",
    memory_type: str = "belief",
    initial_uncertainty: Optional[float] = None,
    confidence: float = 0.8,
) -> MemorySplat:
    """Create a new memory splat from an embedding.

    Initial uncertainty is uniform across all dimensions.
    As evidence accumulates, some dimensions tighten, others widen.
    """
    d = len(embedding)
    mu = np.asarray(embedding, dtype=np.float32)
    if initial_uncertainty is None:
        initial_uncertainty = 2.0 / d  # calibrated default for unit vectors
    sigma = np.full(d, initial_uncertainty, dtype=np.float32)

    splat = MemorySplat(
        memory_id=memory_id,
        mu=mu,
        sigma=sigma,
        alpha=confidence,
        text=text,
        memory_type=memory_type,
        created_at=time.time(),
        last_updated=time.time(),
    )
    splat.snapshot()  # initial state
    return splat


def create_splat_from_type(
    memory_id: str,
    embedding: np.ndarray,
    text: str = "",
    memory_type: str = "belief",
    confidence: float = 0.8,
) -> MemorySplat:
    """Create a splat with type-dependent initial uncertainty.

    Facts start tight. Beliefs start wider. Identity starts medium.
    """
    # Calibrated for 384D unit vectors where typical pairwise distance ~ 1.4
    # and similar vectors (cos > 0.8) are distance ~ 0.6 apart.
    # Per-dimension variance needs to be large enough for distributions to overlap.
    # sigma_i = (expected_distance / sqrt(d))^2 gives the right scale.
    #
    # For d=384: sqrt(384) ~ 19.6
    # Similar vectors (dist~0.6): per-dim sigma ~ (0.6/19.6)^2 ~ 0.001
    # Dissimilar (dist~1.4): per-dim sigma ~ (1.4/19.6)^2 ~ 0.005
    # We want beliefs to overlap at the "similar" scale, so start wider.
    dim = len(embedding)
    scale = 1.0 / dim  # base scale for unit vectors in d dimensions
    type_uncertainty = {
        "fact": 0.5 * scale,        # facts are precise
        "preference": 1.5 * scale,  # preferences have some spread
        "event": 0.8 * scale,       # events are fairly precise
        "belief": 2.0 * scale,      # beliefs are uncertain
        "identity": 1.2 * scale,    # identity is moderately certain
    }
    uncertainty = type_uncertainty.get(memory_type, 0.015)
    return create_splat(memory_id, embedding, text, memory_type,
                        uncertainty, confidence)


# ---------------------------------------------------------------------------
# Distance metrics between splats (all closed-form for diagonal Gaussians)
# ---------------------------------------------------------------------------

def bhattacharyya_distance(a: MemorySplat, b: MemorySplat) -> float:
    """Bhattacharyya distance between two diagonal Gaussian splats.

    DB = 1/8 * (mu1-mu2)^T * Sigma_avg^{-1} * (mu1-mu2)
         + 1/2 * ln(det(Sigma_avg) / sqrt(det(Sigma1) * det(Sigma2)))

    For diagonal: all operations are element-wise, O(d).
    """
    sigma_avg = (a.sigma + b.sigma) / 2.0
    diff = a.mu - b.mu

    # Term 1: Mahalanobis-like distance
    term1 = 0.125 * np.sum(diff ** 2 / sigma_avg)

    # Term 2: Covariance divergence
    # ln(det(Sigma_avg)) = sum(ln(sigma_avg_i))
    # ln(det(Sigma1)) = sum(ln(sigma1_i))
    log_det_avg = np.sum(np.log(sigma_avg))
    log_det_a = np.sum(np.log(a.sigma))
    log_det_b = np.sum(np.log(b.sigma))
    term2 = 0.5 * (log_det_avg - 0.5 * (log_det_a + log_det_b))

    return float(term1 + term2)


def bhattacharyya_coefficient(a: MemorySplat, b: MemorySplat) -> float:
    """Bhattacharyya coefficient: exp(-DB). Range [0, 1].

    BC = 1: identical distributions
    BC = 0: no overlap
    Higher BC + different centers = contradiction territory.
    """
    db = bhattacharyya_distance(a, b)
    return float(np.exp(-db))


def overlap_integral(a: MemorySplat, b: MemorySplat) -> float:
    """Overlap integral: integral of sqrt(p(x) * q(x)) dx.

    For diagonal Gaussians, this equals the Bhattacharyya coefficient.
    This IS our continuous contradiction measure.

    High overlap + different centers = conflict.
    """
    return bhattacharyya_coefficient(a, b)


def kl_divergence(a: MemorySplat, b: MemorySplat) -> float:
    """KL(a || b) for diagonal Gaussians. Asymmetric.

    KL = 1/2 * [tr(Sigma_b^{-1} Sigma_a) + (mu_b-mu_a)^T Sigma_b^{-1} (mu_b-mu_a)
                - d + ln(det(Sigma_b)/det(Sigma_a))]

    For diagonal: O(d).
    """
    d = a.dim
    diff = b.mu - a.mu

    trace_term = np.sum(a.sigma / b.sigma)
    quad_term = np.sum(diff ** 2 / b.sigma)
    log_det_term = np.sum(np.log(b.sigma)) - np.sum(np.log(a.sigma))

    return float(0.5 * (trace_term + quad_term - d + log_det_term))


def cosine_similarity(a: MemorySplat, b: MemorySplat) -> float:
    """Standard cosine similarity between centers (ignoring covariance).
    For comparison with splat-aware metrics."""
    dot = np.dot(a.mu, b.mu)
    na = np.linalg.norm(a.mu)
    nb = np.linalg.norm(b.mu)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# Geometric contradiction detection
# ---------------------------------------------------------------------------

@dataclass
class GeometricContradictionResult:
    """Result of geometric contradiction check between two splats."""
    overlap: float              # Bhattacharyya coefficient [0,1]
    center_distance: float      # Euclidean distance between centers
    cosine_sim: float           # Cosine similarity between centers
    kl_ab: float                # KL(a||b)
    kl_ba: float                # KL(b||a)
    kl_symmetrized: float       # (KL(a||b) + KL(b||a)) / 2
    is_potential_contradiction: bool
    conflict_intensity: float   # 0-1, higher = stronger conflict
    explanation: str


def detect_geometric_contradiction(
    a: MemorySplat,
    b: MemorySplat,
    overlap_threshold: float = 0.3,
    center_divergence_threshold: float = 0.5,
) -> GeometricContradictionResult:
    """Detect contradiction from geometry alone — no NLI needed.

    A contradiction is: high overlap (related topics) + divergent centers
    (different claims about the same thing).

    This is the fast first-pass. NLI confirms on text.
    """
    ov = overlap_integral(a, b)
    cos = cosine_similarity(a, b)
    center_dist = float(np.linalg.norm(a.mu - b.mu))
    kl_ab = kl_divergence(a, b)
    kl_ba = kl_divergence(b, a)
    kl_sym = (kl_ab + kl_ba) / 2

    # In high dimensions (384D), overlap integral between Gaussians is
    # near-zero even for related distributions (curse of dimensionality).
    # Overlap is useful for TRACKING convergence over time, not static detection.
    #
    # For static contradiction detection, use cosine similarity between centers:
    # - High cosine = same topic
    # - Low cosine = different topics
    #
    # Contradiction = same topic (high cosine) + not identical (some distance)
    # The covariance comparison (KL divergence) tells us about uncertainty mismatch.
    related = cos > 0.3  # same broad topic
    not_identical = center_dist > center_divergence_threshold

    is_contra = related and not_identical
    intensity = 0.0
    if is_contra:
        # Intensity: higher cosine (more same-topic) + more distance = stronger
        intensity = min(1.0, cos * center_dist)
        # Uncertainty mismatch amplifies: if one is tight and one is fat, more interesting
        uncertainty_ratio = max(a.total_uncertainty, b.total_uncertainty) / \
                          max(min(a.total_uncertainty, b.total_uncertainty), 1e-8)
        if uncertainty_ratio > 1.5:
            intensity = min(1.0, intensity * 1.2)

    explanation = (
        f"overlap={ov:.3f}, cos={cos:.3f}, dist={center_dist:.3f}, "
        f"kl_sym={kl_sym:.3f}"
    )
    if is_contra:
        explanation = f"CONFLICT (intensity={intensity:.3f}): " + explanation
    else:
        if not related:
            explanation = "Unrelated (low overlap): " + explanation
        else:
            explanation = "Related but aligned: " + explanation

    return GeometricContradictionResult(
        overlap=ov,
        center_distance=center_dist,
        cosine_sim=cos,
        kl_ab=kl_ab,
        kl_ba=kl_ba,
        kl_symmetrized=kl_sym,
        is_potential_contradiction=is_contra,
        conflict_intensity=intensity,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Covariance updates from evidence
# ---------------------------------------------------------------------------

def update_splat_confirming(splat: MemorySplat, learning_rate: float = 0.1):
    """Confirming evidence: tighten the splat (reduce covariance).

    The splat gets more certain. Covariance shrinks.
    """
    splat.sigma *= (1.0 - learning_rate)
    splat.sigma = np.maximum(splat.sigma, 1e-8)
    splat.alpha = min(1.0, splat.alpha + 0.05)
    splat.update_count += 1
    splat.last_updated = time.time()


def update_splat_contradicting(splat: MemorySplat, learning_rate: float = 0.15):
    """Contradicting evidence: widen the splat (increase covariance).

    The splat gets less certain. Covariance grows.
    The thing the system was sure about is now questioned.
    """
    splat.sigma *= (1.0 + learning_rate)
    splat.alpha = max(0.1, splat.alpha - 0.1)
    splat.update_count += 1
    splat.last_updated = time.time()


def update_splat_with_evidence(
    splat: MemorySplat,
    new_embedding: np.ndarray,
    weight: float = 0.2,
):
    """Bayesian-ish update: shift center toward new evidence, adjust covariance.

    If new evidence is close to center → tighten.
    If new evidence is far from center → widen + shift.
    """
    new_emb = np.asarray(new_embedding, dtype=np.float32)
    diff = new_emb - splat.mu
    dist_sq = np.sum(diff ** 2)

    # Shift center toward evidence
    splat.mu = splat.mu + weight * diff

    # Adjust covariance based on surprise
    # Low surprise (close) → tighten
    # High surprise (far) → widen
    expected_dist = np.sum(splat.sigma)  # expected squared distance under the distribution
    surprise_ratio = dist_sq / max(expected_dist, 1e-8)

    if surprise_ratio < 1.0:
        # Evidence is within expected range -- tighten
        splat.sigma *= (1.0 - 0.05 * weight)
    else:
        # Evidence is surprising -- widen proportionally
        # Use log scale so extreme surprises (reversal) hit hard
        # surprise_ratio=2 -> widen 10%, =10 -> widen 23%, =100 -> widen 46%
        widen_factor = min(0.5, 0.1 * math.log(surprise_ratio + 1))
        splat.sigma *= (1.0 + widen_factor)
        # Also reduce confidence proportional to surprise
        confidence_hit = min(0.3, 0.05 * math.log(surprise_ratio + 1))
        splat.alpha = max(0.1, splat.alpha - confidence_hit)

    splat.sigma = np.maximum(splat.sigma, 1e-8)
    splat.update_count += 1
    splat.last_updated = time.time()


# ---------------------------------------------------------------------------
# Trajectory analysis (predictive contradiction detection)
# ---------------------------------------------------------------------------

def predict_trajectory(splat: MemorySplat, steps_ahead: int = 5) -> Optional[np.ndarray]:
    """Linear extrapolation of center trajectory.

    Returns predicted center position steps_ahead into the future.
    Needs at least 2 snapshots.
    """
    if len(splat.trajectory) < 2:
        return None

    # Use last two snapshots for linear extrapolation
    t1 = splat.trajectory[-2]
    t2 = splat.trajectory[-1]

    dt = t2['timestamp'] - t1['timestamp']
    if dt < 1e-6:
        return None

    velocity = (t2['mu'] - t1['mu']) / dt
    predicted = t2['mu'] + velocity * dt * steps_ahead

    return predicted


def predict_overlap_trend(
    a: MemorySplat,
    b: MemorySplat,
    steps: int = 5,
) -> Optional[List[float]]:
    """Predict how overlap between two splats will evolve.

    If overlap is increasing over time → converging → potential future conflict.
    """
    if len(a.trajectory) < 2 or len(b.trajectory) < 2:
        return None

    # Compute overlap at each historical snapshot
    overlaps = []
    n = min(len(a.trajectory), len(b.trajectory))
    for i in range(n):
        snap_a = MemorySplat("tmp_a", a.trajectory[i]['mu'], a.trajectory[i]['sigma'],
                             a.trajectory[i]['alpha'])
        snap_b = MemorySplat("tmp_b", b.trajectory[i]['mu'], b.trajectory[i]['sigma'],
                             b.trajectory[i]['alpha'])
        overlaps.append(overlap_integral(snap_a, snap_b))

    # Linear extrapolation of overlap trend
    if len(overlaps) >= 2:
        trend = overlaps[-1] - overlaps[-2]
        for s in range(steps):
            predicted = overlaps[-1] + trend * (s + 1)
            overlaps.append(max(0.0, min(1.0, predicted)))

    return overlaps


def covariance_velocity(splat: MemorySplat) -> Optional[float]:
    """Rate of change of total uncertainty.

    Positive = uncertainty growing (splat widening)
    Negative = uncertainty shrinking (splat tightening)
    Zero = stable

    This is the urgency signal from THEORY.md Section 6.
    """
    if len(splat.trajectory) < 2:
        return None

    t1 = splat.trajectory[-2]
    t2 = splat.trajectory[-1]
    dt = t2['timestamp'] - t1['timestamp']
    if dt < 1e-6:
        return None

    unc1 = float(np.sum(t1['sigma']))
    unc2 = float(np.sum(t2['sigma']))

    return (unc2 - unc1) / dt


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("MEMORY SPLATS — Phase 1 Test")
    print("=" * 70)

    np.random.seed(42)
    d = 384

    # Create some test embeddings (unit vectors with controlled similarity)
    def make_similar(base, noise_level=0.1):
        noise = np.random.randn(d).astype(np.float32) * noise_level
        v = base + noise
        return v / np.linalg.norm(v)

    base_job = np.random.randn(d).astype(np.float32)
    base_job /= np.linalg.norm(base_job)

    base_food = np.random.randn(d).astype(np.float32)
    base_food /= np.linalg.norm(base_food)

    # --- Create splats ---
    love_job = create_splat_from_type("love_job", make_similar(base_job, 0.05),
                                      "I love my job", "belief", 0.8)
    hate_job = create_splat_from_type("hate_job", make_similar(base_job, 0.05),
                                      "My job is killing me", "belief", 0.7)
    like_sushi = create_splat_from_type("like_sushi", make_similar(base_food, 0.1),
                                        "I like sushi", "preference", 0.9)
    unrelated = create_splat_from_type("weather", np.random.randn(d).astype(np.float32),
                                       "It's sunny today", "event", 0.6)
    # Normalize
    unrelated.mu /= np.linalg.norm(unrelated.mu)

    print(f"\n  --- SPLAT PROPERTIES ---")
    for s in [love_job, hate_job, like_sushi, unrelated]:
        print(f"  {s.memory_id:<12} | dim={s.dim} | alpha={s.alpha:.2f} "
              f"| avg_sigma={s.avg_uncertainty:.4f} | total_sigma={s.total_uncertainty:.2f} "
              f"| storage={s.storage_bytes()} bytes | \"{s.text}\"")

    # --- Distance metrics ---
    print(f"\n  --- PAIRWISE METRICS ---")
    pairs = [
        (love_job, hate_job, "love_job vs hate_job (SAME TOPIC, should conflict)"),
        (love_job, like_sushi, "love_job vs like_sushi (DIFFERENT TOPIC)"),
        (love_job, unrelated, "love_job vs weather (UNRELATED)"),
        (like_sushi, unrelated, "like_sushi vs weather (UNRELATED)"),
    ]

    for a, b, label in pairs:
        bc = bhattacharyya_coefficient(a, b)
        bd = bhattacharyya_distance(a, b)
        cos = cosine_similarity(a, b)
        kl = kl_divergence(a, b)
        print(f"\n  {label}")
        print(f"    BC={bc:.4f}  BD={bd:.2f}  cos={cos:.4f}  KL={kl:.2f}")

    # --- Geometric contradiction detection ---
    print(f"\n  --- GEOMETRIC CONTRADICTION DETECTION ---")
    for a, b, label in pairs:
        result = detect_geometric_contradiction(a, b)
        print(f"\n  {label}")
        print(f"    {result.explanation}")
        print(f"    is_contradiction={result.is_potential_contradiction} "
              f"intensity={result.conflict_intensity:.3f}")

    # --- Evidence updates ---
    print(f"\n  --- EVIDENCE UPDATE TEST ---")
    test_splat = create_splat("test", base_job.copy(), "test belief", "belief", None, 0.7)
    print(f"  Initial: avg_sigma={test_splat.avg_uncertainty:.5f} alpha={test_splat.alpha:.2f}")

    # 3 confirming pieces of evidence
    for i in range(3):
        update_splat_confirming(test_splat)
        test_splat.snapshot()
    print(f"  After 3 confirmations: avg_sigma={test_splat.avg_uncertainty:.5f} "
          f"alpha={test_splat.alpha:.2f}")

    # 1 contradicting piece
    update_splat_contradicting(test_splat)
    test_splat.snapshot()
    print(f"  After 1 contradiction: avg_sigma={test_splat.avg_uncertainty:.5f} "
          f"alpha={test_splat.alpha:.2f}")

    # Evidence from a nearby embedding
    nearby = make_similar(base_job, 0.05)
    update_splat_with_evidence(test_splat, nearby, weight=0.2)
    test_splat.snapshot()
    print(f"  After nearby evidence: avg_sigma={test_splat.avg_uncertainty:.5f} "
          f"alpha={test_splat.alpha:.2f}")

    # Evidence from a far embedding
    far = np.random.randn(d).astype(np.float32)
    far /= np.linalg.norm(far)
    update_splat_with_evidence(test_splat, far, weight=0.2)
    test_splat.snapshot()
    print(f"  After far evidence:    avg_sigma={test_splat.avg_uncertainty:.5f} "
          f"alpha={test_splat.alpha:.2f}")

    # --- Covariance velocity ---
    print(f"\n  --- COVARIANCE VELOCITY (urgency signal) ---")
    vel = covariance_velocity(test_splat)
    if vel is not None:
        print(f"  Test splat velocity: {vel:.6f} per second")
    else:
        print(f"  Test splat velocity: None (need 2+ snapshots)")
    if vel and vel > 0:
        print(f"  >> Uncertainty GROWING (splat widening) — belief under pressure")
    elif vel and vel < 0:
        print(f"  >> Uncertainty SHRINKING (splat tightening) — belief settling")

    # --- Trajectory prediction ---
    print(f"\n  --- TRAJECTORY PREDICTION ---")
    predicted = predict_trajectory(test_splat, steps_ahead=3)
    if predicted is not None:
        current_cos = float(np.dot(test_splat.mu, predicted) /
                           (np.linalg.norm(test_splat.mu) * np.linalg.norm(predicted)))
        print(f"  Current center -> Predicted (3 steps): cosine={current_cos:.4f}")
        print(f"  Trajectory points: {len(test_splat.trajectory)}")

    # --- Overlap trend prediction ---
    print(f"\n  --- OVERLAP TREND (converging splats = future conflict) ---")
    # Simulate two splats drifting toward each other
    splat_a = create_splat("drift_a", base_job.copy(), "Belief A", "belief", 0.02, 0.7)
    drift_target = make_similar(base_job, 0.3)  # somewhat different
    splat_b = create_splat("drift_b", drift_target, "Belief B", "belief", 0.02, 0.7)

    print(f"  Initial overlap: {overlap_integral(splat_a, splat_b):.4f}")

    # Drift splat_a toward splat_b over several steps
    for step in range(5):
        direction = splat_b.mu - splat_a.mu
        splat_a.mu += 0.1 * direction  # drift toward B
        splat_a.snapshot()
        splat_b.snapshot()

    trend = predict_overlap_trend(splat_a, splat_b, steps=3)
    if trend:
        print(f"  Overlap history: {[f'{o:.4f}' for o in trend[:5]]}")
        print(f"  Predicted trend: {[f'{o:.4f}' for o in trend[5:]]}")
        if len(trend) > 2 and trend[-1] > trend[0]:
            print(f"  >> CONVERGING — potential future conflict!")

    # --- Storage comparison ---
    print(f"\n  --- STORAGE COMPARISON ---")
    print(f"  Point embedding (384D float32):  {384 * 4} bytes")
    print(f"  Splat (mu + sigma + alpha):      {love_job.storage_bytes()} bytes")
    print(f"  Overhead: {love_job.storage_bytes() - 384*4} bytes ({love_job.storage_bytes() / (384*4):.1f}x)")
    print(f"  (Worth it? The extra {384*4} bytes encode the SHAPE of uncertainty)")

    print(f"\n{'='*70}")
    print("MEMORY SPLATS TEST COMPLETE")
    print(f"{'='*70}")
