"""Shared-categorical Track-and-Stop for Pareto-SMC v17.

Each type is an arm and each independent *complete canonical run replica*
returns one categorical endpoint: category 0 is outside the certified family and
categories 1..J are the frozen cells.  A single observation therefore updates
all cell indicators for that type.

The module contains:
* the exact characteristic game for simultaneous best-type identification;
* a certified projected-supergradient bracket for the game value;
* the pairwise categorical GLR statistic;
* a Dirichlet-mixture time-uniform stopping threshold valid under adaptive arm
  selection;
* a deterministic C-tracking implementation with forced exploration;
* exact-rational, time-uniform coordinate lower bounds used by the confirm
  planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Callable, Iterable, Sequence


class TrackAndStopError(ValueError):
    pass


def _validate_probability_matrix(probabilities: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(x) for x in row) for row in probabilities)
    if not rows or len(rows[0]) < 2:
        raise TrackAndStopError("at least one type and one certified cell are required")
    k = len(rows[0])
    if any(len(row) != k for row in rows):
        raise TrackAndStopError("ragged categorical probability matrix")
    for row in rows:
        if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in row):
            raise TrackAndStopError("invalid categorical probability")
        if abs(sum(row) - 1.0) > 1e-10:
            raise TrackAndStopError("categorical rows must sum to one")
    return rows


def answer_map(probabilities: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Return the unique best type for every certified cell.

    Category 0 is the outside category.  Ties are rejected because the
    instance-optimal theorem is stated at regular points with unique answers.
    """

    p = _validate_probability_matrix(probabilities)
    r_count = len(p)
    j_count = len(p[0]) - 1
    out: list[int] = []
    for j in range(1, j_count + 1):
        values = [p[r][j] for r in range(r_count)]
        best = max(values)
        winners = [r for r, value in enumerate(values) if value == best]
        if len(winners) != 1:
            raise TrackAndStopError(f"cell {j-1} has a non-unique best type")
        out.append(winners[0])
    return tuple(out)


def minimum_cell_gap(probabilities: Sequence[Sequence[float]]) -> float:
    p = _validate_probability_matrix(probabilities)
    answers = answer_map(p)
    gaps: list[float] = []
    for j0, best in enumerate(answers):
        j = j0 + 1
        second = max(p[r][j] for r in range(len(p)) if r != best)
        gaps.append(p[best][j] - second)
    return min(gaps)


def bernoulli_kl(p: float, q: float) -> float:
    if p < 0.0 or p > 1.0 or q < 0.0 or q > 1.0:
        raise TrackAndStopError("Bernoulli probabilities must lie in [0,1]")
    if p == q:
        return 0.0
    if q == 0.0:
        return math.inf if p > 0.0 else 0.0
    if q == 1.0:
        return math.inf if p < 1.0 else 0.0
    first = 0.0 if p == 0.0 else p * math.log(p / q)
    second = 0.0 if p == 1.0 else (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return first + second


def pair_information(
    probabilities: Sequence[Sequence[float]],
    weights: Sequence[float],
    cell_index: int,
    challenger: int,
) -> tuple[float, tuple[float, ...], int]:
    """Information against one answer-changing alternative.

    ``cell_index`` is zero-based over certified cells.  The best type is derived
    from the supplied model.  The categorical I-projection changes only the
    selected cell probability of the best and challenger arms and preserves the
    conditional proportions of all other categories.  The resulting cost is a
    weighted pair of Bernoulli KL divergences.
    """

    p = _validate_probability_matrix(probabilities)
    if len(weights) != len(p):
        raise TrackAndStopError("weight dimension mismatch")
    if any(x < 0.0 for x in weights) or abs(sum(weights) - 1.0) > 1e-8:
        raise TrackAndStopError("weights must lie in the simplex")
    answers = answer_map(p)
    if cell_index < 0 or cell_index >= len(answers):
        raise TrackAndStopError("cell index out of range")
    best = answers[cell_index]
    if challenger < 0 or challenger >= len(p) or challenger == best:
        raise TrackAndStopError("invalid challenger")
    category = cell_index + 1
    wa = weights[best]
    ws = weights[challenger]
    total = wa + ws
    pa = p[best][category]
    ps = p[challenger][category]
    if total <= 0.0:
        # At the origin of this two-coordinate face, choose any fixed
        # projection point m0.  Since I(v)=inf_m <v,KL(m)> is bounded above by
        # the affine form evaluated at m0, its KL coefficients form a valid
        # supergradient of the concave information function at zero.
        m0 = 0.5 * (pa + ps)
        gradient = [0.0] * len(p)
        gradient[best] = bernoulli_kl(pa, m0)
        gradient[challenger] = bernoulli_kl(ps, m0)
        return 0.0, tuple(gradient), best
    m = (wa * pa + ws * ps) / total
    value = wa * bernoulli_kl(pa, m) + ws * bernoulli_kl(ps, m)
    gradient = [0.0] * len(p)
    gradient[best] = bernoulli_kl(pa, m)
    gradient[challenger] = bernoulli_kl(ps, m)
    return value, tuple(gradient), best


def characteristic_value(
    probabilities: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, tuple[int, int], tuple[float, ...]]:
    p = _validate_probability_matrix(probabilities)
    answers = answer_map(p)
    active_value = math.inf
    active_pair = (-1, -1)
    active_gradient: tuple[float, ...] | None = None
    for j, best in enumerate(answers):
        for challenger in range(len(p)):
            if challenger == best:
                continue
            value, gradient, _ = pair_information(p, weights, j, challenger)
            key = (j, challenger)
            if value < active_value - 1e-15 or (
                abs(value - active_value) <= 1e-15 and key < active_pair
            ):
                active_value = value
                active_pair = key
                active_gradient = gradient
    if active_gradient is None:
        raise TrackAndStopError("no answer-changing alternative exists")
    return active_value, active_pair, active_gradient


def project_simplex(vector: Sequence[float]) -> tuple[float, ...]:
    """Euclidean projection onto the probability simplex."""

    v = [float(x) for x in vector]
    if not v:
        raise TrackAndStopError("cannot project an empty vector")
    u = sorted(v, reverse=True)
    cssv = 0.0
    rho = -1
    theta = 0.0
    for i, value in enumerate(u):
        cssv += value
        candidate = (cssv - 1.0) / (i + 1)
        if value > candidate:
            rho = i
            theta = candidate
    if rho < 0:
        return tuple(1.0 / len(v) for _ in v)
    out = [max(value - theta, 0.0) for value in v]
    total = sum(out)
    if total <= 0.0:
        return tuple(1.0 / len(v) for _ in v)
    return tuple(value / total for value in out)


@dataclass(frozen=True)
class CharacteristicSolution:
    weights: tuple[float, ...]
    lower_bound: float
    upper_bound: float
    gap: float
    iterations: int
    active_cell: int
    active_challenger: int

    @property
    def characteristic_time_upper(self) -> float:
        return math.inf if self.lower_bound <= 0.0 else 1.0 / self.lower_bound

    @property
    def characteristic_time_lower(self) -> float:
        return math.inf if self.upper_bound <= 0.0 else 1.0 / self.upper_bound


def solve_characteristic_game(
    probabilities: Sequence[Sequence[float]],
    *,
    iterations: int = 20_000,
    step_scale: float = 0.5,
    simplex_floor: float = 0.0,
) -> CharacteristicSolution:
    """Return a certified lower/upper bracket for ``Gamma*(P)``.

    For the concave function ``g(w)=min_c I_c(w)``, the active pair's envelope
    gradient is a supergradient.  Hence

      ``max_v g(v) <= g(w) + max_i grad_i - <grad,w>``.

    The best observed value is a valid lower bound and the smallest tangent
    bound is a valid upper bound.  The optimizer is numerical, but the bracket
    semantics do not rely on declaring the iterate globally optimal.
    """

    p = _validate_probability_matrix(probabilities)
    answer_map(p)
    if iterations <= 0:
        raise TrackAndStopError("iterations must be positive")
    if step_scale <= 0.0:
        raise TrackAndStopError("step_scale must be positive")
    r_count = len(p)
    if simplex_floor < 0.0 or simplex_floor * r_count >= 1.0:
        raise TrackAndStopError("invalid simplex floor")
    w = tuple(1.0 / r_count for _ in range(r_count))
    best_w = w
    best_lb = -math.inf
    best_ub = math.inf
    best_pair = (-1, -1)
    average = [0.0] * r_count
    support_floor = min(x for row in p for x in row)
    if support_floor <= 0.0:
        raise TrackAndStopError("certified characteristic solver requires full categorical support")
    diameter = math.sqrt(2.0)
    gradient_bound = math.sqrt(2.0) * math.log(1.0 / support_floor)
    if gradient_bound == 0.0:
        gradient_bound = 1e-15
    step = step_scale * diameter / (gradient_bound * math.sqrt(iterations))
    rate_gap = (diameter * gradient_bound / (2.0 * math.sqrt(iterations))) * (
        step_scale + 1.0 / step_scale
    )

    for t in range(1, iterations + 1):
        value, pair, gradient = characteristic_value(p, w)
        if value > best_lb:
            best_lb = value
            best_w = w
            best_pair = pair
        tangent_ub = value + max(gradient) - sum(g * x for g, x in zip(gradient, w, strict=True))
        best_ub = min(best_ub, tangent_ub)
        # Average the iterate at which the supergradient was evaluated; this is
        # the sequence covered by the standard projected-supergradient proof.
        for r in range(r_count):
            average[r] += w[r]
        averaged = tuple(x / t for x in average)
        avg_value, avg_pair, _ = characteristic_value(p, averaged)
        if avg_value > best_lb:
            best_lb = avg_value
            best_w = averaged
            best_pair = avg_pair
        candidate = project_simplex([x + step * g for x, g in zip(w, gradient, strict=True)])
        if simplex_floor > 0.0:
            scale = 1.0 - r_count * simplex_floor
            candidate = tuple(simplex_floor + scale * x for x in candidate)
        w = candidate

    # Projected supergradient ascent on a concave G-Lipschitz function over a
    # diameter-D domain gives opt - f(average) <=
    # D*G*(step_scale + 1/step_scale)/(2*sqrt(T)).  Since best_lb is at least
    # the averaged value, this is a second globally valid upper certificate.
    best_ub = min(best_ub, best_lb + rate_gap)
    best_ub = max(best_ub, best_lb)
    return CharacteristicSolution(
        weights=best_w,
        lower_bound=best_lb,
        upper_bound=best_ub,
        gap=max(0.0, best_ub - best_lb),
        iterations=iterations,
        active_cell=best_pair[0],
        active_challenger=best_pair[1],
    )


def empirical_probabilities(counts: Sequence[Sequence[int]], *, smoothing: float = 0.0) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(int(x) for x in row) for row in counts)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise TrackAndStopError("invalid categorical count matrix")
    if any(x < 0 for row in rows for x in row):
        raise TrackAndStopError("counts must be nonnegative")
    k = len(rows[0])
    out: list[tuple[float, ...]] = []
    for row in rows:
        n = sum(row)
        if n == 0 and smoothing == 0.0:
            out.append(tuple(1.0 / k for _ in row))
        else:
            denominator = n + smoothing * k
            out.append(tuple((x + smoothing) / denominator for x in row))
    return tuple(out)


def glr_statistic(counts: Sequence[Sequence[int]]) -> tuple[float, tuple[int, ...], tuple[int, int]]:
    """Pairwise GLR for simultaneous best-type identification."""

    rows = tuple(tuple(int(x) for x in row) for row in counts)
    p = empirical_probabilities(rows)
    r_count = len(rows)
    j_count = len(rows[0]) - 1
    answers: list[int] = []
    for j in range(1, j_count + 1):
        values = [p[r][j] for r in range(r_count)]
        best_value = max(values)
        winners = [r for r, value in enumerate(values) if value == best_value]
        if len(winners) != 1:
            return 0.0, tuple(min(winners) for _ in [0]) * j_count, (j - 1, min(winners))
        answers.append(winners[0])

    z = math.inf
    active = (-1, -1)
    for j0, best in enumerate(answers):
        j = j0 + 1
        n_best = sum(rows[best])
        for challenger in range(r_count):
            if challenger == best:
                continue
            n_ch = sum(rows[challenger])
            if n_best == 0 or n_ch == 0:
                value = 0.0
            else:
                p_best = rows[best][j] / n_best
                p_ch = rows[challenger][j] / n_ch
                pooled = (rows[best][j] + rows[challenger][j]) / (n_best + n_ch)
                value = n_best * bernoulli_kl(p_best, pooled) + n_ch * bernoulli_kl(p_ch, pooled)
            key = (j0, challenger)
            if value < z - 1e-15 or (abs(value - z) <= 1e-15 and key < active):
                z = value
                active = key
    return z, tuple(answers), active


def log_binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def dirichlet_mixture_threshold(counts: Sequence[Sequence[int]], delta: float) -> float:
    """Exact-form threshold induced by uniform Dirichlet mixtures.

    With ``K`` categories and adaptive arm selection, the product mixture
    likelihood ratio is a nonnegative martingale.  Ville's inequality yields

      beta_t = log(1/delta) + sum_r log C(N_r+K-1, K-1).
    """

    if not (0.0 < delta < 1.0):
        raise TrackAndStopError("delta must lie in (0,1)")
    rows = tuple(tuple(int(x) for x in row) for row in counts)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise TrackAndStopError("invalid count matrix")
    k = len(rows[0])
    return math.log(1.0 / delta) + sum(
        log_binomial(sum(row) + k - 1, k - 1) for row in rows
    )


def kl_binary_confidence_lower(phat: float, budget: float, *, iterations: int = 120) -> float:
    """Invert Bernoulli KL on the lower side by monotone bisection."""

    if phat <= 0.0:
        return 0.0
    if budget < 0.0:
        raise TrackAndStopError("KL budget must be nonnegative")
    lo, hi = 0.0, phat
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        value = bernoulli_kl(phat, mid)
        if value > budget:
            lo = mid
        else:
            hi = mid
    return lo


def _ceil_sqrt_ratio(numerator: int, denominator: int) -> int:
    """Smallest p with ``denominator*p^2 >= numerator``."""

    if numerator < 0 or denominator <= 0:
        raise TrackAndStopError("invalid square-root ratio")
    p = math.isqrt(numerator // denominator) if numerator >= denominator else 0
    while denominator * p * p < numerator:
        p += 1
    while p > 0 and denominator * (p - 1) * (p - 1) >= numerator:
        p -= 1
    return p


def time_uniform_hoeffding_radius(
    n: int,
    num_types: int,
    num_cells: int,
    alpha: Fraction | str,
    *,
    denominator: int = 10**12,
) -> Fraction:
    """Exact-rational upper radius valid simultaneously over all times.

    Choose the smallest integer ``t`` with

      ``2**t >= 2 R J n(n+1) / alpha``

    and then return the smallest dyadic-decimal rational ``c=p/D`` satisfying
    ``c^2 >= t/(2n)``.  Hoeffding and ``e^{-t} <= 2^{-t}`` give a summable
    familywise bound without any anti-conservative floating-point rounding.
    """

    if n <= 0 or num_types <= 0 or num_cells <= 0 or denominator <= 0:
        raise TrackAndStopError("invalid time-uniform radius arguments")
    a = Fraction(alpha)
    if a <= 0 or a >= 1:
        raise TrackAndStopError("alpha must lie in (0,1)")
    rhs = Fraction(2 * num_types * num_cells * n * (n + 1), 1) / a
    t = 0
    power = 1
    while Fraction(power, 1) < rhs:
        power *= 2
        t += 1
    p = _ceil_sqrt_ratio(t * denominator * denominator, 2 * n)
    return Fraction(p, denominator)


def time_uniform_lower_matrix(
    counts: Sequence[Sequence[int]],
    alpha: Fraction | str,
    *,
    denominator: int = 10**12,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return exact-rational endpoint probability lower bounds for all cells."""

    rows = tuple(tuple(int(x) for x in row) for row in counts)
    if not rows or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise TrackAndStopError("invalid categorical count matrix")
    r_count = len(rows)
    j_count = len(rows[0]) - 1
    out: list[tuple[Fraction, ...]] = []
    for row in rows:
        n = sum(row)
        if n <= 0:
            raise TrackAndStopError("every type must have at least one pilot sample")
        radius = time_uniform_hoeffding_radius(
            n,
            r_count,
            j_count,
            alpha,
            denominator=denominator,
        )
        out.append(
            tuple(max(Fraction(0, 1), Fraction(row[j], n) - radius) for j in range(1, j_count + 1))
        )
    return tuple(out)


def binary_kl_decision_lower_bound(delta: float, gamma_star: float) -> float:
    """Asymptotic transportation lower bound in expected samples."""

    if not (0.0 < delta < 0.5):
        raise TrackAndStopError("delta must lie in (0,1/2)")
    if gamma_star <= 0.0:
        return math.inf
    decision_kl = (1.0 - delta) * math.log((1.0 - delta) / delta) + delta * math.log(delta / (1.0 - delta))
    return decision_kl / gamma_star


@dataclass(frozen=True)
class TrackAndStopConfig:
    delta: Fraction | str | float
    max_samples: int
    optimizer_iterations: int = 4_000
    optimizer_step_scale: float = 0.5
    optimizer_growth_power: float = 0.5
    forced_exploration_scale: float = 1.0
    smoothing: float = 0.5

    def __post_init__(self) -> None:
        delta = self.delta if isinstance(self.delta, Fraction) else Fraction(str(self.delta))
        object.__setattr__(self, "delta", delta)
        if not (Fraction(0, 1) < delta < Fraction(1, 1)):
            raise TrackAndStopError("delta must lie in (0,1)")
        if self.max_samples <= 0 or self.optimizer_iterations <= 0:
            raise TrackAndStopError("sample and optimizer limits must be positive")
        if self.optimizer_growth_power <= 0.0:
            raise TrackAndStopError("optimizer_growth_power must be positive")
        if self.forced_exploration_scale <= 0.0 or self.smoothing <= 0.0:
            raise TrackAndStopError("forced exploration and smoothing must be positive")


@dataclass(frozen=True)
class TrackAndStopResult:
    stopped: bool
    total_samples: int
    counts: tuple[tuple[int, ...], ...]
    answer: tuple[int, ...]
    glr: float
    threshold: float
    active_cell: int
    active_challenger: int
    characteristic_lower: float
    characteristic_upper: float
    characteristic_gap: float
    allocation: tuple[float, ...]
    trace_sha256: str
    glr_likelihood_ratio: str
    threshold_likelihood_ratio: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stopped": self.stopped,
            "total_samples": self.total_samples,
            "counts": [list(row) for row in self.counts],
            "answer": list(self.answer),
            "glr": self.glr,
            "threshold": self.threshold,
            "active_cell": self.active_cell,
            "active_challenger": self.active_challenger,
            "characteristic_lower": self.characteristic_lower,
            "characteristic_upper": self.characteristic_upper,
            "characteristic_gap": self.characteristic_gap,
            "allocation": list(self.allocation),
            "trace_sha256": self.trace_sha256,
            "glr_likelihood_ratio": self.glr_likelihood_ratio,
            "threshold_likelihood_ratio": self.threshold_likelihood_ratio,
        }


def run_track_and_stop(
    num_types: int,
    num_cells: int,
    sampler: Callable[[int, int], int],
    config: TrackAndStopConfig,
) -> TrackAndStopResult:
    """Execute shared-categorical Track-and-Stop.

    ``sampler(type_index, within_type_sample_index)`` must return category 0 for
    outside or category ``1..num_cells``.  The theorem interprets these calls as
    independent complete canonical run replicas from a frozen arm law.
    """

    if num_types < 2 or num_cells < 1:
        raise TrackAndStopError("at least two types and one cell are required")
    k = num_cells + 1
    counts = [[0 for _ in range(k)] for _ in range(num_types)]
    pulls = [0] * num_types
    target_cumulative = [0.0] * num_types
    trace: list[dict[str, object]] = []

    def pull(arm: int) -> None:
        category = int(sampler(arm, pulls[arm]))
        if category < 0 or category >= k:
            raise TrackAndStopError("sampler returned a category outside the frozen family")
        counts[arm][category] += 1
        pulls[arm] += 1
        trace.append({"t": sum(pulls), "arm": arm, "category": category})

    for arm in range(num_types):
        pull(arm)
    # Account for the deterministic one-sample initialization in the cumulative
    # target so the tracking discrepancy has zero total mass.
    target_cumulative = [1.0] * num_types

    try:
        final_solution = solve_characteristic_game(
            empirical_probabilities(counts, smoothing=config.smoothing),
            iterations=config.optimizer_iterations,
            step_scale=config.optimizer_step_scale,
        )
    except TrackAndStopError:
        uniform = tuple(1.0 / num_types for _ in range(num_types))
        final_solution = CharacteristicSolution(
            weights=uniform,
            lower_bound=0.0,
            upper_bound=math.inf,
            gap=math.inf,
            iterations=0,
            active_cell=-1,
            active_challenger=-1,
        )
    stopped = False
    z = 0.0
    answer = tuple(0 for _ in range(num_cells))
    active = (-1, -1)
    threshold = math.inf
    exact_decision: dict[str, object] = {
        "stopped": False,
        "answer": None,
        "active_cell": -1,
        "active_challenger": -1,
        "glr_likelihood_ratio": "1",
        "threshold_likelihood_ratio": "1",
    }

    while sum(pulls) <= config.max_samples:
        z, answer, active = glr_statistic(counts)
        threshold = dirichlet_mixture_threshold(counts, float(config.delta))
        # A floating diagnostic may only delay an exact stop; it can never open
        # the gate.  The expensive exact-rational comparison is invoked once
        # the diagnostic is at or above the threshold (up to a tiny negative
        # prefilter margin).  If rounding underestimates the GLR, the algorithm
        # takes extra samples, which preserves delta-correctness.
        if z >= threshold - 1e-10:
            exact_decision = exact_track_stop_decision(counts, config.delta)
            if bool(exact_decision["stopped"]):
                stopped = True
                if exact_decision["answer"] is not None:
                    answer = tuple(int(x) for x in exact_decision["answer"])
                active = (int(exact_decision["active_cell"]), int(exact_decision["active_challenger"]))
                break
        if sum(pulls) == config.max_samples:
            break
        t = sum(pulls)
        model = empirical_probabilities(counts, smoothing=config.smoothing)
        try:
            final_solution = solve_characteristic_game(
                model,
                iterations=config.optimizer_iterations + int(math.ceil(t ** config.optimizer_growth_power)),
                step_scale=config.optimizer_step_scale,
            )
        except TrackAndStopError:
            uniform = tuple(1.0 / num_types for _ in range(num_types))
            final_solution = CharacteristicSolution(
                weights=uniform,
                lower_bound=0.0,
                upper_bound=math.inf,
                gap=math.inf,
                iterations=0,
                active_cell=-1,
                active_challenger=-1,
            )
        # Exploration is injected into the target allocation itself.  Pure
        # cumulative tracking then has a deterministic discrepancy bound, while
        # sum_s xi_s/R diverges like sqrt(t), guaranteeing every arm is sampled
        # infinitely often without an exceptional override branch.
        xi = min(1.0, config.forced_exploration_scale * num_types / math.sqrt(t + 1.0))
        mixed = tuple((1.0 - xi) * w + xi / num_types for w in final_solution.weights)
        for arm, weight in enumerate(mixed):
            target_cumulative[arm] += weight
        deficits = [target_cumulative[arm] - pulls[arm] for arm in range(num_types)]
        chosen = min(range(num_types), key=lambda arm: (-deficits[arm], arm))
        pull(chosen)

    if exact_decision.get("answer") is None or exact_decision.get("threshold_likelihood_ratio") == "1":
        exact_decision = exact_track_stop_decision(counts, config.delta)
    payload = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return TrackAndStopResult(
        stopped=stopped,
        total_samples=sum(pulls),
        counts=tuple(tuple(row) for row in counts),
        answer=answer,
        glr=z,
        threshold=threshold,
        active_cell=active[0],
        active_challenger=active[1],
        characteristic_lower=final_solution.lower_bound,
        characteristic_upper=final_solution.upper_bound,
        characteristic_gap=final_solution.gap,
        allocation=final_solution.weights,
        trace_sha256=digest,
        glr_likelihood_ratio=str(exact_decision["glr_likelihood_ratio"]),
        threshold_likelihood_ratio=str(exact_decision["threshold_likelihood_ratio"]),
    )

# ---------------------------------------------------------------------------
# Exact rational stopping comparison
# ---------------------------------------------------------------------------

def _fraction_power(base: Fraction, exponent: int) -> Fraction:
    if exponent < 0:
        raise TrackAndStopError("negative exponent in exact GLR")
    if exponent == 0:
        return Fraction(1, 1)
    return base ** exponent


def exact_empirical_answer(counts: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    rows = tuple(tuple(int(x) for x in row) for row in counts)
    if not rows or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise TrackAndStopError("invalid categorical count matrix")
    if any(sum(row) <= 0 for row in rows):
        raise TrackAndStopError("every type needs at least one observation")
    answers: list[int] = []
    for j in range(1, len(rows[0])):
        values = [Fraction(rows[r][j], sum(rows[r])) for r in range(len(rows))]
        best = max(values)
        winners = [r for r, value in enumerate(values) if value == best]
        if len(winners) != 1:
            return None
        answers.append(winners[0])
    return tuple(answers)


def exact_pair_glr_ratio(
    counts: Sequence[Sequence[int]],
    cell_index: int,
    best: int,
    challenger: int,
) -> Fraction:
    """Return ``exp(Z_pair)`` exactly as a rational number."""

    rows = tuple(tuple(int(x) for x in row) for row in counts)
    j = cell_index + 1
    n_a = sum(rows[best])
    n_s = sum(rows[challenger])
    if n_a <= 0 or n_s <= 0:
        return Fraction(1, 1)
    h_a = rows[best][j]
    h_s = rows[challenger][j]
    total_n = n_a + n_s
    total_h = h_a + h_s

    numerator = (
        _fraction_power(Fraction(h_a, n_a), h_a)
        * _fraction_power(Fraction(n_a - h_a, n_a), n_a - h_a)
        * _fraction_power(Fraction(h_s, n_s), h_s)
        * _fraction_power(Fraction(n_s - h_s, n_s), n_s - h_s)
    )
    denominator = (
        _fraction_power(Fraction(total_h, total_n), total_h)
        * _fraction_power(Fraction(total_n - total_h, total_n), total_n - total_h)
    )
    if denominator == 0:
        # This can only occur in a degenerate pooled boundary case.  The
        # empirical proportions then coincide and the GLR is zero.
        return Fraction(1, 1)
    return numerator / denominator


def exact_glr_ratio(
    counts: Sequence[Sequence[int]],
) -> tuple[Fraction, tuple[int, ...] | None, tuple[int, int]]:
    answers = exact_empirical_answer(counts)
    rows = tuple(tuple(int(x) for x in row) for row in counts)
    if answers is None:
        return Fraction(1, 1), None, (-1, -1)
    best_ratio: Fraction | None = None
    active = (-1, -1)
    for j, best in enumerate(answers):
        for challenger in range(len(rows)):
            if challenger == best:
                continue
            ratio = exact_pair_glr_ratio(rows, j, best, challenger)
            key = (j, challenger)
            if best_ratio is None or ratio < best_ratio or (ratio == best_ratio and key < active):
                best_ratio = ratio
                active = key
    if best_ratio is None:
        raise TrackAndStopError("no answer-changing pair")
    return best_ratio, answers, active


def exact_dirichlet_threshold_ratio(
    counts: Sequence[Sequence[int]],
    delta: Fraction | str | int,
) -> Fraction:
    d = Fraction(delta)
    if d <= 0 or d >= 1:
        raise TrackAndStopError("delta must lie in (0,1)")
    rows = tuple(tuple(int(x) for x in row) for row in counts)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise TrackAndStopError("invalid categorical count matrix")
    k = len(rows[0])
    threshold = Fraction(1, 1) / d
    for row in rows:
        threshold *= math.comb(sum(row) + k - 1, k - 1)
    return threshold


def exact_track_stop_decision(
    counts: Sequence[Sequence[int]],
    delta: Fraction | str | int,
) -> dict[str, object]:
    ratio, answers, active = exact_glr_ratio(counts)
    threshold = exact_dirichlet_threshold_ratio(counts, delta)
    return {
        "stopped": answers is not None and ratio >= threshold,
        "answer": None if answers is None else list(answers),
        "active_cell": active[0],
        "active_challenger": active[1],
        "glr_likelihood_ratio": str(ratio),
        "threshold_likelihood_ratio": str(threshold),
    }
