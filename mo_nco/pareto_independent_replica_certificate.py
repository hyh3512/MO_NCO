from __future__ import annotations

"""Exact probability contracts for independent endpoint replicas.

The functions in this module deliberately avoid binary floating-point
probabilities.  Inputs are exact :class:`fractions.Fraction` objects, boundary
integers, or canonical rational strings such as ``"1/1000000000"``.  All
reported probability values are serialized back to canonical rational
strings.

Two distinctions are part of the public contract:

* a Clopper--Pearson lower endpoint is reported as a rational bracket whose
  lower member is conservative (never above the mathematical endpoint); and
* a replica-count result says whether it is the exact minimum or a proved
  conservative upper bound.  The latter is used for counts too large for a
  practical exact integer-power comparison.

The frequentist statements assume independent Bernoulli trials on the
mathematical probability space.  Reproducible, domain-separated pseudo-random
seeds are an implementation mechanism, not a proof of that assumption.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import comb, lcm
import re
from typing import Sequence, TypeAlias


PROBABILITY_CERTIFICATE_SCHEMA_V15 = (
    "pareto_independent_replica_probability_certificate_v15"
)
FALSE_PASS_EVENT_LABEL = "PASS_AND_CONFIRM_FAILURE"
FALSE_PASS_BOUND_SEMANTICS = (
    "joint_probability_of_PASS_and_confirm_failure_not_"
    "conditional_probability_given_PASS"
)

REPLICA_PLAN_EXACT_MINIMUM = "EXACT_MINIMUM"
REPLICA_PLAN_CONSERVATIVE_UPPER = "CONSERVATIVE_UPPER"
REPLICA_PLAN_IMPOSSIBLE = "IMPOSSIBLE"

OCCUPANCY_EVENT_LABEL = (
    "ALL_DECLARED_MUTUALLY_EXCLUSIVE_CELLS_HIT"
)
OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION = (
    "EXACT_RATIONAL_INCLUSION_EXCLUSION"
)
OCCUPANCY_METHOD_BONFERRONI = (
    "BONFERRONI_UNION_BOUND_LOWER"
)
EXACT_OCCUPANCY_CELL_LIMIT = 20

_CANONICAL_RATIONAL_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")

ProbabilityInput: TypeAlias = Fraction | str | int


class ProbabilityCertificateError(ValueError):
    """Raised when an exact probability certificate cannot be formed."""


def canonical_rational_string(value: Fraction | int) -> str:
    """Return the unique reduced ``numerator[/denominator]`` representation."""

    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise ProbabilityCertificateError(
            "A canonical rational value must be a Fraction or integer."
        )
    resolved = Fraction(value)
    if resolved.denominator == 1:
        return str(resolved.numerator)
    return f"{resolved.numerator}/{resolved.denominator}"


def parse_canonical_probability(
    value: ProbabilityInput,
    *,
    label: str = "probability",
) -> Fraction:
    """Parse an exact probability and reject floats/noncanonical strings.

    Nontrivial JSON probabilities should therefore be strings such as
    ``"1/20"``.  Integers are accepted only as exact boundary conveniences.
    """

    if isinstance(value, bool):
        raise ProbabilityCertificateError(f"{label} cannot be Boolean.")
    if isinstance(value, Fraction):
        resolved = value
    elif isinstance(value, int):
        resolved = Fraction(value)
    elif isinstance(value, str):
        if _CANONICAL_RATIONAL_RE.fullmatch(value) is None:
            raise ProbabilityCertificateError(
                f"{label} must be a canonical nonnegative rational string."
            )
        numerator_text, separator, denominator_text = value.partition("/")
        denominator = int(denominator_text) if separator else 1
        resolved = Fraction(int(numerator_text), denominator)
        if canonical_rational_string(resolved) != value:
            raise ProbabilityCertificateError(
                f"{label} is not reduced canonical rational text."
            )
    else:
        raise ProbabilityCertificateError(
            f"{label} must be a Fraction or canonical rational string; "
            "binary floating-point values are forbidden."
        )
    if resolved < 0 or resolved > 1:
        raise ProbabilityCertificateError(
            f"{label} must lie in the closed interval [0, 1]."
        )
    return resolved


def _strict_unit_probability(
    value: ProbabilityInput,
    *,
    label: str,
) -> Fraction:
    resolved = parse_canonical_probability(value, label=label)
    if resolved <= 0 or resolved >= 1:
        raise ProbabilityCertificateError(
            f"{label} must lie in the open interval (0, 1)."
        )
    return resolved


def _nonnegative_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbabilityCertificateError(
            f"{label} must be a nonnegative integer."
        )
    return value


def exact_binomial_survival(
    trials: int,
    at_least: int,
    probability: ProbabilityInput,
) -> Fraction:
    """Return ``P[Bin(trials, probability) >= at_least]`` exactly.

    A common integer denominator is used, so no floating-point tail or
    recurrence rounding can reverse a certificate inequality.
    """

    n = _nonnegative_integer(trials, label="trials")
    if isinstance(at_least, bool) or not isinstance(at_least, int):
        raise ProbabilityCertificateError("at_least must be an integer.")
    p = parse_canonical_probability(probability)
    if at_least <= 0:
        return Fraction(1)
    if at_least > n:
        return Fraction(0)
    if p == 0:
        return Fraction(0)
    if p == 1:
        return Fraction(1)

    success = p.numerator
    denominator_base = p.denominator
    failure = denominator_base - success
    denominator = denominator_base**n

    def weighted_term(successes: int) -> int:
        return (
            comb(n, successes)
            * success**successes
            * failure ** (n - successes)
        )

    tail_term_count = n - at_least + 1
    lower_term_count = at_least
    if lower_term_count < tail_term_count:
        lower_numerator = sum(
            weighted_term(successes)
            for successes in range(at_least)
        )
        numerator = denominator - lower_numerator
    else:
        numerator = sum(
            weighted_term(successes)
            for successes in range(at_least, n + 1)
        )
    return Fraction(numerator, denominator)


binomial_survival_probability = exact_binomial_survival


@dataclass(frozen=True)
class OccupancyLowerBoundCertificate:
    """Lower bound for hitting every declared mutually exclusive cell."""

    replicas: int
    probability_lower_bounds: tuple[Fraction, ...]
    lower_bound: Fraction
    method: str
    exact_for_lower_probability_model: bool

    @property
    def cell_count(self) -> int:
        return len(self.probability_lower_bounds)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": PROBABILITY_CERTIFICATE_SCHEMA_V15,
            "event": OCCUPANCY_EVENT_LABEL,
            "method": self.method,
            "replicas": self.replicas,
            "cell_count": self.cell_count,
            "probability_lower_bounds": [
                canonical_rational_string(value)
                for value in self.probability_lower_bounds
            ],
            "probability_lower_bound_sum": canonical_rational_string(
                sum(self.probability_lower_bounds, Fraction(0))
            ),
            "all_cells_hit_probability_lower_bound": (
                canonical_rational_string(self.lower_bound)
            ),
            "exact_for_lower_probability_model": (
                self.exact_for_lower_probability_model
            ),
            "mutually_exclusive_cells_required": True,
            "independent_categorical_replicas_assumed": True,
        }


def mutually_exclusive_cell_occupancy_lower_bound(
    probability_lower_bounds: Sequence[ProbabilityInput],
    replicas: int,
) -> OccupancyLowerBoundCertificate:
    """Certify that all mutually exclusive cells are hit.

    For at most :data:`EXACT_OCCUPANCY_CELL_LIMIT` cells this evaluates

    ``sum_S (-1)^|S| (1 - sum_{j in S} lower_p[j])**replicas``

    by exact integer inclusion--exclusion.  This is the exact occupancy
    probability for the categorical model with the lower probabilities and
    residual outside mass.  It is a lower bound for any componentwise larger
    cell-probability vector: residual outcomes can be coupled into additional
    cell outcomes, and the all-cells-hit event is increasing under that map.

    Larger families use the explicit Bonferroni lower bound

    ``max(0, 1 - sum_j (1-lower_p[j])**replicas)``.
    """

    m = _nonnegative_integer(replicas, label="replicas")
    if isinstance(probability_lower_bounds, (str, bytes)):
        raise ProbabilityCertificateError(
            "probability_lower_bounds must be a probability sequence."
        )
    try:
        raw_bounds = tuple(probability_lower_bounds)
    except TypeError as error:
        raise ProbabilityCertificateError(
            "probability_lower_bounds must be a probability sequence."
        ) from error
    bounds = tuple(
        parse_canonical_probability(value, label=f"lower_p[{index}]")
        for index, value in enumerate(raw_bounds)
    )
    if sum(bounds, Fraction(0)) > 1:
        raise ProbabilityCertificateError(
            "Mutually exclusive cell probability lower bounds must sum "
            "to at most one."
        )

    if len(bounds) > EXACT_OCCUPANCY_CELL_LIMIT:
        missed_union_upper = sum(
            ((1 - value) ** m for value in bounds),
            Fraction(0),
        )
        lower_bound = max(Fraction(0), 1 - missed_union_upper)
        return OccupancyLowerBoundCertificate(
            replicas=m,
            probability_lower_bounds=bounds,
            lower_bound=lower_bound,
            method=OCCUPANCY_METHOD_BONFERRONI,
            exact_for_lower_probability_model=False,
        )

    if not bounds:
        lower_bound = Fraction(1)
    elif m == 0 or any(value == 0 for value in bounds):
        lower_bound = Fraction(0)
    else:
        common_denominator = 1
        for value in bounds:
            common_denominator = lcm(
                common_denominator,
                value.denominator,
            )
        weights = tuple(
            value.numerator
            * (common_denominator // value.denominator)
            for value in bounds
        )
        denominator_power = common_denominator**m
        signed_numerator = 0
        subset_weight = 0
        previous_gray_code = 0
        for ordinal in range(1 << len(weights)):
            gray_code = ordinal ^ (ordinal >> 1)
            if ordinal:
                changed_bit = gray_code ^ previous_gray_code
                index = changed_bit.bit_length() - 1
                if gray_code & changed_bit:
                    subset_weight += weights[index]
                else:
                    subset_weight -= weights[index]
            term = (common_denominator - subset_weight) ** m
            if gray_code.bit_count() % 2:
                signed_numerator -= term
            else:
                signed_numerator += term
            previous_gray_code = gray_code
        lower_bound = Fraction(signed_numerator, denominator_power)
        if lower_bound < 0 or lower_bound > 1:
            raise AssertionError(
                "Exact occupancy inclusion-exclusion left [0, 1]."
            )

    return OccupancyLowerBoundCertificate(
        replicas=m,
        probability_lower_bounds=bounds,
        lower_bound=lower_bound,
        method=OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION,
        exact_for_lower_probability_model=True,
    )


@dataclass(frozen=True)
class ClopperPearsonLowerBracket:
    """A directed rational bracket for a one-sided CP lower endpoint."""

    trials: int
    successes: int
    alpha: Fraction
    lower: Fraction
    upper: Fraction
    survival_at_lower: Fraction
    survival_at_upper: Fraction
    iterations: int
    exact_endpoint: bool
    tail_equation_applies: bool

    @property
    def conservative_lower(self) -> Fraction:
        return self.lower

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": PROBABILITY_CERTIFICATE_SCHEMA_V15,
            "trials": self.trials,
            "successes": self.successes,
            "alpha": canonical_rational_string(self.alpha),
            "lower": canonical_rational_string(self.lower),
            "upper": canonical_rational_string(self.upper),
            "survival_at_lower": canonical_rational_string(
                self.survival_at_lower
            ),
            "survival_at_upper": canonical_rational_string(
                self.survival_at_upper
            ),
            "iterations": self.iterations,
            "exact_endpoint": self.exact_endpoint,
            "tail_equation_applies": self.tail_equation_applies,
            "lower_rounding_direction": "toward_negative_infinity",
            "upper_rounding_direction": "toward_positive_infinity",
        }


def clopper_pearson_lower_bracket(
    successes: int,
    trials: int,
    alpha: ProbabilityInput,
    *,
    precision_bits: int = 128,
) -> ClopperPearsonLowerBracket:
    """Bracket the exact one-sided Clopper--Pearson lower endpoint.

    For ``successes > 0`` the endpoint is the unique solution of
    ``P_p[Bin(trials, p) >= successes] = alpha``.  Exact rational bisection
    maintains

    ``survival(lower) <= alpha <= survival(upper)``.

    Thus returning ``lower`` as the reported lower confidence endpoint is
    always conservative.  ``successes == 0`` uses the standard exact boundary
    endpoint zero; there is no tail-equation root in that case.
    """

    n = _nonnegative_integer(trials, label="trials")
    x = _nonnegative_integer(successes, label="successes")
    if n == 0:
        raise ProbabilityCertificateError("trials must be positive.")
    if x > n:
        raise ProbabilityCertificateError(
            "successes cannot exceed trials."
        )
    a = _strict_unit_probability(alpha, label="alpha")
    bits = _nonnegative_integer(precision_bits, label="precision_bits")
    if bits == 0 or bits > 4096:
        raise ProbabilityCertificateError(
            "precision_bits must lie in [1, 4096]."
        )

    if x == 0:
        return ClopperPearsonLowerBracket(
            trials=n,
            successes=x,
            alpha=a,
            lower=Fraction(0),
            upper=Fraction(0),
            survival_at_lower=Fraction(1),
            survival_at_upper=Fraction(1),
            iterations=0,
            exact_endpoint=True,
            tail_equation_applies=False,
        )

    lower = Fraction(0)
    upper = Fraction(1)
    survival_at_lower = Fraction(0)
    survival_at_upper = Fraction(1)
    exact_endpoint = False
    iterations = 0

    for iterations in range(1, bits + 1):
        midpoint = (lower + upper) / 2
        survival = exact_binomial_survival(n, x, midpoint)
        if survival == a:
            lower = midpoint
            upper = midpoint
            survival_at_lower = survival
            survival_at_upper = survival
            exact_endpoint = True
            break
        if survival < a:
            lower = midpoint
            survival_at_lower = survival
        else:
            upper = midpoint
            survival_at_upper = survival

    if survival_at_lower > a or survival_at_upper < a:
        raise AssertionError("Directed CP bracket invariant was violated.")
    return ClopperPearsonLowerBracket(
        trials=n,
        successes=x,
        alpha=a,
        lower=lower,
        upper=upper,
        survival_at_lower=survival_at_lower,
        survival_at_upper=survival_at_upper,
        iterations=iterations,
        exact_endpoint=exact_endpoint,
        tail_equation_applies=True,
    )


def verify_clopper_pearson_lower_at_least(
    successes: int,
    trials: int,
    alpha: ProbabilityInput,
    candidate_probability: ProbabilityInput,
) -> bool:
    """Verify ``L_CP(successes; trials, alpha) >= candidate`` exactly."""

    n = _nonnegative_integer(trials, label="trials")
    x = _nonnegative_integer(successes, label="successes")
    if n == 0 or x > n:
        raise ProbabilityCertificateError(
            "Require positive trials and 0 <= successes <= trials."
        )
    a = _strict_unit_probability(alpha, label="alpha")
    candidate = parse_canonical_probability(
        candidate_probability,
        label="candidate_probability",
    )
    if candidate == 0:
        return True
    if x == 0:
        return False
    return exact_binomial_survival(n, x, candidate) <= a


def independent_replica_miss_probability(
    hit_probability: ProbabilityInput,
    replicas: int,
) -> Fraction:
    """Return the exact all-miss probability ``(1-q)**replicas``."""

    q = parse_canonical_probability(
        hit_probability,
        label="hit_probability",
    )
    m = _nonnegative_integer(replicas, label="replicas")
    return (1 - q) ** m


def _miss_is_at_most(
    q: Fraction,
    replicas: int,
    miss_budget: Fraction,
) -> bool:
    """Compare ``(1-q)**replicas <= miss_budget`` by exact integers."""

    base_numerator = q.denominator - q.numerator
    return (
        base_numerator**replicas * miss_budget.denominator
        <= miss_budget.numerator * q.denominator**replicas
    )


def _ceil_log2_reciprocal(value: Fraction) -> int:
    """Smallest ``t`` such that ``2**(-t) <= value``, exactly."""

    if value <= 0 or value > 1:
        raise ProbabilityCertificateError(
            "The dyadic bound requires a probability in (0, 1]."
        )
    if value == 1:
        return 0
    numerator = value.numerator
    denominator = value.denominator
    exponent = max(0, denominator.bit_length() - numerator.bit_length())
    if (numerator << exponent) < denominator:
        exponent += 1
    if exponent > 0 and (numerator << (exponent - 1)) >= denominator:
        exponent -= 1
    return exponent


@dataclass(frozen=True)
class ReplicaCountPlan:
    """An exact-minimum or explicitly conservative replica-count plan."""

    hit_probability: Fraction
    miss_budget: Fraction
    replicas: int | None
    status: str
    is_exact_minimum: bool
    proof_method: str
    certified_miss_upper_bound: Fraction | None
    exact_miss_probability: Fraction | None
    exact_predecessor_miss_probability: Fraction | None
    dyadic_exponent: int | None

    @property
    def feasible(self) -> bool:
        return self.replicas is not None

    def to_jsonable(self) -> dict[str, object]:
        def token(value: Fraction | None) -> str | None:
            return (
                None
                if value is None
                else canonical_rational_string(value)
            )

        return {
            "schema": PROBABILITY_CERTIFICATE_SCHEMA_V15,
            "hit_probability": canonical_rational_string(
                self.hit_probability
            ),
            "miss_budget": canonical_rational_string(self.miss_budget),
            "replicas": self.replicas,
            "status": self.status,
            "is_exact_minimum": self.is_exact_minimum,
            "proof_method": self.proof_method,
            "certified_miss_upper_bound": token(
                self.certified_miss_upper_bound
            ),
            "exact_miss_probability": token(
                self.exact_miss_probability
            ),
            "exact_predecessor_miss_probability": token(
                self.exact_predecessor_miss_probability
            ),
            "dyadic_exponent": self.dyadic_exponent,
            "binary_floating_point_used": False,
        }


def plan_replica_count(
    hit_probability: ProbabilityInput,
    miss_budget: ProbabilityInput,
    *,
    exact_search_limit: int = 4096,
) -> ReplicaCountPlan:
    """Plan replicas without a floating-point loop.

    Exact integer-power binary search is used when the proved upper count is
    no larger than ``exact_search_limit``.  For larger counts, the returned
    count is explicitly marked conservative and follows from

    ``(1-q)**m <= 2**(-q*m) <= 2**(-t) <= miss_budget``.
    """

    q = parse_canonical_probability(
        hit_probability,
        label="hit_probability",
    )
    delta = parse_canonical_probability(
        miss_budget,
        label="miss_budget",
    )
    limit = _nonnegative_integer(
        exact_search_limit,
        label="exact_search_limit",
    )

    if delta == 1:
        return ReplicaCountPlan(
            hit_probability=q,
            miss_budget=delta,
            replicas=0,
            status=REPLICA_PLAN_EXACT_MINIMUM,
            is_exact_minimum=True,
            proof_method="boundary_delta_one",
            certified_miss_upper_bound=Fraction(1),
            exact_miss_probability=Fraction(1),
            exact_predecessor_miss_probability=None,
            dyadic_exponent=0,
        )
    if q == 0:
        return ReplicaCountPlan(
            hit_probability=q,
            miss_budget=delta,
            replicas=None,
            status=REPLICA_PLAN_IMPOSSIBLE,
            is_exact_minimum=False,
            proof_method="zero_hit_probability_has_unit_miss_probability",
            certified_miss_upper_bound=None,
            exact_miss_probability=None,
            exact_predecessor_miss_probability=None,
            dyadic_exponent=None,
        )
    if q == 1:
        return ReplicaCountPlan(
            hit_probability=q,
            miss_budget=delta,
            replicas=1,
            status=REPLICA_PLAN_EXACT_MINIMUM,
            is_exact_minimum=True,
            proof_method="certain_hit_boundary",
            certified_miss_upper_bound=Fraction(0),
            exact_miss_probability=Fraction(0),
            exact_predecessor_miss_probability=Fraction(1),
            dyadic_exponent=None,
        )
    if delta == 0:
        return ReplicaCountPlan(
            hit_probability=q,
            miss_budget=delta,
            replicas=None,
            status=REPLICA_PLAN_IMPOSSIBLE,
            is_exact_minimum=False,
            proof_method="positive_miss_base_never_reaches_zero",
            certified_miss_upper_bound=None,
            exact_miss_probability=None,
            exact_predecessor_miss_probability=None,
            dyadic_exponent=None,
        )

    dyadic_exponent = _ceil_log2_reciprocal(delta)
    conservative_upper = (
        dyadic_exponent * q.denominator + q.numerator - 1
    ) // q.numerator
    dyadic_bound = Fraction(1, 1 << dyadic_exponent)

    if conservative_upper > limit:
        return ReplicaCountPlan(
            hit_probability=q,
            miss_budget=delta,
            replicas=conservative_upper,
            status=REPLICA_PLAN_CONSERVATIVE_UPPER,
            is_exact_minimum=False,
            proof_method=(
                "dyadic_exponential_bound_1_minus_q_le_2_to_minus_q"
            ),
            certified_miss_upper_bound=dyadic_bound,
            exact_miss_probability=None,
            exact_predecessor_miss_probability=None,
            dyadic_exponent=dyadic_exponent,
        )

    lower = 0
    upper = conservative_upper
    while lower < upper:
        midpoint = (lower + upper) // 2
        if _miss_is_at_most(q, midpoint, delta):
            upper = midpoint
        else:
            lower = midpoint + 1
    replicas = lower
    exact_miss = independent_replica_miss_probability(q, replicas)
    predecessor = (
        None
        if replicas == 0
        else independent_replica_miss_probability(q, replicas - 1)
    )
    if exact_miss > delta or (
        predecessor is not None and predecessor <= delta
    ):
        raise AssertionError("Exact minimum replica invariant was violated.")
    return ReplicaCountPlan(
        hit_probability=q,
        miss_budget=delta,
        replicas=replicas,
        status=REPLICA_PLAN_EXACT_MINIMUM,
        is_exact_minimum=True,
        proof_method="exact_integer_power_binary_search",
        certified_miss_upper_bound=exact_miss,
        exact_miss_probability=exact_miss,
        exact_predecessor_miss_probability=predecessor,
        dyadic_exponent=dyadic_exponent,
    )


minimum_replicas_for_miss_probability = plan_replica_count


def pilot_success_threshold(
    trials: int,
    target_probability: ProbabilityInput,
    alpha: ProbabilityInput,
) -> int | None:
    """Return exact ``k*`` for a one-sided CP pilot certificate.

    The root need not be approximated: for ``p0 > 0``,

    ``L_CP(k; n, alpha) >= p0``

    is equivalent to

    ``P_{p0}[Bin(n, p0) >= k] <= alpha``.
    """

    n = _nonnegative_integer(trials, label="trials")
    if n == 0:
        raise ProbabilityCertificateError("trials must be positive.")
    p0 = parse_canonical_probability(
        target_probability,
        label="target_probability",
    )
    a = _strict_unit_probability(alpha, label="alpha")
    if p0 == 0:
        return 0
    if exact_binomial_survival(n, n, p0) > a:
        return None

    lower = 1
    upper = n
    while lower < upper:
        midpoint = (lower + upper) // 2
        if exact_binomial_survival(n, midpoint, p0) <= a:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def pilot_pass_probability(
    trials: int,
    critical_successes: int | None,
    true_probability: ProbabilityInput,
) -> Fraction:
    """Return the exact probability of attaining a frozen pilot threshold."""

    n = _nonnegative_integer(trials, label="trials")
    if n == 0:
        raise ProbabilityCertificateError("trials must be positive.")
    p = parse_canonical_probability(
        true_probability,
        label="true_probability",
    )
    if critical_successes is None:
        return Fraction(0)
    if isinstance(critical_successes, bool) or not isinstance(
        critical_successes, int
    ):
        raise ProbabilityCertificateError(
            "critical_successes must be an integer or None."
        )
    if critical_successes < 0 or critical_successes > n:
        raise ProbabilityCertificateError(
            "critical_successes must lie in [0, trials]."
        )
    return exact_binomial_survival(n, critical_successes, p)


@dataclass(frozen=True)
class PilotPowerCertificate:
    """Exact nonvacuity/power certificate for one pilot cell."""

    trials: int
    target_probability: Fraction
    true_probability_lower_bound: Fraction
    alpha: Fraction
    critical_successes: int | None
    pass_probability_lower_bound: Fraction
    minimum_acceptable_pass_probability: Fraction
    power_gate: bool

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": PROBABILITY_CERTIFICATE_SCHEMA_V15,
            "trials": self.trials,
            "target_probability": canonical_rational_string(
                self.target_probability
            ),
            "true_probability_lower_bound": canonical_rational_string(
                self.true_probability_lower_bound
            ),
            "alpha": canonical_rational_string(self.alpha),
            "critical_successes": self.critical_successes,
            "pass_probability_lower_bound": canonical_rational_string(
                self.pass_probability_lower_bound
            ),
            "minimum_acceptable_pass_probability": (
                canonical_rational_string(
                    self.minimum_acceptable_pass_probability
                )
            ),
            "power_gate": self.power_gate,
            "power_event": "Binomial_success_count_at_least_k_star",
            "power_bound_semantics": (
                "lower_bound_under_true_probability_at_least_p1"
            ),
        }


def certify_pilot_power(
    trials: int,
    target_probability: ProbabilityInput,
    true_probability_lower_bound: ProbabilityInput,
    alpha: ProbabilityInput,
    *,
    minimum_acceptable_pass_probability: ProbabilityInput,
) -> PilotPowerCertificate:
    """Certify a predeclared nonvacuous power gate when ``p >= p1 > p0``."""

    n = _nonnegative_integer(trials, label="trials")
    if n == 0:
        raise ProbabilityCertificateError("trials must be positive.")
    p0 = parse_canonical_probability(
        target_probability,
        label="target_probability",
    )
    if p0 == 0:
        raise ProbabilityCertificateError(
            "target_probability must be positive for a nonvacuous cell-mass "
            "certificate."
        )
    p1 = parse_canonical_probability(
        true_probability_lower_bound,
        label="true_probability_lower_bound",
    )
    if p1 <= p0:
        raise ProbabilityCertificateError(
            "true_probability_lower_bound must be strictly greater than "
            "target_probability."
        )
    a = _strict_unit_probability(alpha, label="alpha")
    minimum_power = _strict_unit_probability(
        minimum_acceptable_pass_probability,
        label="minimum_acceptable_pass_probability",
    )
    threshold = pilot_success_threshold(n, p0, a)
    power = pilot_pass_probability(n, threshold, p1)
    return PilotPowerCertificate(
        trials=n,
        target_probability=p0,
        true_probability_lower_bound=p1,
        alpha=a,
        critical_successes=threshold,
        pass_probability_lower_bound=power,
        minimum_acceptable_pass_probability=minimum_power,
        power_gate=threshold is not None and power >= minimum_power,
    )


def minimum_pilot_trials(
    target_probability: ProbabilityInput,
    true_probability_lower_bound: ProbabilityInput,
    alpha: ProbabilityInput,
    beta: ProbabilityInput,
    *,
    max_trials: int,
) -> PilotPowerCertificate:
    """Find the first ``n <= max_trials`` with exact power at least ``1-beta``.

    The bounded linear search is intentional: the nonrandomized exact
    binomial test has discrete threshold changes, so this function does not
    assume monotonicity in ``n`` that has not been established.
    """

    maximum = _nonnegative_integer(max_trials, label="max_trials")
    if maximum == 0:
        raise ProbabilityCertificateError("max_trials must be positive.")
    miss = _strict_unit_probability(beta, label="beta")
    required_power = 1 - miss
    for trials in range(1, maximum + 1):
        certificate = certify_pilot_power(
            trials,
            target_probability,
            true_probability_lower_bound,
            alpha,
            minimum_acceptable_pass_probability=required_power,
        )
        if certificate.power_gate:
            return certificate
    raise ProbabilityCertificateError(
        "No pilot sample size through max_trials attains the requested "
        "exact power."
    )


def simultaneous_pilot_power_lower_bound(
    individual_power_lower_bounds: Sequence[ProbabilityInput],
) -> Fraction:
    """Bonferroni lower bound for all declared pilot events to pass."""

    powers = tuple(
        parse_canonical_probability(value, label=f"power[{index}]")
        for index, value in enumerate(individual_power_lower_bounds)
    )
    total_failure = sum((1 - value for value in powers), Fraction(0))
    return max(Fraction(0), 1 - total_failure)


@dataclass(frozen=True)
class FalsePassCertificate:
    """The joint false-PASS bound, kept distinct from conditional risk."""

    pilot_familywise_error: Fraction
    confirm_failure_budget: Fraction
    alpha_plus_delta: Fraction
    probability_upper_bound: Fraction
    pass_probability_lower_bound: Fraction | None
    derived_conditional_failure_upper_bound: Fraction | None

    def to_jsonable(self) -> dict[str, object]:
        token = canonical_rational_string
        return {
            "schema": PROBABILITY_CERTIFICATE_SCHEMA_V15,
            "event": FALSE_PASS_EVENT_LABEL,
            "semantics": FALSE_PASS_BOUND_SEMANTICS,
            "pilot_familywise_error": token(
                self.pilot_familywise_error
            ),
            "confirm_failure_budget": token(
                self.confirm_failure_budget
            ),
            "alpha_plus_delta": token(self.alpha_plus_delta),
            "probability_upper_bound": token(
                self.probability_upper_bound
            ),
            "primary_bound_is_conditional_given_pass": False,
            "pass_probability_lower_bound": (
                None
                if self.pass_probability_lower_bound is None
                else token(self.pass_probability_lower_bound)
            ),
            "derived_conditional_failure_upper_bound": (
                None
                if self.derived_conditional_failure_upper_bound is None
                else token(
                    self.derived_conditional_failure_upper_bound
                )
            ),
        }


def build_false_pass_certificate(
    pilot_familywise_error: ProbabilityInput,
    confirm_failure_budget: ProbabilityInput,
    *,
    pass_probability_lower_bound: ProbabilityInput | None = None,
) -> FalsePassCertificate:
    """Build ``P(PASS and confirm failure) <= alpha_P + delta_C``."""

    alpha = parse_canonical_probability(
        pilot_familywise_error,
        label="pilot_familywise_error",
    )
    delta = parse_canonical_probability(
        confirm_failure_budget,
        label="confirm_failure_budget",
    )
    union_bound = alpha + delta
    probability_bound = min(Fraction(1), union_bound)
    q0: Fraction | None = None
    conditional: Fraction | None = None
    if pass_probability_lower_bound is not None:
        q0 = parse_canonical_probability(
            pass_probability_lower_bound,
            label="pass_probability_lower_bound",
        )
        if q0 == 0:
            raise ProbabilityCertificateError(
                "A conditional-risk corollary requires a positive PASS "
                "probability lower bound."
            )
        conditional = min(Fraction(1), probability_bound / q0)
    return FalsePassCertificate(
        pilot_familywise_error=alpha,
        confirm_failure_budget=delta,
        alpha_plus_delta=union_bound,
        probability_upper_bound=probability_bound,
        pass_probability_lower_bound=q0,
        derived_conditional_failure_upper_bound=conditional,
    )


__all__ = [
    "EXACT_OCCUPANCY_CELL_LIMIT",
    "FALSE_PASS_BOUND_SEMANTICS",
    "FALSE_PASS_EVENT_LABEL",
    "OCCUPANCY_EVENT_LABEL",
    "OCCUPANCY_METHOD_BONFERRONI",
    "OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION",
    "PROBABILITY_CERTIFICATE_SCHEMA_V15",
    "REPLICA_PLAN_CONSERVATIVE_UPPER",
    "REPLICA_PLAN_EXACT_MINIMUM",
    "REPLICA_PLAN_IMPOSSIBLE",
    "ClopperPearsonLowerBracket",
    "FalsePassCertificate",
    "OccupancyLowerBoundCertificate",
    "PilotPowerCertificate",
    "ProbabilityCertificateError",
    "ReplicaCountPlan",
    "binomial_survival_probability",
    "build_false_pass_certificate",
    "canonical_rational_string",
    "certify_pilot_power",
    "clopper_pearson_lower_bracket",
    "exact_binomial_survival",
    "independent_replica_miss_probability",
    "minimum_pilot_trials",
    "minimum_replicas_for_miss_probability",
    "mutually_exclusive_cell_occupancy_lower_bound",
    "parse_canonical_probability",
    "pilot_pass_probability",
    "pilot_success_threshold",
    "plan_replica_count",
    "simultaneous_pilot_power_lower_bound",
    "verify_clopper_pearson_lower_at_least",
]
