"""Source-derived minorization certificates for Pareto-SMC v18.

The v17 packet accepted a number named ``epsilon``.  This module removes that
unproved scalar from the certified branch.  A minorization coefficient is
derived from the raw independence-Metropolis mixture contract

    K = (1-gamma) K_local + gamma K_ind,

where ``K_local`` and ``K_ind`` preserve the same target
``pi(dx) proportional to exp(-beta U(x)) mu(dx)`` and the certified potential
span satisfies ``sup U - inf U <= span``.

For the independence proposal ``mu`` one has

    K_ind(x, A) >= exp(-beta*span) pi(A).

The exponential is replaced by an exact rational lower bound
``(1-x/n)^n <= exp(-x)`` with ``n>x``.  Consequently every certificate-facing
quantity is a ``Fraction`` and a failed inequality cannot pass by floating
rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

from .pareto_v17_regeneration import (
    MinorizationBlock,
    RegenerationCertificateError,
    as_fraction,
)




def exact_fraction_payload(value: Fraction) -> object:
    """Serialize an exact ``Fraction`` without relying on giant decimal integers.

    CPython limits conversion of very large integers to base-10 strings.  The
    v18 residual may contain tens of thousands of decimal digits after raising
    a rational one-step bound to a long mutation schedule.  Hexadecimal integer
    conversion is exact and is not subject to that decimal-digit guard.

    Small fractions keep the compact ``"p/q"`` representation for backwards
    readability.  Large fractions use a canonical hexadecimal pair.
    """

    try:
        return str(value)
    except ValueError:
        return {
            "format": "fraction_hex_v1",
            "numerator_hex": hex(value.numerator),
            "denominator_hex": hex(value.denominator),
        }

class MinorizationProvenanceError(RegenerationCertificateError):
    """Raised when a source-derived kernel contract is malformed."""


KERNEL_SEMANTICS_V18 = "independence_mh_mixture_from_bounded_energy_span_v18"
POTENTIAL_SEMANTICS_V18 = "binary64_frozen_box_augmented_tchebycheff_v18"
FINAL_KERNEL_CONTRACT_V18 = (
    "uniform_symmetric_fixed_origin_two_opt_plus_uniform_fixed_origin_"
    "independence_exact_real_mh_v18"
)
LOCAL_PROPOSAL_CONTRACT_V18 = "uniform_fixed_origin_two_opt_involution_v1"
GLOBAL_PROPOSAL_CONTRACT_V18 = "uniform_fixed_origin_tour_independence_v1"
ACCEPTANCE_CONTRACT_V18 = "exact_real_metropolis_for_frozen_energy_v1"
MIXTURE_CONTRACT_V18 = "state_independent_bernoulli_mixture_v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def binary64_augmented_tchebycheff_span_upper(
    reference_weights: Sequence[float | int | str],
    rho: float | int | str,
) -> tuple[Fraction, tuple[str, ...], str]:
    """Return a conservative upper bound for the *implemented* binary64 energy.

    The runtime computes ``max(w_i z_i) + rho * sum(w_i z_i)`` with
    ``0 <= z_i <= 1``.  Positive multiplication cannot exceed the stored
    weight, while every addition/multiplication used for the upper envelope is
    rounded once toward ``+infinity`` with ``nextafter``.  The returned
    ``Fraction`` is the exact real value of that outward binary64 bound.
    """

    weights = tuple(float(value) for value in reference_weights)
    rho_f = float(rho)
    if not weights or any(not math.isfinite(w) or w <= 0.0 for w in weights):
        raise MinorizationProvenanceError("reference weights must be finite and strictly positive")
    if not math.isfinite(rho_f) or rho_f < 0.0:
        raise MinorizationProvenanceError("rho must be finite and nonnegative")
    sum_upper = 0.0
    for weight in weights:
        sum_upper = math.nextafter(sum_upper + weight, math.inf)
    rho_term_upper = math.nextafter(rho_f * sum_upper, math.inf)
    span_upper = math.nextafter(max(weights) + rho_term_upper, math.inf)
    if not math.isfinite(span_upper):
        raise MinorizationProvenanceError("binary64 potential span overflowed")
    return (
        Fraction.from_float(span_upper),
        tuple(weight.hex() for weight in weights),
        rho_f.hex(),
    )


@dataclass(frozen=True)
class Binary64AugmentedTchebycheffPotentialSpec:
    reference_weights: tuple[float, ...]
    rho: float
    semantics: str = POTENTIAL_SEMANTICS_V18
    _span_upper: Fraction = field(init=False, repr=False, compare=False)
    _weight_hex: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _rho_hex: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        span, weight_hex, rho_hex = binary64_augmented_tchebycheff_span_upper(
            self.reference_weights, self.rho
        )
        object.__setattr__(self, "reference_weights", tuple(float(x) for x in self.reference_weights))
        object.__setattr__(self, "rho", float(self.rho))
        if self.semantics != POTENTIAL_SEMANTICS_V18:
            raise MinorizationProvenanceError("unsupported potential semantics")
        object.__setattr__(self, "_span_upper", span)
        object.__setattr__(self, "_weight_hex", weight_hex)
        object.__setattr__(self, "_rho_hex", rho_hex)

    @property
    def energy_span_upper(self) -> Fraction:
        return self._span_upper

    def target_sha256(self, *, context_sha256: str, type_id: str, beta: Fraction) -> str:
        if _HEX64.fullmatch(context_sha256) is None:
            raise MinorizationProvenanceError("context SHA-256 is malformed")
        return _canonical_sha256({
            "semantics": self.semantics,
            "context_sha256": context_sha256,
            "type_id": type_id,
            "beta": exact_fraction_payload(beta),
            "reference_weight_hex": list(self._weight_hex),
            "rho_hex": self._rho_hex,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "reference_weight_hex": list(self._weight_hex),
            "rho_hex": self._rho_hex,
            "energy_span_upper": exact_fraction_payload(self.energy_span_upper),
        }


@dataclass(frozen=True)
class IdealFinalKernelContract:
    """Ideal-kernel contract needed by the regeneration product theorem.

    Minorization alone is insufficient to multiply residual factors over
    several steps. Every block must also preserve the same frozen target.
    This contract fixes the exact ideal proposal/acceptance semantics from
    which reversibility, and hence target invariance, follows. The booleans
    ``proposal_symmetric`` and ``target_invariant`` are derived outputs rather
    than caller-supplied claims.
    """

    semantics: str = FINAL_KERNEL_CONTRACT_V18
    local_proposal: str = LOCAL_PROPOSAL_CONTRACT_V18
    global_proposal: str = GLOBAL_PROPOSAL_CONTRACT_V18
    acceptance: str = ACCEPTANCE_CONTRACT_V18
    mixture: str = MIXTURE_CONTRACT_V18

    def __post_init__(self) -> None:
        expected = (
            FINAL_KERNEL_CONTRACT_V18,
            LOCAL_PROPOSAL_CONTRACT_V18,
            GLOBAL_PROPOSAL_CONTRACT_V18,
            ACCEPTANCE_CONTRACT_V18,
            MIXTURE_CONTRACT_V18,
        )
        actual = (
            self.semantics,
            self.local_proposal,
            self.global_proposal,
            self.acceptance,
            self.mixture,
        )
        if actual != expected:
            raise MinorizationProvenanceError(
                "unsupported ideal final-kernel invariance contract"
            )

    @property
    def target_invariant(self) -> bool:
        return True

    @property
    def proposal_symmetric(self) -> bool:
        return True

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "local_proposal": self.local_proposal,
            "global_proposal": self.global_proposal,
            "acceptance": self.acceptance,
            "mixture": self.mixture,
            "proposal_symmetric": self.proposal_symmetric,
            "target_invariant": self.target_invariant,
            "invariance_proof": (
                "both proposal kernels use exact real Metropolis-Hastings for "
                "the same frozen target; each is reversible and their "
                "state-independent convex mixture is reversible"
            ),
        }


def parse_ideal_final_kernel_contract(
    raw: Mapping[str, object],
) -> IdealFinalKernelContract:
    if not isinstance(raw, Mapping):
        raise MinorizationProvenanceError(
            "ideal_kernel_contract must be a mapping"
        )
    forbidden = {"proposal_symmetric", "target_invariant", "reversible"}.intersection(raw)
    if forbidden:
        raise MinorizationProvenanceError(
            "kernel invariance booleans are derived and must not be supplied"
        )
    return IdealFinalKernelContract(
        semantics=str(raw.get("semantics", "")),
        local_proposal=str(raw.get("local_proposal", "")),
        global_proposal=str(raw.get("global_proposal", "")),
        acceptance=str(raw.get("acceptance", "")),
        mixture=str(raw.get("mixture", "")),
    )


def parse_potential_spec(raw: Mapping[str, object]) -> Binary64AugmentedTchebycheffPotentialSpec:
    if not isinstance(raw, Mapping):
        raise MinorizationProvenanceError("potential_contract must be a mapping")
    if "energy_span_upper" in raw:
        raise MinorizationProvenanceError(
            "the certified packet must derive, not accept, the potential span"
        )
    weights = raw.get("reference_weights")
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
        raise MinorizationProvenanceError("potential contract requires reference_weights")
    return Binary64AugmentedTchebycheffPotentialSpec(
        reference_weights=tuple(float(value) for value in weights),
        rho=float(raw.get("rho")),
        semantics=str(raw.get("semantics", POTENTIAL_SEMANTICS_V18)),
    )


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def rational_exp_neg_lower(
    exponent: Fraction | int | str,
    *,
    subdivisions: int | None = None,
) -> Fraction:
    """Return an exact rational lower bound on ``exp(-exponent)``.

    For ``x>=0`` and any integer ``n>x``, ``1-x/n <= exp(-x/n)`` and hence
    ``(1-x/n)^n <= exp(-x)``.  The default deterministic subdivision count is
    at least 64 and at least ``16*x+1``.  Larger counts improve the bound but
    increase integer sizes.
    """

    x = as_fraction(exponent)
    if x < 0:
        raise MinorizationProvenanceError("exponent must be nonnegative")
    if x == 0:
        return Fraction(1, 1)
    if subdivisions is None:
        subdivisions = max(64, ceil_fraction(16 * x) + 1)
    if not isinstance(subdivisions, int) or subdivisions <= 0:
        raise MinorizationProvenanceError("subdivisions must be a positive integer")
    n = Fraction(subdivisions, 1)
    if n <= x:
        raise MinorizationProvenanceError("subdivisions must be strictly larger than exponent")
    return (Fraction(1, 1) - x / n) ** subdivisions


def rational_exp_neg_upper(
    exponent: Fraction | int | str,
    *,
    subdivisions: int | None = None,
) -> Fraction:
    """Return an exact rational upper bound on ``exp(-exponent)``.

    ``exp(x/n) >= 1+x/n`` implies
    ``exp(-x) <= (1+x/n)^(-n)``.
    """

    x = as_fraction(exponent)
    if x < 0:
        raise MinorizationProvenanceError("exponent must be nonnegative")
    if x == 0:
        return Fraction(1, 1)
    if subdivisions is None:
        subdivisions = max(64, ceil_fraction(16 * x) + 1)
    if not isinstance(subdivisions, int) or subdivisions <= 0:
        raise MinorizationProvenanceError("subdivisions must be a positive integer")
    n = Fraction(subdivisions, 1)
    return Fraction(1, 1) / (Fraction(1, 1) + x / n) ** subdivisions


@dataclass(frozen=True)
class IndependenceMHMinorizationSpec:
    """Raw contract from which a minorization coefficient is derived.

    ``energy_span_upper`` must prove ``sup U - inf U <= span`` for the frozen
    target potential.  For the augmented-Tchebycheff target on a valid frozen
    objective box this can be instantiated with ``1+rho``.
    """

    gamma: Fraction
    beta: Fraction
    energy_span_upper: Fraction
    steps: int
    subdivisions: int = 256
    kernel_semantics: str = KERNEL_SEMANTICS_V18

    def __post_init__(self) -> None:
        gamma = as_fraction(self.gamma)
        beta = as_fraction(self.beta)
        span = as_fraction(self.energy_span_upper)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "energy_span_upper", span)
        if not (Fraction(0, 1) < gamma <= Fraction(1, 1)):
            raise MinorizationProvenanceError("gamma must lie in (0,1]")
        if beta < 0 or span < 0:
            raise MinorizationProvenanceError("beta and energy span must be nonnegative")
        if not isinstance(self.steps, int) or self.steps < 0:
            raise MinorizationProvenanceError("steps must be a nonnegative integer")
        if not isinstance(self.subdivisions, int) or self.subdivisions <= 0:
            raise MinorizationProvenanceError("subdivisions must be positive")
        if self.kernel_semantics != KERNEL_SEMANTICS_V18:
            raise MinorizationProvenanceError("unsupported kernel semantics")

    @property
    def exponent(self) -> Fraction:
        return self.beta * self.energy_span_upper

    @property
    def density_ratio_lower(self) -> Fraction:
        return rational_exp_neg_lower(
            self.exponent,
            subdivisions=self.subdivisions,
        )

    @property
    def epsilon_lower(self) -> Fraction:
        return self.gamma * self.density_ratio_lower

    @property
    def residual_upper(self) -> Fraction:
        return (Fraction(1, 1) - self.epsilon_lower) ** self.steps

    def as_minorization_block(self) -> MinorizationBlock:
        return MinorizationBlock(self.epsilon_lower, self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_semantics": self.kernel_semantics,
            "gamma": exact_fraction_payload(self.gamma),
            "beta": exact_fraction_payload(self.beta),
            "energy_span_upper": exact_fraction_payload(self.energy_span_upper),
            "steps": self.steps,
            "subdivisions": self.subdivisions,
            "exponent": exact_fraction_payload(self.exponent),
            "density_ratio_lower": exact_fraction_payload(self.density_ratio_lower),
            "epsilon_lower": exact_fraction_payload(self.epsilon_lower),
            "residual_upper": exact_fraction_payload(self.residual_upper),
        }


@dataclass(frozen=True)
class DerivedTypeMinorization:
    type_id: str
    target_sha256: str
    potential: Binary64AugmentedTchebycheffPotentialSpec
    ideal_kernel_contract: IdealFinalKernelContract
    pilot: tuple[IndependenceMHMinorizationSpec, ...]
    confirm: tuple[IndependenceMHMinorizationSpec, ...]

    def __post_init__(self) -> None:
        if not self.type_id:
            raise MinorizationProvenanceError("type_id must be nonempty")
        if _HEX64.fullmatch(self.target_sha256) is None:
            raise MinorizationProvenanceError("final target SHA-256 is malformed")
        if not self.pilot or not self.confirm:
            raise MinorizationProvenanceError("pilot and confirm final blocks must be nonempty")
        if not self.ideal_kernel_contract.target_invariant:
            raise MinorizationProvenanceError(
                "regeneration products require target-invariant final kernels"
            )
        betas = {block.beta for block in (*self.pilot, *self.confirm)}
        spans = {block.energy_span_upper for block in (*self.pilot, *self.confirm)}
        if len(betas) != 1:
            raise MinorizationProvenanceError(
                "final-regeneration blocks must preserve one common final-beta target"
            )
        if spans != {self.potential.energy_span_upper}:
            raise MinorizationProvenanceError("minorization span differs from the derived potential span")

    @property
    def pilot_residual_upper(self) -> Fraction:
        result = Fraction(1, 1)
        for block in self.pilot:
            result *= block.residual_upper
        return result

    @property
    def confirm_residual_upper(self) -> Fraction:
        result = Fraction(1, 1)
        for block in self.confirm:
            result *= block.residual_upper
        return result


def parse_independence_mh_spec(
    raw: Mapping[str, object],
    *,
    energy_span_upper: Fraction,
) -> IndependenceMHMinorizationSpec:
    if not isinstance(raw, Mapping):
        raise MinorizationProvenanceError("minorization block must be a mapping")
    forbidden = {"epsilon", "epsilon_lower", "minorization", "energy_span_upper"}.intersection(raw)
    if forbidden:
        raise MinorizationProvenanceError(
            "certificate input must not supply a derived minorization coefficient or span"
        )
    return IndependenceMHMinorizationSpec(
        gamma=as_fraction(raw["gamma"]),
        beta=as_fraction(raw["beta"]),
        energy_span_upper=energy_span_upper,
        steps=int(raw["steps"]),
        subdivisions=int(raw.get("subdivisions", 256)),
        kernel_semantics=str(raw.get("kernel_semantics", KERNEL_SEMANTICS_V18)),
    )


def parse_type_minorization(
    raw: Mapping[str, object],
    *,
    expected_type_id: str,
    context_sha256: str,
) -> DerivedTypeMinorization:
    if not isinstance(raw, Mapping) or str(raw.get("type_id")) != expected_type_id:
        raise MinorizationProvenanceError("minorization type ordering mismatch")
    if "pilot_blocks" in raw or "confirm_blocks" in raw:
        raise MinorizationProvenanceError(
            "legacy stagewise block fields are forbidden; only final-target blocks may be certified"
        )
    potential = parse_potential_spec(raw.get("potential_contract"))  # type: ignore[arg-type]
    ideal_kernel_contract = parse_ideal_final_kernel_contract(
        raw.get("ideal_kernel_contract")  # type: ignore[arg-type]
    )
    pilot_raw = raw.get("pilot_final_blocks")
    confirm_raw = raw.get("confirm_final_blocks")
    if not isinstance(pilot_raw, Sequence) or isinstance(pilot_raw, (str, bytes)) or not pilot_raw:
        raise MinorizationProvenanceError("pilot_final_blocks must be a nonempty sequence")
    if not isinstance(confirm_raw, Sequence) or isinstance(confirm_raw, (str, bytes)) or not confirm_raw:
        raise MinorizationProvenanceError("confirm_final_blocks must be a nonempty sequence")
    pilot = tuple(
        parse_independence_mh_spec(item, energy_span_upper=potential.energy_span_upper)
        for item in pilot_raw
    )
    confirm = tuple(
        parse_independence_mh_spec(item, energy_span_upper=potential.energy_span_upper)
        for item in confirm_raw
    )
    beta = pilot[0].beta
    target_sha = potential.target_sha256(
        context_sha256=context_sha256, type_id=expected_type_id, beta=beta
    )
    return DerivedTypeMinorization(
        type_id=expected_type_id,
        target_sha256=target_sha,
        potential=potential,
        ideal_kernel_contract=ideal_kernel_contract,
        pilot=pilot,
        confirm=confirm,
    )


__all__ = [
    "Binary64AugmentedTchebycheffPotentialSpec",
    "IdealFinalKernelContract",
    "DerivedTypeMinorization",
    "FINAL_KERNEL_CONTRACT_V18",
    "LOCAL_PROPOSAL_CONTRACT_V18",
    "GLOBAL_PROPOSAL_CONTRACT_V18",
    "ACCEPTANCE_CONTRACT_V18",
    "MIXTURE_CONTRACT_V18",
    "exact_fraction_payload",
    "IndependenceMHMinorizationSpec",
    "KERNEL_SEMANTICS_V18",
    "POTENTIAL_SEMANTICS_V18",
    "MinorizationProvenanceError",
    "binary64_augmented_tchebycheff_span_upper",
    "parse_independence_mh_spec",
    "parse_ideal_final_kernel_contract",
    "parse_potential_spec",
    "parse_type_minorization",
    "rational_exp_neg_lower",
    "rational_exp_neg_upper",
]
