from __future__ import annotations

"""Distribution-free calibration-to-test control for a frozen design menu."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence

from .pareto_independent_replica_certificate import canonical_rational_string, parse_canonical_probability

FINITE_MENU_GENERALIZATION_SCHEMA_V16 = "pareto_finite_menu_generalization_certificate_v16"

class FiniteMenuGeneralizationError(ValueError):
    pass

def _loss(value: Fraction | int | str, *, label: str) -> Fraction:
    if isinstance(value, str):
        return parse_canonical_probability(value, label=label)
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise FiniteMenuGeneralizationError(f"{label} must be exact in [0,1].")
    resolved = Fraction(value)
    if resolved < 0 or resolved > 1:
        raise FiniteMenuGeneralizationError(f"{label} must lie in [0,1].")
    return resolved

def _ceil_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise FiniteMenuGeneralizationError("Positive log argument required.")
    if value <= 1:
        return 0
    n, d = value.numerator, value.denominator
    exponent = max(0, n.bit_length() - d.bit_length())
    if d << exponent < n:
        exponent += 1
    while exponent > 0 and d << (exponent - 1) >= n:
        exponent -= 1
    return exponent

def _sqrt_upper(value: Fraction, *, bits: int = 256) -> Fraction:
    if value == 0:
        return Fraction(0)
    scaled = -(-(value.numerator << (2 * bits)) // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << bits)

@dataclass(frozen=True)
class FiniteMenuGeneralizationCertificate:
    schema: str
    design_ids: tuple[str, ...]
    calibration_case_count: int
    confidence_error: str
    uniform_radius_upper: str
    selected_design_id: str
    selected_empirical_loss: str
    out_of_sample_loss_upper: str
    oracle_excess_loss_upper: str
    exact_calibration_means: tuple[tuple[str, str], ...]
    independent_calibration_cases_required: bool
    bounded_loss_range: str
    selection_rule: str
    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exact_calibration_means"] = [
            {"design_id": design, "mean": mean}
            for design, mean in self.exact_calibration_means
        ]
        return payload

def certify_finite_menu_generalization(
    losses_by_design: Mapping[str, Sequence[Fraction | int | str]], *,
    confidence_error: Fraction | str,
) -> FiniteMenuGeneralizationCertificate:
    """Uniform Hoeffding bound and a 2-epsilon oracle inequality.

    The radius is an exact dyadic upper bound.  It uses exp(-x)<=2**(-x),
    avoiding binary floating-point logarithms in the certificate.
    """
    if not losses_by_design:
        raise FiniteMenuGeneralizationError("At least one design is required.")
    design_ids = tuple(sorted(losses_by_design))
    rows: dict[str, tuple[Fraction, ...]] = {}
    case_count: int | None = None
    for design in design_ids:
        values = tuple(_loss(v, label=f"loss[{design},{i}]")
                       for i, v in enumerate(losses_by_design[design]))
        if not values:
            raise FiniteMenuGeneralizationError("Every design needs calibration losses.")
        if case_count is None:
            case_count = len(values)
        elif len(values) != case_count:
            raise FiniteMenuGeneralizationError("All designs need the same calibration cases.")
        rows[design] = values
    assert case_count is not None
    delta = parse_canonical_probability(confidence_error, label="confidence_error")
    if delta <= 0 or delta >= 1:
        raise FiniteMenuGeneralizationError("confidence_error must lie in (0,1).")
    exponent = _ceil_log2_fraction(Fraction(2 * len(design_ids), 1) / delta)
    epsilon = _sqrt_upper(Fraction(exponent, 2 * case_count))
    means = {design: sum(values, Fraction(0)) / case_count for design, values in rows.items()}
    selected = min(design_ids, key=lambda design: (means[design], design))
    empirical = means[selected]
    return FiniteMenuGeneralizationCertificate(
        schema=FINITE_MENU_GENERALIZATION_SCHEMA_V16,
        design_ids=design_ids,
        calibration_case_count=case_count,
        confidence_error=canonical_rational_string(delta),
        uniform_radius_upper=canonical_rational_string(epsilon),
        selected_design_id=selected,
        selected_empirical_loss=canonical_rational_string(empirical),
        out_of_sample_loss_upper=canonical_rational_string(min(Fraction(1), empirical + epsilon)),
        oracle_excess_loss_upper=canonical_rational_string(min(Fraction(1), 2 * epsilon)),
        exact_calibration_means=tuple((d, canonical_rational_string(means[d])) for d in design_ids),
        independent_calibration_cases_required=True,
        bounded_loss_range="[0,1]",
        selection_rule="minimum_exact_empirical_mean_then_lexicographic_id",
    )

__all__ = ["FINITE_MENU_GENERALIZATION_SCHEMA_V16", "FiniteMenuGeneralizationCertificate",
           "FiniteMenuGeneralizationError", "certify_finite_menu_generalization"]
