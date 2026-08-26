"""Finite-horizon implementation-kernel perturbation certificates for v18.

If ideal kernels ``K_t`` and implemented kernels ``Khat_t`` satisfy

    sup_x ||Khat_t(x,.) - K_t(x,.)||_TV <= kappa_t,

then the inhomogeneous products from a common initial law differ by at most
``min(1,sum_t kappa_t)`` in total variation.  This is the discrete-time
Duhamel/telescoping inequality.  It converts an ideal regeneration lower bound
into an implementation lower bound without pretending that binary64 MH is the
same kernel as real-arithmetic MH.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Mapping, Sequence

from .pareto_v17_regeneration import as_fraction


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class KernelPerturbationError(ValueError):
    pass


@dataclass(frozen=True)
class KernelPerturbationCertificate:
    semantics: str
    per_step_tv_upper: tuple[Fraction, ...]
    horizon_tv_upper: Fraction
    proof_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "per_step_tv_upper": [str(x) for x in self.per_step_tv_upper],
            "horizon_tv_upper": str(self.horizon_tv_upper),
            "proof_sha256": self.proof_sha256,
        }


def build_kernel_perturbation_certificate(raw: Mapping[str, object] | None) -> KernelPerturbationCertificate:
    if raw is None:
        raise KernelPerturbationError(
            "kernel arithmetic semantics must be explicit; use ideal_real_mh or a verified TV bound"
        )
    semantics = str(raw.get("semantics"))
    if semantics == "ideal_real_mh":
        forbidden = {"per_step_tv_upper", "proof_sha256"}.intersection(raw)
        if forbidden:
            raise KernelPerturbationError("ideal_real_mh must not carry an implementation perturbation")
        return KernelPerturbationCertificate(
            semantics=semantics,
            per_step_tv_upper=(),
            horizon_tv_upper=Fraction(0, 1),
            proof_sha256=None,
        )
    if semantics != "externally_verified_tv_perturbation":
        raise KernelPerturbationError("unsupported kernel arithmetic semantics")
    values = raw.get("per_step_tv_upper")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise KernelPerturbationError("verified perturbation mode requires per_step_tv_upper")
    kappas = tuple(as_fraction(x) for x in values)
    if any(x < 0 or x > 1 for x in kappas):
        raise KernelPerturbationError("TV perturbation bounds must lie in [0,1]")
    proof_sha = raw.get("proof_sha256")
    if not isinstance(proof_sha, str) or _HEX64.fullmatch(proof_sha) is None:
        raise KernelPerturbationError("verified perturbation mode requires a proof SHA-256")
    return KernelPerturbationCertificate(
        semantics=semantics,
        per_step_tv_upper=kappas,
        horizon_tv_upper=min(Fraction(1, 1), sum(kappas, Fraction(0, 1))),
        proof_sha256=proof_sha,
    )


def implementation_probability_lower(
    ideal_probability_lower: Fraction | int | str,
    horizon_tv_upper: Fraction | int | str,
) -> Fraction:
    return max(Fraction(0, 1), as_fraction(ideal_probability_lower) - as_fraction(horizon_tv_upper))


def ideal_probability_lower_from_implementation(
    implementation_probability_lower: Fraction | int | str,
    horizon_tv_upper: Fraction | int | str,
) -> Fraction:
    return max(Fraction(0, 1), as_fraction(implementation_probability_lower) - as_fraction(horizon_tv_upper))


__all__ = [
    "KernelPerturbationCertificate",
    "KernelPerturbationError",
    "build_kernel_perturbation_certificate",
    "ideal_probability_lower_from_implementation",
    "implementation_probability_lower",
]
