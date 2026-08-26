from __future__ import annotations

"""Adaptive type allocation for the IJOC-oriented Pareto-SMC branch.

The allocator is deliberately separated from the invariant SMC core.  It is
used only by the post-certificate search tail, where each objective evaluation
is treated as one bandit round.  The implementation follows the classical
EXP3 update for rewards in ``[0, 1]``.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Dict, Sequence, Tuple


def derive_domain_separated_seed(
    master_seed: int,
    *,
    context: str,
    domain: str,
) -> int:
    """Derive a reproducible 256-bit seed for one stochastic role.

    Domain separation does not turn Python's deterministic PRNG into an ideal
    product probability space.  It does prevent the implementation from
    consuming allocator draws and counterfactual proposal tapes from the same
    mutable RNG state, which is the source-level nonanticipation contract used
    by the IJOC branch.
    """

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("master_seed must be an integer.")
    if not isinstance(context, str) or not context:
        raise ValueError("context must be a nonempty string.")
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a nonempty string.")
    payload = {
        "schema": "ijoc_domain_separated_seed_v1",
        "master_seed": master_seed,
        "context": context,
        "domain": domain,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


@dataclass(frozen=True)
class SearchRewardWeights:
    """Convex weights for an observable search reward."""

    hypervolume: float = 0.75
    new_cell: float = 0.20
    scalar_improvement: float = 0.05

    def __post_init__(self) -> None:
        values = (self.hypervolume, self.new_cell, self.scalar_improvement)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Search reward weights must be finite and nonnegative.")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Search reward weights must sum to one.")

    def combine(
        self,
        *,
        normalized_hypervolume_gain: float,
        new_cell: bool,
        normalized_scalar_improvement: float,
    ) -> float:
        hv = min(1.0, max(0.0, float(normalized_hypervolume_gain)))
        scalar = min(1.0, max(0.0, float(normalized_scalar_improvement)))
        reward = math.fsum(
            (
                self.hypervolume * hv,
                self.new_cell * float(bool(new_cell)),
                self.scalar_improvement * scalar,
            )
        )
        # Directed clamping protects the EXP3 contract from insignificant
        # floating roundoff while preserving the mathematical [0, 1] range.
        if reward > 1.0 - 4.0 * math.ulp(1.0):
            return 1.0
        return min(1.0, max(0.0, reward))


@dataclass(frozen=True)
class Exp3Snapshot:
    rounds: int
    probabilities: Tuple[float, ...]
    pulls: Tuple[int, ...]
    observed_reward_sums: Tuple[float, ...]
    total_observed_reward: float
    exploration: float
    learning_rate: float


class Exp3TypeAllocator:
    """Numerically stable EXP3 allocator for typed mutation proposals.

    For ``K`` types and exploration parameter ``gamma``, the distribution is

    ``p_i = (1-gamma) w_i / sum_j w_j + gamma / K``

    and the selected arm receives the importance-weighted update

    ``log w_i += gamma * reward / (K * p_i)``.

    Since ``p_i >= gamma / K`` and ``reward <= 1``, the exponent increment is
    at most one, matching the usual finite-horizon EXP3 proof.
    """

    def __init__(
        self,
        num_types: int,
        *,
        exploration: float,
    ) -> None:
        if isinstance(num_types, bool) or not isinstance(num_types, int) or num_types <= 0:
            raise ValueError("num_types must be a positive integer.")
        if not math.isfinite(float(exploration)) or not 0.0 < float(exploration) <= 1.0:
            raise ValueError("exploration must lie in (0, 1].")
        self.num_types = num_types
        self.exploration = float(exploration)
        self.learning_rate = self.exploration / self.num_types
        self._log_weights = [0.0] * num_types
        self._pulls = [0] * num_types
        self._reward_sums = [0.0] * num_types
        self._rounds = 0
        self._last_probabilities: Tuple[float, ...] = tuple(
            1.0 / num_types for _ in range(num_types)
        )
        self._pending_selection: Tuple[int, float] | None = None

    @staticmethod
    def recommended_exploration(num_types: int, horizon: int) -> float:
        if isinstance(num_types, bool) or not isinstance(num_types, int) or num_types <= 0:
            raise ValueError("num_types must be a positive integer.")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        numerator = num_types * math.log(max(2, num_types))
        denominator = (math.e - 1.0) * horizon
        return min(1.0, math.sqrt(numerator / denominator))

    def probabilities(self) -> Tuple[float, ...]:
        maximum = max(self._log_weights)
        scaled = [math.exp(value - maximum) for value in self._log_weights]
        normalizer = sum(scaled)
        base = [value / normalizer for value in scaled]
        uniform = self.exploration / self.num_types
        probabilities = [
            (1.0 - self.exploration) * value + uniform
            for value in base
        ]
        # Apply the tiny normalization residual to the largest coordinate.
        # Unlike dividing every coordinate by a rounded total, this preserves
        # the exact proof-domain floor p_i >= gamma / K.
        residual = 1.0 - math.fsum(probabilities)
        correction_index = max(range(self.num_types), key=probabilities.__getitem__)
        probabilities[correction_index] += residual
        if any(value < uniform for value in probabilities):
            raise RuntimeError("Floating EXP3 probabilities violated the exploration floor.")
        if abs(math.fsum(probabilities) - 1.0) > 4.0 * math.ulp(1.0):
            raise RuntimeError("Floating EXP3 probabilities failed to normalize.")
        resolved = tuple(probabilities)
        self._last_probabilities = resolved
        return resolved

    def select(self, rng: random.Random) -> Tuple[int, float]:
        if self._pending_selection is not None:
            raise RuntimeError(
                "EXP3 requires observing the previous selection before drawing again."
            )
        probabilities = self.probabilities()
        draw = rng.random()
        cumulative = 0.0
        selected = self.num_types - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                selected = index
                break
        selected_probability = probabilities[selected]
        self._pending_selection = (selected, selected_probability)
        return selected, selected_probability

    def observe(self, type_index: int, reward: float, selection_probability: float) -> None:
        if type_index < 0 or type_index >= self.num_types:
            raise IndexError("type_index is out of range.")
        value = float(reward)
        probability = float(selection_probability)
        pending = self._pending_selection
        if pending is None:
            raise RuntimeError("EXP3 observe() requires a preceding select().")
        if type_index != pending[0] or probability != pending[1]:
            raise RuntimeError(
                "EXP3 observation does not match the pending selected arm/probability."
            )
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("EXP3 rewards must lie in [0, 1].")
        if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
            raise ValueError("selection_probability must lie in (0, 1].")
        increment = self.learning_rate * value / probability
        # The exploration floor implies increment <= 1.  Reject a broken
        # probability contract rather than silently changing the algorithm.
        if increment > 1.0 + 1e-12:
            raise RuntimeError("EXP3 importance-weighted update exceeded its proof domain.")
        self._log_weights[type_index] += min(1.0, increment)
        maximum = max(self._log_weights)
        if maximum > 500.0:
            self._log_weights = [weight - maximum for weight in self._log_weights]
        self._pulls[type_index] += 1
        self._reward_sums[type_index] += value
        self._rounds += 1
        self._pending_selection = None

    def regret_upper_bound(self, horizon: int | None = None) -> float:
        """Certified external-regret bound on the realized-state reward sequence.

        For rewards in ``[0,1]`` and a nonanticipating adaptive environment,
        classical EXP3 gives

        ``E[Regret_T] <= (e-1) gamma T + K log(K) / gamma``.

        Regret is also trivially at most ``T``.  This is external regret for
        the one-step counterfactual rewards defined on the actually realized
        search states; it is not policy regret for the trajectory that would
        have resulted from committing to one type from round zero.
        """

        rounds = self._rounds if horizon is None else int(horizon)
        if rounds < 0:
            raise ValueError("horizon must be nonnegative.")
        if self.num_types == 1:
            return 0.0
        classical = (
            (math.e - 1.0) * self.exploration * rounds
            + self.num_types * math.log(self.num_types) / self.exploration
        )
        return min(float(rounds), classical)

    def classical_regret_expression(self, horizon: int | None = None) -> float:
        rounds = self._rounds if horizon is None else int(horizon)
        if rounds < 0:
            raise ValueError("horizon must be nonnegative.")
        if self.num_types == 1:
            return 0.0
        return (
            (math.e - 1.0) * self.exploration * rounds
            + self.num_types * math.log(self.num_types) / self.exploration
        )

    def snapshot(self) -> Exp3Snapshot:
        return Exp3Snapshot(
            rounds=self._rounds,
            probabilities=self.probabilities(),
            pulls=tuple(self._pulls),
            observed_reward_sums=tuple(self._reward_sums),
            total_observed_reward=sum(self._reward_sums),
            exploration=self.exploration,
            learning_rate=self.learning_rate,
        )

    def metadata(self) -> Dict[str, object]:
        snapshot = self.snapshot()
        return {
            "allocator": "exp3_type_allocator_v1",
            "rounds": snapshot.rounds,
            "probabilities": snapshot.probabilities,
            "pulls": snapshot.pulls,
            "observed_reward_sums": snapshot.observed_reward_sums,
            "total_observed_reward": snapshot.total_observed_reward,
            "exploration": snapshot.exploration,
            "learning_rate": snapshot.learning_rate,
            "expected_external_regret_upper_bound": self.regret_upper_bound(),
            "classical_external_regret_expression": (
                self.classical_regret_expression()
            ),
            "regret_scope": (
                "external_regret_on_realized_state_one_step_counterfactual_"
                "reward_vectors_not_policy_regret_or_final_metric_regret"
            ),
            "final_hypervolume_regret_claimed": False,
        }


def normalized_hypervolume_gain(
    before: float,
    after: float,
    *,
    objective_box_volume: float,
) -> float:
    if not all(math.isfinite(value) for value in (before, after, objective_box_volume)):
        raise ValueError("Hypervolume reward inputs must be finite.")
    if objective_box_volume <= 0.0:
        raise ValueError("objective_box_volume must be positive.")
    return min(1.0, max(0.0, (after - before) / objective_box_volume))
