from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry
from .potential import HypervolumeArchivePotential, PotentialContext, ScalarArchivePotential
from .types import ObjectiveVector


class TinyMLP:
    """A tiny dependency-free tanh MLP for scalar potential fitting."""

    def __init__(self, input_dim: int, hidden_units: int, rng: random.Random) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.rng = rng
        scale = 0.2
        self.w1 = [
            [rng.uniform(-scale, scale) for _ in range(input_dim)]
            for _ in range(hidden_units)
        ]
        self.b1 = [0.0 for _ in range(hidden_units)]
        self.w2 = [rng.uniform(-scale, scale) for _ in range(hidden_units)]
        self.b2 = 0.0
        self._numpy_cache: Optional[Tuple[object, object, object, object]] = None

    def predict(self, x: Sequence[float]) -> float:
        hidden = self._hidden(x)
        return self.b2 + sum(w * h for w, h in zip(self.w2, hidden))

    def predict_batch(self, inputs: Sequence[Sequence[float]]) -> List[float]:
        if not inputs:
            return []
        try:
            import numpy as np

            w1, b1, w2, b2 = self._numpy_weights()
            x = np.asarray(inputs, dtype=float)
            hidden = np.tanh(x @ w1.T + b1)
            return (hidden @ w2 + b2).tolist()
        except Exception:
            return [self.predict(x) for x in inputs]

    def _numpy_weights(self) -> Tuple[object, object, object, object]:
        if self._numpy_cache is None:
            import numpy as np

            self._numpy_cache = (
                np.asarray(self.w1, dtype=float),
                np.asarray(self.b1, dtype=float),
                np.asarray(self.w2, dtype=float),
                float(self.b2),
            )
        return self._numpy_cache

    def _invalidate_numpy_cache(self) -> None:
        self._numpy_cache = None

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, payload: dict, rng: random.Random) -> "TinyMLP":
        net = cls(int(payload["input_dim"]), int(payload["hidden_units"]), rng)
        net.w1 = [[float(value) for value in row] for row in payload["w1"]]
        net.b1 = [float(value) for value in payload["b1"]]
        net.w2 = [float(value) for value in payload["w2"]]
        net.b2 = float(payload["b2"])
        net._invalidate_numpy_cache()
        return net

    def fit(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        epochs: int,
        learning_rate: float,
    ) -> None:
        if not inputs:
            return
        try:
            self._fit_numpy(inputs, targets, epochs, learning_rate)
            return
        except Exception:
            pass
        self._fit_python(inputs, targets, epochs, learning_rate)

    def fit_mixed(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        residual_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        ranking_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        hypercone_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        epochs: int,
        learning_rate: float,
        flow_residual_weight: float = 1.0,
        ranking_weight: float = 0.0,
        hypercone_weight: float = 0.0,
        weight_norm_bound: float = 0.0,
        coverage_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]] = (),
        expert_pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]] = (),
        coverage_weight: float = 0.0,
        expert_weight: float = 0.0,
    ) -> None:
        """Fit a scalar oracle with state, flow-residual, ranking, and cone losses.

        The network always outputs a scalar state potential. Pair losses update
        scalar differences, which preserves antisymmetry of potential deltas by
        construction: the caller computes g(y) - g(x), never a direct edge score.
        """

        if epochs <= 0 or learning_rate <= 0.0:
            return
        for _ in range(epochs):
            if inputs:
                self.fit(inputs, targets, 1, learning_rate)
            if flow_residual_weight > 0.0:
                if not self._pair_updates_numpy(residual_pairs, learning_rate, flow_residual_weight, "residual"):
                    self._shuffle_pair_updates(residual_pairs, learning_rate, flow_residual_weight, "residual")
            if ranking_weight > 0.0:
                if not self._pair_updates_numpy(ranking_pairs, learning_rate, ranking_weight, "margin"):
                    self._shuffle_pair_updates(ranking_pairs, learning_rate, ranking_weight, "margin")
            if hypercone_weight > 0.0:
                if not self._pair_updates_numpy(hypercone_pairs, learning_rate, hypercone_weight, "margin"):
                    self._shuffle_pair_updates(hypercone_pairs, learning_rate, hypercone_weight, "margin")
            if coverage_weight > 0.0:
                if not self._pair_updates_numpy(coverage_pairs, learning_rate, coverage_weight, "margin"):
                    self._shuffle_pair_updates(coverage_pairs, learning_rate, coverage_weight, "margin")
            if expert_weight > 0.0:
                if not self._pair_updates_numpy(expert_pairs, learning_rate, expert_weight, "margin"):
                    self._shuffle_pair_updates(expert_pairs, learning_rate, expert_weight, "margin")
            self.clip_weight_norms(weight_norm_bound)

    def _pair_updates_numpy(
        self,
        pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        learning_rate: float,
        loss_weight: float,
        mode: str,
    ) -> bool:
        if not pairs:
            return True
        try:
            import numpy as np

            first = np.asarray([pair[0] for pair in pairs], dtype=float)
            second = np.asarray([pair[1] for pair in pairs], dtype=float)
            value = np.asarray([pair[2] for pair in pairs], dtype=float)
            w1 = np.asarray(self.w1, dtype=float)
            b1 = np.asarray(self.b1, dtype=float)
            w2 = np.asarray(self.w2, dtype=float)
            b2 = float(self.b2)
            h_first = np.tanh(first @ w1.T + b1)
            h_second = np.tanh(second @ w1.T + b1)
            out_first = h_first @ w2 + b2
            out_second = h_second @ w2 + b2
            if mode == "residual":
                err = np.clip((out_second - out_first) - value, -10.0, 10.0)
                grad_first = -loss_weight * err
                grad_second = loss_weight * err
            else:
                violation = value + out_first - out_second
                active = violation > 0.0
                if not bool(np.any(active)):
                    return True
                step = np.clip(loss_weight * violation, 0.0, 2.0)
                grad_first = np.where(active, step, 0.0)
                grad_second = np.where(active, -step, 0.0)
            scale = 1.0 / max(1, len(pairs))
            grad_first *= scale
            grad_second *= scale
            old_w2 = w2.copy()
            grad_w2 = grad_first @ h_first + grad_second @ h_second
            grad_b2 = float(grad_first.sum() + grad_second.sum())
            delta_first = (grad_first[:, None] * old_w2[None, :]) * (1.0 - h_first * h_first)
            delta_second = (grad_second[:, None] * old_w2[None, :]) * (1.0 - h_second * h_second)
            grad_w1 = delta_first.T @ first + delta_second.T @ second
            grad_b1 = delta_first.sum(axis=0) + delta_second.sum(axis=0)
            self.w2 = (w2 - learning_rate * grad_w2).tolist()
            self.b2 = b2 - learning_rate * grad_b2
            self.w1 = (w1 - learning_rate * grad_w1).tolist()
            self.b1 = (b1 - learning_rate * grad_b1).tolist()
            self._invalidate_numpy_cache()
            return True
        except Exception:
            return False

    def _shuffle_pair_updates(
        self,
        pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        learning_rate: float,
        loss_weight: float,
        mode: str,
    ) -> None:
        if not pairs:
            return
        order = list(range(len(pairs)))
        self._shuffle_indices(order)
        for idx in order:
            first, second, value = pairs[idx]
            pred_first = self.predict(first)
            pred_second = self.predict(second)
            if mode == "residual":
                err = max(-10.0, min(10.0, (pred_second - pred_first) - value))
                step = max(-2.0, min(2.0, loss_weight * err))
                self._train_one(second, pred_second - step, learning_rate)
                self._train_one(first, pred_first + step, learning_rate)
            else:
                violation = value + pred_first - pred_second
                if violation <= 0.0:
                    continue
                step = max(0.0, min(2.0, loss_weight * violation))
                self._train_one(first, pred_first - step, learning_rate)
                self._train_one(second, pred_second + step, learning_rate)

    def clip_weight_norms(self, bound: float) -> None:
        if bound <= 0.0:
            return
        try:
            import numpy as np

            w1 = np.asarray(self.w1, dtype=float)
            w2 = np.asarray(self.w2, dtype=float)
            sigma = float(np.linalg.norm(w1, ord=2)) if w1.size else 0.0
            if sigma > bound:
                w1 *= bound / max(sigma, 1e-12)
            norm2 = float(np.linalg.norm(w2)) if w2.size else 0.0
            if norm2 > bound:
                w2 *= bound / max(norm2, 1e-12)
            b1 = np.clip(np.asarray(self.b1, dtype=float), -bound, bound)
            self.w1 = w1.tolist()
            self.w2 = w2.tolist()
            self.b1 = b1.tolist()
            self.b2 = max(-bound, min(bound, float(self.b2)))
            self._invalidate_numpy_cache()
            return
        except Exception:
            pass

        for row in self.w1:
            norm = sum(value * value for value in row) ** 0.5
            if norm > bound:
                scale = bound / max(norm, 1e-12)
                for idx, value in enumerate(row):
                    row[idx] = value * scale
        norm2 = sum(value * value for value in self.w2) ** 0.5
        if norm2 > bound:
            scale = bound / max(norm2, 1e-12)
            for idx, value in enumerate(self.w2):
                self.w2[idx] = value * scale
        self.b1 = [max(-bound, min(bound, value)) for value in self.b1]
        self.b2 = max(-bound, min(bound, self.b2))
        self._invalidate_numpy_cache()

    def spectral_diagnostics(self, bound: float = 0.0) -> dict:
        """Return exact small-network spectral diagnostics for stability audits."""
        try:
            import numpy as np

            w1 = np.asarray(self.w1, dtype=float)
            w2 = np.asarray(self.w2, dtype=float)
            sigma1 = float(np.linalg.norm(w1, ord=2)) if w1.size else 0.0
            norm2 = float(np.linalg.norm(w2)) if w2.size else 0.0
            bias1 = float(np.max(np.abs(np.asarray(self.b1, dtype=float)))) if self.b1 else 0.0
            bias2 = abs(float(self.b2))
        except Exception:
            sigma1 = max((sum(value * value for value in row) ** 0.5 for row in self.w1), default=0.0)
            norm2 = sum(value * value for value in self.w2) ** 0.5
            bias1 = max((abs(value) for value in self.b1), default=0.0)
            bias2 = abs(float(self.b2))
        lipschitz_proxy = sigma1 * norm2
        active_bound = float(bound) if bound > 0.0 else 0.0
        excess1 = sigma1 / active_bound if active_bound > 0.0 else 0.0
        excess2 = norm2 / active_bound if active_bound > 0.0 else 0.0
        return {
            "w1_spectral_norm": sigma1,
            "w2_euclidean_norm": norm2,
            "lipschitz_proxy": lipschitz_proxy,
            "max_abs_b1": bias1,
            "abs_b2": bias2,
            "clip_bound": active_bound,
            "w1_excess_ratio": excess1,
            "w2_excess_ratio": excess2,
            "clip_active": bool(active_bound > 0.0 and (excess1 > 1.000001 or excess2 > 1.000001)),
        }

    def _fit_python(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        epochs: int,
        learning_rate: float,
    ) -> None:
        for _ in range(epochs):
            for x, target in zip(inputs, targets):
                self._train_one(x, target, learning_rate)

    def _train_one(self, x: Sequence[float], target: float, learning_rate: float) -> None:
        hidden = self._hidden(x)
        output = self.b2 + sum(w * h for w, h in zip(self.w2, hidden))
        err = max(-10.0, min(10.0, output - target))

        old_w2 = list(self.w2)
        for j, h in enumerate(hidden):
            self.w2[j] -= learning_rate * err * h
        self.b2 -= learning_rate * err

        for j, h in enumerate(hidden):
            delta = err * old_w2[j] * (1.0 - h * h)
            for i, value in enumerate(x):
                self.w1[j][i] -= learning_rate * delta * value
            self.b1[j] -= learning_rate * delta
        self._invalidate_numpy_cache()

    def _fit_numpy(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        epochs: int,
        learning_rate: float,
    ) -> None:
        import numpy as np

        if epochs <= 0 or learning_rate <= 0.0:
            return
        x = np.asarray(inputs, dtype=float)
        y = np.asarray(targets, dtype=float)
        w1 = np.asarray(self.w1, dtype=float)
        b1 = np.asarray(self.b1, dtype=float)
        w2 = np.asarray(self.w2, dtype=float)
        b2 = float(self.b2)
        batch_size = min(256, max(16, len(x)))
        order = list(range(len(x)))
        for _ in range(epochs):
            self._shuffle_indices(order)
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                xb = x[idx]
                yb = y[idx]
                hidden = np.tanh(xb @ w1.T + b1)
                output = hidden @ w2 + b2
                err = np.clip(output - yb, -10.0, 10.0)
                scale = 1.0 / max(1, len(idx))
                old_w2 = w2.copy()
                grad_w2 = err @ hidden * scale
                grad_b2 = float(err.mean())
                hidden_delta = (err[:, None] * old_w2[None, :]) * (1.0 - hidden * hidden)
                grad_w1 = hidden_delta.T @ xb * scale
                grad_b1 = hidden_delta.mean(axis=0)
                w2 -= learning_rate * grad_w2
                b2 -= learning_rate * grad_b2
                w1 -= learning_rate * grad_w1
                b1 -= learning_rate * grad_b1
        self.w1 = w1.tolist()
        self.b1 = b1.tolist()
        self.w2 = w2.tolist()
        self.b2 = b2
        self._invalidate_numpy_cache()

    def _shuffle_indices(self, indices: List[int]) -> None:
        for idx in range(len(indices) - 1, 0, -1):
            swap = self.rng.randrange(idx + 1)
            indices[idx], indices[swap] = indices[swap], indices[idx]

    def _hidden(self, x: Sequence[float]) -> List[float]:
        import math

        hidden = []
        for row, bias in zip(self.w1, self.b1):
            z = bias + sum(w * value for w, value in zip(row, x))
            hidden.append(math.tanh(z))
        return hidden


class NeuralScalarPotential(HypervolumeArchivePotential):
    """Trainable scalar potential with exact empirical Hamiltonian deltas.

    The network is fitted at archive stopping times and then frozen during the
    next Metropolis epoch. This preserves the scalar-potential structure needed
    by the local detailed-balance argument.
    """

    def __init__(
        self,
        hidden_units: int = 16,
        training_epochs: int = 80,
        learning_rate: float = 0.03,
        seed: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        self._net: Optional[TinyMLP] = None
        self._input_dim: Optional[int] = None

    def fit(
        self,
        objectives: Iterable[ObjectiveVector],
        archive_entries: Iterable[ArchiveEntry],
        context: PotentialContext,
    ) -> None:
        training_objectives = list(objectives) + [entry.objectives for entry in archive_entries]
        if not training_objectives:
            return
        dim = len(training_objectives[0])
        if self._net is None or self._input_dim != dim:
            self._net = TinyMLP(dim, self.hidden_units, self.rng)
            self._input_dim = dim

        inputs = [self.normalize(obj, context) for obj in training_objectives]
        targets = [self._analytic_single_energy(obj, context) for obj in training_objectives]
        self._net.fit(inputs, targets, self.training_epochs, self.learning_rate)

    def single_energy(self, objective: ObjectiveVector, context: PotentialContext) -> float:
        if self._net is None:
            return self._analytic_single_energy(objective, context)
        return self._net.predict(self.normalize(objective, context))

    def _analytic_single_energy(self, objective: ObjectiveVector, context: PotentialContext) -> float:
        return ScalarArchivePotential.single_energy(self, objective, context)
