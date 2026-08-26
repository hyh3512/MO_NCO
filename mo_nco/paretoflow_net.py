from __future__ import annotations

"""ParetoFlow-style scalar potential backend for discrete IPS.

This module adapts the SELU multilayer network shape used by
`mila-iqia/ParetoFlow`'s `VectorFieldNet` into a scalar-potential oracle for
finite-state multi-objective combinatorial optimization.

Attribution: the architectural pattern is adapted from ParetoFlow
(`experiments/paretoflow_nets.py`), MIT License, Copyright (c) 2024 Ye Yuan.
The continuous flow sampler is deliberately not copied here; the IPS code only
consumes the network as a frozen scalar state potential so that edge scores are
formed as exact differences g(y) - g(x).
"""

import math
import random
from typing import List, Sequence, Tuple


class ParetoFlowScalarNet:
    """A dependency-light SELU MLP scalar potential.

    The hidden stack mirrors ParetoFlow's `VectorFieldNet` shape:
    Linear -> SELU -> Linear -> SELU -> Linear -> SELU -> Linear.  ParetoFlow
    outputs a continuous vector field; this adapted backend outputs one scalar
    state potential, keeping the zero-circulation condition used by IPS.
    """

    backend_name = "paretoflow_scalar"

    def __init__(self, input_dim: int, hidden_units: int, rng: random.Random) -> None:
        import numpy as np

        self.input_dim = int(input_dim)
        self.hidden_units = int(hidden_units) if hidden_units > 0 else 512
        self.rng = rng
        dims = [self.input_dim, self.hidden_units, self.hidden_units, self.hidden_units, 1]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(1.0 / max(1, fan_in))
            values = [
                [rng.gauss(0.0, scale) for _ in range(fan_out)]
                for _ in range(fan_in)
            ]
            self.weights.append(np.asarray(values, dtype=float))
            self.biases.append(np.zeros(fan_out, dtype=float))

    def to_dict(self) -> dict:
        return {
            "backend": self.backend_name,
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "weights": [matrix.tolist() for matrix in self.weights],
            "biases": [vector.tolist() for vector in self.biases],
        }

    @classmethod
    def from_dict(cls, payload: dict, rng: random.Random) -> "ParetoFlowScalarNet":
        import numpy as np

        net = cls(int(payload["input_dim"]), int(payload["hidden_units"]), rng)
        net.weights = [np.asarray(matrix, dtype=float) for matrix in payload["weights"]]
        net.biases = [np.asarray(vector, dtype=float) for vector in payload["biases"]]
        return net

    def predict(self, x: Sequence[float]) -> float:
        return float(self.predict_batch([x])[0])

    def predict_batch(self, inputs: Sequence[Sequence[float]]) -> List[float]:
        if not inputs:
            return []
        import numpy as np

        x = np.asarray(inputs, dtype=float)
        output, _, _ = self._forward(x)
        return output.tolist()

    def fit(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[float],
        epochs: int,
        learning_rate: float,
    ) -> None:
        if not inputs or epochs <= 0 or learning_rate <= 0.0:
            return
        import numpy as np

        x = np.asarray(inputs, dtype=float)
        y = np.asarray(targets, dtype=float)
        batch_size = min(256, max(16, len(x)))
        order = list(range(len(x)))
        for _ in range(epochs):
            self._shuffle_indices(order)
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                xb = x[idx]
                yb = y[idx]
                pred, activations, preacts = self._forward(xb)
                grad_out = np.clip(pred - yb, -10.0, 10.0) / max(1, len(idx))
                self._backward_update(activations, preacts, grad_out, learning_rate)

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
        """Fit scalar state, residual-flow, ranking, and hypercone losses."""

        if epochs <= 0 or learning_rate <= 0.0:
            return
        for _ in range(epochs):
            if inputs:
                self.fit(inputs, targets, 1, learning_rate)
            if flow_residual_weight > 0.0:
                self._pair_update(residual_pairs, learning_rate, flow_residual_weight, "residual")
            if ranking_weight > 0.0:
                self._pair_update(ranking_pairs, learning_rate, ranking_weight, "margin")
            if hypercone_weight > 0.0:
                self._pair_update(hypercone_pairs, learning_rate, hypercone_weight, "margin")
            if coverage_weight > 0.0:
                self._pair_update(coverage_pairs, learning_rate, coverage_weight, "margin")
            if expert_weight > 0.0:
                self._pair_update(expert_pairs, learning_rate, expert_weight, "margin")
            self.clip_weight_norms(weight_norm_bound)

    def clip_weight_norms(self, bound: float) -> None:
        if bound <= 0.0:
            return
        import numpy as np

        for idx, matrix in enumerate(self.weights):
            sigma = float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0
            if sigma > bound:
                self.weights[idx] = matrix * (bound / max(sigma, 1e-12))
        for idx, bias in enumerate(self.biases):
            self.biases[idx] = np.clip(bias, -bound, bound)

    def spectral_diagnostics(self, bound: float = 0.0) -> dict:
        import numpy as np

        norms = [
            float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0
            for matrix in self.weights
        ]
        max_bias = max(
            (float(np.max(np.abs(bias))) if bias.size else 0.0 for bias in self.biases),
            default=0.0,
        )
        lipschitz_proxy = 1.0
        for norm in norms:
            lipschitz_proxy *= norm
        active_bound = float(bound) if bound > 0.0 else 0.0
        max_excess = max((norm / active_bound for norm in norms), default=0.0) if active_bound > 0.0 else 0.0
        return {
            "backend": self.backend_name,
            "w1_spectral_norm": norms[0] if norms else 0.0,
            "w2_euclidean_norm": norms[-1] if norms else 0.0,
            "hidden_spectral_norms": norms[1:-1],
            "max_spectral_norm": max(norms, default=0.0),
            "lipschitz_proxy": lipschitz_proxy,
            "max_abs_bias": max_bias,
            "clip_bound": active_bound,
            "max_excess_ratio": max_excess,
            "clip_active": bool(active_bound > 0.0 and max_excess > 1.000001),
        }

    def _pair_update(
        self,
        pairs: Sequence[Tuple[Sequence[float], Sequence[float], float]],
        learning_rate: float,
        loss_weight: float,
        mode: str,
    ) -> None:
        if not pairs:
            return
        import numpy as np

        # ParetoFlow replay can contain hundreds of thousands of pairs per
        # loss kind.  A full pair update forms a huge 2n-by-d feature matrix
        # and makes architecture-sweep baselines unnecessarily slow.  Use a
        # stochastic bounded update instead; over epochs this is the usual SGD
        # estimator for the same pairwise objective.
        max_pairs = min(len(pairs), 8192)
        if len(pairs) > max_pairs:
            sample = self.rng.sample(range(len(pairs)), max_pairs)
            batch = [pairs[idx] for idx in sample]
        else:
            batch = list(pairs)
        first = np.asarray([pair[0] for pair in batch], dtype=float)
        second = np.asarray([pair[1] for pair in batch], dtype=float)
        value = np.asarray([pair[2] for pair in batch], dtype=float)
        x = np.vstack([first, second])
        pred, activations, preacts = self._forward(x)
        n = len(batch)
        pred_first = pred[:n]
        pred_second = pred[n:]
        grad = np.zeros(2 * n, dtype=float)
        if mode == "residual":
            err = np.clip((pred_second - pred_first) - value, -10.0, 10.0)
            scale = loss_weight / max(1, n)
            grad[:n] = -scale * err
            grad[n:] = scale * err
        else:
            violation = value + pred_first - pred_second
            active = violation > 0.0
            if not bool(np.any(active)):
                return
            step = np.clip(loss_weight * violation / max(1, n), 0.0, 2.0)
            grad[:n] = np.where(active, step, 0.0)
            grad[n:] = np.where(active, -step, 0.0)
        self._backward_update(activations, preacts, grad, learning_rate)

    def _forward(self, x):
        activations = [x]
        preacts = []
        h = x
        for matrix, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = h @ matrix + bias
            preacts.append(z)
            h = self._selu(z)
            activations.append(h)
        output = h @ self.weights[-1] + self.biases[-1]
        return output.reshape(-1), activations, preacts

    def _backward_update(self, activations, preacts, grad_out, learning_rate: float) -> None:
        grad = grad_out.reshape(-1, 1)
        grad_w = [None for _ in self.weights]
        grad_b = [None for _ in self.biases]
        grad_w[-1] = activations[-1].T @ grad
        grad_b[-1] = grad.sum(axis=0)
        grad_hidden = grad @ self.weights[-1].T
        for layer in range(len(self.weights) - 2, -1, -1):
            grad_hidden = grad_hidden * self._selu_derivative(preacts[layer])
            grad_w[layer] = activations[layer].T @ grad_hidden
            grad_b[layer] = grad_hidden.sum(axis=0)
            if layer > 0:
                grad_hidden = grad_hidden @ self.weights[layer].T
        for idx in range(len(self.weights)):
            self.weights[idx] = self.weights[idx] - learning_rate * grad_w[idx]
            self.biases[idx] = self.biases[idx] - learning_rate * grad_b[idx]

    @staticmethod
    def _selu(z):
        import numpy as np

        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        z_clip = np.clip(z, -50.0, 50.0)
        return scale * np.where(z_clip > 0.0, z_clip, alpha * (np.exp(z_clip) - 1.0))

    @staticmethod
    def _selu_derivative(z):
        import numpy as np

        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        z_clip = np.clip(z, -50.0, 50.0)
        return scale * np.where(z_clip > 0.0, 1.0, alpha * np.exp(z_clip))

    def _shuffle_indices(self, indices: List[int]) -> None:
        for idx in range(len(indices) - 1, 0, -1):
            swap = self.rng.randrange(idx + 1)
            indices[idx], indices[swap] = indices[swap], indices[idx]
