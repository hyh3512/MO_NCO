from __future__ import annotations

"""Sparse learned move generator for bi-objective TSP IPS.

The generator consumes current-tour edge structure and Pareto/reference
conditioning, then directly samples sparse feasible two-opt moves.  It remains
a proposal policy, not a replacement potential: detailed-balance and zero-curl
claims are attached to the frozen scalar potential used after proposal.
"""

import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .types import ObjectiveVector, Tour


ActionRecord = Tuple[Tuple[float, ...], Tuple[float, ...], float]
JointActionRecord = Tuple[
    Tuple[float, ...],
    Tuple[float, ...],
    float,
    float,
    float,
    float,
]


class SparseMoveGenerator:
    """Shared edge-set encoder with one decomposable proposal distribution.

    Target/HV, flow consistency, mean-field, and basin-crossing objectives have
    separate readouts for auditability, but all heads consume and update the
    same nonlinear pair encoder.  Their weighted sum is the only action logit
    used by the sparse proposal policy.
    """

    backend_name = "shared_edge_set_joint_action_policy"

    def __init__(
        self,
        input_dim: int = 16,
        hidden_units: int = 24,
        learning_rate: float = 0.04,
        rng: Optional[random.Random] = None,
        weight_norm_bound: float = 3.0,
        target_head_weight: float = 1.0,
        flow_head_weight: float = 0.35,
        mean_field_head_weight: float = 0.20,
        conductance_head_weight: float = 0.20,
    ) -> None:
        self.input_dim = max(4, int(input_dim))
        self.hidden_units = max(4, int(hidden_units))
        self.learning_rate = max(0.0, float(learning_rate))
        self.rng = rng or random.Random(0)
        self.weight_norm_bound = max(0.25, float(weight_norm_bound))
        self.target_head_weight = max(0.0, float(target_head_weight))
        self.flow_head_weight = max(0.0, float(flow_head_weight))
        self.mean_field_head_weight = max(0.0, float(mean_field_head_weight))
        self.conductance_head_weight = max(0.0, float(conductance_head_weight))
        scale = 1.0 / math.sqrt(self.input_dim)
        self.node_hidden = [
            [self.rng.uniform(-scale, scale) for _ in range(self.input_dim)]
            for _ in range(self.hidden_units)
        ]
        self.node_hidden_bias = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.pair_hidden = [
            [self.rng.uniform(-scale, scale) for _ in range(self.input_dim)]
            for _ in range(self.hidden_units)
        ]
        self.pair_hidden_bias = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.node_readout = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.pair_readout = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.flow_readout = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.mean_field_readout = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.conductance_readout = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.summary_readout = [self.rng.uniform(-0.03, 0.03) for _ in range(self.hidden_units)]
        self.bias = 0.0
        self.updates = 0

    @property
    def node_weights(self) -> Tuple[float, ...]:
        return tuple(self.node_readout)

    @property
    def pair_weights(self) -> Tuple[float, ...]:
        return tuple(self.pair_readout)

    @classmethod
    def from_dict(cls, payload: dict, rng: Optional[random.Random] = None) -> "SparseMoveGenerator":
        model = payload.get("model", payload)
        generator = cls(
            input_dim=int(model.get("input_dim", payload.get("input_dim", 16))),
            hidden_units=int(model.get("hidden_units", payload.get("hidden_units", 24))),
            learning_rate=float(model.get("learning_rate", payload.get("learning_rate", 0.04))),
            rng=rng,
            weight_norm_bound=float(model.get("weight_norm_bound", payload.get("weight_norm_bound", 3.0))),
            target_head_weight=float(model.get("target_head_weight", 1.0)),
            flow_head_weight=float(model.get("flow_head_weight", 0.35)),
            mean_field_head_weight=float(model.get("mean_field_head_weight", 0.20)),
            conductance_head_weight=float(model.get("conductance_head_weight", 0.20)),
        )
        for name in (
            "node_hidden",
            "node_hidden_bias",
            "pair_hidden",
            "pair_hidden_bias",
            "node_readout",
            "pair_readout",
            "flow_readout",
            "mean_field_readout",
            "conductance_readout",
            "summary_readout",
        ):
            if name in model:
                setattr(generator, name, _as_float_nested(model[name]))
        if "bias" in model:
            generator.bias = float(model["bias"])
        if "updates" in model:
            generator.updates = int(model["updates"])
        generator._clip()
        return generator

    @classmethod
    def load(cls, path: Path, rng: Optional[random.Random] = None) -> "SparseMoveGenerator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("move_generator", payload.get("model", payload))
        return cls.from_dict(model, rng)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend_name,
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "learning_rate": self.learning_rate,
            "weight_norm_bound": self.weight_norm_bound,
            "target_head_weight": self.target_head_weight,
            "flow_head_weight": self.flow_head_weight,
            "mean_field_head_weight": self.mean_field_head_weight,
            "conductance_head_weight": self.conductance_head_weight,
            "node_hidden": self.node_hidden,
            "node_hidden_bias": self.node_hidden_bias,
            "pair_hidden": self.pair_hidden,
            "pair_hidden_bias": self.pair_hidden_bias,
            "node_readout": self.node_readout,
            "pair_readout": self.pair_readout,
            "flow_readout": self.flow_readout,
            "mean_field_readout": self.mean_field_readout,
            "conductance_readout": self.conductance_readout,
            "summary_readout": self.summary_readout,
            "bias": self.bias,
            "updates": self.updates,
        }

    def sample_two_opt(
        self,
        tour: Tour,
        objectives: ObjectiveVector,
        matrices: Sequence[Sequence[Sequence[float]]],
        reference: Tuple[float, float],
        target: Tuple[float, float],
        scale0: float,
        scale1: float,
        sparse_nodes: int,
        sparse_partners: int,
        context: Sequence[float] = (),
    ) -> Optional[Tuple[int, int, Tuple[float, ...], Tuple[float, ...]]]:
        samples = self.sample_two_opt_candidates(
            tour,
            objectives,
            matrices,
            reference,
            target,
            scale0,
            scale1,
            sparse_nodes,
            sparse_partners,
            context,
            sample_count=1,
        )
        return samples[0] if samples else None

    def sample_two_opt_candidates(
        self,
        tour: Tour,
        objectives: ObjectiveVector,
        matrices: Sequence[Sequence[Sequence[float]]],
        reference: Tuple[float, float],
        target: Tuple[float, float],
        scale0: float,
        scale1: float,
        sparse_nodes: int,
        sparse_partners: int,
        context: Sequence[float] = (),
        sample_count: int = 1,
        guidance_scale: float = 1.0,
    ) -> List[Tuple[int, int, Tuple[float, ...], Tuple[float, ...]]]:
        n = len(tour)
        if n < 4:
            return []
        node_pool = self._sample_positions(n, max(4, sparse_nodes), exclude=())
        if not node_pool:
            return []
        node_features = [
            self.node_features(tour, pos, objectives, matrices, reference, target, scale0, scale1, context)
            for pos in node_pool
        ]
        unconditioned_node_features = [self._unconditioned_features(features) for features in node_features]
        node_embeddings = [self._embed(self.node_hidden, self.node_hidden_bias, features) for features in node_features]
        unconditioned_node_embeddings = [
            self._embed(self.node_hidden, self.node_hidden_bias, features)
            for features in unconditioned_node_features
        ]
        summary = self._mean_embedding(node_embeddings)
        unconditioned_summary = self._mean_embedding(unconditioned_node_embeddings)
        node_logits = [
            self._guided_score(
                self._node_score(embedding, summary),
                self._node_score(unconditioned_embedding, unconditioned_summary),
                guidance_scale,
            )
            for embedding, unconditioned_embedding in zip(node_embeddings, unconditioned_node_embeddings)
        ]
        samples: List[Tuple[int, int, Tuple[float, ...], Tuple[float, ...]]] = []
        attempts = 0
        max_attempts = max(4, sample_count * 4)
        while len(samples) < max(1, sample_count) and attempts < max_attempts:
            attempts += 1
            first_idx = self._sample_softmax(node_logits)
            first_pos = node_pool[first_idx]
            exclusions = {first_pos, (first_pos - 1) % n, (first_pos + 1) % n}
            partner_pool = self._sample_positions(n, max(4, sparse_partners), exclude=exclusions)
            partner_pool = [pos for pos in partner_pool if self._valid_two_opt_positions(first_pos, pos, n)]
            if not partner_pool:
                continue
            first_features = node_features[first_idx]
            pair_features = []
            pair_logits = []
            second_feature_cache = []
            for pos in partner_pool:
                second_features = self.node_features(tour, pos, objectives, matrices, reference, target, scale0, scale1, context)
                combined = self._pair_features(first_features, second_features)
                unconditioned_combined = self._pair_features(
                    self._unconditioned_features(first_features),
                    self._unconditioned_features(second_features),
                )
                pair_features.append(combined)
                second_feature_cache.append(second_features)
                pair_logits.append(
                    self._guided_score(
                        self._pair_score(self._embed(self.pair_hidden, self.pair_hidden_bias, combined), summary),
                        self._pair_score(
                            self._embed(self.pair_hidden, self.pair_hidden_bias, unconditioned_combined),
                            unconditioned_summary,
                        ),
                        guidance_scale,
                    )
                )
            second_idx = self._sample_softmax(pair_logits)
            second_pos = partner_pool[second_idx]
            second_features = second_feature_cache[second_idx]
            i, j = first_pos, second_pos
            if i > j:
                i, j = j, i
                first_features = self.node_features(tour, i, objectives, matrices, reference, target, scale0, scale1, context)
                second_features = self.node_features(tour, j, objectives, matrices, reference, target, scale0, scale1, context)
            if self._valid_two_opt_positions(i, j, n):
                samples.append((i, j, first_features, second_features))
        return samples

    def update(self, first_features: Sequence[float], second_features: Sequence[float], reward: float) -> None:
        self.update_joint(first_features, second_features, reward, 0.0, 0.0, 0.0)

    def update_joint(
        self,
        first_features: Sequence[float],
        second_features: Sequence[float],
        target_advantage: float,
        flow_advantage: float,
        mean_field_advantage: float,
        conductance_advantage: float,
    ) -> None:
        advantages = (
            max(-1.0, min(1.0, float(target_advantage))),
            max(-1.0, min(1.0, float(flow_advantage))),
            max(-1.0, min(1.0, float(mean_field_advantage))),
            max(-1.0, min(1.0, float(conductance_advantage))),
        )
        weights = (
            self.target_head_weight,
            self.flow_head_weight,
            self.mean_field_head_weight,
            self.conductance_head_weight,
        )
        joint_advantage = sum(weight * advantage for weight, advantage in zip(weights, advantages))
        if (
            max(abs(value) for value in advantages) <= 1e-12
            or abs(joint_advantage) <= 1e-12
            or self.learning_rate <= 0.0
        ):
            return
        pair_features = self._pair_features(first_features, second_features)
        h1 = self._embed(self.node_hidden, self.node_hidden_bias, first_features)
        h2 = self._embed(self.node_hidden, self.node_hidden_bias, second_features)
        hp = self._embed(self.pair_hidden, self.pair_hidden_bias, pair_features)
        head_vectors = (
            self.pair_readout,
            self.flow_readout,
            self.mean_field_readout,
            self.conductance_readout,
        )
        encoder_signal = [
            sum(weight * advantage * head[idx] for weight, advantage, head in zip(weights, advantages, head_vectors))
            for idx in range(self.hidden_units)
        ]
        step = self.learning_rate * joint_advantage
        for idx in range(self.hidden_units):
            self.node_readout[idx] += step * (h1[idx] + 0.50 * h2[idx])
            self.summary_readout[idx] += 0.10 * step * (h1[idx] + h2[idx] + hp[idx]) / 3.0
            for weight, advantage, head in zip(weights, advantages, head_vectors):
                head[idx] += self.learning_rate * weight * advantage * hp[idx]
            pair_grad = self.learning_rate * encoder_signal[idx] * (1.0 - hp[idx] * hp[idx])
            node_grad = self.learning_rate * joint_advantage * self.node_readout[idx]
            for feature_idx in range(self.input_dim):
                self.pair_hidden[idx][feature_idx] += pair_grad * float(pair_features[feature_idx])
                self.node_hidden[idx][feature_idx] += 0.50 * node_grad * (
                    float(first_features[feature_idx]) + 0.50 * float(second_features[feature_idx])
                )
            self.pair_hidden_bias[idx] += pair_grad
            self.node_hidden_bias[idx] += 0.75 * node_grad
        self.bias += 0.05 * step
        self._clip()
        self.updates += 1

    def fit_actions(
        self,
        actions: Sequence[ActionRecord],
        epochs: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        local_rng = rng or self.rng
        if not actions:
            return
        order = list(range(len(actions)))
        for _ in range(max(1, epochs)):
            local_rng.shuffle(order)
            for idx in order:
                first, second, reward = actions[idx]
                self.update(first, second, reward)

    def fit_joint_actions(
        self,
        actions: Sequence[JointActionRecord],
        epochs: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        local_rng = rng or self.rng
        if not actions:
            return
        order = list(range(len(actions)))
        for _ in range(max(1, epochs)):
            local_rng.shuffle(order)
            for idx in order:
                self.update_joint(*actions[idx])

    def action_score(self, first_features: Sequence[float], second_features: Sequence[float]) -> float:
        pair_features = self._pair_features(first_features, second_features)
        embedding = self._embed(self.pair_hidden, self.pair_hidden_bias, pair_features)
        return self._pair_score(embedding, ())

    def action_probabilities(
        self,
        actions: Sequence[Tuple[Sequence[float], Sequence[float]]],
    ) -> Tuple[float, ...]:
        if not actions:
            return ()
        logits = [self.action_score(first, second) for first, second in actions]
        maximum = max(logits)
        weights = [math.exp(max(-20.0, min(20.0, value - maximum))) for value in logits]
        total = sum(weights)
        if total <= 0.0:
            return tuple(1.0 / len(actions) for _ in actions)
        return tuple(value / total for value in weights)

    def node_features(
        self,
        tour: Tour,
        pos: int,
        objectives: ObjectiveVector,
        matrices: Sequence[Sequence[Sequence[float]]],
        reference: Tuple[float, float],
        target: Tuple[float, float],
        scale0: float,
        scale1: float,
        context: Sequence[float] = (),
    ) -> Tuple[float, ...]:
        n = len(tour)
        prev_city = tour[(pos - 1) % n]
        city = tour[pos]
        next_city = tour[(pos + 1) % n]
        m0, m1 = matrices[0], matrices[1]
        in0 = float(m0[prev_city][city]) / max(scale0, 1e-9)
        out0 = float(m0[city][next_city]) / max(scale0, 1e-9)
        in1 = float(m1[prev_city][city]) / max(scale1, 1e-9)
        out1 = float(m1[city][next_city]) / max(scale1, 1e-9)
        angle = 2.0 * math.pi * pos / max(1, n)
        total0 = float(objectives[0]) / max(scale0 * n, 1e-9)
        total1 = float(objectives[1]) / max(scale1 * n, 1e-9)
        base = (
            self._bound(in0),
            self._bound(out0),
            self._bound(in1),
            self._bound(out1),
            math.sin(angle),
            math.cos(angle),
            self._bound(total0),
            self._bound(total1),
            self._bound(reference[0]),
            self._bound(reference[1]),
            self._bound(target[0]),
            self._bound(target[1]),
        )
        extras = tuple(self._bound(value) for value in context[: self.input_dim - len(base)])
        if len(base) + len(extras) < self.input_dim:
            extras = (*extras, *((0.0,) * (self.input_dim - len(base) - len(extras))))
        return (*base, *extras)[: self.input_dim]

    def _node_score(self, embedding: Sequence[float], summary: Sequence[float]) -> float:
        direct = self._dot(self.node_readout, embedding)
        context = 0.20 * self._dot(self.summary_readout, summary)
        interaction = 0.10 * self._dot(embedding, summary) / max(1, len(embedding))
        return direct + context + interaction + self.bias

    def _pair_score(self, embedding: Sequence[float], summary: Sequence[float]) -> float:
        direct = (
            self.target_head_weight * self._dot(self.pair_readout, embedding)
            + self.flow_head_weight * self._dot(self.flow_readout, embedding)
            + self.mean_field_head_weight * self._dot(self.mean_field_readout, embedding)
            + self.conductance_head_weight * self._dot(self.conductance_readout, embedding)
        )
        context = 0.15 * self._dot(self.summary_readout, summary) if summary else 0.0
        interaction = 0.10 * self._dot(embedding, summary) / max(1, len(embedding)) if summary else 0.0
        return direct + context + interaction

    def _pair_features(self, first: Sequence[float], second: Sequence[float]) -> Tuple[float, ...]:
        values = []
        for a, b in zip(first, second):
            af = float(a)
            bf = float(b)
            values.append(self._bound(0.35 * (af + bf) + 0.35 * abs(af - bf) + 0.30 * af * bf))
        if len(values) < self.input_dim:
            values.extend([0.0] * (self.input_dim - len(values)))
        return tuple(values[: self.input_dim])

    def _unconditioned_features(self, features: Sequence[float]) -> Tuple[float, ...]:
        # Classifier-free-style proposal guidance: retain state/tour-edge
        # coordinates and objective scale, while dropping reference direction,
        # target Pareto point, and archive/mean-field context.
        keep = {0, 1, 2, 3, 4, 5, 6, 7}
        return tuple(float(value) if idx in keep else 0.0 for idx, value in enumerate(features[: self.input_dim]))

    @staticmethod
    def _guided_score(conditioned: float, unconditioned: float, guidance_scale: float) -> float:
        scale = max(0.0, float(guidance_scale))
        if scale == 1.0:
            return conditioned
        return unconditioned + scale * (conditioned - unconditioned)

    def _embed(
        self,
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        features: Sequence[float],
    ) -> Tuple[float, ...]:
        return tuple(math.tanh(self._dot(row, features) + float(b)) for row, b in zip(weights, bias))

    @staticmethod
    def _mean_embedding(embeddings: Sequence[Sequence[float]]) -> Tuple[float, ...]:
        if not embeddings:
            return ()
        hidden = len(embeddings[0])
        return tuple(sum(float(embedding[idx]) for embedding in embeddings) / len(embeddings) for idx in range(hidden))

    def _sample_positions(self, n: int, count: int, exclude: set[int]) -> List[int]:
        feasible = [pos for pos in range(1, n - 1) if pos not in exclude]
        if len(feasible) <= count:
            return feasible
        return self.rng.sample(feasible, count)

    @staticmethod
    def _valid_two_opt_positions(i: int, j: int, n: int) -> bool:
        if i == j or abs(i - j) <= 1:
            return False
        if min(i, j) <= 0 or max(i, j) >= n:
            return False
        return True

    def _sample_softmax(self, logits: Sequence[float]) -> int:
        if not logits:
            return 0
        max_logit = max(logits)
        weights = [math.exp(max(-20.0, min(20.0, logit - max_logit))) for logit in logits]
        total = sum(weights)
        if total <= 0.0:
            return self.rng.randrange(len(logits))
        threshold = self.rng.random() * total
        acc = 0.0
        for idx, weight in enumerate(weights):
            acc += weight
            if acc >= threshold:
                return idx
        return len(logits) - 1

    def _clip(self) -> None:
        self.node_readout = self._clip_vector(self.node_readout)
        self.pair_readout = self._clip_vector(self.pair_readout)
        self.flow_readout = self._clip_vector(self.flow_readout)
        self.mean_field_readout = self._clip_vector(self.mean_field_readout)
        self.conductance_readout = self._clip_vector(self.conductance_readout)
        self.summary_readout = self._clip_vector(self.summary_readout)
        self.node_hidden = [self._clip_vector(row) for row in self.node_hidden]
        self.pair_hidden = [self._clip_vector(row) for row in self.pair_hidden]
        self.node_hidden_bias = [self._bound(value) for value in self.node_hidden_bias]
        self.pair_hidden_bias = [self._bound(value) for value in self.pair_hidden_bias]
        self.bias = self._bound(self.bias)

    def _clip_vector(self, vector: Sequence[float]) -> List[float]:
        clipped = [self._bound(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in clipped))
        if norm > self.weight_norm_bound:
            scale = self.weight_norm_bound / max(norm, 1e-12)
            clipped = [value * scale for value in clipped]
        return clipped

    @staticmethod
    def _dot(weights: Sequence[float], values: Sequence[float]) -> float:
        return sum(float(w) * float(v) for w, v in zip(weights, values))

    @staticmethod
    def _bound(value: float) -> float:
        if value != value:
            return 0.0
        return max(-2.0, min(2.0, float(value)))


def save_move_generator(path: Path, generator: SparseMoveGenerator, metadata: Optional[dict] = None) -> None:
    payload = {
        "kind": "mo_nco_replay_learned_move_prior",
        "metadata": metadata or {},
        "move_generator": generator.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _as_float_nested(value):
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [[float(item) for item in row] for row in value]
    if isinstance(value, list):
        return [float(item) for item in value]
    return value
