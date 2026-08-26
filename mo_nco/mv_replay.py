from __future__ import annotations

"""Offline replay data for MV/MMD-GFEF neural priors.

The records generated here are deliberately independent of the online neural
oracle.  They use public MOTSP instances plus a finite-state MMD-style teacher
computed from warmup particles/archive points.  This avoids training only on
scalar-greedy or local-polish preferences.
"""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .benchmark_suite import BenchmarkCase, BenchmarkSuite
from .instance import MultiObjectiveTSPInstance
from .ips_efficient import TheoryAlignedIPSOptimizer
from .learned_move_generator import SparseMoveGenerator, save_move_generator
from .moves import sample_two_opt_indices, two_opt_at
from .neural_potential import TinyMLP
from .neural_prior import split_suite_cases, write_suite_split
from .paretoflow_net import ParetoFlowScalarNet
from .pcd_net import PCDResidualScalarNet
from .types import ObjectiveVector, Tour


Feature = Tuple[float, ...]
Pair = Tuple[Feature, Feature, float]


@dataclass(frozen=True)
class MVReplayGenerationConfig:
    seed: int = 0
    train_fraction: float = 0.7
    population: int = 32
    warmup_evaluations: int = 256
    log_period: int = 128
    archive_update_period: int = 64
    max_state_examples_per_case: int = 4096
    action_pairs_per_case: int = 2048
    action_candidate_pool: int = 8
    long_horizon_candidates: int = 2
    long_horizon_discount: float = 0.60
    mmd_bandwidth: float = 0.18
    output_feature_dim: int = 24


@dataclass(frozen=True)
class ReplayPriorTrainingConfig:
    seed: int = 0
    backend: str = "pcd"
    hidden_units: int = 48
    training_epochs: int = 32
    learning_rate: float = 0.001
    max_state_examples: int = 100_000
    max_pairs_per_kind: int = 100_000
    flow_residual_weight: float = 0.35
    ranking_weight: float = 0.12
    hypercone_weight: float = 0.10
    coverage_weight: float = 0.10
    expert_weight: float = 0.10
    weight_norm_bound: float = 3.0
    endpoint_only_features: bool = True


@dataclass(frozen=True)
class ReplayMovePolicyTrainingConfig:
    seed: int = 0
    input_dim: int = 16
    hidden_units: int = 32
    training_epochs: int = 16
    learning_rate: float = 0.025
    max_action_examples: int = 250_000
    positive_reweight: float = 1.25
    hard_negative_reweight: float = 1.50
    weight_norm_bound: float = 3.0
    target_head_weight: float = 1.0
    flow_head_weight: float = 0.35
    mean_field_head_weight: float = 0.20
    conductance_head_weight: float = 0.20


def generate_mv_replay_dataset(
    suite: BenchmarkSuite,
    output_dir: Path,
    config: MVReplayGenerationConfig,
) -> dict:
    rng = random.Random(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "mv_mmd_replay.jsonl"
    train_cases, test_cases = split_suite_cases(suite, config.train_fraction)
    split_dir = output_dir / "splits"
    train_suite_path, test_suite_path = write_suite_split(suite, train_cases, test_cases, split_dir)
    case_counts: Dict[str, Dict[str, int]] = {}
    total_records = 0
    with replay_path.open("w", encoding="utf-8") as handle:
        meta = {
            "kind": "meta",
            "dataset": "mv_mmd_gfef_replay",
            "suite": suite.name,
            "config": asdict(config),
            "train_cases": [case.name for case in train_cases],
            "test_cases": [case.name for case in test_cases],
            "train_suite_path": str(train_suite_path),
            "test_suite_path": str(test_suite_path),
            "feature_dim": config.output_feature_dim,
        }
        handle.write(json.dumps(meta, sort_keys=True) + "\n")
        total_records += 1
        for case_idx, case in enumerate(train_cases):
            counts = _write_case_replay(handle, case, case_idx, rng, config)
            case_counts[case.name] = counts
            total_records += sum(counts.values())
    summary = {
        "replay_path": str(replay_path),
        "train_suite_path": str(train_suite_path),
        "test_suite_path": str(test_suite_path),
        "case_counts": case_counts,
        "records": total_records,
        "feature_dim": config.output_feature_dim,
    }
    (output_dir / "mv_mmd_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_prior_from_replay(
    replay_paths: Sequence[Path],
    output_path: Path,
    config: ReplayPriorTrainingConfig,
) -> dict:
    rng = random.Random(config.seed)
    states: List[Tuple[Feature, float]] = []
    pairs: Dict[str, List[Pair]] = {
        "residual": [],
        "ranking": [],
        "hypercone": [],
        "coverage": [],
        "expert": [],
    }
    meta_records = []
    seen_states = 0
    seen_pairs: Dict[str, int] = {kind: 0 for kind in pairs}
    active_pair_kinds = {
        kind
        for kind, weight in {
            "residual": config.flow_residual_weight,
            "ranking": config.ranking_weight,
            "hypercone": config.hypercone_weight,
            "coverage": config.coverage_weight,
            "expert": config.expert_weight,
        }.items()
        if weight > 0.0
    }
    for path in replay_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                kind = item.get("kind")
                if kind == "meta":
                    meta_records.append(item)
                    continue
                if kind == "state":
                    features = tuple(float(v) for v in item["features"])
                    if config.endpoint_only_features:
                        features = _endpoint_state_features(features)
                    seen_states = _reservoir_add(
                        states,
                        (features, float(item["target"])),
                        seen_states,
                        config.max_state_examples,
                        rng,
                    )
                elif kind in pairs:
                    if kind not in active_pair_kinds:
                        seen_pairs[kind] += 1
                        continue
                    first = tuple(float(v) for v in item["first"])
                    second = tuple(float(v) for v in item["second"])
                    if config.endpoint_only_features:
                        first = _endpoint_state_features(first)
                        second = _endpoint_state_features(second)
                    seen_pairs[kind] = _reservoir_add(
                        pairs[kind],
                        (
                            first,
                            second,
                            float(item.get("value", item.get("margin", 0.0))),
                        ),
                        seen_pairs[kind],
                        config.max_pairs_per_kind,
                        rng,
                    )
    if not states:
        raise ValueError("Replay dataset contains no state examples.")

    inputs = [features for features, _ in states]
    targets = [target for _, target in states]
    input_dim = len(inputs[0])
    net = _new_prior_backend(config.backend, input_dim, config.hidden_units, rng)
    if hasattr(net, "fit_mixed"):
        net.fit_mixed(
            inputs,
            targets,
            pairs["residual"],
            pairs["ranking"],
            pairs["hypercone"],
            config.training_epochs,
            config.learning_rate,
            config.flow_residual_weight,
            config.ranking_weight,
            config.hypercone_weight,
            config.weight_norm_bound,
            pairs["coverage"],
            pairs["expert"],
            config.coverage_weight,
            config.expert_weight,
        )
    else:
        net.fit(inputs, targets, config.training_epochs, config.learning_rate)
        net.clip_weight_norms(config.weight_norm_bound)
    payload = {
        "kind": "mo_nco_replay_neural_scalar_prior",
        "config": asdict(config),
        "replay_paths": [str(path) for path in replay_paths],
        "meta": meta_records,
        "training_samples": len(inputs),
        "pair_counts": {kind: len(values) for kind, values in pairs.items()},
        "active_pair_kinds": sorted(active_pair_kinds),
        "scanned_state_records": seen_states,
        "scanned_pair_records": seen_pairs,
        "feature_contract": "endpoint_state_v1" if config.endpoint_only_features else "legacy_mixed_state_action",
        "zeroed_action_feature_indices": [2, 3, 13] if config.endpoint_only_features else [],
        "network": net.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _endpoint_state_features(features: Feature) -> Feature:
    """Project replay features onto the endpoint-only scalar-potential contract.

    The 20/24-dimensional replay layout reserves indices 2, 3 and 13 for
    source-dependent displacement and scalar-delta action features.  A state
    potential must return one value for an endpoint regardless of its parent,
    so these coordinates are zeroed before *all* scalar-prior losses.  The
    separate move-policy trainer still consumes the original action records.
    """

    values = list(features)
    for index in (2, 3, 13):
        if index < len(values):
            values[index] = 0.0
    return tuple(values)


def train_move_policy_from_replay(
    replay_paths: Sequence[Path],
    output_path: Path,
    config: ReplayMovePolicyTrainingConfig,
) -> dict:
    rng = random.Random(config.seed)
    actions: List[Tuple[Feature, Feature, float, float, float, float, str]] = []
    meta_records = []
    seen_actions = 0
    scanned_kind_counts: Dict[str, int] = {}
    for path in replay_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                kind = item.get("kind")
                if kind == "meta":
                    meta_records.append(item)
                    continue
                if kind not in {"move_action", "move_hard_negative"}:
                    continue
                scanned_kind_counts[kind] = scanned_kind_counts.get(kind, 0) + 1
                first = tuple(float(v) for v in item["first"])
                second = tuple(float(v) for v in item["second"])
                legacy_reward = float(item["reward"])
                target_advantage = float(item.get("target_advantage", legacy_reward))
                flow_advantage = float(item.get("flow_advantage", 0.0))
                mean_field_advantage = float(item.get("mean_field_advantage", 0.0))
                conductance_advantage = float(item.get("conductance_advantage", 0.0))
                if kind == "move_action" and legacy_reward > 0.0:
                    target_advantage *= config.positive_reweight
                    flow_advantage *= config.positive_reweight
                    mean_field_advantage *= config.positive_reweight
                    conductance_advantage *= config.positive_reweight
                if kind == "move_hard_negative":
                    target_advantage *= config.hard_negative_reweight
                    flow_advantage *= config.hard_negative_reweight
                    mean_field_advantage *= config.hard_negative_reweight
                    conductance_advantage *= config.hard_negative_reweight
                seen_actions = _reservoir_add(
                    actions,
                    (
                        first,
                        second,
                        max(-1.0, min(1.0, target_advantage)),
                        max(-1.0, min(1.0, flow_advantage)),
                        max(-1.0, min(1.0, mean_field_advantage)),
                        max(-1.0, min(1.0, conductance_advantage)),
                        kind,
                    ),
                    seen_actions,
                    config.max_action_examples,
                    rng,
                )
    if not actions:
        raise ValueError("Replay dataset contains no move_action records.")
    generator = SparseMoveGenerator(
        input_dim=config.input_dim,
        hidden_units=config.hidden_units,
        learning_rate=config.learning_rate,
        rng=rng,
        weight_norm_bound=config.weight_norm_bound,
        target_head_weight=config.target_head_weight,
        flow_head_weight=config.flow_head_weight,
        mean_field_head_weight=config.mean_field_head_weight,
        conductance_head_weight=config.conductance_head_weight,
    )
    generator.fit_joint_actions(
        [
            (first, second, target, flow, mean_field, conductance)
            for first, second, target, flow, mean_field, conductance, _ in actions
        ],
        config.training_epochs,
        rng,
    )
    kind_counts: Dict[str, int] = {}
    for *_, kind in actions:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    signal_means = {
        "target_advantage": sum(item[2] for item in actions) / len(actions),
        "flow_advantage": sum(item[3] for item in actions) / len(actions),
        "mean_field_advantage": sum(item[4] for item in actions) / len(actions),
        "conductance_advantage": sum(item[5] for item in actions) / len(actions),
    }
    payload = {
        "config": asdict(config),
        "replay_paths": [str(path) for path in replay_paths],
        "meta": meta_records,
        "training_samples": len(actions),
        "kind_counts": kind_counts,
        "scanned_action_records": seen_actions,
        "scanned_kind_counts": scanned_kind_counts,
        "signal_means": signal_means,
    }
    save_move_generator(output_path, generator, payload)
    payload["move_generator"] = generator.to_dict()
    return payload


def _write_case_replay(
    handle,
    case: BenchmarkCase,
    case_idx: int,
    rng: random.Random,
    config: MVReplayGenerationConfig,
) -> Dict[str, int]:
    instance = case.load_instance() or MultiObjectiveTSPInstance.random_biobjective(
        case.cities,
        seed=case.instance_seed,
    )
    optimizer = TheoryAlignedIPSOptimizer(
        instance=instance,
        num_particles=int(case.population or config.population),
        evaluations=int(config.warmup_evaluations if config.warmup_evaluations > 0 else (case.evaluations or 256)),
        seed=config.seed + 10_000 * case_idx,
        log_period=config.log_period,
        neighbor_size=8,
        crossover_probability=0.0,
        archive_parent_probability=0.10,
        archive_parent_sample=4,
        archive_update_period=config.archive_update_period,
        proposal="two_opt",
        extra_two_opt_probability=0.0,
        archive_conditioning=True,
        archive_conditioning_weight=3.0,
        neural_scalar_weight=0.0,
        neural_proposal_probability=0.0,
        neural_mean_field_features=True,
        neural_front_reweighting_strength=1.0,
        initialization="mixed_scalar_greedy",
        greedy_candidate_pool=3,
    )
    optimizer.run()
    front = _front_points(optimizer)
    particles = tuple(optimizer._normalize2(obj) for obj in optimizer.objectives)
    source_objectives = _state_objectives(optimizer)
    counts = {
        "state": 0,
        "residual": 0,
        "ranking": 0,
        "hypercone": 0,
        "coverage": 0,
        "expert": 0,
        "move_action": 0,
        "move_hard_negative": 0,
    }
    move_encoder = SparseMoveGenerator(
        input_dim=16,
        hidden_units=8,
        learning_rate=0.0,
        rng=random.Random(config.seed + 77_777 + case_idx),
    )
    state_budget = min(config.max_state_examples_per_case, len(source_objectives) * len(optimizer.weights))
    state_indices = list(range(len(source_objectives) * len(optimizer.weights)))
    if len(state_indices) > state_budget:
        state_indices = rng.sample(state_indices, state_budget)
    for flat_idx in state_indices:
        obj_idx, weight_idx = divmod(flat_idx, len(optimizer.weights))
        objective = source_objectives[obj_idx]
        target = _mmd_teacher_target(optimizer, objective, weight_idx, particles, front, config.mmd_bandwidth)
        features = _replay_features(optimizer, objective, weight_idx, None, config.output_feature_dim)
        _write_record(handle, {"kind": "state", "case": case.name, "features": features, "target": target})
        counts["state"] += 1

    parent_items = list(zip(optimizer.population, optimizer.objectives))
    for action_group in range(config.action_pairs_per_case):
        parent, parent_obj = rng.choice(parent_items)
        weight_idx = rng.randrange(len(optimizer.weights))
        candidates = _sample_action_candidates(optimizer, parent, parent_obj, rng, config.action_candidate_pool)
        action_records = _sample_action_candidate_features(
            optimizer,
            parent,
            parent_obj,
            weight_idx,
            rng,
            config.action_candidate_pool,
            move_encoder,
            particles,
            front,
            config.mmd_bandwidth,
            config.long_horizon_candidates,
            config.long_horizon_discount,
        )
        if len(candidates) < 2:
            continue
        scored = []
        parent_features = _replay_features(optimizer, parent_obj, weight_idx, None, config.output_feature_dim)
        parent_target = _mmd_teacher_target(optimizer, parent_obj, weight_idx, particles, front, config.mmd_bandwidth)
        for child_obj in candidates:
            target = _mmd_teacher_target(optimizer, child_obj, weight_idx, particles, front, config.mmd_bandwidth)
            features = _replay_features(optimizer, child_obj, weight_idx, parent_obj, config.output_feature_dim)
            scored.append((target, features, child_obj))
            _write_record(
                handle,
                {
                    "kind": "residual",
                    "case": case.name,
                    "first": parent_features,
                    "second": features,
                    "value": target - parent_target,
                },
            )
            counts["residual"] += 1
        scored.sort(key=lambda item: item[0])
        best_target, best_features, best_obj = scored[0]
        for other_target, other_features, other_obj in scored[1:]:
            diff = other_target - best_target
            if diff <= 1e-9:
                continue
            margin = min(0.15, 0.01 + 0.25 * diff)
            _write_record(handle, {"kind": "ranking", "case": case.name, "first": best_features, "second": other_features, "margin": margin})
            counts["ranking"] += 1
            best_gap = optimizer._archive_gap_value_from_norm(*optimizer._normalize2(best_obj))
            other_gap = optimizer._archive_gap_value_from_norm(*optimizer._normalize2(other_obj))
            if best_gap > other_gap + 1e-4:
                _write_record(handle, {"kind": "coverage", "case": case.name, "first": best_features, "second": other_features, "margin": margin})
                counts["coverage"] += 1
            best_drift = optimizer._directional_drift_from_norm(*optimizer._normalize2(best_obj), weight_idx)
            other_drift = optimizer._directional_drift_from_norm(*optimizer._normalize2(other_obj), weight_idx)
            if best_drift + 1e-4 < other_drift:
                _write_record(handle, {"kind": "hypercone", "case": case.name, "first": best_features, "second": other_features, "margin": margin})
                counts["hypercone"] += 1
            if _is_scalar_hard_negative(optimizer, best_obj, other_obj, weight_idx):
                _write_record(handle, {"kind": "expert", "case": case.name, "first": best_features, "second": other_features, "margin": margin})
                counts["expert"] += 1
        if action_records:
            best_action_reward = max(float(record["reward"]) for record in action_records)
            for record in action_records:
                _write_record(
                    handle,
                    {
                        "kind": "move_action",
                        "case": case.name,
                        "action_group": action_group,
                        **record,
                    },
                )
                counts["move_action"] += 1
                if (
                    float(record["scalar_delta"]) < -1e-6
                    and float(record["archive_hv_increment"]) <= 1e-12
                    and float(record["long_horizon_advantage"]) <= 1e-12
                    and float(record["reward"]) < best_action_reward - 1e-4
                ):
                    _write_record(
                        handle,
                        {
                            "kind": "move_hard_negative",
                            "case": case.name,
                            "action_group": action_group,
                            **record,
                            "reward": -min(1.0, abs(float(record["flow_advantage"])) + 0.25),
                            "target_advantage": -max(0.25, abs(float(record["target_advantage"]))),
                            "conductance_advantage": -abs(float(record["conductance_advantage"])),
                        },
                    )
                    counts["move_hard_negative"] += 1
    return counts


def _state_objectives(optimizer: TheoryAlignedIPSOptimizer) -> List[ObjectiveVector]:
    objectives = list(optimizer.objectives)
    objectives.extend(entry.objectives for entry in optimizer.archive.entries)
    return objectives


def _front_points(optimizer: TheoryAlignedIPSOptimizer) -> Tuple[Tuple[float, float], ...]:
    points = [optimizer._normalize2(entry.objectives) for entry in optimizer.archive.entries]
    if not points:
        points = [optimizer._normalize2(obj) for obj in optimizer.objectives]
    nondominated = []
    best_y = float("inf")
    for x, y in sorted(points):
        if y < best_y:
            nondominated.append((x, y))
            best_y = y
    if not nondominated:
        return ((0.5, 0.5),)
    return tuple(nondominated)


def _sample_action_candidates(
    optimizer: TheoryAlignedIPSOptimizer,
    parent: Tour,
    parent_obj: ObjectiveVector,
    rng: random.Random,
    pool_size: int,
) -> List[ObjectiveVector]:
    candidates: List[ObjectiveVector] = []
    seen = set()
    for _ in range(max(2, pool_size)):
        i, j = sample_two_opt_indices(len(parent), rng)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        candidate_obj = optimizer._uncounted_symmetric_two_opt_objectives(parent, parent_obj, i, j)
        if candidate_obj is None:
            child = two_opt_at(parent, i, j)
            candidate_obj = optimizer.instance.evaluate(child)
        candidates.append(candidate_obj)
    return candidates


def _sample_action_candidate_features(
    optimizer: TheoryAlignedIPSOptimizer,
    parent: Tour,
    parent_obj: ObjectiveVector,
    weight_idx: int,
    rng: random.Random,
    pool_size: int,
    move_encoder: SparseMoveGenerator,
    particles: Sequence[Tuple[float, float]],
    front: Sequence[Tuple[float, float]],
    bandwidth: float,
    long_horizon_candidates: int,
    long_horizon_discount: float,
) -> List[dict]:
    base = getattr(optimizer.instance, "base", optimizer.instance)
    matrices = getattr(base, "_distance_matrices", None)
    symmetric = getattr(base, "_symmetric_matrices", None)
    if matrices is None or symmetric is None or len(matrices) != 2 or not all(symmetric):
        return []
    reference = (optimizer._weight0[weight_idx], optimizer._weight1[weight_idx])
    target = optimizer._reference_target_point(weight_idx)
    context = _move_replay_context(optimizer, parent_obj, weight_idx)
    parent_target = _mmd_teacher_target(optimizer, parent_obj, weight_idx, particles, front, bandwidth)
    parent_scalar = optimizer._base_scalar2_by_weight(parent_obj, weight_idx)
    parent_mf = optimizer._mean_field_target_reward_from_norm(*optimizer._normalize2(parent_obj), weight_idx)
    records: List[dict] = []
    seen = set()
    for _ in range(max(2, pool_size)):
        i, j = sample_two_opt_indices(len(parent), rng)
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        child_obj = optimizer._uncounted_symmetric_two_opt_objectives(parent, parent_obj, i, j)
        if child_obj is None:
            continue
        first = move_encoder.node_features(
            parent,
            i,
            parent_obj,
            matrices,
            reference,
            target,
            optimizer._edge_scale0,
            optimizer._edge_scale1,
            context,
        )
        second = move_encoder.node_features(
            parent,
            j,
            parent_obj,
            matrices,
            reference,
            target,
            optimizer._edge_scale0,
            optimizer._edge_scale1,
            context,
        )
        child_target = _mmd_teacher_target(optimizer, child_obj, weight_idx, particles, front, bandwidth)
        teacher_delta = child_target - parent_target
        scalar_delta = optimizer._base_scalar2_by_weight(child_obj, weight_idx) - parent_scalar
        archive_hv_increment = optimizer._normalized_archive_hv_gain2(child_obj)
        child = two_opt_at(parent, i, j)
        long_horizon_advantage = _long_horizon_archive_advantage(
            optimizer,
            child,
            child_obj,
            rng,
            config_candidates=max(0, long_horizon_candidates),
            discount=max(0.0, min(1.0, long_horizon_discount)),
            immediate_hv=archive_hv_increment,
        )
        flow_advantage = parent_target - child_target
        target_progress = _target_direction_advantage(optimizer, parent_obj, child_obj, weight_idx)
        child_mf = optimizer._mean_field_target_reward_from_norm(*optimizer._normalize2(child_obj), weight_idx)
        mean_field_advantage = child_mf - parent_mf
        basin_crossing = float(_objective_basin(optimizer, child_obj) != _objective_basin(optimizer, parent_obj))
        hv_signal = _squash_positive(archive_hv_increment, 0.01)
        horizon_signal = _squash_positive(long_horizon_advantage, 0.01)
        target_advantage = max(
            -1.0,
            min(1.0, 0.55 * hv_signal + 0.25 * horizon_signal + 0.20 * target_progress),
        )
        conductance_advantage = basin_crossing * max(0.0, target_advantage + 0.25 * flow_advantage)
        reward = max(
            -1.0,
            min(
                1.0,
                0.70 * target_advantage
                + 0.15 * flow_advantage
                + 0.10 * mean_field_advantage
                + 0.05 * conductance_advantage,
            ),
        )
        records.append(
            {
                "first": first,
                "second": second,
                "reward": reward,
                "target_advantage": target_advantage,
                "flow_advantage": flow_advantage,
                "mean_field_advantage": mean_field_advantage,
                "conductance_advantage": conductance_advantage,
                "archive_hv_increment": archive_hv_increment,
                "long_horizon_advantage": long_horizon_advantage,
                "basin_crossing": basin_crossing,
                "scalar_delta": scalar_delta,
                "teacher_delta": teacher_delta,
            }
        )
    return records


def _long_horizon_archive_advantage(
    optimizer: TheoryAlignedIPSOptimizer,
    child: Tour,
    child_obj: ObjectiveVector,
    rng: random.Random,
    config_candidates: int,
    discount: float,
    immediate_hv: float,
) -> float:
    best_future_hv = immediate_hv
    seen = set()
    for _ in range(config_candidates):
        i, j = sample_two_opt_indices(len(child), rng)
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        grandchild_obj = optimizer._uncounted_symmetric_two_opt_objectives(child, child_obj, i, j)
        if grandchild_obj is None:
            grandchild_obj = optimizer.instance.evaluate(two_opt_at(child, i, j))
        best_future_hv = max(best_future_hv, optimizer._normalized_archive_hv_gain2(grandchild_obj))
    return immediate_hv + discount * max(0.0, best_future_hv - immediate_hv)


def _target_direction_advantage(
    optimizer: TheoryAlignedIPSOptimizer,
    parent_obj: ObjectiveVector,
    child_obj: ObjectiveVector,
    weight_idx: int,
) -> float:
    p0, p1 = optimizer._normalize2(parent_obj)
    z0, z1 = optimizer._normalize2(child_obj)
    target0, target1 = optimizer._reference_target_point(weight_idx)
    parent_distance = abs(p0 - target0) + abs(p1 - target1)
    child_distance = abs(z0 - target0) + abs(z1 - target1)
    dominance_progress = max(0.0, p0 - z0) + max(0.0, p1 - z1)
    return max(-1.0, min(1.0, parent_distance - child_distance + 0.50 * dominance_progress))


def _objective_basin(
    optimizer: TheoryAlignedIPSOptimizer,
    objective: ObjectiveVector,
    bins: int = 6,
) -> Tuple[int, int]:
    z0, z1 = optimizer._normalize2(objective)
    return (
        min(bins - 1, max(0, int(max(0.0, min(0.999999, z0)) * bins))),
        min(bins - 1, max(0, int(max(0.0, min(0.999999, z1)) * bins))),
    )


def _squash_positive(value: float, scale: float) -> float:
    positive = max(0.0, float(value))
    return positive / (max(1e-12, scale) + positive)


def _move_replay_context(
    optimizer: TheoryAlignedIPSOptimizer,
    objective: ObjectiveVector,
    weight_idx: int,
) -> Tuple[float, ...]:
    z0, z1 = optimizer._normalize2(objective)
    target0, target1 = optimizer._reference_target_point(weight_idx)
    return (
        optimizer._archive_gap_value_from_norm(z0, z1),
        optimizer._mean_field_target_reward_from_norm(z0, z1, weight_idx),
        optimizer._weight_extremeness(weight_idx),
        abs(z0 - target0) + abs(z1 - target1),
    )


def _replay_features(
    optimizer: TheoryAlignedIPSOptimizer,
    objective: ObjectiveVector,
    weight_idx: int,
    parent_objective: Optional[ObjectiveVector],
    output_dim: int,
) -> Feature:
    z0, z1 = optimizer._normalize2(objective)
    terms = optimizer._archive_bias_terms2(objective)
    hv_gain = novelty = 0.0
    if terms is not None:
        _, _, hv_gain, novelty = terms
    if parent_objective is None:
        dz0 = dz1 = scalar_delta = 0.0
    else:
        dz0 = (objective[0] - parent_objective[0]) * optimizer._inv0
        dz1 = (objective[1] - parent_objective[1]) * optimizer._inv1
        scalar_delta = optimizer._base_scalar2_by_weight(objective, weight_idx) - optimizer._base_scalar2_by_weight(
            parent_objective,
            weight_idx,
        )
    gap_value = optimizer._archive_gap_value_from_norm(z0, z1)
    drift = optimizer._directional_drift_from_norm(z0, z1, weight_idx)
    extreme = optimizer._weight_extremeness(weight_idx)
    extreme_progress = optimizer._extreme_progress_from_norm(z0, z1, weight_idx)
    state_scalar = optimizer._base_scalar2_by_weight(objective, weight_idx)
    if weight_idx < len(optimizer._particle_direction_summary):
        best, mean_value, std_value, neighbor_best, neighbor_mean, archive_best = optimizer._particle_direction_summary[
            weight_idx
        ]
    else:
        best = mean_value = neighbor_best = neighbor_mean = archive_best = state_scalar
        std_value = 0.0
    particle_summary = (
        state_scalar - best,
        state_scalar - mean_value,
        std_value / (1.0 + abs(std_value)),
        state_scalar - neighbor_best,
        state_scalar - neighbor_mean,
        state_scalar - archive_best,
    )
    target0, target1 = optimizer._reference_target_point(weight_idx)
    target_dist = abs(z0 - target0) + abs(z1 - target1)
    front_weight = optimizer._pcd_front_reweight_from_terms(hv_gain, novelty, gap_value, extreme, extreme_progress)
    values = (
        z0,
        z1,
        dz0,
        dz1,
        optimizer._weight0[weight_idx],
        optimizer._weight1[weight_idx],
        hv_gain,
        novelty,
        gap_value,
        drift,
        extreme,
        extreme_progress,
        state_scalar,
        scalar_delta,
        *particle_summary,
        target0,
        target1,
        target_dist,
        front_weight,
    )
    bounded = tuple(_bound(value) for value in values)
    if output_dim <= len(bounded):
        return bounded[:output_dim]
    return (*bounded, *((0.0,) * (output_dim - len(bounded))))


def _mmd_teacher_target(
    optimizer: TheoryAlignedIPSOptimizer,
    objective: ObjectiveVector,
    weight_idx: int,
    particle_points: Sequence[Tuple[float, float]],
    front_points: Sequence[Tuple[float, float]],
    bandwidth: float,
) -> float:
    z0, z1 = optimizer._normalize2(objective)
    sigma2 = max(1e-6, bandwidth * bandwidth)
    front_kernel = _mean_kernel(z0, z1, front_points, sigma2)
    particle_kernel = _mean_kernel(z0, z1, particle_points, sigma2)
    target0, target1 = optimizer._reference_target_point(weight_idx)
    target_reward = 1.0 / (1.0 + abs(z0 - target0) + abs(z1 - target1))
    gap_reward = optimizer._archive_gap_value_from_norm(z0, z1)
    extreme_reward = optimizer._weight_extremeness(weight_idx) * optimizer._extreme_progress_from_norm(z0, z1, weight_idx)
    mmd_reward = max(0.0, front_kernel - 0.35 * particle_kernel)
    reward = 0.45 * mmd_reward + 0.25 * target_reward + 0.15 * gap_reward + 0.15 * extreme_reward
    return -max(0.0, min(1.0, reward))


def _mean_kernel(
    z0: float,
    z1: float,
    points: Sequence[Tuple[float, float]],
    sigma2: float,
) -> float:
    if not points:
        return 0.0
    total = 0.0
    for x, y in points:
        dist2 = (z0 - x) ** 2 + (z1 - y) ** 2
        total += math.exp(-0.5 * dist2 / sigma2)
    return total / len(points)


def _is_scalar_hard_negative(
    optimizer: TheoryAlignedIPSOptimizer,
    best_obj: ObjectiveVector,
    other_obj: ObjectiveVector,
    weight_idx: int,
) -> bool:
    scalar_best = optimizer._base_scalar2_by_weight(best_obj, weight_idx)
    scalar_other = optimizer._base_scalar2_by_weight(other_obj, weight_idx)
    best_gap = optimizer._archive_gap_value_from_norm(*optimizer._normalize2(best_obj))
    other_gap = optimizer._archive_gap_value_from_norm(*optimizer._normalize2(other_obj))
    return scalar_other + 1e-4 < scalar_best and best_gap > other_gap + 1e-4


def _write_record(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _reservoir_add(bucket: list, item, seen: int, limit: int, rng: random.Random) -> int:
    next_seen = seen + 1
    if limit <= 0:
        return next_seen
    if len(bucket) < limit:
        bucket.append(item)
        return next_seen
    replace_idx = rng.randrange(next_seen)
    if replace_idx < limit:
        bucket[replace_idx] = item
    return next_seen


def _new_prior_backend(backend: str, input_dim: int, hidden_units: int, rng: random.Random):
    name = backend.lower().strip()
    if name == "pcd":
        return PCDResidualScalarNet(input_dim, hidden_units, rng)
    if name == "paretoflow":
        return ParetoFlowScalarNet(input_dim, hidden_units, rng)
    if name == "tiny":
        return TinyMLP(input_dim, hidden_units, rng)
    raise ValueError("backend must be one of: tiny, paretoflow, pcd")


def _bound(value: float) -> float:
    if value != value:
        return 0.0
    return max(-2.0, min(2.0, float(value)))
