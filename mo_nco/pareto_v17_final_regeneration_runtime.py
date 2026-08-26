"""Mechanical final-regeneration runtime seam for interacting Pareto-SMC.

The pre-block population may have arbitrary dependence created by interacting
SMC.  This seam freezes that population and then evolves every terminal
particle with a private domain-separated random tape and no further resampling,
archive feedback, or cross-particle reads.  The mathematical theorem interprets
those tapes as conditionally independent ideal random variables; the Python
implementation provides replay and structural isolation, not a proof of PRNG
independence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Callable, Generic, Mapping, Sequence, TypeVar


StateT = TypeVar("StateT")


class FinalRegenerationRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeBlock:
    steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or self.steps < 0:
            raise FinalRegenerationRuntimeError("block steps must be nonnegative integers")


@dataclass(frozen=True)
class FinalRegenerationRuntimeResult(Generic[StateT]):
    terminal_states: Mapping[str, tuple[StateT, ...]]
    categorical_counts: Mapping[str, tuple[int, ...]]
    trace_sha256: str
    seed_commitment_sha256: str
    conditional_independence_contract: str

    def to_dict(self) -> dict[str, object]:
        return {
            "categorical_counts": {key: list(value) for key, value in self.categorical_counts.items()},
            "trace_sha256": self.trace_sha256,
            "seed_commitment_sha256": self.seed_commitment_sha256,
            "conditional_independence_contract": self.conditional_independence_contract,
        }


def _derive_seed(master_seed: int, context_sha256: str, type_id: str, particle: int) -> int:
    payload = (
        b"pareto-smc-v17-final-regeneration\0"
        + str(master_seed).encode("ascii")
        + b"\0"
        + context_sha256.encode("ascii")
        + b"\0"
        + type_id.encode("utf-8")
        + b"\0"
        + str(particle).encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def run_final_regeneration_block(
    *,
    frozen_states: Mapping[str, Sequence[StateT]],
    blocks_by_type: Mapping[str, Sequence[RuntimeBlock]],
    num_categories: int,
    transition: Callable[[str, StateT, int, int, random.Random], StateT],
    classify: Callable[[StateT], int],
    master_seed: int,
    context_sha256: str,
) -> FinalRegenerationRuntimeResult[StateT]:
    if len(context_sha256) != 64:
        raise FinalRegenerationRuntimeError("context_sha256 must be a 64-character digest")
    if num_categories < 2:
        raise FinalRegenerationRuntimeError("at least outside plus one cell category is required")
    if set(frozen_states) != set(blocks_by_type):
        raise FinalRegenerationRuntimeError("state and block type sets differ")

    terminal: dict[str, tuple[StateT, ...]] = {}
    counts_out: dict[str, tuple[int, ...]] = {}
    trace: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    for type_id in sorted(frozen_states):
        blocks = tuple(blocks_by_type[type_id])
        states_out: list[StateT] = []
        counts = [0] * num_categories
        for particle, initial in enumerate(tuple(frozen_states[type_id])):
            seed = _derive_seed(master_seed, context_sha256, type_id, particle)
            rng = random.Random(seed)
            seed_rows.append({"type_id": type_id, "particle": particle, "seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest()})
            state = initial
            for block_index, block in enumerate(blocks):
                for step_index in range(block.steps):
                    state = transition(type_id, state, block_index, step_index, rng)
            category = int(classify(state))
            if category < 0 or category >= num_categories:
                raise FinalRegenerationRuntimeError("terminal state classified outside the frozen category family")
            counts[category] += 1
            states_out.append(state)
            trace.append(
                {
                    "type_id": type_id,
                    "particle": particle,
                    "category": category,
                    "terminal_repr_sha256": hashlib.sha256(repr(state).encode("utf-8")).hexdigest(),
                }
            )
        terminal[type_id] = tuple(states_out)
        counts_out[type_id] = tuple(counts)

    trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    seed_bytes = json.dumps(seed_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FinalRegenerationRuntimeResult(
        terminal_states=terminal,
        categorical_counts=counts_out,
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        seed_commitment_sha256=hashlib.sha256(seed_bytes).hexdigest(),
        conditional_independence_contract=(
            "private_domain_separated_tapes_no_resampling_no_cross_particle_reads; "
            "mathematical independence remains an ideal-randomness assumption"
        ),
    )
