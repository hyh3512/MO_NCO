"""Scalable blocked occupancy certificates for Pareto-SMC v19.1.

Exact inclusion--exclusion over ``J`` cells costs ``2**J`` terms.  This module
freezes a partition of the certified cells into small blocks and evaluates
inclusion--exclusion exactly inside each block.  Two cross-block aggregations
are available:

* ``union``: the sum of block-failure probabilities;
* ``hunter_pairwise``: Hunter's spanning-tree upper bound, which subtracts a
  maximum spanning tree of exactly computed pairwise block-failure
  intersections.

For maximum block size ``B``, the union model costs ``O(sum_b 2**|B_b|)``.
The Hunter model additionally evaluates unions of two blocks, at cost
``O(number_of_blocks**2 * 2**(2B))`` in the worst case.  Both are rigorous
upper bounds under the dominated-categorical coupling.  The integer allocation
planner is globally optimal for the selected blocked surrogate whenever its
Dijkstra search completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from fractions import Fraction
import heapq
from typing import Sequence

from .pareto_v18_occupancy import OccupancyCertificateError, union_miss_upper
from .pareto_v17_regeneration import as_fraction
from .pareto_v17_multitype_confirm import MultiTypeConfirmProblem, greedy_feasible_allocation


class BlockedOccupancyError(OccupancyCertificateError):
    pass


def _validate_matrix(
    raw: Sequence[Sequence[Fraction | int | str]],
) -> tuple[tuple[Fraction, ...], ...]:
    matrix = tuple(tuple(as_fraction(x) for x in row) for row in raw)
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise BlockedOccupancyError("endpoint lower matrix must be nonempty and rectangular")
    for r, row in enumerate(matrix):
        if any(x < 0 or x > 1 for x in row):
            raise BlockedOccupancyError("cell lower probabilities must lie in [0,1]")
        if sum(row, Fraction(0, 1)) > 1:
            raise BlockedOccupancyError(
                f"type row {r} is not a dominated categorical lower vector"
            )
    for j in range(len(matrix[0])):
        if all(matrix[r][j] == 0 for r in range(len(matrix))):
            raise BlockedOccupancyError(f"cell {j} has no positive lower hit probability")
    return matrix


def canonical_contiguous_blocks(num_cells: int, max_block_size: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(num_cells, int) or num_cells <= 0:
        raise BlockedOccupancyError("num_cells must be positive")
    if not isinstance(max_block_size, int) or max_block_size <= 0:
        raise BlockedOccupancyError("max_block_size must be positive")
    return tuple(
        tuple(range(start, min(num_cells, start + max_block_size)))
        for start in range(0, num_cells, max_block_size)
    )


def validate_partition(
    blocks: Sequence[Sequence[int]],
    num_cells: int,
    *,
    max_block_size: int,
) -> tuple[tuple[int, ...], ...]:
    parsed = tuple(tuple(int(j) for j in block) for block in blocks)
    if not parsed or any(not block for block in parsed):
        raise BlockedOccupancyError("blocks must form a nonempty partition")
    if any(len(block) > max_block_size for block in parsed):
        raise BlockedOccupancyError("a block exceeds max_block_size")
    flat = [j for block in parsed for j in block]
    if sorted(flat) != list(range(num_cells)) or len(set(flat)) != len(flat):
        raise BlockedOccupancyError("blocks must partition every cell exactly once")
    if any(tuple(sorted(block)) != block for block in parsed):
        raise BlockedOccupancyError("cell indices inside a block must be sorted")
    return parsed


def exact_block_all_hit_lower(
    endpoint_lower: Sequence[Sequence[Fraction | int | str]],
    counts: Sequence[int],
    block: Sequence[int],
) -> Fraction:
    matrix = _validate_matrix(endpoint_lower)
    state = tuple(counts)
    if len(state) != len(matrix) or any(not isinstance(x, int) or x < 0 for x in state):
        raise BlockedOccupancyError("invalid allocation counts")
    cells = tuple(int(j) for j in block)
    if not cells or any(j < 0 or j >= len(matrix[0]) for j in cells):
        raise BlockedOccupancyError("invalid block cell index")
    total = Fraction(0, 1)
    for mask in range(1 << len(cells)):
        sign = -1 if mask.bit_count() % 2 else 1
        term = Fraction(1, 1)
        for r, count in enumerate(state):
            mass = sum(
                (matrix[r][cells[k]] for k in range(len(cells)) if mask & (1 << k)),
                Fraction(0, 1),
            )
            term *= (Fraction(1, 1) - mass) ** count
        total += sign * term
    if total < 0 or total > 1:
        raise AssertionError("exact block occupancy probability escaped [0,1]")
    return total


def _maximum_spanning_tree_weight(
    vertex_count: int,
    weighted_edges: Sequence[tuple[Fraction, int, int]],
) -> Fraction:
    if vertex_count <= 1:
        return Fraction(0, 1)
    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    total = Fraction(0, 1)
    used = 0
    for weight, a, b in sorted(
        weighted_edges, key=lambda item: (-item[0], item[1], item[2])
    ):
        if union(a, b):
            total += weight
            used += 1
            if used == vertex_count - 1:
                break
    if used != vertex_count - 1:
        raise AssertionError("complete block graph did not contain a spanning tree")
    return total


def hunter_blocked_simultaneous_miss_upper(
    endpoint_lower: Sequence[Sequence[Fraction | int | str]],
    counts: Sequence[int],
    blocks: Sequence[Sequence[int]],
    *,
    max_block_size: int = 12,
    max_pair_union_size: int | None = None,
) -> Fraction:
    """Hunter spanning-tree upper bound for the union of block failures."""

    matrix = _validate_matrix(endpoint_lower)
    parsed = validate_partition(blocks, len(matrix[0]), max_block_size=max_block_size)
    pair_cap = 2 * max_block_size if max_pair_union_size is None else int(max_pair_union_size)
    if pair_cap <= 0:
        raise BlockedOccupancyError("max_pair_union_size must be positive")
    hits = tuple(exact_block_all_hit_lower(matrix, counts, block) for block in parsed)
    failures = tuple(Fraction(1, 1) - hit for hit in hits)
    if len(parsed) == 1:
        return failures[0]
    edges: list[tuple[Fraction, int, int]] = []
    for a in range(len(parsed)):
        for b in range(a + 1, len(parsed)):
            union_cells = tuple(sorted((*parsed[a], *parsed[b])))
            if len(union_cells) > pair_cap:
                raise BlockedOccupancyError(
                    "a pairwise block union exceeds max_pair_union_size"
                )
            both_hit = exact_block_all_hit_lower(matrix, counts, union_cells)
            both_fail = Fraction(1, 1) - hits[a] - hits[b] + both_hit
            if both_fail < 0 or both_fail > min(failures[a], failures[b]):
                raise AssertionError("pairwise block-failure intersection escaped its probability range")
            edges.append((both_fail, a, b))
    hunter = sum(failures, Fraction(0, 1)) - _maximum_spanning_tree_weight(
        len(parsed), edges
    )
    return min(Fraction(1, 1), max(Fraction(0, 1), hunter))


def blocked_simultaneous_miss_upper(
    endpoint_lower: Sequence[Sequence[Fraction | int | str]],
    counts: Sequence[int],
    blocks: Sequence[Sequence[int]],
    *,
    max_block_size: int = 12,
    combination_mode: str = "hunter_pairwise",
    max_pair_union_size: int | None = None,
) -> Fraction:
    matrix = _validate_matrix(endpoint_lower)
    parsed = validate_partition(blocks, len(matrix[0]), max_block_size=max_block_size)
    if combination_mode == "hunter_pairwise":
        return hunter_blocked_simultaneous_miss_upper(
            matrix, counts, parsed,
            max_block_size=max_block_size,
            max_pair_union_size=max_pair_union_size,
        )
    if combination_mode != "union":
        raise BlockedOccupancyError("unsupported block combination mode")
    total = sum(
        (Fraction(1, 1) - exact_block_all_hit_lower(matrix, counts, block) for block in parsed),
        Fraction(0, 1),
    )
    return min(Fraction(1, 1), total)


@dataclass(frozen=True)
class BlockedOccupancyProblem:
    endpoint_lower: tuple[tuple[Fraction, ...], ...]
    costs: tuple[Fraction, ...]
    delta: Fraction
    blocks: tuple[tuple[int, ...], ...]
    max_block_size: int = 12
    risk_cache_size: int = 100_000
    combination_mode: str = "hunter_pairwise"
    max_pair_union_size: int | None = None
    _block_terms: tuple[tuple[tuple[int, tuple[Fraction, ...]], ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _pair_terms: tuple[tuple[int, int, tuple[tuple[int, tuple[Fraction, ...]], ...]], ...] = field(
        init=False, repr=False, compare=False
    )
    _risk_cache: OrderedDict[tuple[int, ...], Fraction] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        matrix = _validate_matrix(self.endpoint_lower)
        costs = tuple(as_fraction(x) for x in self.costs)
        delta = as_fraction(self.delta)
        blocks = validate_partition(
            self.blocks,
            len(matrix[0]),
            max_block_size=self.max_block_size,
        )
        if len(costs) != len(matrix) or any(x <= 0 for x in costs):
            raise BlockedOccupancyError("one positive rational cost is required per type")
        if not (Fraction(0, 1) < delta < Fraction(1, 1)):
            raise BlockedOccupancyError("delta must lie in (0,1)")
        if self.risk_cache_size <= 0:
            raise BlockedOccupancyError("risk_cache_size must be positive")
        if self.combination_mode not in {"union", "hunter_pairwise"}:
            raise BlockedOccupancyError("unsupported block combination mode")
        pair_cap = (
            2 * self.max_block_size
            if self.max_pair_union_size is None
            else int(self.max_pair_union_size)
        )
        if pair_cap <= 0:
            raise BlockedOccupancyError("max_pair_union_size must be positive")
        if self.combination_mode == "hunter_pairwise" and any(
            len(blocks[a]) + len(blocks[b]) > pair_cap
            for a in range(len(blocks))
            for b in range(a + 1, len(blocks))
        ):
            raise BlockedOccupancyError(
                "a pairwise block union exceeds max_pair_union_size"
            )
        object.__setattr__(self, "max_pair_union_size", pair_cap)
        object.__setattr__(self, "endpoint_lower", matrix)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "blocks", blocks)
        block_terms: list[tuple[tuple[int, tuple[Fraction, ...]], ...]] = []
        for block in blocks:
            terms: list[tuple[int, tuple[Fraction, ...]]] = []
            for mask in range(1 << len(block)):
                sign = -1 if mask.bit_count() % 2 else 1
                bases = tuple(
                    Fraction(1, 1)
                    - sum(
                        (
                            matrix[r][block[k]]
                            for k in range(len(block))
                            if mask & (1 << k)
                        ),
                        Fraction(0, 1),
                    )
                    for r in range(len(matrix))
                )
                terms.append((sign, bases))
            block_terms.append(tuple(terms))
        object.__setattr__(self, "_block_terms", tuple(block_terms))
        pair_terms: list[
            tuple[int, int, tuple[tuple[int, tuple[Fraction, ...]], ...]]
        ] = []
        if self.combination_mode == "hunter_pairwise":
            for a in range(len(blocks)):
                for b in range(a + 1, len(blocks)):
                    union_block = tuple(sorted((*blocks[a], *blocks[b])))
                    terms: list[tuple[int, tuple[Fraction, ...]]] = []
                    for mask in range(1 << len(union_block)):
                        sign = -1 if mask.bit_count() % 2 else 1
                        bases = tuple(
                            Fraction(1, 1)
                            - sum(
                                (
                                    matrix[r][union_block[k]]
                                    for k in range(len(union_block))
                                    if mask & (1 << k)
                                ),
                                Fraction(0, 1),
                            )
                            for r in range(len(matrix))
                        )
                        terms.append((sign, bases))
                    pair_terms.append((a, b, tuple(terms)))
        object.__setattr__(self, "_pair_terms", tuple(pair_terms))
        object.__setattr__(self, "_risk_cache", OrderedDict())

    @property
    def num_types(self) -> int:
        return len(self.endpoint_lower)

    @property
    def num_cells(self) -> int:
        return len(self.endpoint_lower[0])

    @property
    def subset_term_count(self) -> int:
        base = sum(1 << len(block) for block in self.blocks)
        if self.combination_mode == "union":
            return base
        pair = sum(
            1 << (len(self.blocks[a]) + len(self.blocks[b]))
            for a in range(len(self.blocks))
            for b in range(a + 1, len(self.blocks))
        )
        return base + pair

    def validate_counts(self, counts: Sequence[int]) -> tuple[int, ...]:
        state = tuple(counts)
        if len(state) != self.num_types or any(
            not isinstance(x, int) or x < 0 for x in state
        ):
            raise BlockedOccupancyError("invalid allocation counts")
        return state

    def cost(self, counts: Sequence[int]) -> Fraction:
        state = self.validate_counts(counts)
        return sum(
            (self.costs[r] * state[r] for r in range(self.num_types)),
            Fraction(0, 1),
        )

    def risk(self, counts: Sequence[int]) -> Fraction:
        state = self.validate_counts(counts)
        cache = self._risk_cache
        cached = cache.get(state)
        if cached is not None:
            cache.move_to_end(state)
            return cached
        hits: list[Fraction] = []
        for terms in self._block_terms:
            hit = Fraction(0, 1)
            for sign, bases in terms:
                term = Fraction(1, 1)
                for r, count in enumerate(state):
                    term *= bases[r] ** count
                hit += sign * term
            if hit < 0 or hit > 1:
                raise AssertionError("block hit probability escaped [0,1]")
            hits.append(hit)
        failures = tuple(Fraction(1, 1) - hit for hit in hits)
        if self.combination_mode == "union" or len(hits) == 1:
            value = min(Fraction(1, 1), sum(failures, Fraction(0, 1)))
        else:
            edges: list[tuple[Fraction, int, int]] = []
            for a, b, terms in self._pair_terms:
                both_hit = Fraction(0, 1)
                for sign, bases in terms:
                    term = Fraction(1, 1)
                    for r, count in enumerate(state):
                        term *= bases[r] ** count
                    both_hit += sign * term
                both_fail = Fraction(1, 1) - hits[a] - hits[b] + both_hit
                if both_fail < 0 or both_fail > min(failures[a], failures[b]):
                    raise AssertionError(
                        "pairwise block-failure intersection escaped its probability range"
                    )
                edges.append((both_fail, a, b))
            hunter = sum(failures, Fraction(0, 1)) - _maximum_spanning_tree_weight(
                len(hits), edges
            )
            value = min(Fraction(1, 1), max(Fraction(0, 1), hunter))
        if len(cache) >= self.risk_cache_size:
            cache.popitem(last=False)
        cache[state] = value
        return value

    def feasible(self, counts: Sequence[int]) -> bool:
        return self.risk(counts) <= self.delta


@dataclass(frozen=True)
class BlockedOccupancyPlan:
    counts: tuple[int, ...]
    total_cost: Fraction
    blocked_miss_upper: Fraction
    cellwise_union_miss_upper: Fraction
    optimal_for_blocked_surrogate: bool
    explored_nodes: int
    search_exhausted: bool
    subset_term_count: int
    risk_model: str

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_model": self.risk_model,
            "counts": list(self.counts),
            "total_cost": str(self.total_cost),
            "blocked_miss_upper": str(self.blocked_miss_upper),
            "cellwise_union_miss_upper": str(self.cellwise_union_miss_upper),
            "optimal_for_blocked_surrogate": self.optimal_for_blocked_surrogate,
            "explored_nodes": self.explored_nodes,
            "search_exhausted": self.search_exhausted,
            "subset_term_count": self.subset_term_count,
        }


def _union_incumbent(problem: BlockedOccupancyProblem, max_steps: int) -> tuple[int, ...]:
    union_problem = MultiTypeConfirmProblem(
        endpoint_lower=problem.endpoint_lower,
        costs=problem.costs,
        delta=problem.delta,
    )
    return greedy_feasible_allocation(union_problem, max_steps=max_steps)


def exact_minimum_cost_blocked_allocation(
    problem: BlockedOccupancyProblem,
    *,
    max_nodes: int = 2_000_000,
    max_greedy_steps: int = 1_000_000,
) -> BlockedOccupancyPlan:
    if max_nodes <= 0 or max_greedy_steps <= 0:
        raise BlockedOccupancyError("search budgets must be positive")
    incumbent = _union_incumbent(problem, max_greedy_steps)
    incumbent_cost = problem.cost(incumbent)
    zero = tuple(0 for _ in range(problem.num_types))
    queue: list[tuple[Fraction, int, tuple[int, ...]]] = [(Fraction(0, 1), 0, zero)]
    seen = {zero}
    explored = 0
    while queue:
        cost, l1, state = heapq.heappop(queue)
        explored += 1
        if explored > max_nodes:
            return BlockedOccupancyPlan(
                counts=incumbent,
                total_cost=incumbent_cost,
                blocked_miss_upper=problem.risk(incumbent),
                cellwise_union_miss_upper=union_miss_upper(problem.endpoint_lower, incumbent),
                optimal_for_blocked_surrogate=False,
                explored_nodes=explored,
                search_exhausted=True,
                subset_term_count=problem.subset_term_count,
                risk_model=(
                    "partitioned_exact_within_block_hunter_pairwise_v19_1"
                    if problem.combination_mode == "hunter_pairwise"
                    else "partitioned_exact_within_block_union_v19"
                ),
            )
        if cost > incumbent_cost:
            break
        if problem.feasible(state):
            return BlockedOccupancyPlan(
                counts=state,
                total_cost=cost,
                blocked_miss_upper=problem.risk(state),
                cellwise_union_miss_upper=union_miss_upper(problem.endpoint_lower, state),
                optimal_for_blocked_surrogate=True,
                explored_nodes=explored,
                search_exhausted=False,
                subset_term_count=problem.subset_term_count,
                risk_model=(
                    "partitioned_exact_within_block_hunter_pairwise_v19_1"
                    if problem.combination_mode == "hunter_pairwise"
                    else "partitioned_exact_within_block_union_v19"
                ),
            )
        for r in range(problem.num_types):
            child = list(state)
            child[r] += 1
            child_t = tuple(child)
            if child_t in seen:
                continue
            child_cost = cost + problem.costs[r]
            if child_cost > incumbent_cost:
                continue
            seen.add(child_t)
            heapq.heappush(queue, (child_cost, l1 + 1, child_t))
    return BlockedOccupancyPlan(
        counts=incumbent,
        total_cost=incumbent_cost,
        blocked_miss_upper=problem.risk(incumbent),
        cellwise_union_miss_upper=union_miss_upper(problem.endpoint_lower, incumbent),
        optimal_for_blocked_surrogate=True,
        explored_nodes=explored,
        search_exhausted=False,
        subset_term_count=problem.subset_term_count,
        risk_model=(
            "partitioned_exact_within_block_hunter_pairwise_v19_1"
            if problem.combination_mode == "hunter_pairwise"
            else "partitioned_exact_within_block_union_v19"
        ),
    )


__all__ = [
    "BlockedOccupancyError",
    "BlockedOccupancyPlan",
    "BlockedOccupancyProblem",
    "blocked_simultaneous_miss_upper",
    "canonical_contiguous_blocks",
    "exact_block_all_hit_lower",
    "hunter_blocked_simultaneous_miss_upper",
    "exact_minimum_cost_blocked_allocation",
    "validate_partition",
]
