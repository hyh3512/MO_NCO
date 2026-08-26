from __future__ import annotations

"""Authoritative streaming evaluation ledger for V21 experiments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import time
from typing import Callable, Iterator, Sequence, Tuple

from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    Solution,
)
from .types import ObjectiveVector


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _solution_sha256(solution: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_bytes([int(value) for value in solution])).hexdigest()


def _sqlite_real(value: float | None) -> float | None:
    """Canonicalize values to the semantics preserved by SQLite ``REAL``.

    SQLite does not preserve the sign bit of negative zero.  Hashing ``-0.0``
    before insertion would therefore create a semantic chain that a read-only
    verifier cannot reconstruct from the persisted artifact.
    """

    if value is None:
        return None
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


@dataclass(frozen=True)
class EvaluationContext:
    evidence_partition: str
    search_phase_id: str
    stage_id: str
    type_id: int | None
    operator_id: str
    operator_call_id: int
    parent_solutions: Tuple[Solution, ...] = ()
    parent_type_ids: Tuple[int, ...] = ()
    objective_evaluation_kind: str = "full_objective_vector_v1"
    feasible_before_repair: bool | None = None
    repair_applied: bool = False
    repair_operator_id: str | None = None
    local_search_block_id: int | None = None
    local_search_depth: int = 0
    operator_witness: object | None = None


@dataclass(frozen=True)
class LedgerEvaluation:
    evaluation_index: int
    proposal: Solution
    proposal_solution_ref: int
    proposal_sha256: str
    parent_solution_refs: Tuple[int, ...]
    objectives: ObjectiveVector
    duplicate_of_evaluation_index: int | None
    elapsed_monotonic_ns: int


@dataclass(frozen=True)
class DecisionInput:
    accepted_into_population: bool
    population_replacement_count: int
    population_target_type_ids: Tuple[int, ...]
    decision_reason: str
    archive_changed: bool
    retained_after_update: bool
    archive_size_after: int
    scalarization_id: str | None = None
    scalar_parent: float | None = None
    scalar_candidate: float | None = None
    scalar_advantage: float | None = None
    cell_id: str | None = None
    new_evaluated_cell: bool | None = None
    new_nondominated_cell: bool | None = None


class SQLiteEvaluationLedger:
    """One objective call, one evaluation row, and one decision row.

    The raw evaluator is retained only as a private capability of this ledger.
    V21 algorithm code receives the ledger and never invokes ``problem.evaluate``
    directly.  Solutions are family-packed once and referenced by integer IDs.
    """

    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        *,
        database_path: str | Path | None,
    ) -> None:
        self._problem_name = str(problem.name)
        self._family = self._infer_family(problem)
        self._solution_size = int(problem.solution_size)
        self._raw_evaluator: Callable[[Solution], ObjectiveVector] = problem.evaluate
        self._validator = problem.validate_solution
        self._database_path = (
            None if database_path is None else Path(database_path).resolve()
        )
        if self._database_path is not None:
            if self._database_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite V21 trace database: {self._database_path}"
                )
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            location = str(self._database_path)
        else:
            location = ":memory:"
        self._connection = sqlite3.connect(location)
        self._connection.execute("PRAGMA foreign_keys=ON")
        if self._database_path is not None:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self._physical_call_count = 0
        self._evaluation_count = 0
        self._decision_count = 0
        self._mechanism_count = 0
        self._first_evaluation_by_solution_sha: dict[str, int] = {}
        self._solution_ref_by_sha: dict[str, int] = {}
        self._previous_evaluation_hash = "0" * 64
        self._previous_decision_hash = "0" * 64
        self._previous_mechanism_hash = "0" * 64
        self._start_ns = time.monotonic_ns()
        self._finalized = False

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def physical_call_count(self) -> int:
        return self._physical_call_count

    @property
    def database_path(self) -> Path | None:
        return self._database_path

    @classmethod
    def from_problem(
        cls,
        problem: MultiObjectiveCombinatorialProblem,
        *,
        database_path: str | Path | None,
    ) -> "SQLiteEvaluationLedger":
        return cls(problem, database_path=database_path)

    def evaluate(
        self,
        proposal: Solution,
        context: EvaluationContext,
    ) -> LedgerEvaluation:
        if self._finalized:
            raise RuntimeError("A finalized V21 evaluation ledger is immutable.")
        self._validator(proposal)
        proposal_ref, proposal_sha = self._register_solution(proposal)
        parent_refs = tuple(
            self._register_solution(parent)[0] for parent in context.parent_solutions
        )
        self._physical_call_count += 1
        objectives = tuple(float(value) for value in self._raw_evaluator(proposal))
        self._evaluation_count += 1
        index = self._evaluation_count
        duplicate_of = self._first_evaluation_by_solution_sha.get(proposal_sha)
        self._first_evaluation_by_solution_sha.setdefault(proposal_sha, index)
        elapsed_ns = time.monotonic_ns() - self._start_ns
        semantic_payload = {
            "evaluation_index": index,
            "evidence_partition": context.evidence_partition,
            "search_phase_id": context.search_phase_id,
            "stage_id": context.stage_id,
            "type_id": context.type_id,
            "operator_id": context.operator_id,
            "operator_call_id": context.operator_call_id,
            "parent_solution_refs": parent_refs,
            "parent_type_ids": context.parent_type_ids,
            "proposal_solution_ref": proposal_ref,
            "proposal_sha256": proposal_sha,
            "objectives": objectives,
            "objective_evaluation_kind": context.objective_evaluation_kind,
            "feasible_before_repair": context.feasible_before_repair,
            "repair_applied": context.repair_applied,
            "repair_operator_id": context.repair_operator_id,
            "duplicate_of_evaluation_index": duplicate_of,
            "local_search_block_id": context.local_search_block_id,
            "local_search_depth": context.local_search_depth,
            "operator_witness": context.operator_witness,
            "prev_record_sha256": self._previous_evaluation_hash,
        }
        record_hash = hashlib.sha256(_canonical_bytes(semantic_payload)).hexdigest()
        self._connection.execute(
            """
            INSERT INTO evaluations (
                evaluation_index, evidence_partition, search_phase_id, stage_id,
                type_id, operator_id, operator_call_id, parent_solution_refs_json,
                parent_type_ids_json, proposal_solution_ref, proposal_sha256,
                objectives_json, objective_evaluation_kind,
                feasible_before_repair, repair_applied, repair_operator_id,
                duplicate_of_evaluation_index, local_search_block_id,
                local_search_depth, operator_witness_json, elapsed_monotonic_ns,
                prev_record_sha256, record_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                index,
                context.evidence_partition,
                context.search_phase_id,
                context.stage_id,
                context.type_id,
                context.operator_id,
                context.operator_call_id,
                json.dumps(parent_refs, separators=(",", ":")),
                json.dumps(context.parent_type_ids, separators=(",", ":")),
                proposal_ref,
                proposal_sha,
                json.dumps(objectives, separators=(",", ":"), allow_nan=False),
                context.objective_evaluation_kind,
                self._nullable_bool(context.feasible_before_repair),
                int(context.repair_applied),
                context.repair_operator_id,
                duplicate_of,
                context.local_search_block_id,
                context.local_search_depth,
                (
                    None
                    if context.operator_witness is None
                    else json.dumps(
                        context.operator_witness,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
                elapsed_ns,
                self._previous_evaluation_hash,
                record_hash,
            ),
        )
        self._previous_evaluation_hash = record_hash
        return LedgerEvaluation(
            evaluation_index=index,
            proposal=tuple(proposal),
            proposal_solution_ref=proposal_ref,
            proposal_sha256=proposal_sha,
            parent_solution_refs=parent_refs,
            objectives=objectives,
            duplicate_of_evaluation_index=duplicate_of,
            elapsed_monotonic_ns=elapsed_ns,
        )

    def commit_decision(
        self,
        evaluation_index: int,
        decision: DecisionInput,
    ) -> None:
        if self._finalized:
            raise RuntimeError("A finalized V21 evaluation ledger is immutable.")
        if evaluation_index != self._decision_count + 1:
            raise RuntimeError("V21 decisions must be committed in evaluation order.")
        scalar_parent = _sqlite_real(decision.scalar_parent)
        scalar_candidate = _sqlite_real(decision.scalar_candidate)
        scalar_advantage = _sqlite_real(decision.scalar_advantage)
        semantic_payload = {
            "evaluation_index": evaluation_index,
            "accepted_into_population": decision.accepted_into_population,
            "population_replacement_count": decision.population_replacement_count,
            "population_target_type_ids": decision.population_target_type_ids,
            "decision_reason": decision.decision_reason,
            "archive_changed": decision.archive_changed,
            "retained_after_update": decision.retained_after_update,
            "archive_size_after": decision.archive_size_after,
            "scalarization_id": decision.scalarization_id,
            "scalar_parent": scalar_parent,
            "scalar_candidate": scalar_candidate,
            "scalar_advantage": scalar_advantage,
            "cell_id": decision.cell_id,
            "new_evaluated_cell": decision.new_evaluated_cell,
            "new_nondominated_cell": decision.new_nondominated_cell,
            "prev_decision_sha256": self._previous_decision_hash,
        }
        decision_hash = hashlib.sha256(_canonical_bytes(semantic_payload)).hexdigest()
        self._connection.execute(
            """
            INSERT INTO decisions (
                evaluation_index, accepted_into_population,
                population_replacement_count, population_target_type_ids_json,
                decision_reason, archive_changed, retained_after_update,
                archive_size_after, scalarization_id, scalar_parent,
                scalar_candidate, scalar_advantage, cell_id, new_evaluated_cell,
                new_nondominated_cell, decision_elapsed_ns,
                prev_decision_sha256, decision_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                evaluation_index,
                int(decision.accepted_into_population),
                decision.population_replacement_count,
                json.dumps(
                    decision.population_target_type_ids,
                    separators=(",", ":"),
                ),
                decision.decision_reason,
                int(decision.archive_changed),
                int(decision.retained_after_update),
                decision.archive_size_after,
                decision.scalarization_id,
                scalar_parent,
                scalar_candidate,
                scalar_advantage,
                decision.cell_id,
                self._nullable_bool(decision.new_evaluated_cell),
                self._nullable_bool(decision.new_nondominated_cell),
                time.monotonic_ns() - self._start_ns,
                self._previous_decision_hash,
                decision_hash,
            ),
        )
        self._decision_count += 1
        self._previous_decision_hash = decision_hash

    def record_mechanism(
        self,
        *,
        after_evaluation_index: int,
        event_kind: str,
        payload: object,
    ) -> int:
        if self._finalized:
            raise RuntimeError("A finalized V21 evaluation ledger is immutable.")
        if not 0 <= after_evaluation_index <= self._evaluation_count:
            raise ValueError("Mechanism events must bind an observed evaluation prefix.")
        self._mechanism_count += 1
        semantic = {
            "event_index": self._mechanism_count,
            "after_evaluation_index": int(after_evaluation_index),
            "event_kind": str(event_kind),
            "payload": payload,
            "prev_event_sha256": self._previous_mechanism_hash,
        }
        event_hash = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
        self._connection.execute(
            """
            INSERT INTO mechanisms(
                event_index, after_evaluation_index, event_kind,
                payload_json, event_sha256
            ) VALUES (?,?,?,?,?)
            """,
            (
                self._mechanism_count,
                int(after_evaluation_index),
                str(event_kind),
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                event_hash,
            ),
        )
        self._previous_mechanism_hash = event_hash
        return self._mechanism_count

    def iter_evaluated_solutions(
        self,
    ) -> Iterator[tuple[int, Solution, ObjectiveVector]]:
        cursor = self._connection.execute(
            """
            SELECT e.evaluation_index, s.codec, s.solution_size, s.payload,
                   e.objectives_json
            FROM evaluations AS e
            JOIN solutions AS s
              ON s.solution_ref = e.proposal_solution_ref
            ORDER BY e.evaluation_index
            """
        )
        for index, codec, solution_size, payload, objectives_json in cursor:
            yield (
                int(index),
                self._decode_solution(str(codec), int(solution_size), bytes(payload)),
                tuple(float(value) for value in json.loads(objectives_json)),
            )

    def finalize(self, *, expected_budget: int) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("The V21 ledger has already been finalized.")
        evaluation_rows = int(
            self._connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        )
        decision_rows = int(
            self._connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        mechanism_rows = int(
            self._connection.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0]
        )
        bounds = self._connection.execute(
            "SELECT MIN(evaluation_index), MAX(evaluation_index) FROM evaluations"
        ).fetchone()
        integrity = self._connection.execute("PRAGMA integrity_check").fetchone()[0]
        gates = {
            "physical_call_count": self._physical_call_count,
            "evaluation_record_count": evaluation_rows,
            "decision_record_count": decision_rows,
            "expected_budget": int(expected_budget),
            "index_min": bounds[0],
            "index_max": bounds[1],
            "sqlite_integrity": integrity,
            "mechanism_event_count": mechanism_rows,
        }
        if not (
            self._physical_call_count
            == evaluation_rows
            == decision_rows
            == int(expected_budget)
            and bounds == (1, int(expected_budget))
            and integrity == "ok"
        ):
            raise RuntimeError(f"V21 trace finalization gate failed: {gates}")
        for key, value in {
            "schema": "v21_sqlite_evaluation_trace_v1",
            "status": "FINALIZED",
            "problem": self._problem_name,
            "family": self._family,
            "expected_budget": str(expected_budget),
            "physical_call_count": str(self._physical_call_count),
            "terminal_evaluation_chain_sha256": self._previous_evaluation_hash,
            "terminal_decision_chain_sha256": self._previous_decision_hash,
            "terminal_mechanism_chain_sha256": self._previous_mechanism_hash,
        }.items():
            self._connection.execute(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                (key, value),
            )
        self._connection.commit()
        self._finalized = True
        receipt: dict[str, object] = {
            "schema": "v21_trace_finalization_receipt_v1",
            "status": "FINALIZED",
            **gates,
            "terminal_evaluation_chain_sha256": self._previous_evaluation_hash,
            "terminal_decision_chain_sha256": self._previous_decision_hash,
            "terminal_mechanism_chain_sha256": self._previous_mechanism_hash,
            "database_path": (
                None if self._database_path is None else str(self._database_path)
            ),
        }
        self._connection.close()
        if self._database_path is not None:
            raw = self._database_path.read_bytes()
            receipt["database_bytes"] = len(raw)
            receipt["database_sha256"] = hashlib.sha256(raw).hexdigest()
        return receipt

    def _register_solution(self, solution: Solution) -> tuple[int, str]:
        digest = _solution_sha256(solution)
        cached = self._solution_ref_by_sha.get(digest)
        if cached is not None:
            return cached, digest
        codec, payload = self._encode_solution(solution)
        cursor = self._connection.execute(
            """
            INSERT INTO solutions(solution_sha256, family, codec, solution_size, payload)
            VALUES (?,?,?,?,?)
            """,
            (digest, self._family, codec, len(solution), payload),
        )
        solution_ref = int(cursor.lastrowid)
        self._solution_ref_by_sha[digest] = solution_ref
        return solution_ref, digest

    def _encode_solution(self, solution: Sequence[int]) -> tuple[str, bytes]:
        if self._family == "MOKP":
            payload = bytearray((len(solution) + 7) // 8)
            for index, value in enumerate(solution):
                if int(value):
                    payload[index // 8] |= 1 << (index % 8)
            return "mokp-bitpack-lsb-v1", bytes(payload)
        if self._family == "MOTSP":
            if max(solution, default=0) < 2**16:
                return (
                    "motsp-uint16le-v1",
                    struct.pack(f"<{len(solution)}H", *solution),
                )
            return (
                "motsp-uint32le-v1",
                struct.pack(f"<{len(solution)}I", *solution),
            )
        return "generic-json-v1", _canonical_bytes([int(value) for value in solution])

    @staticmethod
    def _decode_solution(codec: str, size: int, payload: bytes) -> Solution:
        if codec == "mokp-bitpack-lsb-v1":
            return tuple(
                (payload[index // 8] >> (index % 8)) & 1
                for index in range(size)
            )
        if codec == "motsp-uint16le-v1":
            return tuple(struct.unpack(f"<{size}H", payload))
        if codec == "motsp-uint32le-v1":
            return tuple(struct.unpack(f"<{size}I", payload))
        if codec == "generic-json-v1":
            return tuple(int(value) for value in json.loads(payload))
        raise ValueError(f"Unsupported V21 solution codec: {codec}")

    @staticmethod
    def _infer_family(problem: MultiObjectiveCombinatorialProblem) -> str:
        if isinstance(problem, MultiObjectiveKnapsackInstance):
            return "MOKP"
        if isinstance(problem, MultiObjectiveTSPProblemAdapter):
            return "MOTSP"
        return "GENERIC"

    @staticmethod
    def _nullable_bool(value: bool | None) -> int | None:
        return None if value is None else int(value)

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE solutions (
                solution_ref INTEGER PRIMARY KEY,
                solution_sha256 TEXT NOT NULL UNIQUE,
                family TEXT NOT NULL,
                codec TEXT NOT NULL,
                solution_size INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE evaluations (
                evaluation_index INTEGER PRIMARY KEY,
                evidence_partition TEXT NOT NULL,
                search_phase_id TEXT NOT NULL,
                stage_id TEXT NOT NULL,
                type_id INTEGER,
                operator_id TEXT NOT NULL,
                operator_call_id INTEGER NOT NULL,
                parent_solution_refs_json TEXT NOT NULL,
                parent_type_ids_json TEXT NOT NULL,
                proposal_solution_ref INTEGER NOT NULL REFERENCES solutions(solution_ref),
                proposal_sha256 TEXT NOT NULL,
                objectives_json TEXT NOT NULL,
                objective_evaluation_kind TEXT NOT NULL,
                feasible_before_repair INTEGER,
                repair_applied INTEGER NOT NULL,
                repair_operator_id TEXT,
                duplicate_of_evaluation_index INTEGER,
                local_search_block_id INTEGER,
                local_search_depth INTEGER NOT NULL,
                operator_witness_json TEXT,
                elapsed_monotonic_ns INTEGER NOT NULL,
                prev_record_sha256 TEXT NOT NULL,
                record_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE decisions (
                evaluation_index INTEGER PRIMARY KEY REFERENCES evaluations(evaluation_index),
                accepted_into_population INTEGER NOT NULL,
                population_replacement_count INTEGER NOT NULL,
                population_target_type_ids_json TEXT NOT NULL,
                decision_reason TEXT NOT NULL,
                archive_changed INTEGER NOT NULL,
                retained_after_update INTEGER NOT NULL,
                archive_size_after INTEGER NOT NULL,
                scalarization_id TEXT,
                scalar_parent REAL,
                scalar_candidate REAL,
                scalar_advantage REAL,
                cell_id TEXT,
                new_evaluated_cell INTEGER,
                new_nondominated_cell INTEGER,
                decision_elapsed_ns INTEGER NOT NULL,
                prev_decision_sha256 TEXT NOT NULL,
                decision_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE mechanisms (
                event_index INTEGER PRIMARY KEY,
                after_evaluation_index INTEGER NOT NULL,
                event_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                event_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


__all__ = [
    "DecisionInput",
    "EvaluationContext",
    "LedgerEvaluation",
    "SQLiteEvaluationLedger",
]
