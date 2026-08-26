from __future__ import annotations

"""Independent replay verifier for the V21 SQLite trace contract."""

import hashlib
import json
from pathlib import Path
import sqlite3
import struct
from typing import Sequence

from .archive import ArchiveEntry, ParetoArchive
from .pareto_ijoc_problem import MultiObjectiveCombinatorialProblem, Solution


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _solution_sha256(solution: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_bytes([int(value) for value in solution])).hexdigest()


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


def _nullable_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def verify_v21_trace_database(
    database_path: str | Path,
    problem: MultiObjectiveCombinatorialProblem,
    *,
    expected_budget: int,
    expected_archive: ParetoArchive | None = None,
) -> dict[str, object]:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact_bytes = path.read_bytes()
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity_check failed.")
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        if metadata.get("status") != "FINALIZED":
            raise ValueError("The trace database is not finalized.")
        if int(metadata.get("expected_budget", "-1")) != int(expected_budget):
            raise ValueError("The trace database binds another budget.")

        solutions: dict[int, tuple[Solution, str]] = {}
        for row in connection.execute(
            "SELECT solution_ref,solution_sha256,codec,solution_size,payload FROM solutions"
        ):
            solution = _decode_solution(
                str(row["codec"]),
                int(row["solution_size"]),
                bytes(row["payload"]),
            )
            digest = _solution_sha256(solution)
            if digest != row["solution_sha256"]:
                raise ValueError("A packed solution fails its SHA-256 binding.")
            problem.validate_solution(solution)
            solutions[int(row["solution_ref"])] = (solution, digest)

        evaluations = list(
            connection.execute("SELECT * FROM evaluations ORDER BY evaluation_index")
        )
        decisions = {
            int(row["evaluation_index"]): row
            for row in connection.execute("SELECT * FROM decisions ORDER BY evaluation_index")
        }
        if len(evaluations) != expected_budget or len(decisions) != expected_budget:
            raise ValueError("Evaluation/decision counts do not match the budget.")

        archive = ParetoArchive(max_size=None, tol=0.0)
        previous_record_hash = "0" * 64
        previous_decision_hash = "0" * 64
        first_by_solution_sha: dict[str, int] = {}
        replayed_objectives: dict[str, tuple[float, ...]] = {}
        for expected_index, row in enumerate(evaluations, start=1):
            index = int(row["evaluation_index"])
            if index != expected_index:
                raise ValueError("Evaluation indices are not contiguous.")
            proposal_ref = int(row["proposal_solution_ref"])
            if proposal_ref not in solutions:
                raise ValueError("An evaluation references an unknown solution.")
            proposal, proposal_sha = solutions[proposal_ref]
            if proposal_sha != row["proposal_sha256"]:
                raise ValueError("Evaluation and solution dictionary hashes disagree.")
            objectives = tuple(float(value) for value in json.loads(row["objectives_json"]))
            cached = replayed_objectives.get(proposal_sha)
            if cached is None:
                replayed = tuple(float(value) for value in problem.evaluate(proposal))
                replayed_objectives[proposal_sha] = replayed
            else:
                replayed = cached
            if replayed != objectives:
                raise ValueError(f"Objective replay failed at evaluation {index}.")
            duplicate_of = first_by_solution_sha.get(proposal_sha)
            recorded_duplicate = row["duplicate_of_evaluation_index"]
            if duplicate_of != recorded_duplicate:
                raise ValueError("duplicate_of_evaluation_index is inconsistent.")
            first_by_solution_sha.setdefault(proposal_sha, index)

            parent_refs = tuple(json.loads(row["parent_solution_refs_json"]))
            if any(int(ref) not in solutions for ref in parent_refs):
                raise ValueError("An evaluation references an unknown parent solution.")
            parent_type_ids = tuple(json.loads(row["parent_type_ids_json"]))
            witness = (
                None
                if row["operator_witness_json"] is None
                else json.loads(row["operator_witness_json"])
            )
            semantic = {
                "evaluation_index": index,
                "evidence_partition": row["evidence_partition"],
                "search_phase_id": row["search_phase_id"],
                "stage_id": row["stage_id"],
                "type_id": row["type_id"],
                "operator_id": row["operator_id"],
                "operator_call_id": int(row["operator_call_id"]),
                "parent_solution_refs": parent_refs,
                "parent_type_ids": parent_type_ids,
                "proposal_solution_ref": proposal_ref,
                "proposal_sha256": proposal_sha,
                "objectives": objectives,
                "objective_evaluation_kind": row["objective_evaluation_kind"],
                "feasible_before_repair": _nullable_bool(
                    row["feasible_before_repair"]
                ),
                "repair_applied": bool(row["repair_applied"]),
                "repair_operator_id": row["repair_operator_id"],
                "duplicate_of_evaluation_index": recorded_duplicate,
                "local_search_block_id": row["local_search_block_id"],
                "local_search_depth": int(row["local_search_depth"]),
                "operator_witness": witness,
                "prev_record_sha256": previous_record_hash,
            }
            record_hash = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
            if (
                row["prev_record_sha256"] != previous_record_hash
                or row["record_sha256"] != record_hash
            ):
                raise ValueError("Evaluation semantic hash chain failed.")
            previous_record_hash = record_hash

            changed = archive.update((ArchiveEntry(proposal, objectives),))
            retained = archive.contains(ArchiveEntry(proposal, objectives))
            decision = decisions[index]
            if bool(decision["archive_changed"]) != changed:
                raise ValueError("archive_changed does not replay.")
            if bool(decision["retained_after_update"]) != retained:
                raise ValueError("retained_after_update does not replay.")
            if int(decision["archive_size_after"]) != len(archive):
                raise ValueError("archive_size_after does not replay.")
            decision_semantic = {
                "evaluation_index": index,
                "accepted_into_population": bool(
                    decision["accepted_into_population"]
                ),
                "population_replacement_count": int(
                    decision["population_replacement_count"]
                ),
                "population_target_type_ids": tuple(
                    json.loads(decision["population_target_type_ids_json"])
                ),
                "decision_reason": decision["decision_reason"],
                "archive_changed": changed,
                "retained_after_update": retained,
                "archive_size_after": len(archive),
                "scalarization_id": decision["scalarization_id"],
                "scalar_parent": decision["scalar_parent"],
                "scalar_candidate": decision["scalar_candidate"],
                "scalar_advantage": decision["scalar_advantage"],
                "cell_id": decision["cell_id"],
                "new_evaluated_cell": _nullable_bool(
                    decision["new_evaluated_cell"]
                ),
                "new_nondominated_cell": _nullable_bool(
                    decision["new_nondominated_cell"]
                ),
                "prev_decision_sha256": previous_decision_hash,
            }
            decision_hash = hashlib.sha256(
                _canonical_bytes(decision_semantic)
            ).hexdigest()
            if (
                decision["prev_decision_sha256"] != previous_decision_hash
                or decision["decision_sha256"] != decision_hash
            ):
                raise ValueError(
                    f"Decision semantic hash chain failed at evaluation {index}."
                )
            previous_decision_hash = decision_hash

        previous_mechanism_hash = "0" * 64
        previous_after = -1
        mechanism_count = 0
        for expected_event, row in enumerate(
            connection.execute("SELECT * FROM mechanisms ORDER BY event_index"),
            start=1,
        ):
            mechanism_count += 1
            if int(row["event_index"]) != expected_event:
                raise ValueError("Mechanism event indices are not contiguous.")
            after = int(row["after_evaluation_index"])
            if after < previous_after or not 0 <= after <= expected_budget:
                raise ValueError("Mechanism event prefixes are not monotone.")
            payload = json.loads(row["payload_json"])
            semantic = {
                "event_index": expected_event,
                "after_evaluation_index": after,
                "event_kind": row["event_kind"],
                "payload": payload,
                "prev_event_sha256": previous_mechanism_hash,
            }
            event_hash = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
            if row["event_sha256"] != event_hash:
                raise ValueError("Mechanism semantic hash chain failed.")
            previous_mechanism_hash = event_hash
            previous_after = after

        if metadata.get("terminal_evaluation_chain_sha256") != previous_record_hash:
            raise ValueError("Terminal evaluation chain binding failed.")
        if metadata.get("terminal_decision_chain_sha256") != previous_decision_hash:
            raise ValueError("Terminal decision chain binding failed.")
        if metadata.get("terminal_mechanism_chain_sha256") != previous_mechanism_hash:
            raise ValueError("Terminal mechanism chain binding failed.")
        if expected_archive is not None and archive.entries != expected_archive.entries:
            raise ValueError("Final all-evaluated archive reconstruction failed.")

        archive_payload = [
            {
                "solution_sha256": _solution_sha256(entry.tour),
                "objectives": entry.objectives,
            }
            for entry in archive.entries
        ]
        return {
            "schema": "v21_trace_independent_replay_receipt_v1",
            "status": "PASS",
            "database_path": str(path),
            "database_bytes": len(artifact_bytes),
            "database_sha256": artifact_sha,
            "evaluation_records": len(evaluations),
            "decision_records": len(decisions),
            "mechanism_records": mechanism_count,
            "unique_solution_replays": len(replayed_objectives),
            "archive_reconstruction": "PASS",
            "archive_size": len(archive),
            "archive_semantic_sha256": hashlib.sha256(
                _canonical_bytes(archive_payload)
            ).hexdigest(),
            "terminal_evaluation_chain_sha256": previous_record_hash,
            "terminal_decision_chain_sha256": previous_decision_hash,
            "terminal_mechanism_chain_sha256": previous_mechanism_hash,
        }
    finally:
        connection.close()


__all__ = ["verify_v21_trace_database"]
