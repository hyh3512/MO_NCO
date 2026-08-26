from __future__ import annotations

"""Trace storage and replay microbenchmark for the V21 scale gate."""

import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

from .pareto_ijoc_problem import MultiObjectiveCombinatorialProblem
from .pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
)
from .pareto_v21_hybrid import V21HybridConfig, V21TypedHybridParetoSearch
from .pareto_v21_trace_verify import verify_v21_trace_database


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _process_peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value * (1024 if sys.platform != "darwin" else 1)
        except (ImportError, OSError, ValueError):
            return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def run_v21_trace_microbenchmark(
    *,
    problem: MultiObjectiveCombinatorialProblem,
    trace_path: str | Path,
    receipt_path: str | Path,
    evaluation_budget: int,
    checkpoint_period: int,
    seed: int,
    candidate_id: str,
    reference_directions: Sequence[Sequence[float]],
) -> dict[str, object]:
    trace_file = Path(trace_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    if trace_file.exists():
        raise FileExistsError(trace_file)
    if receipt_file.exists():
        raise FileExistsError(receipt_file)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    directions = tuple(
        tuple(float(value) for value in direction)
        for direction in reference_directions
    )

    memory_started = time.perf_counter()
    memory_run = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(
            candidate_id=candidate_id,
            reference_directions=directions,
            evaluations=int(evaluation_budget),
            checkpoint_period=int(checkpoint_period),
            seed=int(seed),
            phase="development",
            trace_database=None,
            capture_trace=False,
        ),
    ).run()
    memory_wall = time.perf_counter() - memory_started

    persistent_started = time.perf_counter()
    persistent_run = V21TypedHybridParetoSearch(
        problem,
        V21HybridConfig(
            candidate_id=candidate_id,
            reference_directions=directions,
            evaluations=int(evaluation_budget),
            checkpoint_period=int(checkpoint_period),
            seed=int(seed),
            phase="development",
            trace_database=str(trace_file),
            capture_trace=False,
        ),
    ).run()
    persistent_wall = time.perf_counter() - persistent_started

    verifier_started = time.perf_counter()
    verification = verify_v21_trace_database(
        trace_file,
        problem,
        expected_budget=int(evaluation_budget),
        expected_archive=persistent_run.optimization_result.archive,
    )
    verifier_wall = time.perf_counter() - verifier_started
    memory_metadata = memory_run.optimization_result.metadata
    persistent_metadata = persistent_run.optimization_result.metadata
    chain_keys = (
        "terminal_evaluation_chain_sha256",
        "terminal_decision_chain_sha256",
        "terminal_mechanism_chain_sha256",
    )
    semantic_match = all(
        memory_metadata[key] == persistent_metadata[key] for key in chain_keys
    )
    archive_match = (
        memory_run.optimization_result.archive.entries
        == persistent_run.optimization_result.archive.entries
    )
    trace_bytes = trace_file.stat().st_size
    receipt = {
        "schema": "pareto_v21_trace_microbenchmark_receipt_v1",
        "status": "PASS" if semantic_match and archive_match else "FAIL",
        "evidence_scope": "engineering_scale_gate_not_quality_evidence",
        "problem": problem.name,
        "family": (
            "MOKP"
            if isinstance(problem, MultiObjectiveKnapsackInstance)
            else (
                "MOTSP"
                if isinstance(problem, MultiObjectiveTSPProblemAdapter)
                else "GENERIC"
            )
        ),
        "candidate_id": candidate_id,
        "seed": int(seed),
        "evaluation_budget": int(evaluation_budget),
        "checkpoint_period": int(checkpoint_period),
        "reference_directions": directions,
        "memory_wall_seconds": memory_wall,
        "persistent_wall_seconds": persistent_wall,
        "persistent_vs_memory_wall_ratio": persistent_wall / memory_wall,
        "independent_verifier_wall_seconds": verifier_wall,
        "trace_database_bytes": trace_bytes,
        "trace_bytes_per_evaluation": trace_bytes / float(evaluation_budget),
        "projected_trace_bytes_for_192m_evaluations": (
            trace_bytes / float(evaluation_budget) * 192_000_000
        ),
        "process_lifetime_peak_working_set_bytes": (
            _process_peak_working_set_bytes()
        ),
        "peak_rss_scope": (
            "process_lifetime_peak; run in a fresh process for run-local interpretation"
        ),
        "trace_verification_status": verification["status"],
        "trace_database_sha256": verification["database_sha256"],
        "unique_solution_replays": verification["unique_solution_replays"],
        "semantic_reproducibility_gate": "PASS" if semantic_match else "FAIL",
        "archive_reproducibility_gate": "PASS" if archive_match else "FAIL",
        "semantic_chain_sha256": {
            key: persistent_metadata[key] for key in chain_keys
        },
    }
    if receipt["status"] != "PASS":
        raise RuntimeError(f"V21 trace microbenchmark gate failed: {receipt}")
    receipt_file.write_bytes(_canonical_bytes(receipt))
    receipt["receipt_sha256"] = hashlib.sha256(
        receipt_file.read_bytes()
    ).hexdigest()
    return receipt


__all__ = ["run_v21_trace_microbenchmark"]
