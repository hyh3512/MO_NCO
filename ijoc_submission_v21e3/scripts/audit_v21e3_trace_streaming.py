from __future__ import annotations

"""Emit a bounded-memory engineering receipt for the V21e3 trace primitive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tracemalloc
from typing import Iterable

from mo_nco.pareto_v21e3_trace_chunks import (
    iter_trace_chunk_archive,
    write_trace_chunk_archive,
)


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_new_fsynced(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_streaming_audit(
    *,
    output_directory: str | Path,
    receipt_path: str | Path,
    record_count: int = 50_000,
    payload_bytes: int = 128,
    chunk_records: int = 128,
    max_chunk_uncompressed_bytes: int = 64 * 1024,
    max_record_bytes: int = 1024,
    max_peak_python_bytes: int = 16 * 1024 * 1024,
) -> dict[str, object]:
    """Measure a generated single-pass write and lazy replay at small scale."""

    for name, value in {
        "record_count": record_count,
        "payload_bytes": payload_bytes,
        "chunk_records": chunk_records,
        "max_chunk_uncompressed_bytes": max_chunk_uncompressed_bytes,
        "max_record_bytes": max_record_bytes,
        "max_peak_python_bytes": max_peak_python_bytes,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    output = Path(output_directory).resolve()
    receipt_file = Path(receipt_path).resolve()
    if receipt_file.exists():
        raise FileExistsError(receipt_file)

    def records() -> Iterable[dict[str, object]]:
        for index in range(1, record_count + 1):
            yield {
                "evaluation_index": index,
                "payload": "x" * payload_bytes,
            }

    tracemalloc.start()
    try:
        export = write_trace_chunk_archive(
            records(),
            output,
            chunk_records=chunk_records,
            max_chunk_uncompressed_bytes=max_chunk_uncompressed_bytes,
            max_record_bytes=max_record_bytes,
        )
        replayed_count = sum(
            1 for _ in iter_trace_chunk_archive(output / "manifest.json")
        )
        current_python_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    manifest_raw = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    repo_root = Path(__file__).resolve().parents[2]
    implementation_path = repo_root / "mo_nco" / "pareto_v21e3_trace_chunks.py"
    implementation_raw = implementation_path.read_bytes()
    audit_script_path = Path(__file__).resolve()
    audit_script_raw = audit_script_path.read_bytes()
    max_observed_chunk_bytes = max(
        int(entry["uncompressed_bytes"]) for entry in manifest["chunks"]
    )
    max_observed_chunk_records = max(
        int(entry["record_count"]) for entry in manifest["chunks"]
    )
    gates = {
        "replayed_record_count": replayed_count == record_count,
        "writer_single_pass_contract": (
            export["writer_mode"] == "single_pass_bounded_iterator_v1"
        ),
        "chunk_byte_bound": (
            max_observed_chunk_bytes <= max_chunk_uncompressed_bytes
        ),
        "chunk_record_bound": max_observed_chunk_records <= chunk_records,
        "peak_python_allocator_bound": peak_python_bytes <= max_peak_python_bytes,
        "no_uncommitted_temp_files": not any(output.glob("*.tmp")),
        "final_manifest_present": (output / "manifest.json").is_file(),
    }
    status = (
        "PASS_SMALL_SCALE_STREAMING_ENGINEERING_ONLY"
        if all(gates.values())
        else "FAIL_SMALL_SCALE_STREAMING_ENGINEERING_GATE"
    )
    receipt = {
        "schema": "pareto_v21e3r1_trace_streaming_small_scale_receipt_v2",
        "status": status,
        "scientific_scope": "engineering_memory_probe_not_target_scale_evidence",
        "formal_authorized": False,
        "formal_cases_status": "NOT_MATERIALIZED",
        "target_scale_capacity_status": "NOT_RUN",
        "objective_replay_status": "NOT_CONNECTED_IN_THIS_PROBE",
        "durability_scope": {
            "python_exception_injection": "PASS_IN_TEST_SUITE",
            "process_kill_injection": "NOT_PERFORMED",
            "power_loss_injection": "NOT_PERFORMED",
            "parent_directory_fsync": "NOT_ESTABLISHED_ON_THIS_WINDOWS_PROBE",
        },
        "measurement": "python_tracemalloc_peak_bytes",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "source_bindings": [
            {
                "path": "mo_nco/pareto_v21e3_trace_chunks.py",
                "bytes": len(implementation_raw),
                "sha256": _sha256(implementation_raw),
            },
            {
                "path": "ijoc_submission_v21e3/scripts/"
                "audit_v21e3_trace_streaming.py",
                "bytes": len(audit_script_raw),
                "sha256": _sha256(audit_script_raw),
            },
        ],
        "configuration": {
            "record_count": record_count,
            "payload_bytes": payload_bytes,
            "chunk_records": chunk_records,
            "max_chunk_uncompressed_bytes": max_chunk_uncompressed_bytes,
            "max_record_bytes": max_record_bytes,
            "max_peak_python_bytes": max_peak_python_bytes,
        },
        "observed": {
            "trace_output_directory": str(output),
            "manifest_path": str(output / "manifest.json"),
            "manifest_bytes": len(manifest_raw),
            "replayed_record_count": replayed_count,
            "chunk_count": int(export["chunk_count"]),
            "max_observed_chunk_uncompressed_bytes": max_observed_chunk_bytes,
            "max_observed_chunk_records": max_observed_chunk_records,
            "current_python_bytes": current_python_bytes,
            "peak_python_bytes": peak_python_bytes,
            "manifest_sha256": _sha256(manifest_raw),
            "terminal_chunk_sha256": export["terminal_chunk_sha256"],
        },
        "gates": gates,
    }
    _write_new_fsynced(receipt_file, _canonical(receipt))
    if not all(gates.values()):
        raise RuntimeError(f"Trace streaming audit failed: {gates}")
    result = dict(receipt)
    result["receipt_path"] = str(receipt_file)
    result["receipt_sha256"] = _sha256(receipt_file.read_bytes())
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V21e3 small-scale streaming trace audit."
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--record-count", type=int, default=50_000)
    parser.add_argument("--payload-bytes", type=int, default=128)
    parser.add_argument("--chunk-records", type=int, default=128)
    parser.add_argument(
        "--max-chunk-uncompressed-bytes",
        type=int,
        default=64 * 1024,
    )
    parser.add_argument("--max-record-bytes", type=int, default=1024)
    parser.add_argument(
        "--max-peak-python-bytes",
        type=int,
        default=16 * 1024 * 1024,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_streaming_audit(
        output_directory=args.output_directory,
        receipt_path=args.receipt_path,
        record_count=args.record_count,
        payload_bytes=args.payload_bytes,
        chunk_records=args.chunk_records,
        max_chunk_uncompressed_bytes=args.max_chunk_uncompressed_bytes,
        max_record_bytes=args.max_record_bytes,
        max_peak_python_bytes=args.max_peak_python_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
