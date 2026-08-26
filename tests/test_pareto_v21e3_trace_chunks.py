from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mo_nco.pareto_v21e3_trace_chunks import (
    iter_trace_chunk_archive,
    restore_trace_chunk_archive,
    write_trace_chunk_archive,
)


AUDIT_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21e3"
    / "scripts"
    / "audit_v21e3_trace_streaming.py"
)


def test_trace_chunk_prototype_roundtrips_and_hash_chains(tmp_path: Path) -> None:
    records = [
        {"evaluation_index": index, "solution": [index, index + 1], "value": -index}
        for index in range(1, 6)
    ]
    output = tmp_path / "trace-export"

    receipt = write_trace_chunk_archive(records, output, chunk_records=2)
    restored = restore_trace_chunk_archive(output / "manifest.json")

    assert restored == records
    assert receipt["status"] == "PROTOTYPE_PASS_FORMAL_NOT_AUTHORIZED"
    assert receipt["record_count"] == 5
    assert receipt["chunk_count"] == 3
    assert receipt["formal_authorized"] is False
    assert receipt["compression_ratio"] < 1.0
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["chunks"][0]["previous_chunk_sha256"] == "0" * 64
    assert manifest["chunks"][1]["previous_chunk_sha256"] == manifest["chunks"][0]["sha256"]

    chunk = output / manifest["chunks"][1]["path"]
    raw = bytearray(chunk.read_bytes())
    raw[-1] ^= 1
    chunk.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256"):
        restore_trace_chunk_archive(output / "manifest.json")


def test_trace_chunk_writer_commits_before_exhausting_single_pass_input(
    tmp_path: Path,
) -> None:
    output = tmp_path / "streaming-export"

    def records():
        yield {"evaluation_index": 1, "value": -1}
        yield {"evaluation_index": 2, "value": -2}
        assert (output / "chunk-000001.bin").is_file()
        assert not (output / "manifest.json").exists()
        yield {"evaluation_index": 3, "value": -3}

    receipt = write_trace_chunk_archive(records(), output, chunk_records=2)

    assert receipt["record_count"] == 3


def test_trace_chunk_replay_validates_full_archive_before_first_yield(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lazy-replay"
    records = [{"evaluation_index": index} for index in range(1, 5)]
    write_trace_chunk_archive(records, output, chunk_records=2)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    second_chunk = output / manifest["chunks"][1]["path"]
    raw = bytearray(second_chunk.read_bytes())
    raw[-1] ^= 1
    second_chunk.write_bytes(raw)

    replay = iter_trace_chunk_archive(output / "manifest.json")

    with pytest.raises(ValueError, match="SHA-256"):
        next(replay)


def test_trace_chunk_crash_injection_never_commits_a_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "interrupted-export"

    def crash(event: str, chunk_index: int) -> None:
        if event == "after_chunk_commit" and chunk_index == 1:
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        write_trace_chunk_archive(
            ({"evaluation_index": index} for index in range(1, 5)),
            output,
            chunk_records=2,
            failure_injector=crash,
        )

    assert (output / "chunk-000001.bin").is_file()
    assert not (output / "manifest.json").exists()
    assert list(output.glob("*.tmp")) == []
    with pytest.raises(FileNotFoundError):
        list(iter_trace_chunk_archive(output / "manifest.json"))


def test_trace_chunk_writer_enforces_byte_bound_independent_of_record_count(
    tmp_path: Path,
) -> None:
    output = tmp_path / "byte-bounded-export"
    records = [
        {"evaluation_index": index, "payload": "x" * 32}
        for index in range(1, 8)
    ]

    receipt = write_trace_chunk_archive(
        iter(records),
        output,
        chunk_records=100,
        max_chunk_uncompressed_bytes=128,
        max_record_bytes=96,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["record_count"] == len(records)
    assert receipt["bounded_memory_gate"] == "PASS"
    assert manifest["max_chunk_uncompressed_bytes"] == 128
    assert max(chunk["uncompressed_bytes"] for chunk in manifest["chunks"]) <= 128
    assert manifest["chunk_count"] > 1


def test_trace_chunk_reader_rejects_unbounded_manifest_before_replay(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unbounded-manifest"
    write_trace_chunk_archive(
        ({"evaluation_index": index} for index in range(1, 3)),
        output,
        chunk_records=1,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["max_chunk_uncompressed_bytes"] = 1 << 40
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="bounded positive integer"):
        next(iter_trace_chunk_archive(manifest_path))


def test_trace_chunk_reader_rejects_truncated_committed_chunk(
    tmp_path: Path,
) -> None:
    output = tmp_path / "truncated-chunk"
    write_trace_chunk_archive(
        ({"evaluation_index": index} for index in range(1, 4)),
        output,
        chunk_records=2,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    first_chunk = output / manifest["chunks"][0]["path"]
    raw = first_chunk.read_bytes()
    first_chunk.write_bytes(raw[:-7])

    with pytest.raises(ValueError, match="byte count"):
        next(iter_trace_chunk_archive(output / "manifest.json"))


def test_trace_chunk_reader_rejects_uncommitted_or_extra_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ambiguous-archive"
    write_trace_chunk_archive(
        ({"evaluation_index": index} for index in range(1, 3)),
        output,
        chunk_records=1,
    )
    (output / "chunk-999999.bin.tmp").write_bytes(b"partial")

    with pytest.raises(ValueError, match="unexpected files"):
        next(iter_trace_chunk_archive(output / "manifest.json"))


def test_trace_chunk_reader_rejects_zero_record_archive(tmp_path: Path) -> None:
    output = tmp_path / "zero-record-archive"
    write_trace_chunk_archive([{"evaluation_index": 1}], output, chunk_records=1)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for chunk in output.glob("chunk-*.bin"):
        chunk.unlink()
    manifest.update(
        {
            "chunks": [],
            "chunk_count": 0,
            "record_count": 0,
            "uncompressed_payload_bytes": 0,
            "compressed_payload_bytes": 0,
            "terminal_chunk_sha256": "0" * 64,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="positive integer"):
        next(iter_trace_chunk_archive(manifest_path))


def test_trace_chunk_reader_rejects_numeric_string_coercion(tmp_path: Path) -> None:
    output = tmp_path / "numeric-string-archive"
    write_trace_chunk_archive([{"evaluation_index": 1}], output, chunk_records=1)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = str(manifest["record_count"])
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="positive integer"):
        next(iter_trace_chunk_archive(manifest_path))


def test_streaming_audit_receipt_preserves_nonformal_boundary(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("trace_streaming_audit", AUDIT_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_streaming_audit(
        output_directory=tmp_path / "trace",
        receipt_path=tmp_path / "receipt.json",
        record_count=1_000,
        payload_bytes=32,
        chunk_records=32,
        max_chunk_uncompressed_bytes=8 * 1024,
        max_record_bytes=512,
        max_peak_python_bytes=8 * 1024 * 1024,
    )

    assert result["status"] == "PASS_SMALL_SCALE_STREAMING_ENGINEERING_ONLY"
    assert result["target_scale_capacity_status"] == "NOT_RUN"
    assert result["objective_replay_status"] == "NOT_CONNECTED_IN_THIS_PROBE"
    assert result["formal_authorized"] is False
    assert result["gates"]["writer_single_pass_contract"] is True
    assert result["gates"]["replayed_record_count"] is True
    assert result["durability_scope"]["process_kill_injection"] == "NOT_PERFORMED"
    assert [binding["path"] for binding in result["source_bindings"]] == [
        "mo_nco/pareto_v21e3_trace_chunks.py",
        "ijoc_submission_v21e3/scripts/audit_v21e3_trace_streaming.py",
    ]
