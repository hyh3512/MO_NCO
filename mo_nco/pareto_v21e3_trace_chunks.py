from __future__ import annotations

"""Chunked compressed-trace engineering prototype for V21e3.

The format is deliberately marked non-formal.  It supplies a deterministic
round-trip, CRC, per-chunk SHA-256, and a previous-chunk hash chain so that the
restore/replay seam can be benchmarked before any future formal case exists.
"""

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile
from typing import Callable, Iterable, Mapping
import zlib


_MAGIC = b"V21E3R1-TRACE-CHUNK-V2\n"
_ZERO_HASH = "0" * 64
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_CHUNK_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_RECORD_BYTES = 64 * 1024 * 1024
_MAX_CHUNK_FILE_BYTES = 300 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024
_MAX_CHUNK_RECORDS = 100_000
_MAX_CHUNK_COUNT = 100_000


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


def _required_positive_int(
    payload: Mapping[str, object],
    key: str,
    *,
    maximum: int | None = None,
) -> int:
    value = payload.get(key)
    if type(value) is not int or value <= 0 or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{key} must be a bounded positive integer.")
    return value


def _required_hex(payload: Mapping[str, object], key: str, length: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is None:
        raise ValueError(f"{key} is not a canonical lowercase hex digest.")
    return value


def _atomic_commit(path: Path, raw: bytes) -> None:
    """Durably flush bytes before atomically publishing the final filename."""

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


def write_trace_chunk_archive(
    records: Iterable[Mapping[str, object]],
    output_directory: str | Path,
    *,
    chunk_records: int = 4096,
    max_chunk_uncompressed_bytes: int = 32 * 1024 * 1024,
    max_record_bytes: int = 8 * 1024 * 1024,
    failure_injector: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    """Stream records into bounded, fsynced chunks and commit manifest last."""

    limits = {
        "chunk_records": chunk_records,
        "max_chunk_uncompressed_bytes": max_chunk_uncompressed_bytes,
        "max_record_bytes": max_record_bytes,
    }
    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        chunk_records > _MAX_CHUNK_RECORDS
        or max_chunk_uncompressed_bytes > _MAX_CHUNK_UNCOMPRESSED_BYTES
        or max_record_bytes > _MAX_RECORD_BYTES
    ):
        raise ValueError("Requested bounded-memory limits exceed hard safety caps.")
    if max_record_bytes > max_chunk_uncompressed_bytes:
        raise ValueError(
            "max_record_bytes cannot exceed max_chunk_uncompressed_bytes."
        )

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    chunks: list[dict[str, object]] = []
    previous_hash = _ZERO_HASH
    total_uncompressed = 0
    total_compressed = 0
    record_count = 0
    buffered_records: list[bytes] = []
    buffered_bytes = 0

    def commit_buffer() -> None:
        nonlocal previous_hash
        nonlocal total_uncompressed
        nonlocal total_compressed
        nonlocal record_count
        nonlocal buffered_records
        nonlocal buffered_bytes
        if not buffered_records:
            return
        if len(chunks) >= _MAX_CHUNK_COUNT:
            raise ValueError("Trace archive exceeds the hard chunk-count limit.")
        chunk_index = len(chunks) + 1
        payload = b"".join(buffered_records)
        compressed = zlib.compress(payload, level=9)
        header = {
            "schema": "pareto_v21e3r1_trace_chunk_header_v2",
            "chunk_index": chunk_index,
            "first_record_index": record_count + 1,
            "record_count": len(buffered_records),
            "uncompressed_bytes": len(payload),
            "compressed_bytes": len(compressed),
            "uncompressed_crc32_hex": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
            "uncompressed_sha256": _sha256(payload),
            "compressed_sha256": _sha256(compressed),
            "previous_chunk_sha256": previous_hash,
        }
        header_raw = _canonical(header)
        chunk_raw = _MAGIC + struct.pack(">I", len(header_raw)) + header_raw + compressed
        chunk_hash = _sha256(chunk_raw)
        filename = f"chunk-{chunk_index:06d}.bin"
        _atomic_commit(output / filename, chunk_raw)
        chunks.append(
            {
                "path": filename,
                "sha256": chunk_hash,
                "bytes": len(chunk_raw),
                **header,
            }
        )
        previous_hash = chunk_hash
        total_uncompressed += len(payload)
        total_compressed += len(compressed)
        record_count += len(buffered_records)
        buffered_records = []
        buffered_bytes = 0
        if failure_injector is not None:
            failure_injector("after_chunk_commit", chunk_index)

    for record in records:
        canonical_record = _canonical(dict(record))
        if len(canonical_record) > max_record_bytes:
            raise ValueError("Trace record exceeds max_record_bytes.")
        if buffered_records and (
            len(buffered_records) >= chunk_records
            or buffered_bytes + len(canonical_record)
            > max_chunk_uncompressed_bytes
        ):
            commit_buffer()
        buffered_records.append(canonical_record)
        buffered_bytes += len(canonical_record)
        if (
            len(buffered_records) >= chunk_records
            or buffered_bytes >= max_chunk_uncompressed_bytes
        ):
            commit_buffer()
    commit_buffer()
    if record_count == 0:
        raise ValueError("At least one trace record is required.")

    manifest = {
        "schema": "pareto_v21e3r1_trace_chunk_manifest_v2",
        "status": "PROTOTYPE_PASS_FORMAL_NOT_AUTHORIZED",
        "evidence_scope": "engineering_restore_replay_prototype_not_formal_evidence",
        "codec": "canonical_jsonl_zlib_bounded_chunk_v2",
        "chunk_hash_chain": "sha256_header_and_compressed_payload_v2",
        "record_count": record_count,
        "chunk_count": len(chunks),
        "chunk_records": chunk_records,
        "max_chunk_uncompressed_bytes": max_chunk_uncompressed_bytes,
        "max_record_bytes": max_record_bytes,
        "hard_limits": {
            "max_manifest_bytes": _MAX_MANIFEST_BYTES,
            "max_chunk_count": _MAX_CHUNK_COUNT,
            "max_chunk_records": _MAX_CHUNK_RECORDS,
            "max_chunk_file_bytes": _MAX_CHUNK_FILE_BYTES,
            "max_chunk_uncompressed_bytes": _MAX_CHUNK_UNCOMPRESSED_BYTES,
            "max_record_bytes": _MAX_RECORD_BYTES,
        },
        "writer_mode": "single_pass_bounded_iterator_v1",
        "reader_mode": "full_archive_validated_disk_spooled_iterator_v3",
        "commit_protocol": "file_fsync_then_atomic_replace_manifest_last_v1",
        "incomplete_archive_policy": "manifest_absent_or_invalid_fail_closed",
        "power_loss_directory_entry_durability": (
            "NOT_ESTABLISHED_REQUIRES_TARGET_FILESYSTEM_QUALIFICATION"
        ),
        "uncompressed_payload_bytes": total_uncompressed,
        "compressed_payload_bytes": total_compressed,
        "terminal_chunk_sha256": previous_hash,
        "formal_authorized": False,
        "formal_cases_status": "NOT_MATERIALIZED",
        "chunks": chunks,
    }
    manifest_raw = _canonical(manifest)
    if len(manifest_raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("Trace manifest exceeds the bounded-memory limit.")
    manifest_path = output / "manifest.json"
    if failure_injector is not None:
        failure_injector("before_manifest_commit", len(chunks))
    _atomic_commit(manifest_path, manifest_raw)
    restored_count = sum(1 for _ in iter_trace_chunk_archive(manifest_path))
    if restored_count != record_count:
        raise RuntimeError("Immediate trace chunk restore gate failed.")
    return {
        "schema": "pareto_v21e3r1_trace_chunk_export_receipt_v2",
        "status": manifest["status"],
        "evidence_scope": manifest["evidence_scope"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_raw),
        "record_count": record_count,
        "chunk_count": len(chunks),
        "uncompressed_payload_bytes": total_uncompressed,
        "compressed_payload_bytes": total_compressed,
        "compression_ratio": total_compressed / float(total_uncompressed),
        "restore_roundtrip_gate": "PASS",
        "bounded_memory_gate": "PASS",
        "writer_mode": manifest["writer_mode"],
        "reader_mode": manifest["reader_mode"],
        "commit_protocol": manifest["commit_protocol"],
        "power_loss_directory_entry_durability": manifest[
            "power_loss_directory_entry_durability"
        ],
        "max_chunk_uncompressed_bytes": max_chunk_uncompressed_bytes,
        "max_record_bytes": max_record_bytes,
        "hard_limits": manifest["hard_limits"],
        "terminal_chunk_sha256": previous_hash,
        "formal_authorized": False,
        "formal_cases_status": "NOT_MATERIALIZED",
    }


def _iter_trace_chunk_archive_once(
    manifest_path: str | Path,
) -> Iterable[dict[str, object]]:
    """Perform one bounded verification-and-decode pass over the archive."""

    supplied_path = Path(manifest_path)
    if supplied_path.is_symlink():
        raise ValueError("Trace manifest cannot be a symlink.")
    path = supplied_path.resolve()
    if path.name != "manifest.json":
        raise ValueError("Trace manifest must use the canonical filename.")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Trace manifest exceeds the bounded-memory limit.")
    manifest_raw = path.read_bytes()
    if len(manifest_raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("Trace manifest changed beyond the bounded-memory limit.")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict) or _canonical(manifest) != manifest_raw:
        raise ValueError("Trace manifest is not canonical JSON.")
    if not (
        manifest.get("schema") == "pareto_v21e3r1_trace_chunk_manifest_v2"
        and manifest.get("status") == "PROTOTYPE_PASS_FORMAL_NOT_AUTHORIZED"
        and manifest.get("formal_authorized") is False
        and manifest.get("formal_cases_status") == "NOT_MATERIALIZED"
        and manifest.get("codec") == "canonical_jsonl_zlib_bounded_chunk_v2"
        and manifest.get("chunk_hash_chain")
        == "sha256_header_and_compressed_payload_v2"
        and manifest.get("writer_mode") == "single_pass_bounded_iterator_v1"
        and manifest.get("reader_mode")
        == "full_archive_validated_disk_spooled_iterator_v3"
        and manifest.get("commit_protocol")
        == "file_fsync_then_atomic_replace_manifest_last_v1"
        and manifest.get("incomplete_archive_policy")
        == "manifest_absent_or_invalid_fail_closed"
        and manifest.get("power_loss_directory_entry_durability")
        == "NOT_ESTABLISHED_REQUIRES_TARGET_FILESYSTEM_QUALIFICATION"
    ):
        raise ValueError("Trace manifest is not the frozen non-formal prototype.")
    chunk_count = _required_positive_int(
        manifest,
        "chunk_count",
        maximum=_MAX_CHUNK_COUNT,
    )
    expected_record_count = _required_positive_int(manifest, "record_count")
    expected_total_uncompressed = _required_positive_int(
        manifest,
        "uncompressed_payload_bytes",
    )
    expected_total_compressed = _required_positive_int(
        manifest,
        "compressed_payload_bytes",
    )
    terminal_chunk_sha256 = _required_hex(
        manifest,
        "terminal_chunk_sha256",
        64,
    )
    chunks = manifest.get("chunks")
    if (
        not isinstance(chunks, list)
        or len(chunks) != chunk_count
    ):
        raise ValueError("Trace manifest chunk count is inconsistent.")
    expected_files = {"manifest.json"}
    for entry in chunks:
        if not isinstance(entry, dict):
            raise ValueError("Trace manifest chunk entry is invalid.")
        expected_files.add(str(entry.get("path")))
    remaining_files = set(expected_files)
    observed_count = 0
    for child in path.parent.iterdir():
        observed_count += 1
        if observed_count > len(expected_files) or child.name not in remaining_files:
            raise ValueError("Trace archive contains missing or unexpected files.")
        remaining_files.remove(child.name)
    if remaining_files:
        raise ValueError("Trace archive contains missing or unexpected files.")
    chunk_records = _required_positive_int(
        manifest,
        "chunk_records",
        maximum=_MAX_CHUNK_RECORDS,
    )
    max_chunk_uncompressed_bytes = _required_positive_int(
        manifest,
        "max_chunk_uncompressed_bytes",
        maximum=_MAX_CHUNK_UNCOMPRESSED_BYTES,
    )
    max_record_bytes = _required_positive_int(
        manifest,
        "max_record_bytes",
        maximum=_MAX_RECORD_BYTES,
    )
    if not (
        max_record_bytes <= max_chunk_uncompressed_bytes
    ):
        raise ValueError("Trace manifest bounded-memory limits are invalid.")
    expected_hard_limits = {
        "max_manifest_bytes": _MAX_MANIFEST_BYTES,
        "max_chunk_count": _MAX_CHUNK_COUNT,
        "max_chunk_records": _MAX_CHUNK_RECORDS,
        "max_chunk_file_bytes": _MAX_CHUNK_FILE_BYTES,
        "max_chunk_uncompressed_bytes": _MAX_CHUNK_UNCOMPRESSED_BYTES,
        "max_record_bytes": _MAX_RECORD_BYTES,
    }
    if _canonical(manifest.get("hard_limits")) != _canonical(
        expected_hard_limits
    ):
        raise ValueError("Trace manifest hard limits are not the frozen policy.")
    previous_hash = _ZERO_HASH
    record_count = 0
    total_uncompressed = 0
    total_compressed = 0
    for expected_index, entry in enumerate(chunks, start=1):
        entry_path = entry.get("path")
        if not isinstance(entry_path, str):
            raise ValueError("Trace chunk path is not a string.")
        relative = PurePosixPath(entry_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != f"chunk-{expected_index:06d}.bin"
        ):
            raise ValueError("Trace chunk path is not a safe canonical name.")
        chunk_path = path.parent / relative.name
        if chunk_path.is_symlink():
            raise ValueError("Trace chunk cannot be a symlink.")
        expected_chunk_bytes = _required_positive_int(
            entry,
            "bytes",
            maximum=_MAX_CHUNK_FILE_BYTES,
        )
        entry_sha256 = _required_hex(entry, "sha256", 64)
        if chunk_path.stat().st_size != expected_chunk_bytes:
            raise ValueError("Trace chunk byte count mismatch.")
        chunk_raw = chunk_path.read_bytes()
        if _sha256(chunk_raw) != entry_sha256:
            raise ValueError("Trace chunk SHA-256 mismatch.")
        if len(chunk_raw) != expected_chunk_bytes:
            raise ValueError("Trace chunk byte count mismatch.")
        if not chunk_raw.startswith(_MAGIC) or len(chunk_raw) < len(_MAGIC) + 4:
            raise ValueError("Trace chunk magic is invalid.")
        header_size = struct.unpack(
            ">I", chunk_raw[len(_MAGIC) : len(_MAGIC) + 4]
        )[0]
        if not (0 < header_size <= _MAX_HEADER_BYTES):
            raise ValueError("Trace chunk header exceeds its bounded-memory limit.")
        header_start = len(_MAGIC) + 4
        header_end = header_start + header_size
        if header_end >= len(chunk_raw):
            raise ValueError("Trace chunk header is truncated.")
        header_raw = chunk_raw[header_start:header_end]
        compressed = chunk_raw[header_end:]
        header = json.loads(header_raw)
        if not isinstance(header, dict) or _canonical(header) != header_raw:
            raise ValueError("Trace chunk header is not canonical JSON.")
        if header != {
            key: entry[key]
            for key in (
                "schema",
                "chunk_index",
                "first_record_index",
                "record_count",
                "uncompressed_bytes",
                "compressed_bytes",
                "uncompressed_crc32_hex",
                "uncompressed_sha256",
                "compressed_sha256",
                "previous_chunk_sha256",
            )
        }:
            raise ValueError("Trace chunk header and manifest disagree.")
        header_chunk_index = _required_positive_int(
            header,
            "chunk_index",
            maximum=_MAX_CHUNK_COUNT,
        )
        header_first_record_index = _required_positive_int(
            header,
            "first_record_index",
        )
        header_record_count = _required_positive_int(
            header,
            "record_count",
            maximum=chunk_records,
        )
        header_uncompressed_bytes = _required_positive_int(
            header,
            "uncompressed_bytes",
            maximum=max_chunk_uncompressed_bytes,
        )
        header_compressed_bytes = _required_positive_int(
            header,
            "compressed_bytes",
            maximum=_MAX_CHUNK_FILE_BYTES,
        )
        if header.get("schema") != "pareto_v21e3r1_trace_chunk_header_v2":
            raise ValueError("Trace chunk header schema is not frozen.")
        header_compressed_sha256 = _required_hex(
            header,
            "compressed_sha256",
            64,
        )
        _required_hex(header, "uncompressed_sha256", 64)
        _required_hex(header, "uncompressed_crc32_hex", 8)
        _required_hex(header, "previous_chunk_sha256", 64)
        if not (
            header_chunk_index == expected_index
            and header_first_record_index == record_count + 1
            and header["previous_chunk_sha256"] == previous_hash
            and _sha256(compressed) == header_compressed_sha256
            and len(compressed) == header_compressed_bytes
        ):
            raise ValueError("Trace chunk compressed/hash-chain binding failed.")
        expected_uncompressed = header_uncompressed_bytes
        try:
            decompressor = zlib.decompressobj()
            payload = decompressor.decompress(
                compressed,
                expected_uncompressed + 1,
            )
            if (
                len(payload) > expected_uncompressed
                or not decompressor.eof
                or decompressor.unconsumed_tail
                or decompressor.unused_data
            ):
                raise ValueError(
                    "Trace chunk decompression exceeded or violated its bound."
                )
            payload += decompressor.flush()
        except zlib.error as exc:
            raise ValueError("Trace chunk decompression failed.") from exc
        if not (
            len(payload) == header_uncompressed_bytes
            and _sha256(payload) == header["uncompressed_sha256"]
            and f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
            == header["uncompressed_crc32_hex"]
        ):
            raise ValueError("Trace chunk uncompressed integrity gate failed.")
        verified_records: list[dict[str, object]] = []
        for raw_line in io.BytesIO(payload):
            if not raw_line.endswith(b"\n") or len(raw_line) > max_record_bytes:
                raise ValueError("Trace record violates its framed byte bound.")
            line = raw_line[:-1]
            record = json.loads(line)
            if not isinstance(record, dict) or _canonical(record).rstrip(b"\n") != line:
                raise ValueError("Trace record is not canonical JSON.")
            verified_records.append(record)
        if len(verified_records) != header_record_count:
            raise ValueError("Trace chunk record count is inconsistent.")
        previous_hash = entry_sha256
        total_uncompressed += len(payload)
        total_compressed += len(compressed)
        record_count += len(verified_records)
        yield from verified_records
    if not (
        record_count == expected_record_count
        and previous_hash == terminal_chunk_sha256
        and total_uncompressed == expected_total_uncompressed
        and total_compressed == expected_total_compressed
    ):
        raise ValueError("Trace archive terminal bindings are inconsistent.")


def iter_trace_chunk_archive(
    manifest_path: str | Path,
) -> Iterable[dict[str, object]]:
    """Validate the entire archive, then replay it with bounded memory.

    A private disk spool prevents consumers from treating a valid prefix as
    trusted before a later chunk or terminal binding has been checked.  Only
    after the archive pass completes does replay begin; RAM remains chunk-bounded
    and the source directory is never read a second time.
    """

    with tempfile.TemporaryFile(prefix="pareto-v21e3r1-trace-replay-") as spool:
        for record in _iter_trace_chunk_archive_once(manifest_path):
            spool.write(_canonical(record))
        spool.flush()
        spool.seek(0)
        for raw_line in spool:
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise RuntimeError("Private verified replay spool is invalid.")
            yield record


def restore_trace_chunk_archive(
    manifest_path: str | Path,
) -> list[dict[str, object]]:
    """Non-formal compatibility wrapper that explicitly materializes all records.

    Target-scale and future formal callers must use ``iter_trace_chunk_archive``.
    """

    return list(iter_trace_chunk_archive(manifest_path))


__all__ = [
    "iter_trace_chunk_archive",
    "restore_trace_chunk_archive",
    "write_trace_chunk_archive",
]
