from __future__ import annotations

"""Run and audit the complete repository pytest suite for V21e3r1."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = REPO_ROOT / "ijoc_submission_v21e3r1"
DEFAULT_LOG = (
    SUBMISSION_ROOT / "provenance" / "V21E3R1_FULL_PYTEST_Q_V4.log"
)
DEFAULT_RECEIPT = (
    SUBMISSION_ROOT
    / "provenance"
    / "V21E3R1_FULL_PYTEST_RECEIPT_V4.json"
)
_SUMMARY_TAIL_BYTES = 256 * 1024
_HEX = frozenset("0123456789abcdef")


_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<label>passed|failed|errors?)\b")
_DURATION_RE = re.compile(
    r"\bin\s+(?P<seconds>\d+(?:\.\d+)?)s"
    r"(?:\s+\([^\r\n)]+\))?(?:\s*=+)?\s*$"
)


def parse_pytest_summary(output: str) -> dict[str, object]:
    """Parse pass/failure/error counts from pytest's terminal summary."""

    for raw_line in reversed(output.splitlines()):
        line = raw_line.strip()
        duration_match = _DURATION_RE.search(line)
        if duration_match is None:
            continue
        counts = {"passed": 0, "failed": 0, "errors": 0}
        found = False
        for match in _COUNT_RE.finditer(line[: duration_match.start()]):
            label = match.group("label")
            key = "errors" if label in {"error", "errors"} else label
            counts[key] = int(match.group("count"))
            found = True
        if found:
            return {
                "summary_parsed": True,
                **counts,
                "pytest_reported_duration_seconds": float(
                    duration_match.group("seconds")
                ),
            }
    return {
        "summary_parsed": False,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "pytest_reported_duration_seconds": None,
    }


def classify_pytest_result(
    *, exit_code: int, summary_parsed: bool, passed: int
) -> str:
    """Return PASS only for a successful run with a positive parsed pass count."""

    return (
        "PASS"
        if exit_code == 0 and summary_parsed is True and passed > 0
        else "FAIL"
    )


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, *, root: Path, label: str) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the repository root.") from error


def _write_exclusive_canonical(path: Path, payload: object) -> None:
    raw = _canonical_bytes(payload)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_log_tail(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - _SUMMARY_TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace")


def _stream_full_pytest(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    display_stream: TextIO,
) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("xb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for chunk in iter(process.stdout.readline, ""):
            raw = chunk.encode("utf-8")
            log_handle.write(raw)
            log_handle.flush()
            os.fsync(log_handle.fileno())
            display_stream.write(chunk)
            display_stream.flush()
        process.stdout.close()
        exit_code = process.wait()
        log_handle.flush()
        os.fsync(log_handle.fileno())
    return exit_code, time.perf_counter() - started


def run_full_pytest(
    *,
    repo_root: str | Path = REPO_ROOT,
    log_path: str | Path = DEFAULT_LOG,
    receipt_path: str | Path = DEFAULT_RECEIPT,
    prospective_source_root_sha256: str,
    display_stream: TextIO | None = None,
) -> dict[str, object]:
    """Run ``python -m pytest -q`` and exclusively commit its audit receipt."""

    root = Path(repo_root).resolve()
    source_root = str(prospective_source_root_sha256)
    if len(source_root) != 64 or any(character not in _HEX for character in source_root):
        raise ValueError("prospective_source_root_sha256 must be lowercase SHA-256.")
    if not root.is_dir():
        raise NotADirectoryError(root)
    log = Path(log_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    relative_log = _repo_relative(log, root=root, label="pytest log")
    _repo_relative(receipt_file, root=root, label="pytest receipt")
    if log == receipt_file:
        raise ValueError("The pytest log and receipt must be different files.")
    if log.exists():
        raise FileExistsError(f"Refusing to replace pytest log: {log}")
    if receipt_file.exists():
        raise FileExistsError(f"Refusing to replace pytest receipt: {receipt_file}")
    log.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)

    # Preserve a virtual environment's launcher path; resolving a POSIX venv
    # symlink could silently switch the child away from the current runtime.
    executable = Path(os.path.abspath(sys.executable))
    command = [str(executable), "-m", "pytest", "-q"]
    exit_code, duration_seconds = _stream_full_pytest(
        command=command,
        cwd=root,
        log_path=log,
        display_stream=sys.stdout if display_stream is None else display_stream,
    )
    summary = parse_pytest_summary(_read_log_tail(log))
    status = classify_pytest_result(
        exit_code=exit_code,
        summary_parsed=bool(summary["summary_parsed"]),
        passed=int(summary["passed"]),
    )
    receipt: dict[str, object] = {
        "schema": "pareto_v21e3r1_full_pytest_receipt_v1",
        "status": status,
        "suite_scope": "repository_full_pytest_q_v1",
        "prospective_source_root_sha256": source_root,
        "scientific_scope": (
            "repository_software_regression_evidence_only_not_formal_evidence"
        ),
        "command": command,
        "cwd": ".",
        "cwd_path_semantics": "repo_root_self_v1",
        "runtime": {
            "implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "executable": str(executable),
        "executable_sha256": _sha256_file(executable),
        "artifact_path_semantics": "repo_root_relative_posix_v1",
        "log_path": relative_log,
        "log_sha256": _sha256_file(log),
        "log_bytes": log.stat().st_size,
        "output_capture": "merged_stdout_stderr_streamed_to_fsynced_log_v1",
        "exit_code": exit_code,
        **summary,
        "duration_seconds": duration_seconds,
        "selection_authorization": "PROHIBITED",
        "formal_authorized": False,
    }
    _write_exclusive_canonical(receipt_file, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete repository pytest -q suite and write an exclusive "
            "V21e3r1 audit receipt."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--prospective-source-root-sha256", required=True)
    args = parser.parse_args(argv)
    result = run_full_pytest(
        repo_root=args.repo_root,
        log_path=args.log,
        receipt_path=args.receipt,
        prospective_source_root_sha256=args.prospective_source_root_sha256,
    )
    print(
        json.dumps(
            {
                "receipt": str(args.receipt.resolve()),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    if result["status"] == "PASS":
        return 0
    exit_code = int(result["exit_code"])
    return exit_code if 1 <= exit_code <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
