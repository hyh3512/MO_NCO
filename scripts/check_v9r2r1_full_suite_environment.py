"""Fail-fast environment preflight for the V9R2R1 full repository suite.

The scoped V9 tests do not require pymoo.  The repository-wide suite does, and
on Windows an installed native extension can be present yet unusable because a
Code Integrity policy rejects the DLL/PYD at load time.  This preflight tests
the real import seam and emits a self-hashed HOLD receipt instead of allowing a
long full-suite run to mix environment failures with source regressions.

Passing this preflight authorizes only starting the test command.  It does not
authorize experiments or any scientific phase.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BACKEND_MODULES = (
    "moocore",
    "pymoo.algorithms.moo.nsga2",
    "pymoo.algorithms.moo.moead",
)


class FullSuiteEnvironmentError(ValueError):
    """Raised when preflight inputs or exclusive receipt output are invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact_nonempty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise FullSuiteEnvironmentError(f"{label} must be a nonempty exact string")
    return value


def _native_artifacts(distribution_name: str) -> list[dict[str, object]]:
    """Return stable hashes for native files without importing the package."""

    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return []
    records: list[dict[str, object]] = []
    for relative in distribution.files or ():
        if relative.suffix.lower() not in {".pyd", ".dll"}:
            continue
        path = Path(distribution.locate_file(relative)).resolve()
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise FullSuiteEnvironmentError(
                f"native artifact changed during stable read: {path}"
            )
        records.append(
            {
                "path": str(path),
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
    return sorted(records, key=lambda item: str(item["path"]).casefold())


def evaluate_full_suite_environment(
    *,
    expected_python_executable: str,
    expected_python_version_prefix: str,
    expected_pymoo_version: str,
    expected_moocore_version: str,
    observed_python_executable: str | None = None,
    observed_python_version: str | None = None,
    distribution_version: Callable[[str], str] = metadata.version,
    module_importer: Callable[[str], object] = import_module,
    native_artifact_collector: Callable[
        [str], list[dict[str, object]]
    ] = _native_artifacts,
) -> tuple[dict[str, object], int]:
    """Evaluate exact identity plus the backend imports used by full tests."""

    expected_executable = str(Path(_exact_nonempty_string(
        expected_python_executable,
        label="expected_python_executable",
    )).resolve())
    expected_version_prefix = _exact_nonempty_string(
        expected_python_version_prefix,
        label="expected_python_version_prefix",
    )
    expected_versions = {
        "pymoo": _exact_nonempty_string(
            expected_pymoo_version,
            label="expected_pymoo_version",
        ),
        "moocore": _exact_nonempty_string(
            expected_moocore_version,
            label="expected_moocore_version",
        ),
    }
    executable = str(Path(observed_python_executable or sys.executable).resolve())
    version = observed_python_version or sys.version
    interpreter_checks = {
        "executable_exact_match": os.path.normcase(executable)
        == os.path.normcase(expected_executable),
        "version_prefix_match": version.startswith(expected_version_prefix),
    }

    distributions: dict[str, object] = {}
    hold_reasons: list[str] = []
    for name, expected in expected_versions.items():
        try:
            observed = distribution_version(name)
            error = None
        except metadata.PackageNotFoundError as exception:
            observed = None
            error = f"{type(exception).__name__}: {exception}"
        except Exception as exception:  # fail closed on broken metadata backends
            observed = None
            error = f"{type(exception).__name__}: {exception}"
        exact_match = observed == expected
        distributions[name] = {
            "expected_version": expected,
            "observed_version": observed,
            "exact_match": exact_match,
            "metadata_error": error,
        }
        if not exact_match:
            hold_reasons.append(f"distribution_version_mismatch:{name}")

    imports: dict[str, object] = {}
    for module_name in _BACKEND_MODULES:
        try:
            module_importer(module_name)
        except Exception as exception:
            imports[module_name] = {
                "status": "FAIL",
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
            }
            hold_reasons.append(f"backend_import_failed:{module_name}")
        else:
            imports[module_name] = {
                "status": "PASS",
                "exception_type": None,
                "exception_message": None,
            }

    native_artifacts: dict[str, object] = {}
    for name in expected_versions:
        try:
            artifacts = native_artifact_collector(name)
        except Exception as exception:
            artifacts = []
            hold_reasons.append(f"native_artifact_inventory_failed:{name}")
            native_artifacts[name] = {
                "status": "FAIL",
                "error": f"{type(exception).__name__}: {exception}",
                "files": artifacts,
            }
        else:
            native_artifacts[name] = {
                "status": "PASS",
                "error": None,
                "files": artifacts,
            }

    for name, passed in interpreter_checks.items():
        if not passed:
            hold_reasons.append(f"interpreter_{name}")
    hold_reasons = sorted(set(hold_reasons))
    passed = not hold_reasons
    core: dict[str, object] = {
        "schema": "v21e3r1_v9r2r1_full_suite_environment_preflight_v1",
        "status": (
            "PASS_FULL_SUITE_ENVIRONMENT_PREFLIGHT"
            if passed
            else "HOLD_FULL_SUITE_ENVIRONMENT"
        ),
        "identity": {
            "distribution": "mo-nco",
            "version": "0.21.3.14",
            "revision": "V21E3R1_V9R2R1",
        },
        "interpreter": {
            "expected_executable": expected_executable,
            "observed_executable": executable,
            "expected_version_prefix": expected_version_prefix,
            "observed_version": version,
            "checks": interpreter_checks,
        },
        "distributions": distributions,
        "backend_imports": imports,
        "native_artifacts": native_artifacts,
        "hold_reasons": hold_reasons,
        "full_suite_execution_preflight_passed": passed,
        "full_suite_execution_recommended": passed,
        "scoped_v9_tests_affected": False,
        "environment_lock_requirement_satisfied": False,
        "full_development_matrix_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }
    receipt = {**core, "receipt_payload_sha256": _sha256(_canonical_bytes(core))}
    return receipt, 0 if passed else 2


def _write_exclusive(path: Path, receipt: Mapping[str, object]) -> None:
    raw = _canonical_bytes(receipt) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exception:
        raise FullSuiteEnvironmentError(f"refusing to overwrite: {path}") from exception


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-python-executable", required=True)
    parser.add_argument("--expected-python-version-prefix", required=True)
    parser.add_argument("--expected-pymoo-version", required=True)
    parser.add_argument("--expected-moocore-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt, exit_code = evaluate_full_suite_environment(
            expected_python_executable=args.expected_python_executable,
            expected_python_version_prefix=args.expected_python_version_prefix,
            expected_pymoo_version=args.expected_pymoo_version,
            expected_moocore_version=args.expected_moocore_version,
        )
        _write_exclusive(args.output.resolve(), receipt)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return exit_code
    except (FullSuiteEnvironmentError, OSError) as exception:
        print(
            json.dumps(
                {
                    "schema": (
                        "v21e3r1_v9r2r1_full_suite_environment_preflight_error_v1"
                    ),
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(exception),
                    "full_suite_execution_preflight_passed": False,
                    "full_development_matrix_authorized": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullSuiteEnvironmentError",
    "evaluate_full_suite_environment",
    "main",
]
