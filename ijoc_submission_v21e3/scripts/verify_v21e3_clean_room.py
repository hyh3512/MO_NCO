from __future__ import annotations

"""Verify V21e3 in a new venv and prove deterministic package rebuild."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Iterable, Sequence
import zipfile


_SCHEMA = "ijoc_v21e3_standalone_release_manifest_v1"
_VERSION = "0.21.3"
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_ENTRY_COUNT = 10_000
_MAX_ENTRY_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def verify_release_archive(
    archive_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Verify archive bytes, metadata, entry set, and every entry binding."""

    archive_file = Path(archive_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if archive_file.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise ValueError("Release archive exceeds the clean-room size cap.")
    if manifest_file.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Release manifest exceeds the clean-room size cap.")
    archive_raw = archive_file.read_bytes()
    manifest_raw = manifest_file.read_bytes()
    if len(archive_raw) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Release archive changed beyond the clean-room size cap.")
    if len(manifest_raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("Release manifest changed beyond the clean-room size cap.")
    manifest = json.loads(manifest_raw)
    if _canonical(manifest) != manifest_raw:
        raise ValueError("Release manifest is not canonical JSON.")
    if not (
        manifest.get("schema") == _SCHEMA
        and manifest.get("project_version") == _VERSION
        and manifest.get("formal_authorized") is False
        and manifest.get("formal_status") == "NOT_MATERIALIZED"
        and manifest.get("dependency_closure", {}).get("gate") == "PASS"
        and manifest.get("dependency_closure", {}).get(
            "unresolved_internal_imports"
        )
        == []
    ):
        raise ValueError("Release manifest fails the V21e3 boundary gate.")
    archive_sha = _sha256(archive_raw)
    if not (
        archive_sha == manifest.get("archive", {}).get("sha256")
        and len(archive_raw) == int(manifest.get("archive", {}).get("bytes", -1))
    ):
        raise ValueError("Release archive bytes do not match the manifest.")
    prefix = str(manifest.get("archive_prefix"))
    expected_entries = list(manifest.get("files", ()))
    if (
        not expected_entries
        or len(expected_entries) != int(manifest.get("file_count", -1))
        or len(expected_entries) > _MAX_ENTRY_COUNT
    ):
        raise ValueError("Release manifest file count is inconsistent.")
    declared_total = 0
    for entry in expected_entries:
        if not isinstance(entry, dict) or type(entry.get("bytes")) is not int:
            raise ValueError("Release manifest entry byte count is invalid.")
        entry_bytes = int(entry["bytes"])
        if not (0 <= entry_bytes <= _MAX_ENTRY_BYTES):
            raise ValueError("Release manifest entry exceeds the size cap.")
        declared_total += entry_bytes
    if declared_total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("Release manifest total size exceeds the clean-room cap.")
    expected_names = [
        f"{prefix}/{entry['archive_path']}" for entry in expected_entries
    ]
    if expected_names != sorted(expected_names) or len(expected_names) != len(
        set(expected_names)
    ):
        raise ValueError("Release manifest archive paths are not unique and sorted.")
    try:
        with zipfile.ZipFile(archive_file) as archive:
            infos = archive.infolist()
            observed_names = [info.filename for info in infos]
            if observed_names != expected_names:
                raise ValueError("Release archive entry set/order differs from manifest.")
            for info, entry in zip(infos, expected_entries):
                pure = PurePosixPath(info.filename)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or info.is_dir()
                    or info.date_time != _TIMESTAMP
                    or (info.external_attr >> 16) != 0o100644
                    or info.flag_bits & 0x1
                    or info.file_size != int(entry["bytes"])
                    or info.file_size > _MAX_ENTRY_BYTES
                    or info.compress_size > _MAX_ARCHIVE_BYTES
                ):
                    raise ValueError("Release archive entry metadata is unsafe.")
                raw = archive.read(info)
                if not (
                    len(raw) == int(entry["bytes"])
                    and _sha256(raw) == entry["sha256"]
                ):
                    raise ValueError(
                        f"Release archive entry binding failed: {info.filename}"
                    )
            wheelhouse_gate = manifest.get("offline_wheelhouse_gate")
            if wheelhouse_gate == "PASS":
                wheel_manifest_raw = archive.read(
                    f"{prefix}/wheelhouse_manifest.json"
                )
                wheel_manifest = json.loads(wheel_manifest_raw)
                if not (
                    _canonical(wheel_manifest) == wheel_manifest_raw
                    and wheel_manifest.get("schema")
                    == "pareto_v21e3_offline_wheelhouse_manifest_v1"
                    and wheel_manifest.get("install_contract")
                    == "pip_no_index_require_hashes_v1"
                ):
                    raise ValueError("Offline wheelhouse manifest is not frozen.")
                wheel_items = wheel_manifest.get("files")
                if not isinstance(wheel_items, list) or not wheel_items:
                    raise ValueError("Offline wheelhouse manifest is empty or invalid.")
                wheel_names = [str(item.get("filename")) for item in wheel_items]
                if wheel_names != sorted(wheel_names) or len(wheel_names) != len(
                    set(wheel_names)
                ):
                    raise ValueError("Offline wheel names are not unique and sorted.")
                expected_wheels = {}
                for item, name in zip(wheel_items, wheel_names):
                    if not (
                        isinstance(item, dict)
                        and PurePosixPath(name).name == name
                        and "\\" not in name
                        and name.endswith(".whl")
                        and type(item.get("bytes")) is int
                        and 0 < int(item["bytes"]) <= _MAX_ENTRY_BYTES
                        and isinstance(item.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    ):
                        raise ValueError("Offline wheel binding is unsafe.")
                    expected_wheels[name] = item
                observed_wheels = {
                    name.removeprefix(f"{prefix}/wheelhouse/"): name
                    for name in observed_names
                    if name.startswith(f"{prefix}/wheelhouse/")
                }
                if set(expected_wheels) != set(observed_wheels):
                    raise ValueError("Offline wheelhouse entry set mismatch.")
                for name, item in expected_wheels.items():
                    wheel_raw = archive.read(observed_wheels[name])
                    if not (
                        len(wheel_raw) == int(item["bytes"])
                        and _sha256(wheel_raw) == item["sha256"]
                    ):
                        raise ValueError(f"Offline wheel binding failed: {name}")
            elif wheelhouse_gate != "NOT_INCLUDED_TEST_FIXTURE":
                raise ValueError("Release manifest has an invalid wheelhouse gate.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Release archive is not a valid ZIP.") from exc
    return {
        "schema": "ijoc_v21e3_release_archive_verification_v1",
        "status": "PASS",
        "archive_sha256": archive_sha,
        "manifest_sha256": _sha256(manifest_raw),
        "entry_count": len(expected_entries),
        "archive_prefix": prefix,
        "project_version": _VERSION,
        "offline_wheelhouse_gate": manifest["offline_wheelhouse_gate"],
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
        "archive_safety_caps": {
            "max_archive_bytes": _MAX_ARCHIVE_BYTES,
            "max_manifest_bytes": _MAX_MANIFEST_BYTES,
            "max_entry_count": _MAX_ENTRY_COUNT,
            "max_entry_bytes": _MAX_ENTRY_BYTES,
            "max_total_uncompressed_bytes": _MAX_TOTAL_UNCOMPRESSED_BYTES,
        },
    }


def _extract_verified(
    archive_source: Path | bytes,
    manifest: dict[str, object],
    destination: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    prefix = str(manifest["archive_prefix"])
    package_root = destination / prefix
    source = io.BytesIO(archive_source) if isinstance(archive_source, bytes) else archive_source
    with zipfile.ZipFile(source) as archive:
        for entry in manifest["files"]:
            name = f"{prefix}/{entry['archive_path']}"
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.resolve().relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(name)
            if not (
                len(raw) == int(entry["bytes"])
                and _sha256(raw) == entry["sha256"]
            ):
                raise ValueError("Pinned archive changed during extraction.")
            target.write_bytes(raw)
    return package_root


def _venv_python(venv: Path) -> Path:
    return (
        venv / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv / "bin" / "python"
    )


def _console_script(venv: Path) -> Path:
    return (
        venv / "Scripts" / "mo-nco-v21e3.exe"
        if os.name == "nt"
        else venv / "bin" / "mo-nco-v21e3"
    )


def validate_offline_requirements_lock(
    lock_path: str | Path,
) -> dict[str, object]:
    """Reject lock directives that can bypass the local wheelhouse."""

    path = Path(lock_path).resolve()
    raw = path.read_bytes()
    try:
        text_value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Requirements lock is not UTF-8.") from exc
    active_lines = [
        line.strip()
        for line in text_value.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    required_options = (
        "--no-index",
        "--find-links wheelhouse",
        "--require-hashes",
    )
    if active_lines[:3] != list(required_options):
        raise ValueError("Requirements lock lacks the frozen offline options.")
    requirement_pattern = re.compile(
        r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.!+_-]*"
        r'(?:;\s*platform_system\s*==\s*"(?:Windows|Linux|Darwin)")?\s+\\\Z'
    )
    hash_pattern = re.compile(r"--hash=sha256:[0-9a-f]{64}\Z")
    requirement_lines = active_lines[3:]
    if not requirement_lines or len(requirement_lines) % 2:
        raise ValueError("Requirements lock has an invalid pinned-entry structure.")
    package_names: list[str] = []
    for requirement, digest in zip(
        requirement_lines[0::2], requirement_lines[1::2]
    ):
        if not requirement_pattern.fullmatch(requirement):
            raise ValueError("Requirements lock contains a network-capable source.")
        if not hash_pattern.fullmatch(digest):
            raise ValueError("Requirements lock contains an invalid hash binding.")
        package_names.append(requirement.split("==", 1)[0].lower().replace("_", "-"))
    if len(package_names) != len(set(package_names)):
        raise ValueError("Requirements lock contains duplicate package pins.")
    return {
        "schema": "pareto_v21e3r1_offline_requirements_lock_gate_v1",
        "status": "PASS",
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "required_options": list(required_options),
        "package_count": len(package_names),
        "network_capable_sources": [],
    }


def _run(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    logs: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    log_path = logs / f"{name}.log"
    with log_path.open("xb") as log_handle:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    log_bytes = log_path.stat().st_size
    with log_path.open("rb") as log_handle:
        log_handle.seek(max(0, log_bytes - 64 * 1024))
        tail_text = log_handle.read().decode("utf-8", errors="replace")
    log_tail = tail_text.splitlines()[-30:]
    result = {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
        "log_bytes": log_bytes,
        "log_tail": log_tail,
        "subprocess_output_capture": "streamed_to_fsynced_log_bounded_64k_tail_v1",
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Clean-room step {name!r} failed with exit code "
            f"{completed.returncode}: {tail_text[-2000:]}"
        )
    return result


def run_clean_room_gate(
    *,
    archive_path: str | Path,
    manifest_path: str | Path,
    work_directory: str | Path,
    receipt_path: str | Path,
    base_python: str | Path = sys.executable,
) -> dict[str, object]:
    """Run install/import/test/CLI/rebuild gates in a fresh isolated venv."""

    archive_file = Path(archive_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    work = Path(work_directory).resolve()
    receipt_file = Path(receipt_path).resolve()
    if work.exists():
        raise FileExistsError(work)
    if receipt_file.exists():
        raise FileExistsError(receipt_file)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    verification = verify_release_archive(archive_file, manifest_file)
    if verification["offline_wheelhouse_gate"] != "PASS":
        raise ValueError("Clean-room execution requires the offline wheelhouse.")
    pinned_archive_raw = archive_file.read_bytes()
    pinned_manifest_raw = manifest_file.read_bytes()
    if not (
        _sha256(pinned_archive_raw) == verification["archive_sha256"]
        and _sha256(pinned_manifest_raw) == verification["manifest_sha256"]
    ):
        raise RuntimeError("Release inputs changed after verification.")
    manifest = json.loads(pinned_manifest_raw)
    work.mkdir(parents=True, exist_ok=False)
    logs = work / "logs"
    logs.mkdir()
    steps: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema": "ijoc_v21e3_clean_room_gate_receipt_v1",
        "status": "RUNNING",
        "scientific_scope": "software_portability_gate_not_formal_evidence",
        "archive_verification": verification,
        "base_python": str(Path(base_python).resolve()),
        "work_directory": str(work),
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
        "pip_network_policy": {
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "explicit_no_index_each_install": True,
            "wheel_source": "verified_package_root/wheelhouse",
            "pip_resolution_network_disabled": True,
            "os_network_namespace_isolation": "NOT_PERFORMED",
            "arbitrary_test_network_isolation": "NOT_PERFORMED",
        },
        "pinned_input_bytes_gate": "PASS",
        "steps": steps,
    }
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    try:
        package_root = _extract_verified(
            pinned_archive_raw,
            manifest,
            work / "extracted",
        )
        wheelhouse = package_root / "wheelhouse"
        requirements_gate = validate_offline_requirements_lock(
            package_root / "requirements-test.lock"
        )
        receipt["requirements_lock_gate"] = requirements_gate
        environment.update(
            {
                "PIP_FIND_LINKS": str(wheelhouse),
                "PIP_INDEX_URL": "",
                "PIP_EXTRA_INDEX_URL": "",
            }
        )
        venv = work / "venv"
        steps.append(
            _run(
                "01_create_venv",
                (str(base_python), "-m", "venv", str(venv)),
                cwd=work,
                logs=logs,
                environment=environment,
            )
        )
        python = _venv_python(venv)
        steps.append(
            _run(
                "02_install_locked_test_dependencies",
                (
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    str(wheelhouse),
                    "--requirement",
                    str(package_root / "requirements-test.lock"),
                ),
                cwd=package_root,
                logs=logs,
                environment=environment,
            )
        )
        steps.append(
            _run(
                "03_install_package",
                (
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    str(wheelhouse),
                    "--no-build-isolation",
                    "--no-deps",
                    ".",
                ),
                cwd=package_root,
                logs=logs,
                environment=environment,
            )
        )
        modules = sorted(
            {
                str(entry["archive_path"])
                .removesuffix(".py")
                .replace("/", ".")
                for entry in manifest["files"]
                if str(entry["archive_path"]).startswith("mo_nco/")
                and str(entry["archive_path"]).endswith(".py")
                and not str(entry["archive_path"]).endswith("/__init__.py")
            }
        )
        import_code = (
            "import importlib,json,pathlib,sysconfig; modules=json.loads("
            + repr(json.dumps(modules))
            + "); package=importlib.import_module('mo_nco'); "
            + "origin=pathlib.Path(package.__file__).resolve(); "
            + "roots=[pathlib.Path(sysconfig.get_paths()[key]).resolve() "
            + "for key in ('purelib','platlib')]; "
            + "assert any(origin.is_relative_to(root) for root in roots), "
            + "f'uninstalled package origin: {origin}'; "
            + "[importlib.import_module(name) for name in modules]; "
            + "print(json.dumps({'status':'PASS','module_count':len(modules),"
            + "'package_origin':str(origin)}))"
        )
        steps.append(
            _run(
                "04_import_all_packaged_modules",
                (str(python), "-c", import_code),
                cwd=work,
                logs=logs,
                environment=environment,
            )
        )
        tests = sorted(
            str(package_root / str(entry["archive_path"]))
            for entry in manifest["files"]
            if str(entry["archive_path"]).startswith("tests/test_pareto_v21")
            and str(entry["archive_path"]).endswith(".py")
        )
        if not tests:
            raise RuntimeError("The standalone release contains no V21 tests.")
        isolated_pytest_config = work / "pytest-clean-room.ini"
        isolated_pytest_config.write_text("[pytest]\n", encoding="utf-8")
        steps.append(
            _run(
                "05_run_all_packaged_v21_tests",
                (
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "--import-mode=importlib",
                    "-c",
                    str(isolated_pytest_config),
                    *tests,
                ),
                cwd=work,
                logs=logs,
                environment=environment,
            )
        )
        steps.append(
            _run(
                "06_console_cli_status_smoke",
                (str(_console_script(venv)), "--status"),
                cwd=work,
                logs=logs,
                environment=environment,
            )
        )
        rebuild = work / "rebuild"
        rebuild.mkdir()
        steps.append(
            _run(
                "07_deterministic_rebuild",
                (
                    str(python),
                    str(
                        package_root
                        / "ijoc_submission_v21e3"
                        / "scripts"
                        / "build_v21e3_code_release.py"
                    ),
                    "--repo-root",
                    str(package_root),
                    "--archive-path",
                    str(rebuild / "rebuilt.zip"),
                    "--manifest-path",
                    str(rebuild / "rebuilt.manifest.json"),
                    "--checksum-path",
                    str(rebuild / "rebuilt.zip.sha256"),
                ),
                cwd=package_root,
                logs=logs,
                environment=environment,
            )
        )
        rebuilt_raw = (rebuild / "rebuilt.zip").read_bytes()
        if rebuilt_raw != pinned_archive_raw:
            raise RuntimeError(
                "Deterministic clean-room rebuild digest/bytes differ from input."
            )
        steps.append(
            _run(
                "08_installed_versions",
                (str(python), "-m", "pip", "freeze", "--all"),
                cwd=work,
                logs=logs,
                environment=environment,
            )
        )
        receipt.update(
            {
                "status": "PASS",
                "package_root": str(package_root),
                "venv_python": str(python),
                "imported_module_count": len(modules),
                "packaged_v21_test_file_count": len(tests),
                "deterministic_rebuild_gate": "PASS",
                "rebuilt_archive_sha256": _sha256(rebuilt_raw),
                "standalone_install_gate": "PASS",
                "offline_no_index_gate": "PASS",
                "all_module_import_gate": "PASS",
                "installed_distribution_origin_gate": "PASS",
                "isolated_pytest_import_mode_gate": "PASS",
                "all_packaged_v21_tests_gate": "PASS",
                "console_cli_smoke_gate": "PASS",
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        receipt_file.write_bytes(_canonical(receipt))
        raise
    receipt_file.write_bytes(_canonical(receipt))
    result = dict(receipt)
    result["receipt_path"] = str(receipt_file)
    result["receipt_sha256"] = _sha256(receipt_file.read_bytes())
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V21e3 clean-room gate.")
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--base-python", type=Path, default=Path(sys.executable))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_clean_room_gate(
        archive_path=args.archive_path,
        manifest_path=args.manifest_path,
        work_directory=args.work_directory,
        receipt_path=args.receipt_path,
        base_python=args.base_python,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
