from __future__ import annotations

"""Verify V21e3r1 in a new venv and prove deterministic package rebuild."""

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Iterable, Iterator, Sequence
import zipfile


_SCHEMA = "ijoc_v21e3r1_standalone_release_manifest_v1"
_VERSION = "0.21.3.1"
_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_RUNTIME_PROBE_CODE = (
    "import json,platform,sqlite3,sys;"
    "print(json.dumps({"
    "'sys_version':sys.version,"
    "'implementation':platform.python_implementation(),"
    "'platform':platform.platform(),"
    "'sqlite_version':sqlite3.sqlite_version"
    "},sort_keys=True,separators=(',',':')))"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


@contextmanager
def _hold_executable_bytes(executable: Path) -> Iterator[str]:
    """Prevent Windows executable-byte replacement across process launch.

    Windows ``CreateProcess`` accepts a path rather than a pre-opened executable
    handle.  Holding a read handle that grants only read sharing closes the ABA
    window: writers and rename/delete replacement are denied until the child
    has started and completed.  Other platforms retain explicit pre/post byte
    revalidation, but do not advertise an OS-level handle pin.
    """

    if os.name != "nt":
        yield "NOT_PERFORMED_UNSUPPORTED_PLATFORM"
        return

    import ctypes
    from ctypes import wintypes

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_attribute_normal = 0x00000080
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(executable),
        generic_read,
        file_share_read,
        None,
        open_existing,
        file_attribute_normal,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield "PASS_WINDOWS_READ_HANDLE_DENIES_WRITE_DELETE_AND_RENAME"
    finally:
        if not close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _base_python_runtime_binding(base_python: str | Path) -> dict[str, object]:
    """Bind executable bytes and runtime facts emitted by that exact interpreter."""

    executable = Path(base_python).resolve(strict=True)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    with _hold_executable_bytes(executable) as os_handle_pin:
        executable_raw = executable.read_bytes()
        executable_sha256 = _sha256(executable_raw)
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
            environment.pop(name, None)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        completed = subprocess.run(
            (str(executable), "-c", _RUNTIME_PROBE_CODE),
            cwd=executable.parent,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "The base Python runtime probe failed with exit code "
                f"{completed.returncode}: {completed.stderr[-2000:]}"
            )
        try:
            runtime = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "The base Python runtime probe returned non-JSON output."
            ) from error
        expected_keys = {
            "sys_version",
            "implementation",
            "platform",
            "sqlite_version",
        }
        if not isinstance(runtime, dict) or set(runtime) != expected_keys or any(
            not isinstance(runtime[key], str) or not runtime[key]
            for key in expected_keys
        ):
            raise RuntimeError(
                "The base Python runtime probe returned an invalid payload."
            )
        post_probe_raw = executable.read_bytes()
        if not (
            len(post_probe_raw) == len(executable_raw)
            and _sha256(post_probe_raw) == executable_sha256
        ):
            raise RuntimeError("The base Python executable drifted during runtime probe.")
    return {
        "schema": "ijoc_v21e3r1_base_python_runtime_binding_v1",
        "status": "PASS",
        "canonical_resolved_path": str(executable),
        "bytes": len(executable_raw),
        "sha256": executable_sha256,
        **runtime,
        "runtime_probe_executable_revalidation": "PASS_BEFORE_AND_AFTER_PROBE",
        "venv_creation_executable_revalidation": "NOT_RUN",
        "os_level_executable_handle_pin": os_handle_pin,
    }


def _pending_base_python_binding(base_python: str | Path) -> dict[str, object]:
    """Bind executable bytes before any subprocess is allowed to execute."""

    executable = Path(base_python).resolve(strict=True)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    raw = executable.read_bytes()
    return {
        "schema": "ijoc_v21e3r1_base_python_runtime_binding_v1",
        "status": "RUNTIME_PROBE_PENDING",
        "canonical_resolved_path": str(executable),
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "sys_version": "NOT_RUN",
        "implementation": "NOT_RUN",
        "platform": "NOT_RUN",
        "sqlite_version": "NOT_RUN",
        "runtime_probe_executable_revalidation": "NOT_RUN",
        "venv_creation_executable_revalidation": "NOT_RUN",
        "os_level_executable_handle_pin": "NOT_PERFORMED",
    }


def _assert_base_python_bytes(
    executable: Path, binding: dict[str, object], *, stage: str
) -> None:
    raw = executable.read_bytes()
    if not (
        binding.get("canonical_resolved_path") == str(executable)
        and binding.get("bytes") == len(raw)
        and binding.get("sha256") == _sha256(raw)
    ):
        raise RuntimeError(f"The base Python executable drifted at {stage}.")


def verify_release_archive(
    archive_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Verify archive bytes, metadata, entry set, and every entry binding."""

    archive_file = Path(archive_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    archive_raw = archive_file.read_bytes()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
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
        raise ValueError("Release manifest fails the V21e3r1 boundary gate.")
    archive_sha = _sha256(archive_raw)
    if not (
        archive_sha == manifest.get("archive", {}).get("sha256")
        and len(archive_raw) == int(manifest.get("archive", {}).get("bytes", -1))
    ):
        raise ValueError("Release archive bytes do not match the manifest.")
    prefix = str(manifest.get("archive_prefix"))
    expected_entries = list(manifest.get("files", ()))
    if len(expected_entries) != int(manifest.get("file_count", -1)):
        raise ValueError("Release manifest file count is inconsistent.")
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
                wheel_manifest = json.loads(
                    archive.read(f"{prefix}/wheelhouse_manifest.json")
                )
                expected_wheels = {
                    str(item["filename"]): item
                    for item in wheel_manifest.get("files", ())
                }
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
        "schema": "ijoc_v21e3r1_release_archive_verification_v1",
        "status": "PASS",
        "archive_sha256": archive_sha,
        "manifest_sha256": _sha256(manifest_raw),
        "entry_count": len(expected_entries),
        "archive_prefix": prefix,
        "project_version": _VERSION,
        "offline_wheelhouse_gate": manifest["offline_wheelhouse_gate"],
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
    }


def _extract_verified(
    archive_source: Path | bytes,
    manifest: dict[str, object],
    destination: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    prefix = str(manifest["archive_prefix"])
    package_root = destination / prefix
    source = (
        io.BytesIO(archive_source)
        if isinstance(archive_source, bytes)
        else archive_source
    )
    with zipfile.ZipFile(source) as archive:
        for entry in manifest["files"]:
            name = f"{prefix}/{entry['archive_path']}"
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.resolve().relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return package_root


def _venv_python(venv: Path) -> Path:
    return (
        venv / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv / "bin" / "python"
    )


def _console_script(venv: Path) -> Path:
    return (
        venv / "Scripts" / "mo-nco-v21e3r1.exe"
        if os.name == "nt"
        else venv / "bin" / "mo-nco-v21e3r1"
    )


def validate_offline_requirements_lock(
    lock_path: str | Path,
) -> dict[str, object]:
    """Reject lock directives or entries that can bypass the wheelhouse."""

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
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    raw = completed.stdout.encode("utf-8", errors="replace")
    log_path = logs / f"{name}.log"
    log_path.write_bytes(raw)
    result = {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "log_path": str(log_path),
        "log_sha256": _sha256(raw),
        "log_bytes": len(raw),
        "log_tail": completed.stdout.splitlines()[-30:],
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"Clean-room step {name!r} failed with exit code "
            f"{completed.returncode}: {completed.stdout[-2000:]}"
        )
    return result


def _persist_receipt(
    path: Path, payload: dict[str, object], *, exclusive: bool = False
) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(_canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _seal_failure_receipt(
    path: Path, receipt: dict[str, object], error: BaseException
) -> None:
    receipt.update(
        {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    _persist_receipt(path, receipt)


def _prepare_clean_room_inputs(
    *,
    archive_file: Path,
    manifest_file: Path,
    work: Path,
    receipt: dict[str, object],
) -> dict[str, object]:
    """Verify and pin release inputs without starting an execution step."""

    if work.exists():
        raise FileExistsError(work)
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
    pinned_inputs = work / "pinned-inputs"
    pinned_inputs.mkdir()
    pinned_manifest_file = pinned_inputs / "release.manifest.verified.json"
    with pinned_manifest_file.open("xb") as handle:
        handle.write(pinned_manifest_raw)
        handle.flush()
        os.fsync(handle.fileno())
    if pinned_manifest_file.read_bytes() != pinned_manifest_raw:
        raise RuntimeError("Pinned manifest copy failed its byte-for-byte check.")
    receipt.update(
        {
            "archive_verification": verification,
            "pinned_input_bytes_gate": "PASS",
            "extraction_source_gate": "PASS_PINNED_VERIFIED_ARCHIVE_BYTES",
            "pinned_inputs": {
                "archive_sha256": _sha256(pinned_archive_raw),
                "archive_bytes": len(pinned_archive_raw),
                "manifest_sha256": _sha256(pinned_manifest_raw),
                "manifest_bytes": len(pinned_manifest_raw),
                "rebuild_manifest_copy_path": str(pinned_manifest_file),
            },
            "rebuild_provenance_manifest_source_gate": "NOT_RUN",
            "deterministic_rebuild_comparison_gate": "NOT_RUN",
            "deterministic_rebuild_comparison_source": (
                "PINNED_VERIFIED_ARCHIVE_BYTES"
            ),
        }
    )
    environment = dict(os.environ)
    for inherited_python_variable in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
    ):
        environment.pop(inherited_python_variable, None)
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_INDEX_URL": "",
            "PIP_EXTRA_INDEX_URL": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return {
        "pinned_archive_raw": pinned_archive_raw,
        "pinned_manifest_raw": pinned_manifest_raw,
        "manifest": manifest,
        "logs": logs,
        "pinned_manifest_file": pinned_manifest_file,
        "environment": environment,
    }


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
    if receipt_file.exists():
        raise FileExistsError(receipt_file)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    requested_base_python = str(Path(base_python).resolve(strict=False))
    steps: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema": "ijoc_v21e3r1_clean_room_gate_receipt_v2",
        "status": "RUNNING",
        "scientific_scope": "software_portability_gate_not_formal_evidence",
        "base_python": requested_base_python,
        "base_python_runtime_binding": {
            "schema": "ijoc_v21e3r1_base_python_runtime_binding_v1",
            "status": "BASE_RUNTIME_BINDING_UNAVAILABLE",
            "canonical_resolved_path": requested_base_python,
            "bytes": "NOT_AVAILABLE",
            "sha256": "NOT_AVAILABLE",
            "sys_version": "NOT_RUN",
            "implementation": "NOT_RUN",
            "platform": "NOT_RUN",
            "sqlite_version": "NOT_RUN",
            "runtime_probe_executable_revalidation": "NOT_RUN",
            "venv_creation_executable_revalidation": "NOT_RUN",
            "os_level_executable_handle_pin": "NOT_RUN",
        },
        "work_directory": str(work),
        "formal_authorized": False,
        "formal_status": "NOT_MATERIALIZED",
        "pip_network_policy": {
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "explicit_no_index_each_install": True,
            "wheel_source": "verified_package_root/wheelhouse",
            "requirements_lock_semantic_validation": "NOT_RUN",
            "pip_resolution_network_disabled": True,
            "os_network_namespace_isolation": "NOT_PERFORMED",
            "arbitrary_test_network_isolation": "NOT_PERFORMED",
        },
        "python_environment_isolation": {
            "policy": "scrub_inherited_python_bootstrap_variables_v1",
            "scrubbed_variables": [
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONSTARTUP",
            ],
            "host_pythonpath_inheritance": "PROHIBITED",
        },
        "steps": steps,
    }
    _persist_receipt(receipt_file, receipt, exclusive=True)
    try:
        runtime_binding = _pending_base_python_binding(base_python)
        resolved_base_python = Path(
            str(runtime_binding["canonical_resolved_path"])
        )
        receipt["base_python"] = str(resolved_base_python)
        receipt["base_python_runtime_binding"] = runtime_binding
        _persist_receipt(receipt_file, receipt)
    except BaseException as exc:
        _seal_failure_receipt(receipt_file, receipt, exc)
        raise
    try:
        probed_binding = _base_python_runtime_binding(resolved_base_python)
        if not all(
            probed_binding.get(field) == runtime_binding.get(field)
            for field in ("canonical_resolved_path", "bytes", "sha256")
        ):
            raise RuntimeError(
                "The base Python executable drifted before runtime binding completed."
            )
        runtime_binding = probed_binding
        receipt["base_python_runtime_binding"] = runtime_binding
        _persist_receipt(receipt_file, receipt)
    except BaseException as exc:
        _seal_failure_receipt(receipt_file, receipt, exc)
        raise
    try:
        prepared = _prepare_clean_room_inputs(
            archive_file=archive_file,
            manifest_file=manifest_file,
            work=work,
            receipt=receipt,
        )
    except BaseException as exc:
        _seal_failure_receipt(receipt_file, receipt, exc)
        raise
    pinned_archive_raw = prepared["pinned_archive_raw"]
    pinned_manifest_raw = prepared["pinned_manifest_raw"]
    manifest = prepared["manifest"]
    logs = prepared["logs"]
    pinned_manifest_file = prepared["pinned_manifest_file"]
    environment = prepared["environment"]
    if not (
        isinstance(pinned_archive_raw, bytes)
        and isinstance(pinned_manifest_raw, bytes)
        and isinstance(manifest, dict)
        and isinstance(logs, Path)
        and isinstance(pinned_manifest_file, Path)
        and isinstance(environment, dict)
    ):
        error = RuntimeError("Clean-room input preparation returned invalid types.")
        _seal_failure_receipt(receipt_file, receipt, error)
        raise error
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
        pip_network_policy = receipt["pip_network_policy"]
        if not isinstance(pip_network_policy, dict):
            raise RuntimeError("Clean-room pip network policy is malformed.")
        pip_network_policy["requirements_lock_semantic_validation"] = "PASS"
        venv = work / "venv"
        with _hold_executable_bytes(resolved_base_python) as venv_handle_pin:
            _assert_base_python_bytes(
                resolved_base_python,
                runtime_binding,
                stage="immediately_before_venv_creation",
            )
            steps.append(
                _run(
                    "01_create_venv",
                    (str(resolved_base_python), "-m", "venv", str(venv)),
                    cwd=work,
                    logs=logs,
                    environment=environment,
                )
            )
            _assert_base_python_bytes(
                resolved_base_python,
                runtime_binding,
                stage="immediately_after_venv_creation",
            )
        if venv_handle_pin != runtime_binding["os_level_executable_handle_pin"]:
            raise RuntimeError(
                "Runtime-probe and venv-creation executable pin modes differ."
            )
        runtime_binding["venv_creation_executable_revalidation"] = (
            "PASS_BEFORE_AND_AFTER_VENV_CREATION"
        )
        runtime_binding["os_level_executable_handle_pin"] = (
            venv_handle_pin + "_FOR_RUNTIME_PROBE_AND_VENV_CREATION"
            if venv_handle_pin.startswith("PASS_")
            else venv_handle_pin
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
                "04_import_installed_distribution_modules",
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
        steps.append(
            _run(
                "05_run_extracted_tree_v21_tests",
                (
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "--import-mode=prepend",
                    *tests,
                ),
                cwd=package_root,
                logs=logs,
                environment=environment,
            )
        )
        steps.append(
            _run(
                "06_run_installed_distribution_console_cli",
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
                        / "ijoc_submission_v21e3r1"
                        / "scripts"
                        / "build_v21e3r1_code_release.py"
                    ),
                    "--repo-root",
                    str(package_root),
                    "--archive-path",
                    str(rebuild / "rebuilt.zip"),
                    "--manifest-path",
                    str(rebuild / "rebuilt.manifest.json"),
                    "--checksum-path",
                    str(rebuild / "rebuilt.zip.sha256"),
                    "--rebuild-provenance-manifest",
                    str(pinned_manifest_file),
                ),
                cwd=package_root,
                logs=logs,
                environment=environment,
            )
        )
        if pinned_manifest_file.read_bytes() != pinned_manifest_raw:
            raise RuntimeError(
                "Pinned manifest copy changed during deterministic rebuild."
            )
        receipt["rebuild_provenance_manifest_source_gate"] = (
            "PASS_PINNED_VERIFIED_MANIFEST_COPY"
        )
        rebuilt_raw = (rebuild / "rebuilt.zip").read_bytes()
        if rebuilt_raw != pinned_archive_raw:
            raise RuntimeError(
                "Deterministic clean-room rebuild digest/bytes differ from input."
            )
        receipt["deterministic_rebuild_comparison_gate"] = "PASS"
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
                "deterministic_rebuild_mode": (
                    "verified_release_manifest_exact_file_list_replay_v1"
                ),
                "rebuilt_archive_sha256": _sha256(rebuilt_raw),
                "standalone_install_gate": "PASS",
                "offline_no_index_gate": "PASS",
                "installed_distribution_import_gate": "PASS",
                "extracted_tree_pytest_gate": "PASS",
                "installed_distribution_console_cli_gate": "PASS",
                "execution_evidence": {
                    "installed_distribution_import": {
                        "gate": "PASS",
                        "step": "04_import_installed_distribution_modules",
                        "artifact_under_test": (
                            "fresh_venv_installed_distribution"
                        ),
                        "origin_constraint": "venv_purelib_or_platlib",
                        "module_count": len(modules),
                    },
                    "extracted_tree_pytest": {
                        "gate": "PASS",
                        "step": "05_run_extracted_tree_v21_tests",
                        "artifact_under_test": (
                            "verified_extracted_release_tree"
                        ),
                        "test_file_count": len(tests),
                        "installed_distribution_test_claim": "NOT_CLAIMED",
                    },
                    "installed_distribution_console_cli": {
                        "gate": "PASS",
                        "step": (
                            "06_run_installed_distribution_console_cli"
                        ),
                        "artifact_under_test": (
                            "fresh_venv_installed_console_script"
                        ),
                    },
                },
            }
        )
    except BaseException as exc:
        _seal_failure_receipt(receipt_file, receipt, exc)
        raise
    _persist_receipt(receipt_file, receipt)
    result = dict(receipt)
    result["receipt_path"] = str(receipt_file)
    result["receipt_sha256"] = _sha256(receipt_file.read_bytes())
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V21e3r1 clean-room gate.")
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
