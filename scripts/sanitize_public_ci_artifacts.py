"""Create and verify generic path-sanitized public CI artifact bundles.

Inputs and outputs are paired by explicit logical names.  Actual filesystem
paths are never serialized into the receipt.  The transformation delegates to
the versioned public-checkout sanitization engine, while this wrapper binds the
logical set, raw/output hashes, both sanitizer sources, replacement counts,
reference commit/tree, and per-artifact semantic contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET


RECEIPT_SCHEMA = "v21e3r1_v9r2r1_public_ci_artifact_sanitization_receipt_v2"
PASS_STATUS = "PASS_GENERIC_PUBLIC_CI_ARTIFACT_SANITIZATION"
IDENTITY = {
    "distribution": "mo-nco",
    "revision": "V21E3R1_V9R2R1",
    "version": "0.21.3.14",
}
GENERIC_SOURCE_LOGICAL_PATH = "scripts/sanitize_public_ci_artifacts.py"
ENGINE_SOURCE_LOGICAL_PATH = "scripts/sanitize_public_checkout_outputs.py"
HISTORICAL_INTERPRETER = r"C:\miniconda3\envs\ssm_env\python.exe"
HISTORICAL_INTERPRETER_REPLACEMENT = "__HISTORICAL_INTERPRETER__"
RULE_IDS = (
    "historical_interpreter",
    "repository_root",
    "temp_root",
    "user_home",
    "environment_prefix",
    "username",
    "host_name",
)
REPLACEMENTS = {
    "historical_interpreter": HISTORICAL_INTERPRETER_REPLACEMENT,
    "repository_root": "__REPO_ROOT__",
    "temp_root": "__TEMP_ROOT__",
    "user_home": "__USER_HOME__",
    "environment_prefix": "__PYTHON_PREFIX__",
    "username": "__USERNAME__",
    "host_name": "__HOSTNAME__",
}
ARTIFACT_KINDS = frozenset(
    {
        "PYTEST_JUNIT_XML",
        "PYTEST_LOG",
        "STRICT_JSON",
        "UTF8_TEXT",
    }
)
_LOGICAL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UNC_OR_DEVICE_PATH_RE = re.compile(
    rb"(?i)(?<![a-z0-9\\])\\{2,}"
    rb"(?:(?:\?|\.)(?:\\+UNC)?\\+)?[a-z0-9_$.-]+\\+"
)
_FORWARD_UNC_PATH_RE = re.compile(
    rb"(?i)(?<![:a-z0-9/])//[a-z0-9_$.-]+/"
)
_FILE_UNC_URI_RE = re.compile(
    rb"(?i)\bfile:(?:/{2,}|\\{2,})[a-z0-9_$.-]+[/\\]"
)
_WINDOWS_VOLUME_DEVICE_RE = re.compile(
    rb"(?i)(?<![a-z0-9\\])\\{2,}[?.]\\+Volume\{[0-9a-f-]+\}\\+"
)
_WINDOWS_ROOTED_PROFILE_RE = re.compile(
    rb"(?i)(?<![a-z0-9\\])\\+(?:Users|ProgramData|Windows|"
    rb"Documents[ ]and[ ]Settings)\\+"
)
_HISTORICAL_INTERPRETER_VARIANTS = tuple(
    variant.encode("ascii")
    for variant in dict.fromkeys(
        [
            HISTORICAL_INTERPRETER.replace("\\", "\\" * depth)
            for depth in (4, 2, 1)
        ]
        + [HISTORICAL_INTERPRETER.replace("\\", "/")]
    )
)
_HISTORICAL_INTERPRETER_LEFT_DELIMITER = (
    rb"(?:\A|(?<=[\x09\x0a\x0d =\"]))"
)
_HISTORICAL_INTERPRETER_RIGHT_DELIMITER = (
    rb"(?=\Z|[\x09\x0a\x0d\"<]|;[ ]observed[ ])"
)
_HISTORICAL_INTERPRETER_PATH_SHAPE = re.compile(
    rb"C:[\\/]+miniconda3[\\/]+envs[\\/]+ssm_env[\\/]+python[.]exe",
    re.IGNORECASE,
)
_PYTEST_TERMINAL_RE = re.compile(
    r"^(?P<body>(?:\d+ (?:failed|passed|skipped|xfailed|xpassed|warnings?|"
    r"errors?|deselected|reruns?|subtests passed|subtests failed))"
    r"(?:, \d+ (?:failed|passed|skipped|xfailed|xpassed|warnings?|errors?|"
    r"deselected|reruns?|subtests passed|subtests failed))*) in "
    r"(?P<duration>\d+(?:\.\d+)?)s(?: \((?P<wall_clock>[^\r\n]+)\))?\r?$",
    re.MULTILINE,
)
_PYTEST_FAILURE_RE = re.compile(
    r"^(?P<kind>FAILED|ERROR|SUBFAILED(?:\([^\r\n]*\)))\s+"
    r"(?P<node>tests[/\\][^\r\n]+?)\s*$",
    re.MULTILINE,
)


class CIArtifactSanitizationError(ValueError):
    """Raised when a generic CI artifact contract cannot be proven."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_record(raw: bytes) -> dict[str, object]:
    return {"bytes": len(raw), "sha256": _sha256(raw)}


def _source_record(raw: bytes, logical_path: str) -> dict[str, object]:
    return {**_artifact_record(raw), "path": logical_path}


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _assert_no_link_in_existing_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = absolute
    while True:
        if _is_link_or_junction(current):
            raise CIArtifactSanitizationError(
                f"symlink/reparse path is prohibited: {path}"
            )
        if current.parent == current:
            break
        current = current.parent


def _stable_read_regular(path: Path, *, label: str) -> tuple[Path, bytes]:
    _assert_no_link_in_existing_chain(path)
    resolved = path.resolve(strict=True)
    try:
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise CIArtifactSanitizationError(
                    f"{label} is not a regular file"
                )
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise CIArtifactSanitizationError(f"cannot read {label}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(raw) != after.st_size:
        raise CIArtifactSanitizationError(f"{label} changed while being read")
    _assert_no_link_in_existing_chain(path)
    final_resolved = path.resolve(strict=True)
    final = final_resolved.stat(follow_symlinks=False)
    if final_resolved != resolved or (final.st_dev, final.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise CIArtifactSanitizationError(f"{label} path changed while being read")
    return resolved, raw


def _canonical_source_paths() -> tuple[Path, Path]:
    generic = Path(__file__).resolve(strict=True)
    engine = generic.with_name("sanitize_public_checkout_outputs.py").resolve(
        strict=True
    )
    return generic, engine


def _assert_canonical_sources(
    generic_source_path: Path, engine_source_path: Path
) -> tuple[Path, Path]:
    canonical_generic, canonical_engine = _canonical_source_paths()
    requested_generic = generic_source_path.resolve(strict=True)
    requested_engine = engine_source_path.resolve(strict=True)
    if requested_generic != canonical_generic:
        raise CIArtifactSanitizationError(
            "generic sanitizer source substitution is prohibited"
        )
    if requested_engine != canonical_engine:
        raise CIArtifactSanitizationError(
            "sanitization engine source substitution is prohibited"
        )
    return canonical_generic, canonical_engine


def _load_engine(path: Path) -> tuple[ModuleType, bytes]:
    resolved, raw = _stable_read_regular(path, label="sanitization engine source")
    module = ModuleType("public_checkout_sanitization_engine")
    module.__file__ = str(resolved)
    try:
        code = compile(raw, str(resolved), "exec")
        exec(code, module.__dict__)
    except Exception as error:
        raise CIArtifactSanitizationError(
            f"cannot load sanitization engine: {type(error).__name__}"
        ) from error
    required = {
        "_PUBLIC_WINDOWS_PATH_FIXTURES",
        "_mask_public_source_fixtures",
        "_public_fixture_counts_and_residuals",
        "_pytest_node_id",
        "_sanitize_one",
        "_strict_utf8",
        "junit_observation",
        "log_observation",
    }
    if any(not hasattr(module, name) for name in required):
        raise CIArtifactSanitizationError("sanitization engine API drifted")
    return module, raw


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CIArtifactSanitizationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise CIArtifactSanitizationError(f"non-finite JSON value prohibited: {value}")


def _strict_json(path: Path) -> tuple[dict[str, object], bytes]:
    _resolved, raw = _stable_read_regular(path, label="strict JSON receipt")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CIArtifactSanitizationError(f"invalid strict JSON: {path}") from error
    if type(payload) is not dict:
        raise CIArtifactSanitizationError("receipt must be a JSON object")
    if raw != _canonical_json(payload) + b"\n":
        raise CIArtifactSanitizationError(
            "receipt must be canonical JSON followed by exactly one newline"
        )
    return payload, raw


def _validate_logical_name(name: object) -> str:
    if type(name) is not str or not _LOGICAL_NAME_RE.fullmatch(name):
        raise CIArtifactSanitizationError(f"invalid logical artifact name: {name!r}")
    return name


def _normalize_windows_drive_absolute(value: str, *, label: str) -> str:
    """Normalize slash spelling without accepting UNC or drive-relative paths."""

    if type(value) is not str or not re.fullmatch(
        r"[A-Za-z]:[\\/][^\\/\r\n][^\r\n]*", value
    ):
        raise CIArtifactSanitizationError(
            f"{label} must be a drive-absolute Windows path"
        )
    return value.replace("/", "\\")


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``name=path`` arguments with duplicate rejection."""

    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CIArtifactSanitizationError(
                f"named path must use name=path syntax: {value!r}"
            )
        name, raw_path = value.split("=", 1)
        _validate_logical_name(name)
        if not raw_path:
            raise CIArtifactSanitizationError(f"empty path for logical name: {name}")
        if name in result:
            raise CIArtifactSanitizationError(
                f"duplicate logical artifact name: {name}"
            )
        result[name] = Path(raw_path)
    if not result:
        raise CIArtifactSanitizationError("at least one named artifact is required")
    return result


def parse_named_kinds(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``name=KIND`` arguments with exact kind validation."""

    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise CIArtifactSanitizationError(
                f"named kind must use name=KIND syntax: {value!r}"
            )
        name, kind = value.split("=", 1)
        _validate_logical_name(name)
        if name in result:
            raise CIArtifactSanitizationError(
                f"duplicate logical artifact kind: {name}"
            )
        if kind not in ARTIFACT_KINDS:
            raise CIArtifactSanitizationError(
                f"invalid artifact kind for {name}: {kind!r}"
            )
        result[name] = kind
    if not result:
        raise CIArtifactSanitizationError("at least one artifact kind is required")
    return result


def _assert_exact_logical_set(
    left: Mapping[str, Path], right: Mapping[str, Path]
) -> list[str]:
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))
        unexpected = sorted(set(right) - set(left))
        raise CIArtifactSanitizationError(
            f"logical artifact name set drifted; missing={missing!r}; "
            f"unexpected={unexpected!r}"
        )
    return sorted(left)


def _assert_distinct_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        _assert_no_link_in_existing_chain(path)
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise CIArtifactSanitizationError(
            "input, output, receipt, and source paths must be distinct"
        )
    existing = [path for path in resolved if path.exists()]
    for index, left in enumerate(existing):
        for right in existing[index + 1 :]:
            try:
                same_file = os.path.samefile(left, right)
            except OSError as error:
                raise CIArtifactSanitizationError(
                    "cannot prove physical artifact-path distinctness"
                ) from error
            if same_file:
                raise CIArtifactSanitizationError(
                    "input, output, receipt, and source paths must be "
                    "physically distinct"
                )


def _assert_public_logical_names(
    logical_names: Sequence[str], *, user_home: str, host_name: str
) -> None:
    username = Path(user_home).name
    sensitive_tokens = [token.casefold() for token in (username, host_name) if token]
    for name in logical_names:
        folded = name.casefold()
        if any(token in folded for token in sensitive_tokens):
            raise CIArtifactSanitizationError(
                f"logical artifact name contains a sensitive identity token: {name!r}"
            )


def _git_checkout_identity(repository_root: str) -> tuple[str, str]:
    requested_root = Path(repository_root)
    _assert_no_link_in_existing_chain(requested_root)
    root = requested_root.resolve(strict=True)
    if not root.is_dir():
        raise CIArtifactSanitizationError("repository_root is not a directory")

    def git_value(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise CIArtifactSanitizationError(
                "cannot resolve current Git checkout identity"
            ) from error
        value = completed.stdout.strip()
        if completed.returncode != 0 or not value:
            raise CIArtifactSanitizationError(
                "cannot resolve current Git checkout identity"
            )
        return value

    top = Path(git_value("rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != root:
        raise CIArtifactSanitizationError(
            "repository_root must be the current Git checkout root"
        )
    commit = git_value("rev-parse", "--verify", "HEAD")
    tree = git_value("rev-parse", "--verify", "HEAD^{tree}")
    if not _SHA1_RE.fullmatch(commit) or not _SHA1_RE.fullmatch(tree):
        raise CIArtifactSanitizationError("current Git checkout is not SHA-1 bound")
    return commit, tree


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_link_in_existing_chain(path.parent)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CIArtifactSanitizationError(f"refusing to overwrite: {path}") from error
    _resolved, observed = _stable_read_regular(
        path, label="newly written public CI artifact"
    )
    if observed != raw:
        raise CIArtifactSanitizationError(
            "newly written public CI artifact changed after write"
        )


def _validate_artifact_record(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"bytes", "sha256"}:
        raise CIArtifactSanitizationError("artifact record shape drifted")
    if type(value.get("bytes")) is not int or value["bytes"] < 0:
        raise CIArtifactSanitizationError("artifact byte count is invalid")
    digest = value.get("sha256")
    if type(digest) is not str or not _SHA256_RE.fullmatch(digest):
        raise CIArtifactSanitizationError("artifact SHA-256 is invalid")
    return value


def _validate_source_record(
    value: object, *, expected_path: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"bytes", "sha256", "path"}:
        raise CIArtifactSanitizationError("source record shape drifted")
    if value.get("path") != expected_path:
        raise CIArtifactSanitizationError("source logical path drifted")
    _validate_artifact_record(
        {"bytes": value.get("bytes"), "sha256": value.get("sha256")}
    )
    return value


def _public_fixture_counts_and_residuals(
    engine: ModuleType, raw: bytes
) -> tuple[dict[str, int], int]:
    fixture_counts, drive_count = engine._public_fixture_counts_and_residuals(raw)
    masked, _counts, _restorations = engine._mask_public_source_fixtures(raw)
    extra_path_patterns = (
        _UNC_OR_DEVICE_PATH_RE,
        _FORWARD_UNC_PATH_RE,
        _FILE_UNC_URI_RE,
        _WINDOWS_VOLUME_DEVICE_RE,
        _WINDOWS_ROOTED_PROFILE_RE,
    )
    unc_or_device_count = sum(
        len(pattern.findall(masked)) for pattern in extra_path_patterns
    )
    text = engine._strict_utf8(raw, "public CI artifact")
    decoded = text
    for _index in range(3):
        expanded = html.unescape(decoded)
        if expanded == decoded:
            break
        decoded = expanded
    decoded_count = 0
    if decoded != text:
        decoded_raw = decoded.encode("utf-8")
        decoded_fixtures, decoded_drive_count = (
            engine._public_fixture_counts_and_residuals(decoded_raw)
        )
        decoded_masked, _counts, _restorations = (
            engine._mask_public_source_fixtures(decoded_raw)
        )
        decoded_unc_count = sum(
            len(pattern.findall(decoded_masked))
            for pattern in extra_path_patterns
        )
        fixture_counts = {
            literal: max(fixture_counts[literal], decoded_fixtures[literal])
            for literal in fixture_counts
        }
        decoded_count = decoded_drive_count + decoded_unc_count
    return fixture_counts, drive_count + unc_or_device_count + decoded_count


def _replace_historical_interpreter(raw: bytes) -> tuple[bytes, int]:
    """Replace only the frozen historical interpreter path spellings."""

    result = raw
    total = 0
    replacement = HISTORICAL_INTERPRETER_REPLACEMENT.encode("ascii")
    for variant in _HISTORICAL_INTERPRETER_VARIANTS:
        pattern = re.compile(
            _HISTORICAL_INTERPRETER_LEFT_DELIMITER
            + re.escape(variant)
            + _HISTORICAL_INTERPRETER_RIGHT_DELIMITER,
            re.IGNORECASE,
        )
        result, count = pattern.subn(lambda _match: replacement, result)
        total += count
    if _HISTORICAL_INTERPRETER_PATH_SHAPE.search(result) is not None:
        raise CIArtifactSanitizationError(
            "sensitive historical interpreter path occurs outside the exact "
            "delimiter contract"
        )
    return result, total


def _junit_observation(raw: bytes, engine: ModuleType) -> dict[str, object]:
    """Return the delegated projection after enforcing a strict generic shape."""

    observation = engine.junit_observation(raw)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise CIArtifactSanitizationError("invalid JUnit XML") from error
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if len(suites) != 1:
        raise CIArtifactSanitizationError(
            "generic JUnit must contain exactly one testsuite"
        )
    suite = suites[0]
    testcases = list(suite.iter("testcase"))
    declared_tests = suite.attrib.get("tests")
    if declared_tests is None or not declared_tests.isdecimal():
        raise CIArtifactSanitizationError(
            "generic JUnit lacks an integer declared test count"
        )
    if int(declared_tests) < len(testcases):
        raise CIArtifactSanitizationError(
            "declared/observed generic JUnit testcase count drifted"
        )
    failure_or_error_nodes: list[str] = []
    for testcase in testcases:
        outcomes = [
            child
            for child in testcase
            if child.tag in {"failure", "error", "skipped"}
        ]
        if len(outcomes) > 1:
            raise CIArtifactSanitizationError(
                "generic JUnit testcase has multiple terminal outcomes"
            )
        if outcomes and outcomes[0].tag in {"failure", "error"}:
            failure_or_error_nodes.append(
                engine._pytest_node_id(testcase).replace("\\", "/")
            )
    failure_or_error_nodes.sort()
    return {
        **observation,
        "failure_or_error_node_count": len(failure_or_error_nodes),
        "failure_or_error_node_ids_sha256": _sha256(
            _canonical_json(failure_or_error_nodes)
        ),
    }


def _pytest_log_observation(raw: bytes, engine: ModuleType) -> dict[str, object]:
    text = engine._strict_utf8(raw, "pytest CI log")
    terminal = list(_PYTEST_TERMINAL_RE.finditer(text))
    if len(terminal) != 1:
        raise CIArtifactSanitizationError(
            "pytest log must contain exactly one supported terminal summary"
        )
    match = terminal[0]
    counts: dict[str, int] = {}
    for item in match.group("body").split(", "):
        count_text, label = item.split(" ", 1)
        label = {
            "error": "errors",
            "warning": "warnings",
            "rerun": "reruns",
        }.get(label, label)
        if label in counts:
            raise CIArtifactSanitizationError(
                f"duplicate pytest terminal summary label: {label}"
            )
        counts[label] = int(count_text)
    entries: list[str] = []
    failure_nodes: list[str] = []
    for failure in _PYTEST_FAILURE_RE.finditer(text):
        node = failure.group("node")
        if " - " in node:
            node = node.split(" - ", 1)[0]
        node = node.replace("\\", "/")
        failure_nodes.append(node)
        entries.append(f"{failure.group('kind')} {node}")
    if len(entries) != len(set(entries)):
        raise CIArtifactSanitizationError(
            "pytest short-summary failure entries are not unique"
        )
    expected_entries = counts.get("failed", 0) + counts.get("errors", 0)
    if len(entries) != expected_entries:
        raise CIArtifactSanitizationError(
            "pytest terminal failure count does not match short-summary entries"
        )
    entries.sort()
    failure_nodes.sort()
    return {
        "terminal_counts": counts,
        "failure_summary_entry_count": len(entries),
        "failure_summary_entries_sha256": _sha256(_canonical_json(entries)),
        "failure_or_error_node_count": len(failure_nodes),
        "failure_or_error_node_ids_sha256": _sha256(
            _canonical_json(failure_nodes)
        ),
    }


def _iter_json_strings(value: object):
    if type(value) is str:
        yield value
    elif type(value) is list:
        for item in value:
            yield from _iter_json_strings(item)
    elif type(value) is dict:
        for key, item in value.items():
            yield key
            yield from _iter_json_strings(item)


def _validate_strict_json_artifact(
    raw: bytes,
    engine: ModuleType,
    *,
    require_public_path_safety: bool = True,
) -> None:
    text = engine._strict_utf8(raw, "strict JSON CI artifact")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CIArtifactSanitizationError("invalid strict JSON CI artifact") from error
    if not require_public_path_safety:
        return
    for value in _iter_json_strings(payload):
        fixtures, residual_count = _public_fixture_counts_and_residuals(
            engine, value.encode("utf-8")
        )
        if residual_count or any(fixtures.values()):
            raise CIArtifactSanitizationError(
                "strict JSON decodes to a prohibited Windows absolute path"
            )


def _expected_junit_host_replacements(raw: bytes, host_name: str) -> int:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise CIArtifactSanitizationError("invalid JUnit XML") from error
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if len(suites) != 1:
        raise CIArtifactSanitizationError(
            "generic JUnit must contain exactly one testsuite"
        )
    hostname = suites[0].attrib.get("hostname")
    return int(
        hostname is not None and hostname.casefold() == host_name.casefold()
    )


def _validate_pytest_bundle_crosscheck(
    artifact_contracts: Mapping[str, Mapping[str, object]],
    kinds: Mapping[str, str],
) -> None:
    junit_names = {
        name.removesuffix(".junit.xml"): name
        for name, kind in kinds.items()
        if kind == "PYTEST_JUNIT_XML" and name.endswith(".junit.xml")
    }
    log_names = {
        name.removesuffix(".log"): name
        for name, kind in kinds.items()
        if kind == "PYTEST_LOG" and name.endswith(".log")
    }
    paired_junits: set[str] = set()
    for stem in sorted(set(junit_names) & set(log_names)):
        junit_name = junit_names[stem]
        log_name = log_names[stem]
        paired_junits.add(junit_name)
        junit = artifact_contracts[junit_name]["junit_after"]
        log = artifact_contracts[log_name]["log_after"]
        junit_counts = junit["counts"]
        terminal = log["terminal_counts"]
        terminal_tests = sum(
            terminal.get(label, 0)
            for label in (
                "failed",
                "passed",
                "skipped",
                "errors",
                "xfailed",
                "xpassed",
                "subtests passed",
            )
        )
        if junit_counts["declared_tests"] != terminal_tests:
            raise CIArtifactSanitizationError(
                "paired JUnit/pytest-log total test counts drifted"
            )
        if junit_counts["failure_elements"] != terminal.get("failed", 0):
            raise CIArtifactSanitizationError(
                "paired JUnit/pytest-log failure counts drifted"
            )
        if junit_counts["error_elements"] != terminal.get("errors", 0):
            raise CIArtifactSanitizationError(
                "paired JUnit/pytest-log error counts drifted"
            )
        if junit_counts["skipped_elements"] != (
            terminal.get("skipped", 0) + terminal.get("xfailed", 0)
        ):
            raise CIArtifactSanitizationError(
                "paired JUnit/pytest-log skipped counts drifted"
            )
        if junit_counts["passing_testcase_elements"] != (
            terminal.get("passed", 0) + terminal.get("xpassed", 0)
        ):
            raise CIArtifactSanitizationError(
                "paired JUnit/pytest-log passing testcase counts drifted"
            )
        if (
            junit["failure_or_error_node_count"]
            != log["failure_or_error_node_count"]
            or junit["failure_or_error_node_ids_sha256"]
            != log["failure_or_error_node_ids_sha256"]
        ):
            raise CIArtifactSanitizationError(
                "paired JUnit/pytest-log failure node identities drifted"
            )
    for name, kind in kinds.items():
        if kind != "PYTEST_JUNIT_XML" or name in paired_junits:
            continue
        counts = artifact_contracts[name]["junit_after"]["counts"]
        if counts["declared_tests"] != counts["testcase_elements"]:
            raise CIArtifactSanitizationError(
                "generic JUnit with implicit subtests requires a paired pytest log"
            )


def create_ci_artifact_bundle(
    *,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    kinds: Mapping[str, str],
    receipt_path: Path,
    repository_root: str,
    temp_root: str,
    user_home: str,
    environment_prefix: str,
    host_name: str,
    reference_commit: str,
    reference_tree: str,
    generic_source_path: Path,
    engine_source_path: Path,
) -> dict[str, object]:
    """Create a generic, hash-bound public CI artifact bundle."""

    repository_root = _normalize_windows_drive_absolute(
        repository_root, label="repository_root"
    )
    temp_root = _normalize_windows_drive_absolute(temp_root, label="temp_root")
    user_home = _normalize_windows_drive_absolute(user_home, label="user_home")
    environment_prefix = _normalize_windows_drive_absolute(
        environment_prefix, label="environment_prefix"
    )
    logical_names = _assert_exact_logical_set(inputs, outputs)
    _assert_exact_logical_set(inputs, {name: Path(kind) for name, kind in kinds.items()})
    if any(kind not in ARTIFACT_KINDS for kind in kinds.values()):
        raise CIArtifactSanitizationError("artifact kind set drifted")
    _assert_public_logical_names(
        logical_names, user_home=user_home, host_name=host_name
    )
    if not _SHA1_RE.fullmatch(reference_commit):
        raise CIArtifactSanitizationError(
            "reference_commit must be a lowercase Git SHA-1"
        )
    if not _SHA1_RE.fullmatch(reference_tree):
        raise CIArtifactSanitizationError(
            "reference_tree must be a lowercase Git SHA-1"
        )
    current_commit, current_tree = _git_checkout_identity(repository_root)
    if (reference_commit, reference_tree) != (current_commit, current_tree):
        raise CIArtifactSanitizationError(
            "declared reference commit/tree do not match the current Git checkout"
        )
    canonical_generic, canonical_engine = _assert_canonical_sources(
        generic_source_path, engine_source_path
    )
    all_paths = [
        *inputs.values(),
        *outputs.values(),
        receipt_path,
        generic_source_path,
        engine_source_path,
    ]
    _assert_distinct_paths(all_paths)
    for output in [*outputs.values(), receipt_path]:
        if output.exists():
            raise CIArtifactSanitizationError(f"refusing to overwrite: {output}")

    engine, engine_source_raw = _load_engine(canonical_engine)
    _generic_resolved, generic_source_raw = _stable_read_regular(
        canonical_generic, label="generic sanitizer source"
    )
    raw_inputs: dict[str, dict[str, object]] = {}
    sanitized_outputs: dict[str, dict[str, object]] = {}
    artifact_contracts: dict[str, dict[str, object]] = {}
    replacement_counts = {
        rule_id: {} for rule_id in RULE_IDS
    }
    fixture_literals = tuple(engine._PUBLIC_WINDOWS_PATH_FIXTURES)
    fixture_counts: dict[str, dict[str, int]] = {
        literal: {} for literal in fixture_literals
    }
    output_bytes: dict[str, bytes] = {}

    for name in logical_names:
        kind = kinds[name]
        _input_resolved, raw = _stable_read_regular(
            inputs[name], label=f"raw CI artifact {name}"
        )
        engine._strict_utf8(raw, f"raw CI artifact {name}")
        if any(token.encode("ascii") in raw for token in REPLACEMENTS.values()):
            raise CIArtifactSanitizationError(
                f"sanitization replacement token collides with raw input: {name}"
            )
        fixed_sensitive_paths, historical_interpreter_count = (
            _replace_historical_interpreter(raw)
        )
        sanitized, engine_replacements, observed_fixtures = engine._sanitize_one(
            fixed_sensitive_paths,
            repository_root=repository_root,
            temp_root=temp_root,
            user_home=user_home,
            environment_prefix=environment_prefix,
            host_name=host_name,
        )
        observed_replacements = {
            "historical_interpreter": historical_interpreter_count,
            **engine_replacements,
        }
        semantic_label = {
            "PYTEST_JUNIT_XML": "JUnit",
            "PYTEST_LOG": "pytest log",
            "STRICT_JSON": "strict JSON",
            "UTF8_TEXT": "text artifact",
        }[kind]
        if observed_replacements["username"] != 0:
            raise CIArtifactSanitizationError(
                f"{semantic_label} semantic safety requires zero username "
                f"replacements: {name}"
            )
        expected_host_replacements = (
            _expected_junit_host_replacements(raw, host_name)
            if kind == "PYTEST_JUNIT_XML"
            else 0
        )
        if observed_replacements["host_name"] != expected_host_replacements:
            raise CIArtifactSanitizationError(
                f"{semantic_label} semantic safety prohibits host-name "
                f"replacement outside the exact JUnit testsuite hostname: {name}"
            )
        for rule_id in RULE_IDS:
            replacement_counts[rule_id][name] = observed_replacements[rule_id]
            if sanitized.count(REPLACEMENTS[rule_id].encode("ascii")) != (
                observed_replacements[rule_id]
            ):
                raise CIArtifactSanitizationError(
                    f"replacement-token count drifted for {rule_id}: {name}"
                )
        for literal in fixture_literals:
            fixture_counts[literal][name] = observed_fixtures[literal]
        raw_inputs[name] = _artifact_record(raw)
        sanitized_outputs[name] = _artifact_record(sanitized)
        output_bytes[name] = sanitized
        if kind == "STRICT_JSON":
            observed_fixture_counts, residual_count = (
                engine._public_fixture_counts_and_residuals(sanitized)
            )
        else:
            observed_fixture_counts, residual_count = (
                _public_fixture_counts_and_residuals(engine, sanitized)
            )
        if residual_count:
            raise CIArtifactSanitizationError(
                f"sanitized output contains an unallowlisted Windows path: {name}"
            )
        if any(observed_fixture_counts.values()):
            raise CIArtifactSanitizationError(
                f"public-source Windows fixtures are prohibited in generic CI artifacts: {name}"
            )
        if kind == "PYTEST_JUNIT_XML":
            before = _junit_observation(raw, engine)
            after = _junit_observation(sanitized, engine)
            if before != after:
                raise CIArtifactSanitizationError(
                    f"JUnit semantics changed during sanitization: {name}"
                )
            artifact_contracts[name] = {
                "kind": kind,
                "exact_transform_applied": True,
                "junit_before": before,
                "junit_after": after,
            }
        elif kind == "PYTEST_LOG":
            before = _pytest_log_observation(raw, engine)
            after = _pytest_log_observation(sanitized, engine)
            if before != after:
                raise CIArtifactSanitizationError(
                    f"pytest log semantics changed during sanitization: {name}"
                )
            artifact_contracts[name] = {
                "kind": kind,
                "exact_transform_applied": True,
                "log_before": before,
                "log_after": after,
            }
        elif kind == "STRICT_JSON":
            _validate_strict_json_artifact(
                raw, engine, require_public_path_safety=False
            )
            _validate_strict_json_artifact(sanitized, engine)
            artifact_contracts[name] = {
                "kind": kind,
                "exact_transform_applied": True,
                "strict_json_before": True,
                "strict_json_after": True,
            }
        else:
            artifact_contracts[name] = {
                "kind": kind,
                "exact_transform_applied": True,
            }

    _validate_pytest_bundle_crosscheck(artifact_contracts, kinds)

    rules = [
        {
            "id": rule_id,
            "replacement": REPLACEMENTS[rule_id],
            "match_counts": replacement_counts[rule_id],
        }
        for rule_id in RULE_IDS
    ]
    public_fixtures = [
        {
            "literal": literal,
            "classification": "PUBLIC_SOURCE_SYNTHETIC_FIXTURE_NOT_MACHINE_OBSERVATION",
            "occurrence_counts": fixture_counts[literal],
        }
        for literal in fixture_literals
    ]
    core: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": PASS_STATUS,
        "identity": IDENTITY,
        "reference_checkout": {
            "commit_sha1": reference_commit,
            "git_tree_sha1": reference_tree,
        },
        "logical_names": logical_names,
        "generic_sanitizer_source": _source_record(
            generic_source_raw,
            GENERIC_SOURCE_LOGICAL_PATH,
        ),
        "engine_source": _source_record(
            engine_source_raw,
            ENGINE_SOURCE_LOGICAL_PATH,
        ),
        "raw_inputs": raw_inputs,
        "sanitized_outputs": sanitized_outputs,
        "replacement_contract": {
            "encoding": "UTF-8_STRICT",
            "replacement_order": list(RULE_IDS),
            "rules": rules,
            "preserved_public_source_windows_path_fixtures": public_fixtures,
            "residual_unallowlisted_windows_absolute_path_count": 0,
            "sensitive_values_or_absolute_io_paths_recorded_in_receipt": False,
        },
        "artifact_contracts": artifact_contracts,
        "claim_boundary": {
            "repository_wide_green": False,
            "scientific_stage_authorized": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_authorized": False,
            "ijoc_submission_authorized": False,
        },
    }
    receipt = {
        **core,
        "receipt_payload_sha256": _sha256(_canonical_json(core)),
    }
    for name in logical_names:
        _write_exclusive(outputs[name], output_bytes[name])
    _write_exclusive(receipt_path, _canonical_json(receipt) + b"\n")
    return receipt


def _validate_receipt_contract(
    receipt: Mapping[str, object],
    *,
    engine: ModuleType,
    expected_kinds: Mapping[str, str],
) -> list[str]:
    required = {
        "schema",
        "status",
        "identity",
        "reference_checkout",
        "logical_names",
        "generic_sanitizer_source",
        "engine_source",
        "raw_inputs",
        "sanitized_outputs",
        "replacement_contract",
        "artifact_contracts",
        "claim_boundary",
        "receipt_payload_sha256",
    }
    if set(receipt) != required:
        raise CIArtifactSanitizationError("receipt key set drifted")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != PASS_STATUS:
        raise CIArtifactSanitizationError("receipt identity/status drifted")
    if receipt.get("identity") != IDENTITY:
        raise CIArtifactSanitizationError("distribution identity drifted")
    checkout = receipt.get("reference_checkout")
    if type(checkout) is not dict or set(checkout) != {
        "commit_sha1",
        "git_tree_sha1",
    }:
        raise CIArtifactSanitizationError("reference checkout shape drifted")
    for field in ("commit_sha1", "git_tree_sha1"):
        value = checkout.get(field)
        if type(value) is not str or not _SHA1_RE.fullmatch(value):
            raise CIArtifactSanitizationError(f"invalid reference {field}")

    logical_names = receipt.get("logical_names")
    if type(logical_names) is not list or not logical_names:
        raise CIArtifactSanitizationError("logical_names must be a non-empty list")
    if any(type(name) is not str for name in logical_names):
        raise CIArtifactSanitizationError("logical_names must contain only strings")
    if (
        logical_names != sorted(logical_names)
        or len(logical_names) != len(set(logical_names))
    ):
        raise CIArtifactSanitizationError("logical_names must be unique and sorted")
    for name in logical_names:
        _validate_logical_name(name)
    for field in ("raw_inputs", "sanitized_outputs", "artifact_contracts"):
        value = receipt.get(field)
        if type(value) is not dict or set(value) != set(logical_names):
            raise CIArtifactSanitizationError(f"{field} logical-name set drifted")
    if set(expected_kinds) != set(logical_names) or any(
        kind not in ARTIFACT_KINDS for kind in expected_kinds.values()
    ):
        raise CIArtifactSanitizationError("expected artifact kind set drifted")
    for record in receipt["raw_inputs"].values():
        _validate_artifact_record(record)
    for record in receipt["sanitized_outputs"].values():
        _validate_artifact_record(record)
    _validate_source_record(
        receipt.get("generic_sanitizer_source"),
        expected_path=GENERIC_SOURCE_LOGICAL_PATH,
    )
    _validate_source_record(
        receipt.get("engine_source"),
        expected_path=ENGINE_SOURCE_LOGICAL_PATH,
    )

    replacement = receipt.get("replacement_contract")
    if type(replacement) is not dict or set(replacement) != {
        "encoding",
        "replacement_order",
        "rules",
        "preserved_public_source_windows_path_fixtures",
        "residual_unallowlisted_windows_absolute_path_count",
        "sensitive_values_or_absolute_io_paths_recorded_in_receipt",
    }:
        raise CIArtifactSanitizationError("replacement contract shape drifted")
    if replacement.get("encoding") != "UTF-8_STRICT":
        raise CIArtifactSanitizationError("encoding contract drifted")
    if replacement.get("replacement_order") != list(RULE_IDS):
        raise CIArtifactSanitizationError("replacement order drifted")
    rules = replacement.get("rules")
    if type(rules) is not list or len(rules) != len(RULE_IDS):
        raise CIArtifactSanitizationError("replacement rule list drifted")
    for rule_id, rule in zip(RULE_IDS, rules, strict=True):
        if type(rule) is not dict or set(rule) != {
            "id",
            "replacement",
            "match_counts",
        }:
            raise CIArtifactSanitizationError("replacement rule shape drifted")
        if rule.get("id") != rule_id:
            raise CIArtifactSanitizationError("replacement rule id/order drifted")
        if rule.get("replacement") != REPLACEMENTS[rule_id]:
            raise CIArtifactSanitizationError("replacement token drifted")
        counts = rule.get("match_counts")
        if type(counts) is not dict or set(counts) != set(logical_names):
            raise CIArtifactSanitizationError("replacement count name set drifted")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise CIArtifactSanitizationError("replacement count is invalid")
        if rule_id == "username" and any(counts.values()):
            raise CIArtifactSanitizationError(
                "generic CI receipt admits username replacement"
            )
        if rule_id == "host_name":
            for name, count in counts.items():
                allowed = (
                    count in {0, 1}
                    if expected_kinds[name] == "PYTEST_JUNIT_XML"
                    else count == 0
                )
                if not allowed:
                    raise CIArtifactSanitizationError(
                        "generic CI receipt admits unsafe host-name replacement"
                    )

    fixtures = replacement.get("preserved_public_source_windows_path_fixtures")
    expected_fixtures = tuple(engine._PUBLIC_WINDOWS_PATH_FIXTURES)
    if type(fixtures) is not list or len(fixtures) != len(expected_fixtures):
        raise CIArtifactSanitizationError("public fixture allowlist drifted")
    for literal, fixture in zip(expected_fixtures, fixtures, strict=True):
        if type(fixture) is not dict or set(fixture) != {
            "literal",
            "classification",
            "occurrence_counts",
        }:
            raise CIArtifactSanitizationError("public fixture record shape drifted")
        if fixture.get("literal") != literal:
            raise CIArtifactSanitizationError("public fixture literal/order drifted")
        if fixture.get("classification") != (
            "PUBLIC_SOURCE_SYNTHETIC_FIXTURE_NOT_MACHINE_OBSERVATION"
        ):
            raise CIArtifactSanitizationError("public fixture classification drifted")
        counts = fixture.get("occurrence_counts")
        if type(counts) is not dict or set(counts) != set(logical_names):
            raise CIArtifactSanitizationError("public fixture count name set drifted")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise CIArtifactSanitizationError("public fixture count is invalid")
        if any(count != 0 for count in counts.values()):
            raise CIArtifactSanitizationError(
                "generic CI artifacts may not preserve absolute-path fixtures"
            )
    if replacement.get("residual_unallowlisted_windows_absolute_path_count") != 0:
        raise CIArtifactSanitizationError("receipt admits unallowlisted Windows paths")
    if (
        replacement.get("sensitive_values_or_absolute_io_paths_recorded_in_receipt")
        is not False
    ):
        raise CIArtifactSanitizationError("receipt admits sensitive/path disclosure")

    for name, contract in receipt["artifact_contracts"].items():
        kind = expected_kinds[name]
        if kind == "PYTEST_JUNIT_XML":
            if type(contract) is not dict or set(contract) != {
                "kind",
                "exact_transform_applied",
                "junit_before",
                "junit_after",
            }:
                raise CIArtifactSanitizationError("JUnit contract shape drifted")
            if contract.get("kind") != "PYTEST_JUNIT_XML":
                raise CIArtifactSanitizationError("JUnit contract kind drifted")
            if contract.get("junit_before") != contract.get("junit_after"):
                raise CIArtifactSanitizationError("receipt records JUnit semantic drift")
        elif kind == "PYTEST_LOG":
            if type(contract) is not dict or set(contract) != {
                "kind",
                "exact_transform_applied",
                "log_before",
                "log_after",
            }:
                raise CIArtifactSanitizationError("pytest log contract shape drifted")
            if contract.get("kind") != kind:
                raise CIArtifactSanitizationError("pytest log contract kind drifted")
            if contract.get("log_before") != contract.get("log_after"):
                raise CIArtifactSanitizationError(
                    "receipt records pytest log semantic drift"
                )
        elif kind == "STRICT_JSON":
            if contract != {
                "kind": kind,
                "exact_transform_applied": True,
                "strict_json_before": True,
                "strict_json_after": True,
            }:
                raise CIArtifactSanitizationError("strict JSON contract drifted")
        else:
            if contract != {
                "kind": "UTF8_TEXT",
                "exact_transform_applied": True,
            }:
                raise CIArtifactSanitizationError("text artifact contract drifted")
        if contract.get("exact_transform_applied") is not True:
            raise CIArtifactSanitizationError("exact-transform verdict drifted")

    _validate_pytest_bundle_crosscheck(receipt["artifact_contracts"], expected_kinds)

    expected_boundary = {
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }
    if receipt.get("claim_boundary") != expected_boundary:
        raise CIArtifactSanitizationError("claim boundary drifted")
    return logical_names


def verify_ci_artifact_bundle(
    *,
    receipt_path: Path,
    outputs: Mapping[str, Path],
    kinds: Mapping[str, str],
    repository_root: str,
    user_home: str,
    host_name: str,
    generic_source_path: Path,
    engine_source_path: Path,
) -> dict[str, object]:
    """Verify a generic bundle without requiring its private raw inputs."""

    repository_root = _normalize_windows_drive_absolute(
        repository_root, label="repository_root"
    )
    user_home = _normalize_windows_drive_absolute(user_home, label="user_home")
    receipt, _receipt_raw = _strict_json(receipt_path)
    declared_hash = receipt.get("receipt_payload_sha256")
    if type(declared_hash) is not str or not _SHA256_RE.fullmatch(declared_hash):
        raise CIArtifactSanitizationError("invalid receipt payload SHA-256")
    core = dict(receipt)
    del core["receipt_payload_sha256"]
    if declared_hash != _sha256(_canonical_json(core)):
        raise CIArtifactSanitizationError("receipt payload self-hash mismatch")

    canonical_generic, canonical_engine = _assert_canonical_sources(
        generic_source_path, engine_source_path
    )
    _assert_distinct_paths(
        [receipt_path, *outputs.values(), canonical_generic, canonical_engine]
    )
    _stable_read_regular(receipt_path, label="public CI sanitization receipt")
    engine, engine_raw = _load_engine(canonical_engine)
    logical_names = _validate_receipt_contract(
        receipt, engine=engine, expected_kinds=kinds
    )
    if set(outputs) != set(logical_names):
        missing = sorted(set(logical_names) - set(outputs))
        unexpected = sorted(set(outputs) - set(logical_names))
        raise CIArtifactSanitizationError(
            f"output logical-name set drifted; missing={missing!r}; "
            f"unexpected={unexpected!r}"
        )
    _assert_public_logical_names(
        logical_names, user_home=user_home, host_name=host_name
    )
    current_commit, current_tree = _git_checkout_identity(repository_root)
    if receipt["reference_checkout"] != {
        "commit_sha1": current_commit,
        "git_tree_sha1": current_tree,
    }:
        raise CIArtifactSanitizationError(
            "receipt reference commit/tree do not match the current Git checkout"
        )

    _generic_resolved, generic_raw = _stable_read_regular(
        canonical_generic, label="generic sanitizer source"
    )
    if receipt.get("generic_sanitizer_source") != _source_record(
        generic_raw,
        GENERIC_SOURCE_LOGICAL_PATH,
    ):
        raise CIArtifactSanitizationError("generic sanitizer source drifted")
    if receipt.get("engine_source") != _source_record(
        engine_raw,
        ENGINE_SOURCE_LOGICAL_PATH,
    ):
        raise CIArtifactSanitizationError("sanitization engine source drifted")

    fixture_records = receipt["replacement_contract"][
        "preserved_public_source_windows_path_fixtures"
    ]
    receipt_rule_counts = {
        rule["id"]: rule["match_counts"]
        for rule in receipt["replacement_contract"]["rules"]
    }
    for name in logical_names:
        _output_resolved, raw = _stable_read_regular(
            outputs[name], label=f"sanitized CI artifact {name}"
        )
        engine._strict_utf8(raw, f"sanitized CI artifact {name}")
        if receipt["sanitized_outputs"][name] != _artifact_record(raw):
            raise CIArtifactSanitizationError(
                f"sanitized output hash/size drifted: {name}"
            )
        kind = kinds[name]
        if kind == "STRICT_JSON":
            observed_fixtures, residual_count = (
                engine._public_fixture_counts_and_residuals(raw)
            )
        else:
            observed_fixtures, residual_count = (
                _public_fixture_counts_and_residuals(engine, raw)
            )
        if residual_count:
            raise CIArtifactSanitizationError(
                f"sanitized output contains an unallowlisted Windows path: {name}"
            )
        for fixture in fixture_records:
            literal = fixture["literal"]
            if fixture["occurrence_counts"][name] != observed_fixtures[literal]:
                raise CIArtifactSanitizationError(
                    f"public fixture occurrence count drifted: {name}"
                )
        if any(observed_fixtures.values()):
            raise CIArtifactSanitizationError(
                f"sanitized output preserves an absolute-path fixture: {name}"
            )
        for rule_id, replacement in REPLACEMENTS.items():
            if raw.count(replacement.encode("ascii")) != (
                receipt_rule_counts[rule_id][name]
            ):
                raise CIArtifactSanitizationError(
                    f"sanitized replacement-token count drifted for {rule_id}: {name}"
                )
        if kind == "PYTEST_JUNIT_XML":
            observation = _junit_observation(raw, engine)
            if observation != receipt["artifact_contracts"][name]["junit_after"]:
                raise CIArtifactSanitizationError(
                    f"sanitized JUnit semantics drifted: {name}"
                )
            expected_host_count = receipt_rule_counts["host_name"][name]
            observed_host_count = _expected_junit_host_replacements(
                raw, REPLACEMENTS["host_name"]
            )
            if observed_host_count != expected_host_count:
                raise CIArtifactSanitizationError(
                    f"sanitized JUnit host placeholder placement drifted: {name}"
                )
        elif kind == "PYTEST_LOG":
            observation = _pytest_log_observation(raw, engine)
            if observation != receipt["artifact_contracts"][name]["log_after"]:
                raise CIArtifactSanitizationError(
                    f"sanitized pytest log semantics drifted: {name}"
                )
        elif kind == "STRICT_JSON":
            _validate_strict_json_artifact(raw, engine)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create an exclusive bundle")
    create.add_argument("--input", action="append", required=True)
    create.add_argument("--output", action="append", required=True)
    create.add_argument("--kind", action="append", required=True)
    create.add_argument("--receipt", type=Path, required=True)
    create.add_argument("--repository-root", required=True)
    create.add_argument("--temp-root", required=True)
    create.add_argument("--user-home", required=True)
    create.add_argument("--environment-prefix", required=True)
    create.add_argument("--host-name", required=True)
    create.add_argument("--reference-commit", required=True)
    create.add_argument("--reference-tree", required=True)
    create.add_argument(
        "--generic-source",
        type=Path,
        default=Path(__file__),
    )
    create.add_argument(
        "--engine-source",
        type=Path,
        default=Path(__file__).with_name("sanitize_public_checkout_outputs.py"),
    )
    verify = subparsers.add_parser("verify", help="verify a public bundle")
    verify.add_argument("--output", action="append", required=True)
    verify.add_argument("--kind", action="append", required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--repository-root", required=True)
    verify.add_argument("--user-home", required=True)
    verify.add_argument("--host-name", required=True)
    verify.add_argument(
        "--generic-source",
        type=Path,
        default=Path(__file__),
    )
    verify.add_argument(
        "--engine-source",
        type=Path,
        default=Path(__file__).with_name("sanitize_public_checkout_outputs.py"),
    )
    args = parser.parse_args(argv)
    try:
        outputs = parse_named_paths(args.output)
        if args.command == "create":
            receipt = create_ci_artifact_bundle(
                inputs=parse_named_paths(args.input),
                outputs=outputs,
                kinds=parse_named_kinds(args.kind),
                receipt_path=args.receipt,
                repository_root=args.repository_root,
                temp_root=args.temp_root,
                user_home=args.user_home,
                environment_prefix=args.environment_prefix,
                host_name=args.host_name,
                reference_commit=args.reference_commit,
                reference_tree=args.reference_tree,
                generic_source_path=args.generic_source,
                engine_source_path=args.engine_source,
            )
        else:
            receipt = verify_ci_artifact_bundle(
                receipt_path=args.receipt,
                outputs=outputs,
                kinds=parse_named_kinds(args.kind),
                repository_root=args.repository_root,
                user_home=args.user_home,
                host_name=args.host_name,
                generic_source_path=args.generic_source,
                engine_source_path=args.engine_source,
            )
    except (CIArtifactSanitizationError, FileNotFoundError, OSError, ValueError) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CIArtifactSanitizationError",
    "PASS_STATUS",
    "RECEIPT_SCHEMA",
    "create_ci_artifact_bundle",
    "main",
    "parse_named_kinds",
    "parse_named_paths",
    "verify_ci_artifact_bundle",
]
