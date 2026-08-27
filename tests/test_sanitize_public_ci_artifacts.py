from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERIC_SCRIPT = ROOT / "scripts" / "sanitize_public_ci_artifacts.py"
ENGINE_SCRIPT = ROOT / "scripts" / "sanitize_public_checkout_outputs.py"
HISTORICAL_INTERPRETER = r"C:\miniconda3\envs\ssm_env\python.exe"


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


REFERENCE_COMMIT = _git_value("rev-parse", "--verify", "HEAD")
REFERENCE_TREE = _git_value("rev-parse", "--verify", "HEAD^{tree}")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SANITIZER = _load_module("sanitize_public_ci_artifacts_under_test", GENERIC_SCRIPT)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _create_bundle(
    tmp_path: Path,
    artifacts: dict[str, bytes],
    *,
    kinds: dict[str, str] | None = None,
    generic_source_path: Path = GENERIC_SCRIPT,
    engine_source_path: Path = ENGINE_SCRIPT,
    repository_root: str = str(ROOT),
    temp_root: str = r"D:\work\temporary",
    user_home: str = r"C:\Users\test_secret",
    environment_prefix: str = r"C:\Python311",
):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "sanitized"
    raw_root.mkdir()
    inputs: dict[str, Path] = {}
    outputs: dict[str, Path] = {}
    for index, (name, raw) in enumerate(artifacts.items()):
        input_path = raw_root / f"artifact-{index}.raw"
        input_path.write_bytes(raw)
        inputs[name] = input_path
        outputs[name] = output_root / f"artifact-{index}.sanitized"
    receipt_path = tmp_path / "sanitization.receipt.json"
    receipt = SANITIZER.create_ci_artifact_bundle(
        inputs=inputs,
        outputs=outputs,
        kinds=(
            kinds
            if kinds is not None
            else {name: "UTF8_TEXT" for name in artifacts}
        ),
        receipt_path=receipt_path,
        repository_root=repository_root,
        temp_root=temp_root,
        user_home=user_home,
        environment_prefix=environment_prefix,
        host_name="CI-HOST",
        reference_commit=REFERENCE_COMMIT,
        reference_tree=REFERENCE_TREE,
        generic_source_path=generic_source_path,
        engine_source_path=engine_source_path,
    )
    return receipt, outputs, receipt_path


def test_create_accepts_drive_absolute_windows_paths_with_forward_slashes(
    tmp_path: Path,
) -> None:
    repository_root = str(ROOT).replace("\\", "/")
    temp_root = "D:/work/temporary"
    user_home = "C:/Users/test_secret"
    environment_prefix = "C:/Python311"
    raw = (
        f"repo={repository_root}\n"
        f"temp={temp_root}\n"
        f"home={user_home}\n"
        f"prefix={environment_prefix}\n"
    ).encode("utf-8")

    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"diagnostic.txt": raw},
        repository_root=repository_root,
        temp_root=temp_root,
        user_home=user_home,
        environment_prefix=environment_prefix,
    )

    assert outputs["diagnostic.txt"].read_bytes() == (
        b"repo=__REPO_ROOT__\n"
        b"temp=__TEMP_ROOT__\n"
        b"home=__USER_HOME__\n"
        b"prefix=__PYTHON_PREFIX__\n"
    )
    rules = {
        rule["id"]: rule["match_counts"]["diagnostic.txt"]
        for rule in receipt["replacement_contract"]["rules"]
    }
    assert rules == {
        "historical_interpreter": 0,
        "repository_root": 1,
        "temp_root": 1,
        "user_home": 1,
        "environment_prefix": 1,
        "username": 0,
        "host_name": 0,
    }
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        repository_root=repository_root,
        user_home=user_home,
    ) == receipt


@pytest.mark.parametrize(
    "value",
    [
        None,
        Path(r"C:\typed\path"),
        False,
        r"C:relative\path",
        r"\rooted\path",
        r"\\server\share\path",
        "//server/share/path",
        "file://server/share/path",
        "file:///C:/private/path",
        r"\\?\C:\device\path",
        r"\\?\UNC\server\share\path",
        r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\path",
        r"\\.\PhysicalDrive0",
        "C:/",
        "C://server/share/path",
        "C:\\\\server\\share\\path",
        "C:/line\nbreak",
    ],
    ids=[
        "none",
        "path-object",
        "boolean",
        "drive-relative",
        "rooted",
        "unc",
        "forward-unc",
        "file-unc-uri",
        "file-drive-uri",
        "device-drive",
        "device-unc",
        "volume-device",
        "physical-drive",
        "drive-root",
        "double-forward-root-separator",
        "double-backslash-root-separator",
        "newline",
    ],
)
def test_declared_sensitive_roots_reject_non_drive_absolute_forms(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="drive-absolute Windows path"):
        SANITIZER._normalize_windows_drive_absolute(value, label="test_root")


def test_current_source_workflow_preserves_failed_pytest_for_sanitization() -> None:
    workflow = (ROOT / ".github/workflows/current-source.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("$global:LASTEXITCODE = 0") == 2
    for environment_name in ("TARGETED_PYTEST_EXIT", "BACKEND_PYTEST_EXIT"):
        capture = workflow.index(f'"{environment_name}=$pytestExit"')
        reset = workflow.index("$global:LASTEXITCODE = 0", capture)
        sanitize = workflow.index("Sanitize and verify", reset)
        assert capture < reset < sanitize


def test_clean_room_workflow_accepts_only_the_expected_gate_hold_exit() -> None:
    workflow = (ROOT / ".github/workflows/clean-room-package.yml").read_text(
        encoding="utf-8"
    )

    capture = workflow.index("$gateExit = $LASTEXITCODE")
    exact_hold = workflow.index("if ($gateExit -ne 2)", capture)
    boundary = workflow.index("$gate.status -ne 'PRE_DEVELOPMENT_HOLD'", exact_hold)
    reset = workflow.index("$global:LASTEXITCODE = 0", boundary)
    assert capture < exact_hold < boundary < reset


def _verify_bundle(
    *,
    receipt_path: Path,
    outputs: dict[str, Path],
    kinds: dict[str, str] | None = None,
    generic_source_path: Path = GENERIC_SCRIPT,
    engine_source_path: Path = ENGINE_SCRIPT,
    repository_root: str = str(ROOT),
    user_home: str = r"C:\Users\test_secret",
):
    return SANITIZER.verify_ci_artifact_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=(
            kinds
            if kinds is not None
            else {name: "UTF8_TEXT" for name in outputs}
        ),
        repository_root=repository_root,
        user_home=user_home,
        host_name="CI-HOST",
        generic_source_path=generic_source_path,
        engine_source_path=engine_source_path,
    )


def _write_rehashed_receipt(path: Path, receipt: dict[str, object]) -> None:
    core = dict(receipt)
    core.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json(core)
    ).hexdigest()
    path.write_bytes(_canonical_json(receipt) + b"\n")


def test_log_semantic_drift_is_rejected(tmp_path: Path) -> None:
    raw_log = (
        b"FAILED tests/test_secret.py::test_failure - AssertionError: boom\n"
        b"1 failed, 0 passed, 0 skipped, 0 subtests passed in 0.10s\n"
    )

    with pytest.raises(ValueError, match="log.*semantic|semantic.*log"):
        _create_bundle(
            tmp_path,
            {"pytest.log": raw_log},
            kinds={"pytest.log": "PYTEST_LOG"},
        )


def test_host_replacement_outside_junit_hostname_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="semantic.*host-name"):
        _create_bundle(tmp_path, {"diagnostic.txt": b"machine=CI-HOST\n"})


def test_log_summary_and_failure_nodes_are_preserved(tmp_path: Path) -> None:
    raw_log = (
        f"FAILED tests/test_alpha.py::test_failure - {ROOT}\\detail\n"
        "1 failed, 3 passed, 2 skipped, 4 subtests passed in 0.10s\n"
    ).encode("utf-8")
    kinds = {"pytest.log": "PYTEST_LOG"}
    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"pytest.log": raw_log},
        kinds=kinds,
    )

    contract = receipt["artifact_contracts"]["pytest.log"]
    assert contract["log_before"] == contract["log_after"]
    assert contract["log_after"]["terminal_counts"] == {
        "failed": 1,
        "passed": 3,
        "skipped": 2,
        "subtests passed": 4,
    }
    assert str(ROOT).encode("utf-8") not in outputs["pytest.log"].read_bytes()
    receipt_raw = receipt_path.read_bytes()
    for sensitive in (str(ROOT), r"C:\Users\test_secret", "test_secret", "CI-HOST"):
        assert sensitive.encode("utf-8") not in receipt_raw
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=kinds,
    )["status"] == SANITIZER.PASS_STATUS


@pytest.mark.parametrize(
    "private_path",
    [
        rb"\\server\share\private\result.txt",
        rb"\\?\UNC\server\share\private\result.txt",
        rb"\\.\PhysicalDrive0",
        rb"//server/share/private/result.txt",
        rb"file://server/share/private/result.txt",
        rb"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\private.txt",
        rb"\Users\Alice\private\result.txt",
    ],
    ids=(
        "unc",
        "device-unc",
        "device",
        "forward-unc",
        "file-unc",
        "volume-device",
        "rooted-profile",
    ),
)
def test_unc_and_device_paths_are_rejected(
    tmp_path: Path,
    private_path: bytes,
) -> None:
    with pytest.raises(ValueError, match="Windows|absolute path|sensitive"):
        _create_bundle(tmp_path, {"diagnostic.txt": b"path=" + private_path + b"\n"})


def test_generic_artifacts_reject_public_source_fixture_literals(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fixture|Windows|absolute path"):
        _create_bundle(tmp_path, {"diagnostic.txt": b"path=C:/trace.sqlite3\n"})


def test_historical_interpreter_variants_are_replaced_and_receipted(
    tmp_path: Path,
) -> None:
    variants = [
        HISTORICAL_INTERPRETER.replace("\\", "\\" * depth)
        for depth in (4, 2, 1)
    ]
    variants.append(HISTORICAL_INTERPRETER.replace("\\", "/"))
    raw = "".join(f"runtime={variant}\n" for variant in variants).encode("utf-8")

    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"runtime.txt": raw},
    )

    expected = b"runtime=__HISTORICAL_INTERPRETER__\n" * len(variants)
    assert outputs["runtime.txt"].read_bytes() == expected
    assert HISTORICAL_INTERPRETER.encode("ascii") not in receipt_path.read_bytes()
    assert receipt["schema"] == (
        "v21e3r1_v9r2r1_public_ci_artifact_sanitization_receipt_v2"
    )
    historical_rule = receipt["replacement_contract"]["rules"][0]
    assert historical_rule == {
        "id": "historical_interpreter",
        "replacement": "__HISTORICAL_INTERPRETER__",
        "match_counts": {"runtime.txt": len(variants)},
    }
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
    ) == receipt

    tampered = copy.deepcopy(receipt)
    tampered["replacement_contract"]["rules"][0]["match_counts"][
        "runtime.txt"
    ] -= 1
    tampered_path = tmp_path / "historical-count-tampered.receipt.json"
    _write_rehashed_receipt(tampered_path, tampered)
    with pytest.raises(
        SANITIZER.CIArtifactSanitizationError,
        match="replacement-token count",
    ):
        _verify_bundle(
            receipt_path=tampered_path,
            outputs=outputs,
        )


def test_historical_interpreter_in_paired_pytest_evidence_preserves_contract(
    tmp_path: Path,
) -> None:
    historical = HISTORICAL_INTERPRETER.encode("ascii")
    raw_junit = (
        b'<testsuite tests="1" failures="1" errors="0" skipped="0">'
        b'<testcase classname="tests.test_runtime" name="test_failure">'
        b'<failure message="RuntimeError: '
        + historical
        + b'">interpreter='
        + historical
        + b"</failure></testcase></testsuite>"
    )
    raw_log = (
        b"FAILED tests/test_runtime.py::test_failure - RuntimeError: "
        + historical
        + b"\n1 failed in 0.01s\n"
    )
    kinds = {
        "run.junit.xml": "PYTEST_JUNIT_XML",
        "run.log": "PYTEST_LOG",
    }

    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"run.junit.xml": raw_junit, "run.log": raw_log},
        kinds=kinds,
    )

    for output in outputs.values():
        sanitized = output.read_bytes()
        assert historical not in sanitized
        assert b"__HISTORICAL_INTERPRETER__" in sanitized
    historical_rule = receipt["replacement_contract"]["rules"][0]
    assert historical_rule["match_counts"] == {
        "run.junit.xml": 2,
        "run.log": 1,
    }
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=kinds,
    ) == receipt


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("", ".bak"),
        ("", r"\child"),
        ("", ":alternate-stream"),
        ("", "-backup"),
        ("", "@backup"),
        ("", "$backup"),
        ("", "%20backup"),
        ("", "?query"),
        ("", "#fragment"),
        ("", "!backup"),
        ("", "&backup"),
        ("", "(backup"),
        ("", ")backup"),
        ("", "+backup"),
        ("", ",backup"),
        ("", ";backup"),
        ("", "=backup"),
        ("", "[backup"),
        ("", "]backup"),
        ("", "{backup"),
        ("", "}backup"),
        ("", "^backup"),
        ("", "~backup"),
        ("", "'backup"),
        ("", "`backup"),
        ("", " backup"),
        ("-", ""),
        ("@", ""),
        ("$", ""),
        ("%20", ""),
        ("?", ""),
        ("#", ""),
        ("!", ""),
        ("&", ""),
        ("(", ""),
        (")", ""),
        ("+", ""),
        (",", ""),
        (";", ""),
        ("[", ""),
        ("]", ""),
        ("{", ""),
        ("}", ""),
        ("^", ""),
        ("~", ""),
        ("'", ""),
        ("`", ""),
        ("file:///", ""),
    ],
)
def test_historical_interpreter_rule_rejects_path_concatenations(
    tmp_path: Path,
    prefix: str,
    suffix: str,
) -> None:
    raw = f"path={prefix}{HISTORICAL_INTERPRETER}{suffix}\n".encode("ascii")

    with pytest.raises(ValueError, match="Windows|absolute path|sensitive"):
        _create_bundle(tmp_path, {"runtime.txt": raw})


@pytest.mark.parametrize(
    "mixed_path",
    [
        pytest.param(
            r"C:\\miniconda3\envs\\ssm_env\python.exe",
            id="mixed-one-two-one",
        ),
        pytest.param(
            r"C:/miniconda3\envs/ssm_env\\python.exe",
            id="mixed-forward-one-two",
        ),
        pytest.param(
            r"C:\\\miniconda3\\\envs\\\ssm_env\\\python.exe",
            id="unsupported-depth-three",
        ),
    ],
)
def test_historical_interpreter_rule_rejects_mixed_or_unsupported_slash_depths(
    tmp_path: Path,
    mixed_path: str,
) -> None:
    with pytest.raises(ValueError, match="sensitive"):
        _create_bundle(
            tmp_path,
            {"runtime.txt": f"path={mixed_path}\n".encode("ascii")},
        )


def test_actual_historical_interpreter_error_shape_is_sanitized(
    tmp_path: Path,
) -> None:
    observed_prefix = r"C:\hostedtoolcache\windows\Python\3.13.12\x64"
    raw = (
        "RuntimeError: Helper must use the exact historical main-job interpreter "
        f"{HISTORICAL_INTERPRETER}; observed {observed_prefix}\\python.exe\n"
    ).encode("ascii")

    receipt, outputs, _receipt_path = _create_bundle(
        tmp_path,
        {"runtime.txt": raw},
        environment_prefix=observed_prefix,
    )

    assert outputs["runtime.txt"].read_bytes() == (
        b"RuntimeError: Helper must use the exact historical main-job interpreter "
        b"__HISTORICAL_INTERPRETER__; observed __PYTHON_PREFIX__\\python.exe\n"
    )
    rules = {
        rule["id"]: rule["match_counts"]["runtime.txt"]
        for rule in receipt["replacement_contract"]["rules"]
    }
    assert rules["historical_interpreter"] == 1
    assert rules["environment_prefix"] == 1


def test_xml_character_reference_encoded_windows_path_is_rejected(
    tmp_path: Path,
) -> None:
    raw_junit = (
        b'<testsuite tests="1" failures="1" errors="0" skipped="0">'
        b'<testcase classname="tests.test_encoded" name="test_path">'
        b'<failure message="C&#58;&#92;Private&#92;result.txt">boom</failure>'
        b"</testcase></testsuite>"
    )

    with pytest.raises(ValueError, match="Windows|absolute path|encoded"):
        _create_bundle(
            tmp_path,
            {"pytest.xml": raw_junit},
            kinds={"pytest.xml": "PYTEST_JUNIT_XML"},
        )


@pytest.mark.parametrize(
    "encoded_path",
    [
        b'{"path":"C\\u003a\\\\Private\\\\result.txt"}',
        b'{"path":"C\\u003a\\u005cPrivate\\u005cresult.txt"}',
    ],
    ids=("unicode-colon-json-slashes", "fully-unicode-escaped-path"),
)
def test_strict_json_decoded_windows_path_is_rejected(
    tmp_path: Path,
    encoded_path: bytes,
) -> None:
    with pytest.raises(ValueError, match="JSON|Windows|absolute path"):
        _create_bundle(
            tmp_path,
            {"diagnostic.json": encoded_path},
            kinds={"diagnostic.json": "STRICT_JSON"},
        )


def test_strict_json_raw_private_path_is_sanitized_before_public_scan(
    tmp_path: Path,
) -> None:
    raw_json = b'{"path":"C:\\\\Python311\\\\Lib\\\\module.pyd"}'
    kinds = {"diagnostic.json": "STRICT_JSON"}
    _receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"diagnostic.json": raw_json},
        kinds=kinds,
    )

    assert json.loads(outputs["diagnostic.json"].read_text("utf-8")) == {
        "path": "__PYTHON_PREFIX__\\Lib\\module.pyd"
    }
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=kinds,
    )["status"] == SANITIZER.PASS_STATUS


def test_strict_json_line_log_preserves_encoded_relative_suffix(
    tmp_path: Path,
) -> None:
    raw_log = _canonical_json(
        {"native_artifact": r"C:\Python311\Lib\site-packages\module.pyd"}
    ) + b"\n"
    kinds = {"environment-preflight.log": "STRICT_JSON"}
    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"environment-preflight.log": raw_log},
        kinds=kinds,
    )

    assert json.loads(
        outputs["environment-preflight.log"].read_text("utf-8")
    ) == {
        "native_artifact": "__PYTHON_PREFIX__\\Lib\\site-packages\\module.pyd"
    }
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=kinds,
    ) == receipt


def test_live_preflight_stdout_is_declared_as_strict_json() -> None:
    workflow = (
        ROOT / ".github/workflows/full-repository-contract.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count(
        '--kind "environment-preflight.log=STRICT_JSON"'
    ) == 2
    assert 'environment-preflight.log=UTF8_TEXT' not in workflow


def test_live_failure_evidence_is_sanitized_even_after_contract_failure() -> None:
    workflow = (
        ROOT / ".github/workflows/full-repository-contract.yml"
    ).read_text(encoding="utf-8")
    step = workflow.split(
        "      - name: Sanitize and reverify live public-checkout evidence\n",
        maxsplit=1,
    )[1].split("\n      - name:", maxsplit=1)[0]

    assert "        if: always()\n" in step
    assert "scripts/sanitize_public_ci_artifacts.py create" in step
    assert "scripts/sanitize_public_ci_artifacts.py verify" in step
    assert "scripts/sanitize_public_checkout_outputs.py" not in step
    for expected in (
        '--input "full_repository.junit.xml=$rawJunit"',
        '--input "full_repository.log=$rawLog"',
        '--output "full_repository.junit.xml=$safeJunit"',
        '--output "full_repository.log=$safeLog"',
        '--kind "full_repository.junit.xml=PYTEST_JUNIT_XML"',
        '--kind "full_repository.log=PYTEST_LOG"',
        "'full_repository.sanitized.junit.xml'",
        "'full_repository.sanitized.log'",
        "'V9R2R1_RAW_OUTPUT_SANITIZATION_RECEIPT.json'",
    ):
        assert expected in step
    create = step.index("scripts/sanitize_public_ci_artifacts.py create")
    verify = step.index("scripts/sanitize_public_ci_artifacts.py verify")
    failure_contract = step.index(
        "scripts/verify_expected_public_checkout_failure_set.py"
    )
    assert create < verify < failure_contract

    checked_in_step = workflow.split(
        "      - name: Verify checked-in sanitized public-checkout evidence\n",
        maxsplit=1,
    )[1].split("\n      - name:", maxsplit=1)[0]
    assert "scripts/sanitize_public_checkout_outputs.py verify" in checked_in_step
    assert (
        "provenance/V9R2R1_RAW_OUTPUT_SANITIZATION_RECEIPT.json"
        in checked_in_step
    )
    assert "scripts/sanitize_public_ci_artifacts.py" not in checked_in_step


def test_junit_host_replacement_is_limited_to_testsuite_hostname(
    tmp_path: Path,
) -> None:
    raw_junit = (
        b'<testsuite tests="1" failures="0" errors="0" skipped="0" '
        b'hostname="CI-HOST">'
        b'<testcase classname="tests.test_host" name="test_only"/>'
        b"</testsuite>"
    )
    kinds = {"pytest.xml": "PYTEST_JUNIT_XML"}
    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"pytest.xml": raw_junit},
        kinds=kinds,
    )

    sanitized = outputs["pytest.xml"].read_bytes()
    assert b'hostname="__HOSTNAME__"' in sanitized
    assert b"CI-HOST" not in sanitized
    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=kinds,
    )["receipt_payload_sha256"] == receipt["receipt_payload_sha256"]


def test_pytest_bundle_accepts_junit_implicit_passed_subtests(
    tmp_path: Path,
) -> None:
    raw_junit = (
        b'<testsuite tests="3" failures="0" errors="0" skipped="0">'
        b'<testcase classname="tests.test_pair" name="test_parent"/>'
        b"</testsuite>"
    )
    raw_log = b"1 passed, 2 subtests passed in 0.10s\n"
    kinds = {
        "run.junit.xml": "PYTEST_JUNIT_XML",
        "run.log": "PYTEST_LOG",
    }
    _receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"run.junit.xml": raw_junit, "run.log": raw_log},
        kinds=kinds,
    )

    assert _verify_bundle(
        receipt_path=receipt_path,
        outputs=outputs,
        kinds=kinds,
    )["status"] == SANITIZER.PASS_STATUS


def test_pytest_bundle_rejects_junit_log_count_mismatch(tmp_path: Path) -> None:
    raw_junit = (
        b'<testsuite tests="3" failures="0" errors="0" skipped="0">'
        b'<testcase classname="tests.test_pair" name="test_parent"/>'
        b"</testsuite>"
    )
    raw_log = b"1 passed, 1 subtests passed in 0.10s\n"

    with pytest.raises(ValueError, match="JUnit.*log|counts drifted"):
        _create_bundle(
            tmp_path,
            {"run.junit.xml": raw_junit, "run.log": raw_log},
            kinds={
                "run.junit.xml": "PYTEST_JUNIT_XML",
                "run.log": "PYTEST_LOG",
            },
        )


def test_pytest_bundle_rejects_failure_node_identity_mismatch(
    tmp_path: Path,
) -> None:
    raw_junit = (
        b'<testsuite tests="1" failures="1" errors="0" skipped="0">'
        b'<testcase classname="tests.test_alpha" name="test_failure">'
        b'<failure message="boom">boom</failure>'
        b"</testcase></testsuite>"
    )
    raw_log = (
        b"FAILED tests/test_beta.py::test_failure - AssertionError: boom\n"
        b"1 failed in 0.10s\n"
    )

    with pytest.raises(ValueError, match="node identities drifted"):
        _create_bundle(
            tmp_path,
            {"run.junit.xml": raw_junit, "run.log": raw_log},
            kinds={
                "run.junit.xml": "PYTEST_JUNIT_XML",
                "run.log": "PYTEST_LOG",
            },
        )


def test_junit_declared_test_count_must_match_testcase_elements(
    tmp_path: Path,
) -> None:
    raw_junit = (
        b'<testsuite tests="2" failures="0" errors="0" skipped="0">'
        b'<testcase classname="tests.test_count" name="test_only"/>'
        b"</testsuite>"
    )

    with pytest.raises(ValueError, match="test|count"):
        _create_bundle(
            tmp_path,
            {"pytest.xml": raw_junit},
            kinds={"pytest.xml": "PYTEST_JUNIT_XML"},
        )


@pytest.mark.parametrize(
    "outcomes",
    [
        b'<failure message="boom"/><skipped message="also skipped"/>',
        b'<failure message="first"/><failure message="second"/>',
    ],
)
def test_junit_rejects_contradictory_or_multiple_terminal_outcomes(
    tmp_path: Path,
    outcomes: bytes,
) -> None:
    failures = b"2" if outcomes.count(b"<failure") == 2 else b"1"
    skipped = b"1" if b"<skipped" in outcomes else b"0"
    raw_junit = (
        b'<testsuite tests="1" failures="'
        + failures
        + b'" errors="0" skipped="'
        + skipped
        + b'">'
        + b'<testcase classname="tests.test_outcome" name="test_only">'
        + outcomes
        + b"</testcase></testsuite>"
    )

    with pytest.raises(ValueError, match="outcome|terminal|failure|skipped"):
        _create_bundle(
            tmp_path,
            {"pytest.xml": raw_junit},
            kinds={"pytest.xml": "PYTEST_JUNIT_XML"},
        )


@pytest.mark.parametrize("logical_name", ["test_secret.log", "CI-HOST.log"])
def test_sensitive_identity_is_rejected_in_logical_name(
    tmp_path: Path,
    logical_name: str,
) -> None:
    with pytest.raises(
        SANITIZER.CIArtifactSanitizationError,
        match="logical artifact name.*sensitive",
    ):
        _create_bundle(tmp_path, {logical_name: b"safe\n"})


def test_mixed_type_logical_names_fail_with_domain_error(tmp_path: Path) -> None:
    receipt, outputs, _receipt_path = _create_bundle(
        tmp_path,
        {"diagnostic.txt": b"safe\n"},
    )
    malformed = copy.deepcopy(receipt)
    malformed["logical_names"] = ["diagnostic.txt", 7]
    malformed_path = tmp_path / "mixed-logical-names.receipt.json"
    _write_rehashed_receipt(malformed_path, malformed)

    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _verify_bundle(receipt_path=malformed_path, outputs=outputs)


@pytest.mark.parametrize("substituted_source", ["generic", "engine"])
def test_create_rejects_substituted_sanitizer_source(
    tmp_path: Path,
    substituted_source: str,
) -> None:
    canonical = GENERIC_SCRIPT if substituted_source == "generic" else ENGINE_SCRIPT
    shadow = tmp_path / canonical.name
    shadow.write_bytes(canonical.read_bytes())
    generic_source = shadow if substituted_source == "generic" else GENERIC_SCRIPT
    engine_source = shadow if substituted_source == "engine" else ENGINE_SCRIPT

    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _create_bundle(
            tmp_path,
            {"diagnostic.txt": b"safe\n"},
            generic_source_path=generic_source,
            engine_source_path=engine_source,
        )


@pytest.mark.parametrize("substituted_source", ["generic", "engine"])
def test_verify_rejects_substituted_sanitizer_source(
    tmp_path: Path,
    substituted_source: str,
) -> None:
    _receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"diagnostic.txt": b"safe\n"},
    )
    canonical = GENERIC_SCRIPT if substituted_source == "generic" else ENGINE_SCRIPT
    shadow = tmp_path / f"shadow-{canonical.name}"
    shadow.write_bytes(canonical.read_bytes())

    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _verify_bundle(
            receipt_path=receipt_path,
            outputs=outputs,
            generic_source_path=(
                shadow if substituted_source == "generic" else GENERIC_SCRIPT
            ),
            engine_source_path=(
                shadow if substituted_source == "engine" else ENGINE_SCRIPT
            ),
        )


def test_verify_binds_expected_reference_checkout(tmp_path: Path) -> None:
    receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"diagnostic.txt": b"safe\n"},
    )
    verified = _verify_bundle(receipt_path=receipt_path, outputs=outputs)
    assert verified["reference_checkout"] == {
        "commit_sha1": REFERENCE_COMMIT,
        "git_tree_sha1": REFERENCE_TREE,
    }

    tampered = copy.deepcopy(receipt)
    tampered["reference_checkout"]["commit_sha1"] = "3" * 40
    tampered_path = tmp_path / "wrong-reference.receipt.json"
    _write_rehashed_receipt(tampered_path, tampered)
    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _verify_bundle(receipt_path=tampered_path, outputs=outputs)


def test_verify_rejects_duplicate_output_paths(tmp_path: Path) -> None:
    _receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"alpha.txt": b"same\n", "beta.txt": b"same\n"},
    )
    duplicate_outputs = {
        "alpha.txt": outputs["alpha.txt"],
        "beta.txt": outputs["alpha.txt"],
    }

    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _verify_bundle(receipt_path=receipt_path, outputs=duplicate_outputs)


def test_verify_rejects_hard_linked_output_paths(tmp_path: Path) -> None:
    _receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"alpha.txt": b"same\n", "beta.txt": b"same\n"},
    )
    outputs["beta.txt"].unlink()
    os.link(outputs["alpha.txt"], outputs["beta.txt"])
    assert os.path.samefile(outputs["alpha.txt"], outputs["beta.txt"])

    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _verify_bundle(receipt_path=receipt_path, outputs=outputs)


def test_verify_rejects_symlinked_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _receipt, outputs, receipt_path = _create_bundle(
        tmp_path,
        {"alpha.txt": b"same\n", "beta.txt": b"same\n"},
    )
    outputs["beta.txt"].unlink()
    try:
        outputs["beta.txt"].symlink_to(outputs["alpha.txt"])
    except OSError:
        target = outputs["beta.txt"].resolve()
        original = SANITIZER._is_link_or_junction

        def simulated_link(path: Path) -> bool:
            return path.resolve() == target or original(path)

        monkeypatch.setattr(SANITIZER, "_is_link_or_junction", simulated_link)

    with pytest.raises(SANITIZER.CIArtifactSanitizationError):
        _verify_bundle(receipt_path=receipt_path, outputs=outputs)
