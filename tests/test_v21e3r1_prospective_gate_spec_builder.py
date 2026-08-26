from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "build_v21e3r1_prospective_gate_spec.py"
)
EVALUATOR = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "evaluate_v21e3r1_prospective_authorization.py"
)
BOUNDARY_FREEZER = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "freeze_v21e3r1_prospective_boundaries.py"
)
CONSUMER_TEST = ROOT / "tests" / "test_v21e3r1_prospective_authorization.py"
COMMON_BINDINGS = (
    "historical_preservation",
    "exact_504_diagnostic",
    "corrected_reanalysis",
    "successor_source_freeze",
    "successor_development_promotion",
    "same_implementation_coverage",
    "baseline_registry",
    "external_algorithm_replay",
    "simultaneous_inference_spec",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fixture_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_consumer_fixture", CONSUMER_TEST
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common_fixture(
    root: Path, **kwargs: object
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    module = _fixture_module()
    return module._make_common_evidence(root, **kwargs)  # type: ignore[attr-defined]


def _builder_command(
    root: Path,
    bindings: dict[str, dict[str, str]],
    *,
    output: Path,
    requested: str = "selection",
    study_id: str = "v21e3r1-prospective-test",
    candidate_id: str = "C1",
    extra: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--evidence-root",
        str(root),
        "--output",
        str(output),
        "--requested-authorization",
        requested,
        "--study-id",
        study_id,
        "--candidate-id",
        candidate_id,
    ]
    for binding_id in COMMON_BINDINGS:
        command.extend(
            [
                "--" + binding_id.replace("_", "-"),
                str(root / bindings[binding_id]["path"]),
            ]
        )
    if extra:
        command.extend(extra)
    return command


def _run_builder(
    root: Path,
    bindings: dict[str, dict[str, str]],
    *,
    output: Path,
    requested: str = "selection",
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _builder_command(
            root,
            bindings,
            output=output,
            requested=requested,
            extra=extra,
        ),
        text=True,
        capture_output=True,
        check=False,
    )


def _replace_cli_value(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def _rewrite_payload_receipt(path: Path, mutate: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("receipt_payload_sha256", None)
    mutate(value)
    value["receipt_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    path.write_bytes(_canonical(value))


def _rewrite_canonical_json(path: Path, mutate: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(_canonical(value))


def test_selection_gate_spec_is_canonical_hash_bound_and_consumable(
    tmp_path: Path,
) -> None:
    bindings, expected_identity = _common_fixture(tmp_path)
    output = tmp_path / "selection.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["binding_count"] == 9
    raw = output.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    assert raw == _canonical(spec)
    assert set(spec) == {"schema", "requested_authorization", "identity", "bindings"}
    assert spec["schema"] == "v21e3r1_prospective_authorization_gate_spec_v3"
    assert spec["requested_authorization"] == "selection"
    assert spec["identity"] == expected_identity
    assert set(spec["bindings"]) == set(COMMON_BINDINGS)
    for binding_id, binding in spec["bindings"].items():
        artifact = tmp_path / binding["path"]
        assert artifact.is_file()
        assert binding["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert "authorized" not in raw.decode("utf-8").casefold()
    assert "authority" not in raw.decode("utf-8").casefold()

    receipt = tmp_path / "selection.authorization.receipt.json"
    evaluated = subprocess.run(
        [
            sys.executable,
            "-I",
            str(EVALUATOR),
            "--gate-spec",
            str(output),
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 2, evaluated.stderr
    evaluated_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert evaluated_receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert evaluated_receipt["selection_authorized"] is False


def test_builder_rejects_inner_only_same_implementation_evidence(
    tmp_path: Path,
) -> None:
    bindings, _ = _common_fixture(
        tmp_path, use_recovery_bound_same_implementation=False
    )
    output = tmp_path / "inner-only.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "recovery-bound" in completed.stderr
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


def test_builder_accepts_valid_statistical_hold_promotion_and_evaluator_holds(
    tmp_path: Path,
) -> None:
    bindings, _identity = _common_fixture(tmp_path, promotion_passed=False)
    output = tmp_path / "promotion-hold.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["binding_count"] == 9
    receipt_path = tmp_path / "promotion-hold.authorization.receipt.json"
    evaluated = subprocess.run(
        [
            sys.executable,
            "-I",
            str(EVALUATOR),
            "--gate-spec",
            str(output),
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(receipt_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 2, evaluated.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["gates"]["successor_development_promotion_gate_passed"] is False
    assert receipt["selection_authorized"] is False


def test_builder_accepts_zero_se_hold_promotion_and_evaluator_holds(
    tmp_path: Path,
) -> None:
    bindings, _identity = _common_fixture(
        tmp_path,
        promotion_passed=False,
        promotion_zero_standard_error=True,
    )
    output = tmp_path / "promotion-zero-se-hold.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 0, completed.stderr
    receipt_path = tmp_path / "promotion-zero-se.authorization.receipt.json"
    evaluated = subprocess.run(
        [
            sys.executable,
            "-I",
            str(EVALUATOR),
            "--gate-spec",
            str(output),
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(receipt_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 2, evaluated.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_PROSPECTIVE_AUTHORIZATION_PREREQUISITES"
    assert receipt["gates"]["successor_development_promotion_gate_passed"] is False


def test_builder_rejects_upstream_integrity_hold_as_ninth_common_binding(
    tmp_path: Path,
) -> None:
    bindings, _identity = _common_fixture(tmp_path)
    promotion_path = tmp_path / bindings["successor_development_promotion"]["path"]
    integrity_core = {
        "schema": "v21e3r1_successor_development_factorial_evaluation_integrity_hold_v1",
        "status": "HOLD_INTEGRITY_ERROR",
        "phase": "development",
        "promotion_scope": "SUCCESSOR_DEVELOPMENT_PROMOTION_ONLY_HASH_BOUND_PRODUCER_RECEIPT_NO_PROSPECTIVE_108_ROW_RECOMPUTATION_NO_SCIENTIFIC_CLAIM",
        "matrix_directory": "successor-factorial",
        "error": "synthetic upstream integrity failure",
        "integrity_bindings_validated": False,
        "development_promotion_gate_passed": False,
        "selection_cases_materialized": False,
        "confirmation_cases_materialized": False,
        "formal_cases_materialized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_study_authorized": False,
        "scientific_claim_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD",
    }
    integrity_receipt = dict(integrity_core)
    integrity_receipt["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(integrity_core)
    ).hexdigest()
    promotion_path.write_bytes(_canonical(integrity_receipt) + b"\n")
    output = tmp_path / "promotion-integrity-hold.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


@pytest.mark.parametrize(
    "field",
    [
        "successor_source_sha256",
        "successor_config_sha256",
        "source_freeze_receipt_sha256",
    ],
)
def test_builder_rejects_promotion_cross_identity_drift_before_output(
    tmp_path: Path, field: str
) -> None:
    bindings, _identity = _common_fixture(tmp_path)
    promotion_path = tmp_path / bindings["successor_development_promotion"]["path"]
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion.pop("receipt_payload_sha256")
    promotion[field] = "e" * 64
    promotion["receipt_payload_sha256"] = hashlib.sha256(
        _canonical(promotion)
    ).hexdigest()
    promotion_path.write_bytes(_canonical(promotion) + b"\n")
    output = tmp_path / f"promotion-{field}-drift.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert field in completed.stderr


def test_builder_rejects_promotion_payload_tampering_before_output(
    tmp_path: Path,
) -> None:
    bindings, _identity = _common_fixture(tmp_path)
    promotion_path = tmp_path / bindings["successor_development_promotion"]["path"]
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["matrix_row_count"] = 107
    promotion_path.write_bytes(_canonical(promotion) + b"\n")
    output = tmp_path / "promotion-payload-tamper.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "payload digest drifted" in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    ["parent_plan_sha256", "row_policy", "input_binding", "aggregate_schema"],
)
def test_builder_rejects_hash_consistent_wrong_successor_factorial_design(
    tmp_path: Path, mutation: str
) -> None:
    module = _fixture_module()
    bindings, _identity = module._make_common_evidence(tmp_path)  # type: ignore[attr-defined]
    module._mutate_reseal_factorial_design(  # type: ignore[attr-defined]
        tmp_path, bindings, mutation=mutation
    )
    output = tmp_path / f"wrong-factorial-{mutation}.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3, completed.stderr
    assert not output.exists()
    assert "HOLD_INTEGRITY_ERROR" in completed.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    (("bootstrap_samples", 9999), ("bootstrap_seed", 20260824)),
)
def test_builder_rejects_resealed_nonexact_simultaneous_bootstrap_contract(
    tmp_path: Path, field: str, value: int
) -> None:
    module = _fixture_module()
    bindings, identity = module._make_common_evidence(tmp_path)  # type: ignore[attr-defined]
    module._mutate_bound_simultaneous_spec(  # type: ignore[attr-defined]
        tmp_path,
        bindings,
        identity,
        lambda spec: spec.__setitem__(field, value),
    )
    output = tmp_path / f"simultaneous-{field}-drift.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert field in completed.stderr


def test_declared_nested_hash_tampering_is_rejected_before_output(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    source_receipt = tmp_path / bindings["successor_source_freeze"]["path"]
    _rewrite_payload_receipt(
        source_receipt,
        lambda value: value.__setitem__("semantic_config_sha256", "f" * 64),
    )
    output = tmp_path / "tampered-hash.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "semantic_config_sha256" in completed.stderr


def test_cross_receipt_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    baseline = tmp_path / bindings["baseline_registry"]["path"]
    _rewrite_canonical_json(
        baseline, lambda value: value.__setitem__("candidate_id", "C2")
    )
    output = tmp_path / "cross-identity.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "baseline.candidate_id" in completed.stderr


def test_confirmation_adds_one_identity_bound_selection_result(tmp_path: Path) -> None:
    module = _fixture_module()
    bindings, identity = module._make_common_evidence(tmp_path)  # type: ignore[attr-defined]
    selection = module._write_phase_result(  # type: ignore[attr-defined]
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    output = tmp_path / "confirmation.gate-spec.json"

    completed = _run_builder(
        tmp_path,
        bindings,
        output=output,
        requested="confirmation",
        extra=["--selection-result", str(tmp_path / selection["path"])],
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["binding_count"] == 10
    spec = json.loads(output.read_text(encoding="utf-8"))
    assert set(spec["bindings"]) == {*COMMON_BINDINGS, "selection_result"}
    assert spec["bindings"]["selection_result"]["sha256"] == selection["sha256"]

    receipt = tmp_path / "confirmation.authorization.receipt.json"
    evaluated = subprocess.run(
        [
            sys.executable,
            "-I",
            str(EVALUATOR),
            "--gate-spec",
            str(output),
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 2, evaluated.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))[
        "confirmation_authorized"
    ] is False


def test_confirmation_rejects_selection_result_identity_drift(tmp_path: Path) -> None:
    module = _fixture_module()
    bindings, identity = module._make_common_evidence(tmp_path)  # type: ignore[attr-defined]
    selection = module._write_phase_result(  # type: ignore[attr-defined]
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    selection_path = tmp_path / selection["path"]
    _rewrite_payload_receipt(
        selection_path,
        lambda value: value.__setitem__("selected_candidate", "C2"),
    )
    output = tmp_path / "drifted-selection.gate-spec.json"

    completed = _run_builder(
        tmp_path,
        bindings,
        output=output,
        requested="confirmation",
        extra=["--selection-result", str(selection_path)],
    )

    assert completed.returncode == 3
    assert not output.exists()
    assert "selection_result.selected_candidate" in completed.stderr


def test_formal_request_adds_identity_bound_selection_and_confirmation_results(
    tmp_path: Path,
) -> None:
    module = _fixture_module()
    bindings, identity = module._make_common_evidence(tmp_path)  # type: ignore[attr-defined]
    selection = module._write_phase_result(  # type: ignore[attr-defined]
        tmp_path,
        phase="selection",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
    )
    confirmation = module._write_phase_result(  # type: ignore[attr-defined]
        tmp_path,
        phase="confirmation",
        identity=identity,
        source_freeze_sha256=bindings["successor_source_freeze"]["sha256"],
        selection_receipt_sha256=selection["sha256"],
        external_replay_sha256=bindings["external_algorithm_replay"]["sha256"],
        custody_receipt_sha256=module.HEX["custody"],  # type: ignore[attr-defined]
    )
    output = tmp_path / "formal.gate-spec.json"

    completed = _run_builder(
        tmp_path,
        bindings,
        output=output,
        requested="formal_input_materialization",
        extra=[
            "--selection-result",
            str(tmp_path / selection["path"]),
            "--confirmation-result",
            str(tmp_path / confirmation["path"]),
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["binding_count"] == 11
    spec = json.loads(output.read_text(encoding="utf-8"))
    assert set(spec["bindings"]) == {
        *COMMON_BINDINGS,
        "selection_result",
        "confirmation_result",
    }
    receipt = tmp_path / "formal.authorization.receipt.json"
    evaluated = subprocess.run(
        [
            sys.executable,
            "-I",
            str(EVALUATOR),
            "--gate-spec",
            str(output),
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 2, evaluated.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))[
        "formal_input_materialization_authorized"
    ] is False


def test_binding_path_with_lexical_traversal_is_rejected(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    output = tmp_path / "traversal.gate-spec.json"
    command = _builder_command(tmp_path, bindings, output=output)
    traversal = tmp_path / "unused" / ".." / bindings["historical_preservation"]["path"]
    _replace_cli_value(command, "--historical-preservation", str(traversal))

    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 3
    assert not output.exists()
    assert "traversal segment" in completed.stderr


def test_symbolic_link_binding_is_rejected(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    target = tmp_path / bindings["historical_preservation"]["path"]
    linked_directory = tmp_path / "linked-evidence"
    cleanup_link = False
    if os.name == "nt":
        real_directory = tmp_path / "real-linked-evidence"
        real_directory.mkdir()
        (real_directory / target.name).write_bytes(target.read_bytes())
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_directory), str(real_directory)],
            text=True,
            capture_output=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip(f"junctions unavailable in this environment: {created.stderr}")
        cleanup_link = True
        linked = linked_directory / target.name
    else:
        try:
            os.symlink(target.parent, linked_directory, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symbolic links unavailable in this environment: {error}")
        cleanup_link = True
        linked = linked_directory / target.name
    output = tmp_path / "linked.gate-spec.json"
    command = _builder_command(tmp_path, bindings, output=output)
    _replace_cli_value(command, "--historical-preservation", str(linked))

    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
    finally:
        if cleanup_link:
            os.rmdir(linked_directory)

    assert completed.returncode == 3
    assert not output.exists()
    assert "symbolic link" in completed.stderr or "reparse point" in completed.stderr


def test_noncanonical_bound_receipt_is_rejected(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    historical = tmp_path / bindings["historical_preservation"]["path"]
    value = json.loads(historical.read_text(encoding="utf-8"))
    historical.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "noncanonical.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "is not canonical JSON" in completed.stderr


def test_unpaired_surrogate_in_bound_json_is_rejected_safely(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    historical = tmp_path / bindings["historical_preservation"]["path"]
    historical.write_bytes(
        b'{"schema":"v21e3r1_v4_v6_historical_preservation_receipt_v1",'
        b'"unpaired":"\\ud800"}'
    )
    output = tmp_path / "surrogate.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    error = json.loads(completed.stderr)
    assert error["status"] == "HOLD_INTEGRITY_ERROR"
    assert "surrogates" in error["error"] or "encoding" in error["error"]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    output = tmp_path / "existing.gate-spec.json"
    original = b"preexisting-custody-bytes"
    output.write_bytes(original)

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert output.read_bytes() == original
    assert "exclusive create" in completed.stderr


@pytest.mark.parametrize(
    ("requested", "extra"),
    [
        ("selection", ["--selection-result", "historical.json"]),
        ("confirmation", []),
        (
            "confirmation",
            [
                "--selection-result",
                "historical.json",
                "--confirmation-result",
                "diagnostic.json",
            ],
        ),
        (
            "formal_input_materialization",
            ["--selection-result", "historical.json"],
        ),
    ],
)
def test_phase_specific_binding_set_is_fail_closed(
    tmp_path: Path, requested: str, extra: list[str]
) -> None:
    bindings, _ = _common_fixture(tmp_path)
    output = tmp_path / f"{requested}.invalid-phase-set.json"

    completed = _run_builder(
        tmp_path,
        bindings,
        output=output,
        requested=requested,
        extra=extra,
    )

    assert completed.returncode == 3
    assert not output.exists()


def test_real_boundary_receipts_build_a_consumable_v3_hold_spec(tmp_path: Path) -> None:
    bindings, original_identity = _common_fixture(tmp_path)
    frozen = tmp_path / "prospective-boundaries"
    freeze = subprocess.run(
        [
            sys.executable,
            "-I",
            str(BOUNDARY_FREEZER),
            "--repository-root",
            str(ROOT),
            "--output-directory",
            str(frozen),
            "--study-id",
            original_identity["study_id"],
            "--candidate-id",
            original_identity["candidate_id"],
            "--successor-source-sha256",
            original_identity["successor_source_sha256"],
            "--successor-config-sha256",
            original_identity["successor_config_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert freeze.returncode == 0, freeze.stderr
    bindings["baseline_registry"]["path"] = (
        frozen / "baseline-registry.receipt.json"
    ).relative_to(tmp_path).as_posix()
    bindings["external_algorithm_replay"]["path"] = (
        frozen / "external-algorithm-replay.receipt.json"
    ).relative_to(tmp_path).as_posix()
    metric_raw = (frozen / "study.metric-spec.json").read_bytes()
    sim_raw = (frozen / "simultaneous-inference.spec.json").read_bytes()
    metric_path = tmp_path / "successor-freeze/study.metric-spec.json"
    sim_path = tmp_path / "successor-freeze/simultaneous-inference.spec.json"
    metric_path.write_bytes(metric_raw)
    sim_path.write_bytes(sim_raw)
    metric_sha = hashlib.sha256(metric_raw).hexdigest()
    sim_sha = hashlib.sha256(sim_raw).hexdigest()
    source_receipt = tmp_path / bindings["successor_source_freeze"]["path"]

    def bind_frozen_specs(value: dict[str, object]) -> None:
        value["study_metric_spec_sha256"] = metric_sha
        value["simultaneous_inference_spec_sha256"] = sim_sha

    _rewrite_payload_receipt(source_receipt, bind_frozen_specs)
    bindings["successor_source_freeze"]["sha256"] = hashlib.sha256(
        source_receipt.read_bytes()
    ).hexdigest()
    bindings["simultaneous_inference_spec"]["path"] = (
        frozen / "simultaneous-inference.spec.json"
    ).relative_to(
        tmp_path
    ).as_posix()
    updated_identity = dict(original_identity)
    updated_identity["study_metric_spec_sha256"] = metric_sha
    updated_identity["simultaneous_inference_spec_sha256"] = sim_sha
    module = _fixture_module()
    bindings["successor_development_promotion"] = module._write_development_promotion(  # type: ignore[attr-defined]
        tmp_path,
        identity=updated_identity,
        source_freeze=bindings["successor_source_freeze"],
        passed=True,
    )
    output = tmp_path / "v3-selection.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 0, completed.stderr
    spec = json.loads(output.read_text(encoding="utf-8"))
    assert spec["identity"]["study_metric_spec_sha256"] == metric_sha
    assert spec["identity"]["simultaneous_inference_spec_sha256"] == sim_sha
    receipt = tmp_path / "v3-selection.authorization.receipt.json"
    evaluated = subprocess.run(
        [
            sys.executable,
            "-I",
            str(EVALUATOR),
            "--gate-spec",
            str(output),
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 2, evaluated.stderr
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["baseline_eligible_count_by_family"] == {"MOKP": 0, "MOTSP": 0}
    assert result["selection_authorized"] is False


def test_cli_never_accepts_self_declared_authority_booleans(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    output = tmp_path / "self-authorized.gate-spec.json"
    command = _builder_command(tmp_path, bindings, output=output)
    command.extend(["--selection-authorized", "true"])

    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )

    assert completed.returncode != 0
    assert not output.exists()
    assert "unrecognized arguments" in completed.stderr


def test_successor_source_identity_must_be_recomputed_from_manifest_entries(
    tmp_path: Path,
) -> None:
    bindings, _ = _common_fixture(tmp_path)
    forged_root = "f" * 64
    manifest_path = tmp_path / "successor-freeze/source.manifest.json"
    _rewrite_canonical_json(
        manifest_path,
        lambda value: value.__setitem__("source_root_sha256", forged_root),
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sim_path = tmp_path / bindings["simultaneous_inference_spec"]["path"]

    def forge_sim(value: dict[str, object]) -> None:
        value["successor_source_sha256"] = forged_root

    _rewrite_payload_receipt(sim_path, forge_sim)
    sim_sha = hashlib.sha256(sim_path.read_bytes()).hexdigest()
    source_receipt = tmp_path / bindings["successor_source_freeze"]["path"]

    def forge_source(value: dict[str, object]) -> None:
        value["source_snapshot_sha256"] = forged_root
        value["source_manifest_sha256"] = manifest_sha
        value["simultaneous_inference_spec_sha256"] = sim_sha

    _rewrite_payload_receipt(source_receipt, forge_source)
    external = tmp_path / bindings["external_algorithm_replay"]["path"]
    _rewrite_canonical_json(
        external,
        lambda value: value.__setitem__("successor_source_sha256", forged_root),
    )
    output = tmp_path / "forged-source-root.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "canonical source-root" in completed.stderr


def test_v1_v2_metric_alias_hybrid_is_rejected(tmp_path: Path) -> None:
    bindings, _ = _common_fixture(tmp_path)
    baseline = tmp_path / bindings["baseline_registry"]["path"]

    def add_v2_alias(value: dict[str, object]) -> None:
        value["study_metric_spec_sha256"] = value["metric_spec_sha256"]

    _rewrite_canonical_json(baseline, add_v2_alias)
    output = tmp_path / "hybrid-alias.gate-spec.json"

    completed = _run_builder(tmp_path, bindings, output=output)

    assert completed.returncode == 3
    assert not output.exists()
    assert "hybrid metric identity aliases" in completed.stderr

