from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "independent_reproduction"
    / "recompute_v21e3r1_simultaneous_bounds.py"
)
FAMILIES = ("MOKP", "MOTSP")
METHOD = "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
SELECTION_HYPOTHESES = tuple(
    f"{family}:{candidate}-{reference}"
    for family in FAMILIES
    for candidate, reference in (
        ("C1", "C0"),
        ("C2", "C0"),
        ("C2", "C1"),
        ("C3", "C0"),
        ("C3", "C2"),
    )
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_input(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _selection_payload() -> dict[str, object]:
    case_ids_by_family = {
        family: [f"{family.lower()}-selection-{index:02d}" for index in range(8)]
        for family in FAMILIES
    }
    seeds = [17, 29]
    rows: list[dict[str, object]] = []
    base_effects = {"C0": 0.0, "C1": 0.30, "C2": 0.65, "C3": 1.05}
    slopes = {"C0": 0.0, "C1": 0.012, "C2": 0.017, "C3": 0.020}
    for family_index, family in enumerate(FAMILIES):
        for case_index, case_id in enumerate(case_ids_by_family[family]):
            centered_case = case_index - 3.5
            for seed_index, seed in enumerate(seeds):
                baseline = 2.0 + family_index * 0.2 + case_index * 0.03
                baseline += seed_index * 0.001
                for candidate in ("C0", "C1", "C2", "C3"):
                    rows.append(
                        {
                            "family": family,
                            "case_id": case_id,
                            "seed": seed,
                            "candidate": candidate,
                            "score": baseline
                            + base_effects[candidate]
                            + slopes[candidate] * centered_case,
                        }
                    )
    return {
        "schema": "v21e3r1_simultaneous_evaluation_input_v1",
        "phase": "selection",
        "study_id": "v21e3r1-prospective-fixture",
        "study_freeze_sha256": "1" * 64,
        "phase_manifest_sha256": "2" * 64,
        "matrix_receipt_sha256": "3" * 64,
        "source_root_sha256": "4" * 64,
        "metric_spec_sha256": "5" * 64,
        "decision_spec_sha256": "6" * 64,
        "effect_direction": "larger_is_better",
        "case_ids_by_family": case_ids_by_family,
        "seeds": seeds,
        "inference": {
            "method": METHOD,
            "alpha": 0.05,
            "bootstrap_samples": 199,
            "bootstrap_seed": 20260823,
        },
        "thresholds": {"primary": 0.0, "adjacent": 0.005},
        "selection_binding": None,
        "confirmation_controls": None,
        "rows": rows,
    }


def _run(
    payload: dict[str, object], tmp_path: Path
) -> tuple[subprocess.CompletedProcess[str], Path]:
    source = tmp_path / "matrix input.json"
    output = tmp_path / "simultaneous receipt.json"
    _write_input(source, payload)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, output


def _run_raw(
    raw: bytes, tmp_path: Path
) -> tuple[subprocess.CompletedProcess[str], Path]:
    source = tmp_path / "raw matrix input.json"
    output = tmp_path / "raw simultaneous receipt.json"
    source.write_bytes(raw)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, output


def _confirmation_payload(selected_candidate: str) -> dict[str, object]:
    payload = _selection_payload()
    predecessor = {"C1": "C0", "C2": "C1", "C3": "C2"}[
        selected_candidate
    ]
    required_candidates = {"C0", selected_candidate, predecessor}
    payload["phase"] = "confirmation"
    payload["phase_manifest_sha256"] = "a" * 64
    payload["matrix_receipt_sha256"] = "b" * 64
    payload["selection_binding"] = {
        "selection_receipt_sha256": "c" * 64,
        "selection_status": "PASS_SELECTION",
        "selected_candidate": selected_candidate,
    }
    payload["confirmation_controls"] = {
        "external_producer": True,
        "external_producer_receipt_sha256": "d" * 64,
        "independent_custody": True,
        "custody_receipt_sha256": "e" * 64,
        "independent_statistics": True,
        "statistics_source_sha256": _file_sha256(SCRIPT),
    }
    payload["rows"] = [
        row
        for row in payload["rows"]
        if row["candidate"] in required_candidates
    ]
    return payload


def test_selection_jointly_certifies_fixed_ten_cells_and_selects_c3(
    tmp_path: Path,
) -> None:
    completed, output = _run(_selection_payload(), tmp_path)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_SELECTION"
    assert receipt["selected_candidate"] == "C3"
    assert receipt["reached_candidates"] == ["C1", "C2", "C3"]
    assert receipt["not_reached_candidates"] == []
    assert receipt["hypothesis_order"] == list(SELECTION_HYPOTHESES)
    assert len(receipt["cells"]) == 10
    assert receipt["matrix_row_count"] == 128
    assert receipt["expected_matrix_row_count"] == 128
    assert receipt["simultaneous_coverage_certified"] is True
    assert receipt["inference"]["familywise_scope"] == "JOINT_ACROSS_BOTH_FAMILIES"
    assert receipt["inference"]["method"] == METHOD
    assert receipt["inference"]["cluster_unit"] == "PAIRED_CASE"
    assert receipt["inference"]["seed_aggregation"] == "MEAN_WITHIN_CASE_ARM"
    assert receipt["inference"]["case_resampling"] == (
        "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_SHARED_ACROSS_CELLS_WITHIN_FAMILY"
    )
    assert receipt["inference"]["quantile_convention"] == (
        "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC"
    )
    assert receipt["inference"]["critical_value_floor"] == 0.0
    assert math.isfinite(receipt["inference"]["critical_value"])
    assert receipt["inference"]["critical_value"] == pytest.approx(
        1.732050807568932,
        rel=0.0,
        abs=1e-15,
    )
    assert receipt["inference"]["bootstrap_maxima_sha256"] == (
        "11095c0c7cd0f101bfd60fb7c5c0cfd7f972f060b5e3002e37cf8539dadf62ed"
    )
    assert receipt["source_sha256"] == _file_sha256(SCRIPT)
    receipt_core = dict(receipt)
    embedded_hash = receipt_core.pop("receipt_payload_sha256")
    assert embedded_hash == _sha256_payload(receipt_core)


def test_confirmation_c2_uses_four_cells_and_binds_but_does_not_claim_independence(
    tmp_path: Path,
) -> None:
    completed, output = _run(_confirmation_payload("C2"), tmp_path)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_CONFIRMATION"
    assert receipt["selected_candidate"] == "C2"
    assert receipt["candidate_order"] == ["C0", "C1", "C2"]
    assert receipt["hypothesis_order"] == [
        "MOKP:C2-C0",
        "MOKP:C2-C1",
        "MOTSP:C2-C0",
        "MOTSP:C2-C1",
    ]
    assert len(receipt["cells"]) == 4
    assert receipt["confirmation_control_bindings_validated"] is True
    assert receipt["confirmation_control_bindings"]["external_producer"] is True
    assert receipt["confirmation_control_bindings"]["independent_custody"] is True
    assert receipt["confirmation_control_bindings"]["independent_statistics"] is True
    assert receipt["confirmation_control_bindings_scope"] == (
        "INPUT_DECLARATIONS_AND_HASH_BINDINGS_ONLY_NOT_AUTHENTICATION"
    )
    assert receipt["external_independence_claim_authorized"] is False
    assert receipt["scientific_independence"] is False
    assert receipt["formal_authority"] is False


@pytest.mark.parametrize(
    ("candidate", "candidate_order", "suffixes"),
    (
        ("C1", ["C0", "C1"], ["C1-C0"]),
        ("C3", ["C0", "C2", "C3"], ["C3-C0", "C3-C2"]),
    ),
)
def test_confirmation_hypothesis_family_is_fixed_by_selected_candidate(
    tmp_path: Path,
    candidate: str,
    candidate_order: list[str],
    suffixes: list[str],
) -> None:
    completed, output = _run(_confirmation_payload(candidate), tmp_path)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["candidate_order"] == candidate_order
    assert receipt["hypothesis_order"] == [
        f"{family}:{suffix}" for family in FAMILIES for suffix in suffixes
    ]


def test_zero_observed_case_cluster_se_writes_hold_receipt(tmp_path: Path) -> None:
    payload = _selection_payload()
    case_index = {
        case_id: index
        for family_cases in payload["case_ids_by_family"].values()
        for index, case_id in enumerate(family_cases)
    }
    constant_effect = {"C0": 0, "C1": 10, "C2": 20, "C3": 30}
    for row in payload["rows"]:
        baseline = 100 * FAMILIES.index(row["family"])
        baseline += 10 * case_index[row["case_id"]]
        baseline += payload["seeds"].index(row["seed"])
        row["score"] = baseline + constant_effect[row["candidate"]]

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_ZERO_STANDARD_ERROR"
    assert receipt["simultaneous_coverage_certified"] is False
    assert receipt["zero_standard_error_hypotheses"] == list(SELECTION_HYPOTHESES)
    assert receipt["external_independence_claim_authorized"] is False


def test_selection_cannot_skip_a_failed_c2_to_reach_strong_c3(
    tmp_path: Path,
) -> None:
    payload = _selection_payload()
    baseline = {
        (row["family"], row["case_id"], row["seed"]): row["score"]
        for row in payload["rows"]
        if row["candidate"] == "C0"
    }
    case_position = {
        case_id: position
        for family_cases in payload["case_ids_by_family"].values()
        for position, case_id in enumerate(family_cases)
    }
    for row in payload["rows"]:
        if row["candidate"] == "C2":
            key = (row["family"], row["case_id"], row["seed"])
            row["score"] = baseline[key] + 0.005 + 0.001 * case_position[row["case_id"]]

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_SELECTION"
    assert receipt["selected_candidate"] == "C1"
    assert receipt["reached_candidates"] == ["C1", "C2"]
    assert receipt["not_reached_candidates"] == ["C3"]
    assert receipt["blocked_candidate"] == "C2"
    assert any(reason.endswith(":adjacent_lower_bound") for reason in receipt["gate_reasons"])


def test_c1_minus_c0_cell_serves_both_primary_and_adjacent_thresholds(
    tmp_path: Path,
) -> None:
    known_dir = tmp_path / "known-c1-bound"
    known_dir.mkdir()
    completed, output = _run(_selection_payload(), known_dir)
    assert completed.returncode == 0, completed.stderr
    known = json.loads(output.read_text(encoding="utf-8"))
    lower_bound = next(
        cell["simultaneous_lower_bound"]
        for cell in known["cells"]
        if cell["hypothesis_id"] == "MOKP:C1-C0"
    )

    payload = _selection_payload()
    shift = 0.0025 - lower_bound
    for row in payload["rows"]:
        if row["candidate"] == "C1":
            row["score"] += shift
    evaluation_dir = tmp_path / "c1-between-thresholds"
    evaluation_dir.mkdir()

    completed, output = _run(payload, evaluation_dir)

    assert completed.returncode == 2
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "STOP_SELECTION_NO_CANDIDATE"
    assert receipt["selected_candidate"] == "C0"
    assert receipt["reached_candidates"] == ["C1"]
    assert receipt["not_reached_candidates"] == ["C2", "C3"]
    assert "MOKP:adjacent_lower_bound" in receipt["gate_reasons"]
    assert "MOTSP:adjacent_lower_bound" in receipt["gate_reasons"]
    assert "MOKP:primary_lower_bound" not in receipt["gate_reasons"]
    assert "MOTSP:primary_lower_bound" not in receipt["gate_reasons"]


@pytest.mark.parametrize("value", (0.0, 0.004, 0.006, 0.01, -0.005))
def test_practical_threshold_is_exactly_frozen_at_point_zero_zero_five(
    tmp_path: Path,
    value: float,
) -> None:
    payload = _selection_payload()
    payload["thresholds"]["adjacent"] = value

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert "frozen value 0.005" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_legacy_ambiguous_method_identifier_is_rejected(tmp_path: Path) -> None:
    payload = _selection_payload()
    payload["inference"]["method"] = (
        "one_sided_max_centered_paired_case_bootstrap_v1"
    )

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert METHOD in json.loads(completed.stderr)["error"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("external_producer", False),
        ("independent_custody", False),
        ("independent_statistics", False),
        ("external_producer", 1),
        ("independent_statistics", "true"),
    ),
)
def test_confirmation_requires_three_exact_true_independence_bindings(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _confirmation_payload("C2")
    payload["confirmation_controls"][field] = value

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert "must be exact true" in json.loads(completed.stderr)["error"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"phase":"selection","phase":"selection"}\n', "duplicate JSON key"),
        (b'{"score":NaN}\n', "non-finite JSON constant"),
        (b'{"score":Infinity}\n', "non-finite JSON constant"),
    ),
)
def test_strict_json_rejects_duplicates_and_nonfinite_constants_without_receipt(
    tmp_path: Path,
    raw: bytes,
    message: str,
) -> None:
    completed, output = _run_raw(raw, tmp_path)

    assert completed.returncode == 3
    assert message in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_strict_json_rejects_lone_unicode_surrogate_without_receipt(
    tmp_path: Path,
) -> None:
    raw = _canonical_bytes(_selection_payload())
    raw = raw.replace(
        b'"study_id":"v21e3r1-prospective-fixture"',
        b'"study_id":"\\ud800"',
    )

    completed, output = _run_raw(raw + b"\n", tmp_path)

    assert completed.returncode == 3
    assert "Unicode scalar" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_finite_row_scores_that_overflow_aggregation_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _selection_payload()
    for row in payload["rows"]:
        row["score"] = 1e308 if row["candidate"] == "C0" else -1e308

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert "numeric aggregation" in json.loads(completed.stderr)["error"]
    assert not output.exists()


@pytest.mark.parametrize(
    "target",
    ("seed", "score", "bootstrap_samples", "primary_threshold"),
)
def test_boolean_values_never_pass_exact_numeric_fields(
    tmp_path: Path,
    target: str,
) -> None:
    payload = _selection_payload()
    if target == "seed":
        payload["seeds"][0] = True
    elif target == "score":
        payload["rows"][0]["score"] = False
    elif target == "bootstrap_samples":
        payload["inference"]["bootstrap_samples"] = True
    else:
        payload["thresholds"]["primary"] = False

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert "exact" in json.loads(completed.stderr)["error"]
    assert not output.exists()


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_case_by_seed_by_candidate_coverage_is_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _selection_payload()
    if mutation == "missing":
        payload["rows"].pop()
    else:
        payload["rows"].append(dict(payload["rows"][0]))

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    error = json.loads(completed.stderr)["error"]
    assert (
        "full coverage" in error
        if mutation == "missing"
        else "duplicate case x seed x candidate row" in error
    )
    assert not output.exists()


def test_unknown_row_field_is_rejected_before_statistics(tmp_path: Path) -> None:
    payload = _selection_payload()
    payload["rows"][0]["unfrozen_extra"] = 1

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert "exact frozen key set" in json.loads(completed.stderr)["error"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("selection_status", "selection_status=PASS_SELECTION"),
        ("selected_candidate", "must be one of C1, C2, C3"),
        ("statistics_source", "does not bind this independent evaluator source"),
        ("reused_evidence_hash", "must be pairwise distinct"),
    ),
)
def test_confirmation_bindings_fail_closed_before_statistics(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payload = _confirmation_payload("C2")
    if mutation == "selection_status":
        payload["selection_binding"]["selection_status"] = "PASS"
    elif mutation == "selected_candidate":
        payload["selection_binding"]["selected_candidate"] = "C4"
    elif mutation == "statistics_source":
        payload["confirmation_controls"]["statistics_source_sha256"] = "f" * 64
    else:
        payload["confirmation_controls"]["custody_receipt_sha256"] = (
            payload["confirmation_controls"]["external_producer_receipt_sha256"]
        )

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 3
    assert message in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_receipt_is_exclusive_create_and_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "matrix.json"
    output = tmp_path / "existing receipt.json"
    _write_input(source, _selection_payload())
    output.write_bytes(b"preexisting-custody-bytes")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "exclusive create required" in json.loads(completed.stderr)["error"]
    assert output.read_bytes() == b"preexisting-custody-bytes"


def test_independent_evaluator_imports_only_python_standard_library() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "mo_nco" not in imported_roots
    assert imported_roots <= set(sys.stdlib_module_names) | {"__future__"}


def test_confirmation_effect_failure_writes_fail_receipt_without_authority(
    tmp_path: Path,
) -> None:
    payload = _confirmation_payload("C2")
    baseline = {
        (row["family"], row["case_id"], row["seed"]): row["score"]
        for row in payload["rows"]
        if row["candidate"] == "C0"
    }
    case_position = {
        case_id: position
        for family_cases in payload["case_ids_by_family"].values()
        for position, case_id in enumerate(family_cases)
    }
    for row in payload["rows"]:
        if row["candidate"] == "C2":
            key = (row["family"], row["case_id"], row["seed"])
            row["score"] = baseline[key] + 0.005 + 0.001 * case_position[row["case_id"]]

    completed, output = _run(payload, tmp_path)

    assert completed.returncode == 2
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL_CONFIRMATION"
    assert receipt["simultaneous_coverage_certified"] is True
    assert receipt["confirmation_control_bindings_validated"] is True
    assert receipt["external_independence_claim_authorized"] is False
    assert receipt["formal_authority"] is False

