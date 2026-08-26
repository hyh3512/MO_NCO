from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mo_nco import pareto_v21e3r1_v9_runner as runner_module
from mo_nco.pareto_v21e3_hybrid import V21E3ResourceLimitExceeded
from mo_nco.pareto_v21e3r1_v9_runner import (
    main,
    run_v9r1_development_case,
)


MOKP_CASE = Path(
    "ijoc_submission_v21e3/development_partitions_v1/instances/"
    "v21e3-mokp-development-n100-s00.json"
)
MOTSP_CASE = Path(
    "ijoc_submission_v21e3/development_partitions_v1/instances/"
    "v21e3-motsp-development-n100-s00.json"
)

_EXPECTED_IDS = {
    "MOKP": {
        "LEGACY": "V21E3R1_V9_LEGACY_MOKP",
        "SCREEN": "V21E3R1_V9_INFORMATION_SCREEN_MOKP",
        "LYAP": "V21E3R1_V9_LYAPUNOV_MOKP",
        "BOTH": "V21E3R1_V9_INFORMATION_LYAPUNOV_MOKP",
    },
    "MOTSP": {
        "LEGACY": "V21E3R1_V9_LEGACY_MOTSP",
        "SCREEN": "V21E3R1_V9_INFORMATION_SCREEN_MOTSP",
        "LYAP": "V21E3R1_V9_LYAPUNOV_MOTSP",
        "BOTH": "V21E3R1_V9_INFORMATION_LYAPUNOV_MOTSP",
    },
}


def _arguments(case: Path, output: Path) -> list[str]:
    return [
        "--case",
        str(case),
        "--outdir",
        str(output),
        "--seed",
        "1701",
        "--directions",
        "[[0.25,0.75],[0.75,0.25]]",
        "--charged-evaluations",
        "8",
        "--attempt-cap",
        "256",
        "--structural-screening-cap",
        "50000",
        "--wall-time-cap-seconds",
        "60",
        "--candidate-screening-cap",
        "4",
        "--archive-tradeoff-lambda",
        "0.5",
        "--checkpoint-period",
        "4",
    ]


def test_runner_refuses_to_run_without_exposed_development_acknowledgement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v9r1"

    with pytest.raises(SystemExit) as caught:
        main(_arguments(MOKP_CASE, output))

    assert caught.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "case,family",
    ((MOKP_CASE, "MOKP"), (MOTSP_CASE, "MOTSP")),
)
def test_runner_materializes_four_exact_development_arms(
    tmp_path: Path,
    case: Path,
    family: str,
) -> None:
    output = tmp_path / family.lower()
    arguments = _arguments(case, output)
    arguments.append("--acknowledge-exposed-development-only")

    assert main(arguments) == 0

    summary_path = output / "summary.json"
    raw_summary = summary_path.read_bytes()
    summary = json.loads(raw_summary)
    assert raw_summary == (
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert summary["schema"] == (
        "pareto_v21e3r1_v9r2_single_case_four_arm_summary_v2"
    )
    assert summary["status"] == "SUCCESS_ENGINEERING_ONLY"
    assert summary["case"]["family"] == family
    assert summary["case"]["development_manifest_binding"]["split"] == (
        "development"
    )
    assert summary["evidence_partition"] == "EXPOSED_DEVELOPMENT_ONLY"
    assert summary["candidate_id"] == "C0"
    assert summary["phase"] == "development"
    assert summary["implementation_identity"] == (
        "same_repository_same_implementation_v1"
    )
    assert summary["implementation_independence"] is False
    assert summary["scientific_independence"] is False
    assert summary["selection_authorized"] is False
    assert summary["confirmation_authorized"] is False
    assert summary["formal_authorized"] is False
    assert summary["ijoc_submission_authorized"] is False
    protocol = summary["predevelopment_protocol"]
    assert protocol["status"] == "PRE_DEVELOPMENT_HOLD"
    assert protocol["execution_authorization"]["single_case_smoke"] is True
    assert protocol["execution_authorization"]["full_development_matrix"] is False
    assert all(
        value is False for value in protocol["later_phase_authorization"].values()
    )
    assert summary["validated_resource_caps"]["resource_caps"] == {
        "B": 8,
        "A": 256,
        "S": 50_000,
        "T": 60.0,
    }
    protocol_path = output / "predevelopment_protocol.json"
    assert protocol_path.is_file()
    assert protocol["artifact"]["sha256"] == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()

    assert set(summary["arms"]) == set(_EXPECTED_IDS[family])
    for arm, diagnostic_id in _EXPECTED_IDS[family].items():
        arm_summary = summary["arms"][arm]
        config = arm_summary["algorithm_config"]
        trace = output / arm / "trace.sqlite3"
        terminal = output / arm / "terminal.json"
        assert trace.is_file()
        assert terminal.is_file()
        diagnostic = output / arm / "diagnostic.json"
        assert diagnostic.is_file()
        assert {path.name for path in (output / arm).iterdir()} == {
            "trace.sqlite3",
            "terminal.json",
            "diagnostic.json",
            "branch_replay.json",
        }
        assert arm_summary["development_diagnostic_id"] == diagnostic_id
        assert config["development_diagnostic_id"] == diagnostic_id
        assert config["candidate_id"] == "C0"
        assert config["phase"] == "development"
        assert config["attempt_cap"] == 256
        assert config["wall_time_cap_seconds"] == pytest.approx(60.0)
        assert config["structural_screening_cap"] == (
            50_000 if arm in {"SCREEN", "BOTH"} else 0
        )
        assert config["archive_tradeoff_lambda"] == pytest.approx(
            0.5 if arm in {"LYAP", "BOTH"} else 0.0
        )
        receipt = json.loads(terminal.read_text(encoding="utf-8"))
        assert receipt["status"] == "SUCCESS"
        assert receipt["resource_accounting"][
            "all_configured_caps_satisfied"
        ] is True
        assert arm_summary["trace_verification"]["status"] == (
            "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS"
        )
        assert arm_summary["trace_verification"][
            "v9_resource_contract_replay"
        ] == "PASS"
        assert arm_summary["read_only_operator_diagnostic"]["status"] == (
            "DEVELOPMENT_ONLY_NO_LATER_PHASE_AUTHORIZATION"
        )
        assert arm_summary["read_only_operator_diagnostic"][
            "development_diagnostic_id"
        ] == diagnostic_id
        assert all(
            value is False
            for value in arm_summary["read_only_operator_diagnostic"][
                "authorization"
            ].values()
        )
        replay = arm_summary["same_implementation_branch_replay"]
        assert replay["status"] == (
            "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION"
        )
        assert replay["source_binding"]["source_closure_verified"] is True
        assert replay["implementation_independence"] is False
        assert replay["scientific_independence"] is False
        assert replay["third_party_replication"] is False
        replay_path = output / arm / "branch_replay.json"
        assert replay_path.is_file()
        assert arm_summary["branch_replay_report"]["sha256"] == hashlib.sha256(
            replay_path.read_bytes()
        ).hexdigest()

    summary_core = dict(summary)
    summary_digest = summary_core.pop("summary_payload_sha256")
    assert summary_digest == hashlib.sha256(
        json.dumps(
            summary_core,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_runner_refuses_to_overwrite_an_existing_output_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "belongs-to-user.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    arguments = _arguments(MOKP_CASE, output)
    arguments.append("--acknowledge-exposed-development-only")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(arguments)

    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert list(output.iterdir()) == [marker]


def test_programmatic_runner_cannot_bypass_development_acknowledgement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unacknowledged"

    with pytest.raises(PermissionError, match="acknowledgement"):
        run_v9r1_development_case(
            case=MOKP_CASE,
            outdir=output,
            seed=1701,
            reference_directions=((0.25, 0.75), (0.75, 0.25)),
            charged_evaluations=8,
            attempt_cap=256,
            structural_screening_cap=50_000,
            wall_time_cap_seconds=60.0,
            candidate_screening_cap=4,
            archive_tradeoff_lambda=0.5,
            checkpoint_period=4,
            acknowledge_exposed_development_only=False,
        )

    assert not output.exists()


def test_runner_rejects_case_bytes_outside_frozen_exposed_manifest(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "development_partitions_v1"
    instances = partition / "instances"
    instances.mkdir(parents=True)
    manifest = Path(
        "ijoc_submission_v21e3/development_partitions_v1/case_manifest.json"
    )
    (partition / "case_manifest.json").write_bytes(manifest.read_bytes())
    tampered_case = instances / "tampered.json"
    payload = json.loads(MOKP_CASE.read_text(encoding="utf-8"))
    payload["capacity"] += 1
    tampered_case.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tampered-output"
    arguments = _arguments(tampered_case, output)
    arguments.append("--acknowledge-exposed-development-only")

    with pytest.raises(ValueError, match="drifted"):
        main(arguments)

    assert not output.exists()


def test_resource_cap_failure_never_writes_success_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resource-failure"
    arguments = _arguments(MOKP_CASE, output)
    cap_index = arguments.index("--structural-screening-cap") + 1
    arguments[cap_index] = "4"
    arguments.append("--acknowledge-exposed-development-only")

    with pytest.raises(V21E3ResourceLimitExceeded):
        main(arguments)

    assert not (output / "summary.json").exists()
    failure = json.loads(
        (output / "SCREEN" / "terminal.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "FAILURE"
    assert failure["failure_code"] == "V9_RESOURCE_CAP_EXHAUSTED"


def test_runner_rejects_weakened_trace_replay_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_verify = runner_module.verify_v21e3_trace_database

    def weakened_verify(*args: object, **kwargs: object) -> dict[str, object]:
        report = dict(original_verify(*args, **kwargs))
        report.update(
            {
                "full_algorithm_decision_replay": "PASS",
                "v9_population_policy_replay": "NOT_APPLICABLE",
                "v9_population_policy_decisions_verified": 0,
                "v9_candidate_screen_witness_replay": "PASS",
                "v9_candidate_screen_witnesses_verified": 999,
                "v9_lyapunov_policy_witness_replay": "PASS",
                "v9_lyapunov_policy_witnesses_verified": 999,
            }
        )
        return report

    monkeypatch.setattr(
        runner_module,
        "verify_v21e3_trace_database",
        weakened_verify,
    )
    output = tmp_path / "weakened-trace"
    arguments = _arguments(MOKP_CASE, output)
    arguments.append("--acknowledge-exposed-development-only")

    with pytest.raises(RuntimeError, match="trace verification"):
        main(arguments)

    assert not (output / "summary.json").exists()


def test_runner_rejects_weakened_diagnostic_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_analyze = runner_module.analyze_v9_trace_database

    def weakened_analyze(*args: object, **kwargs: object) -> dict[str, object]:
        report = dict(original_analyze(*args, **kwargs))
        validation = dict(report["validation"])  # type: ignore[arg-type]
        validation.update(
            {
                "attempt_semantic_hash_chain": False,
                "terminal_chain_bindings": False,
                "detached_terminal_receipt_exact_match": False,
                "detached_terminal_receipt_external_sha256_bound": False,
                "lyapunov_witness_durable_state_arithmetic": "PASS",
            }
        )
        report.update(
            {
                "schema": "v21e3r1_v9_operator_productivity_diagnostic_v2",
                "objective_function_replay": "PASS",
                "full_algorithm_decision_replay": "PASS",
                "attempt_count": int(report["attempt_count"]) + 1,
                "decision_count": 0,
                "lyapunov_witness_count": 999,
                "lyapunov_witness_replay": "PASS",
                "exact_per_evaluation_left_continuous_hv_auc": -0.25,
                "post_initialization_incremental_hv_gain": -0.5,
                "final_normalized_hv": 1.25,
                "validation": validation,
            }
        )
        return report

    monkeypatch.setattr(
        runner_module,
        "analyze_v9_trace_database",
        weakened_analyze,
    )
    output = tmp_path / "weakened-diagnostic"
    arguments = _arguments(MOKP_CASE, output)
    arguments.append("--acknowledge-exposed-development-only")

    with pytest.raises(RuntimeError, match="read-only diagnostic"):
        main(arguments)

    assert not (output / "summary.json").exists()


def test_runner_rejects_weakened_branch_replay_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reexecute = runner_module.reexecute_and_compare

    def weakened_reexecute(*args: object, **kwargs: object) -> dict[str, object]:
        report = dict(original_reexecute(*args, **kwargs))
        checks = dict(report["checks"])  # type: ignore[arg-type]
        checks["attempts"] = False
        report["checks"] = checks
        report["first_mismatch"] = {
            "semantic_group": "attempts",
            "ordinal": 1,
            "original_sha256": "0" * 64,
            "replay_sha256": "1" * 64,
        }
        report["receipt_payload_sha256"] = "0" * 64
        output_receipt = Path(kwargs["output_receipt"])  # type: ignore[arg-type]
        output_receipt.write_bytes(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        return report

    monkeypatch.setattr(
        runner_module,
        "reexecute_and_compare",
        weakened_reexecute,
    )
    output = tmp_path / "weakened-branch"
    arguments = _arguments(MOKP_CASE, output)
    arguments.append("--acknowledge-exposed-development-only")

    with pytest.raises(RuntimeError, match="branch replay"):
        main(arguments)

    assert not (output / "summary.json").exists()
