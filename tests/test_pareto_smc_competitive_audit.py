from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mo_nco.pareto_smc_competitive_audit import (
    audit_competitive_results,
    load_competitive_protocol,
)

TEST_CASES = (
    ("c100a", 100),
    ("c100b", 100),
    ("c100c", 100),
    ("c200a", 200),
    ("c200b", 200),
    ("c200c", 200),
)
TEST_INSTANCE_SHA256 = "b" * 64
TEST_REFERENCE_CASE = {
    "contract": "frozen_external_v1",
    "hypervolume_reference": [10.0, 10.0],
    "ideal": [0.0, 0.0],
    "nadir": [9.0, 9.0],
    "reference_front": [[1.0, 8.0], [8.0, 1.0]],
}
TEST_REFERENCE_SHA256 = hashlib.sha256(
    json.dumps(
        TEST_REFERENCE_CASE,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
TEST_SUITE_BYTES = json.dumps(
    {
        "name": "test_suite",
        "cases": [
            {
                "name": case,
                "cities": cities,
                "evaluations": 100,
                "instance_sha256": TEST_INSTANCE_SHA256,
            }
            for case, cities in TEST_CASES
        ],
    },
    sort_keys=True,
).encode("utf-8")
TEST_REFERENCE_BYTES = json.dumps(
    {
        "schema_version": 1,
        "contract": "frozen_external_v1",
        "created_utc": "2026-07-27T00:00:00+00:00",
        "calibration_only": True,
        "evaluated_theory_arms_forbidden": ["smc"],
        "calibration_algorithms": ["calibration-baseline"],
        "reference_margin": 0.1,
        "source_files": [
            {
                "path": "calibration.csv",
                "sha256": "d" * 64,
            }
        ],
        "cases": {
            case: {
                **TEST_REFERENCE_CASE,
                "reference_sha256": TEST_REFERENCE_SHA256,
            }
            for case, _ in TEST_CASES
        },
    },
    sort_keys=True,
).encode("utf-8")
TEST_INFORMATION_PAYLOAD = {
    "schema": "pareto_smc_competitive_information_contract_v2",
    "objective_sense": "minimize_all",
    "search_time_information": [
        "complete_problem_instance",
        "predeclared_objective_evaluation_budget",
    ],
    "forbidden_search_time_information": [
        "competitor_outputs",
        "evaluated_test_seed_outcomes",
    ],
    "postprocessing_information_shared_by_all_algorithms": [
        "frozen_metric_reference",
        "frozen_normalization_box",
    ],
    "metric_reference_scope": "frozen_external_reference_v1",
    "budget_scope": (
        "matched_total_objective_evaluations_including_pilot_confirm"
    ),
    "timing_scope": "entire_algorithm_and_required_postprocessing",
    "runtime_scope": "uninstrumented_wall_clock_main_run",
    "memory_scope": "python_allocator_peak_increment_v1",
    "objective_output_scope": "local_full_tour_replay_v1",
    "archive_output_scope": "unbounded_then_common_cap_v1",
    "anytime_front_scope": "cumulative_nondominated_v1",
    "evaluation_evidence_scope": "exact_budget_v1",
    "configuration_scope": "predeclared_hashed_configuration_v1",
    "claim_limit": "No general state-of-the-art inference.",
}
TEST_INFORMATION_BYTES = json.dumps(
    TEST_INFORMATION_PAYLOAD,
    sort_keys=True,
).encode("utf-8")


def _test_algorithm_configuration(
    case: str,
    algorithm: str,
    seed: int,
) -> dict[str, object]:
    return {
        "schema": "mo_nco_predeclared_algorithm_configuration_v2",
        "case": case,
        "algorithm": algorithm,
        "seed": seed,
        "population": 4,
    }


def _test_algorithm_configuration_sha256(
    case: str,
    algorithm: str,
    seed: int,
) -> str:
    return hashlib.sha256(
        json.dumps(
            _test_algorithm_configuration(case, algorithm, seed),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _configuration_bytes_for_anchor(
    anchor: str,
    *,
    certificate_mode: object,
    protocol_version: object,
    certificate_specification_sha256: object = "e" * 64,
    certificate_manifest_sha256: object = "f" * 64,
) -> bytes:
    manifest = json.loads(TEST_CONFIGURATION_BYTES.decode("utf-8"))
    for row in manifest["runs"]:
        if row["algorithm"] != "smc":
            continue
        row["algorithm"] = anchor
        configuration = row["algorithm_configuration"]
        configuration["algorithm"] = anchor
        configuration["algorithm_specific"] = {
            "certificate_mode": certificate_mode,
            "pilot_confirm_protocol_version": protocol_version,
            "certificate_specification_sha256": (
                certificate_specification_sha256
            ),
            "certificate_manifest_sha256": certificate_manifest_sha256,
        }
        row["algorithm_configuration_sha256"] = hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return json.dumps(manifest, sort_keys=True).encode("utf-8")


TEST_CONFIGURATION_BYTES = json.dumps(
    {
        "schema": "pareto_smc_algorithm_configuration_manifest_v2",
        "suite_sha256": hashlib.sha256(TEST_SUITE_BYTES).hexdigest(),
        "runs": [
            {
                "case": case,
                "algorithm": algorithm,
                "seed": seed,
                "population": 4,
                "algorithm_configuration_sha256": (
                    _test_algorithm_configuration_sha256(
                        case,
                        algorithm,
                        seed,
                    )
                ),
                "search_evaluations": (
                    0 if algorithm == "smc" else 100
                ),
                "pilot_evaluations": (
                    50 if algorithm == "smc" else 0
                ),
                "confirm_evaluations": (
                    50 if algorithm == "smc" else 0
                ),
                "algorithm_configuration": (
                    _test_algorithm_configuration(
                        case,
                        algorithm,
                        seed,
                    )
                ),
            }
            for case, _ in TEST_CASES
            for algorithm in ("smc", "baseline")
            for seed in (10, 20)
        ],
    },
    sort_keys=True,
).encode("utf-8")
TEST_SUITE_SHA256 = hashlib.sha256(TEST_SUITE_BYTES).hexdigest()
TEST_REFERENCE_MANIFEST_SHA256 = hashlib.sha256(
    TEST_REFERENCE_BYTES
).hexdigest()
TEST_INFORMATION_SHA256 = hashlib.sha256(
    TEST_INFORMATION_BYTES
).hexdigest()
TEST_CONFIGURATION_SHA256 = hashlib.sha256(
    TEST_CONFIGURATION_BYTES
).hexdigest()


FIELDS = (
    "case",
    "algorithm",
    "seed",
    "population",
    "algorithm_configuration_sha256",
    "search_evaluations",
    "pilot_evaluations",
    "confirm_evaluations",
    "publication_certificate_packet_gate",
    "suite_sha256",
    "reference_manifest_sha256",
    "num_cities",
    "instance_sha256",
    "evaluations",
    "budget_scope",
    "archive_size",
    "archive_limit",
    "reference_sha256",
    "information_signature_sha256",
    "case_relative_hypervolume_2d",
    "case_relative_anytime_hv_eval_auc",
    "igd_plus",
    "additive_epsilon",
    "runtime_seconds",
    "python_peak_traced_memory_bytes",
    "runtime_measurement_contract",
    "execution_order_contract",
    "memory_measurement_contract",
    "memory_replay_order_contract",
    "memory_replay_state_equivalence_gate",
    "output_objective_equivalence_gate",
    "output_objective_max_abs_error",
    "output_objective_equivalence_contract",
    "anytime_objective_equivalence_gate",
    "anytime_objective_equivalence_contract",
    "evaluation_evidence_gate",
    "evaluation_evidence_contract",
    "native_archive_completeness_gate",
    "native_archive_completeness_contract",
    "anytime_front_semantics",
    "anytime_checkpoint_gate",
    "anytime_checkpoint_contract",
    "anytime_checkpoint_period",
    "anytime_checkpoint_count",
    "anytime_auc_integration_contract",
    "anytime_time_auc_status",
    "max_diagnostic_archive_size",
    "diagnostic_archive_limit_gate",
    "diagnostic_archive_limit_contract",
)


def _protocol_payload() -> dict[str, object]:
    return {
        "schema": "pareto_smc_competitive_protocol_v2",
        "anchor": "smc",
        "algorithms": ["smc", "baseline"],
        "suite_path": "suite.json",
        "suite_sha256": TEST_SUITE_SHA256,
        "reference_manifest_path": "reference.json",
        "reference_manifest_sha256": TEST_REFERENCE_MANIFEST_SHA256,
        "information_contract_path": "information.json",
        "information_contract_sha256": TEST_INFORMATION_SHA256,
        "algorithm_configuration_manifest_path": "configurations.json",
        "algorithm_configuration_manifest_sha256": (
            TEST_CONFIGURATION_SHA256
        ),
        "cases": {
            # Six independent case clusters are the smallest all-positive
            # fixture for which the exact two-sided sign randomization test
            # can cross the predeclared 0.05 gate (2 / 2**6).
            "expected_count": 6,
            "expected_ids": [
                "c100a",
                "c100b",
                "c100c",
                "c200a",
                "c200b",
                "c200c",
            ],
            "minimum_cities": 100,
            "required_city_sizes": [100, 200],
        },
        "seeds": [10, 20],
        "fairness": {
            "evaluations_per_run": 100,
            "budget_scope": (
                "matched_total_objective_evaluations_including_pilot_confirm"
            ),
            "archive_limit": 10,
            "runtime_measurement_contract": (
                "uninstrumented_wall_clock_inprocess_v1"
            ),
            "execution_order_contract": "seed-major-balanced-v1",
            "memory_measurement_contract": (
                "python_tracemalloc_separate_replay_peak_increment_v1"
            ),
            "memory_replay_order_contract": (
                "all_case_timed_runs_before_case_memory_replays_v1"
            ),
            "output_objective_equivalence_contract": (
                "local_full_tour_exact_on_integer_domain_else_"
                "rel1e-12_abs1e-12_v1"
            ),
            "native_archive_completeness_contract": (
                "unbounded_nondominated_all_evaluated_candidates_v1"
            ),
            "anytime_front_semantics": (
                "cumulative_nondominated_best_so_far_v1"
            ),
            "anytime_checkpoint_contract": (
                "exact_common_evaluation_checkpoint_archive_snapshot_v1"
            ),
            "anytime_checkpoint_period": 10,
            "anytime_auc_integration_contract": (
                "left_continuous_step_on_evaluation_snapshots_v1"
            ),
            "anytime_time_auc_status": (
                "descriptive_only_not_formal_quality_gate_v1"
            ),
            "diagnostic_archive_limit_contract": (
                "deterministic_nondominated_crowding_"
                "truncation_per_snapshot_v1"
            ),
        },
        "gates": {
            "igd_plus_noninferiority_margin": 0.01,
            "maximum_runtime_ratio": 1.25,
            "maximum_python_memory_ratio": 1.25,
        },
        "analysis": {
            "bootstrap_repetitions": 50,
            "randomization_repetitions": 50,
            "random_seed": 7,
        },
    }


def _row(
    case: str,
    cities: int,
    algorithm: str,
    seed: int,
    *,
    strong: bool,
) -> dict[str, object]:
    is_pilot_confirm = algorithm in {
        "smc",
        "pareto-smc-pilot-confirm-v11",
        "pareto-smc-pilot-confirm-v12",
    }
    return {
        "case": case,
        "algorithm": algorithm,
        "seed": seed,
        "population": 4,
        "algorithm_configuration_sha256": (
            _test_algorithm_configuration_sha256(
                case,
                algorithm,
                seed,
            )
        ),
        "search_evaluations": 0 if is_pilot_confirm else 100,
        "pilot_evaluations": 50 if is_pilot_confirm else 0,
        "confirm_evaluations": 50 if is_pilot_confirm else 0,
        "publication_certificate_packet_gate": (
            "PASS"
            if algorithm == "pareto-smc-pilot-confirm-v12"
            else "NOT_APPLICABLE"
        ),
        "suite_sha256": TEST_SUITE_SHA256,
        "reference_manifest_sha256": TEST_REFERENCE_MANIFEST_SHA256,
        "num_cities": cities,
        "instance_sha256": TEST_INSTANCE_SHA256,
        "evaluations": 100,
        "budget_scope": (
            "matched_total_objective_evaluations_including_pilot_confirm"
        ),
        "archive_size": 8,
        "archive_limit": 10,
        "reference_sha256": TEST_REFERENCE_SHA256,
        "information_signature_sha256": TEST_INFORMATION_SHA256,
        "case_relative_hypervolume_2d": 1.0 if strong else 0.8,
        "case_relative_anytime_hv_eval_auc": 1.0 if strong else 0.8,
        "igd_plus": 0.1 if strong else 0.2,
        "additive_epsilon": 0.2 if strong else 0.3,
        "runtime_seconds": 1.0 if strong else 1.1,
        "python_peak_traced_memory_bytes": 1000 if strong else 1100,
        "runtime_measurement_contract": (
            "uninstrumented_wall_clock_inprocess_v1"
        ),
        "execution_order_contract": "seed-major-balanced-v1",
        "memory_measurement_contract": (
            "python_tracemalloc_separate_replay_peak_increment_v1"
        ),
        "memory_replay_order_contract": (
            "all_case_timed_runs_before_case_memory_replays_v1"
        ),
        "memory_replay_state_equivalence_gate": "PASS",
        "output_objective_equivalence_gate": "PASS",
        "output_objective_max_abs_error": 0.0,
        "output_objective_equivalence_contract": (
            "local_full_tour_exact_on_integer_domain_else_"
            "rel1e-12_abs1e-12_v1"
        ),
        "anytime_objective_equivalence_gate": "PASS",
        "anytime_objective_equivalence_contract": (
            "internal_diagnostic_front_from_local_evaluations_v1"
        ),
        "evaluation_evidence_gate": "PASS",
        "evaluation_evidence_contract": (
            "inprocess_counting_instance_exact_budget_v1"
        ),
        "native_archive_completeness_gate": "PASS",
        "native_archive_completeness_contract": (
            "unbounded_nondominated_all_evaluated_candidates_v1"
        ),
        "anytime_front_semantics": (
            "cumulative_nondominated_best_so_far_v1"
        ),
        "anytime_checkpoint_gate": "PASS",
        "anytime_checkpoint_contract": (
            "exact_common_evaluation_checkpoint_archive_snapshot_v1"
        ),
        "anytime_checkpoint_period": 10,
        "anytime_checkpoint_count": 10,
        "anytime_auc_integration_contract": (
            "left_continuous_step_on_evaluation_snapshots_v1"
        ),
        "anytime_time_auc_status": (
            "descriptive_only_not_formal_quality_gate_v1"
        ),
        "max_diagnostic_archive_size": 8,
        "diagnostic_archive_limit_gate": "PASS",
        "diagnostic_archive_limit_contract": (
            "deterministic_nondominated_crowding_"
            "truncation_per_snapshot_v1"
        ),
    }


class CompetitiveAuditTests(unittest.TestCase):
    def _artifacts(self, directory: Path):  # type: ignore[no-untyped-def]
        (directory / "suite.json").write_bytes(TEST_SUITE_BYTES)
        (directory / "reference.json").write_bytes(
            TEST_REFERENCE_BYTES
        )
        (directory / "information.json").write_bytes(
            TEST_INFORMATION_BYTES
        )
        (directory / "configurations.json").write_bytes(
            TEST_CONFIGURATION_BYTES
        )
        protocol_path = directory / "protocol.json"
        protocol_path.write_text(
            json.dumps(_protocol_payload()),
            encoding="utf-8",
        )
        aggregate = directory / "aggregate.csv"
        rows = []
        for case, cities in TEST_CASES:
            for seed in (10, 20):
                rows.append(_row(case, cities, "smc", seed, strong=True))
                rows.append(_row(case, cities, "baseline", seed, strong=False))
        with aggregate.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return protocol_path, aggregate, rows

    def _anchor_protocol(
        self,
        directory: Path,
        *,
        anchor: str,
        certificate_mode: object,
        protocol_version: object,
        certificate_specification_sha256: object = "e" * 64,
        certificate_manifest_sha256: object = "f" * 64,
    ) -> Path:
        protocol_path, _, _ = self._artifacts(directory)
        configuration_bytes = _configuration_bytes_for_anchor(
            anchor,
            certificate_mode=certificate_mode,
            protocol_version=protocol_version,
            certificate_specification_sha256=(
                certificate_specification_sha256
            ),
            certificate_manifest_sha256=certificate_manifest_sha256,
        )
        (directory / "configurations.json").write_bytes(
            configuration_bytes
        )
        protocol_payload = _protocol_payload()
        protocol_payload["anchor"] = anchor
        protocol_payload["algorithms"] = [anchor, "baseline"]
        protocol_payload["algorithm_configuration_manifest_sha256"] = (
            hashlib.sha256(configuration_bytes).hexdigest()
        )
        protocol_path.write_text(
            json.dumps(protocol_payload),
            encoding="utf-8",
        )
        return protocol_path

    def _v12_audit_artifacts(
        self,
        directory: Path,
    ) -> tuple[Path, Path, list[dict[str, object]]]:
        anchor = "pareto-smc-pilot-confirm-v12"
        protocol_path = self._anchor_protocol(
            directory,
            anchor=anchor,
            certificate_mode="regeneration",
            protocol_version="v12_regeneration",
        )
        configuration_manifest = json.loads(
            (directory / "configurations.json").read_text(
                encoding="utf-8"
            )
        )
        configuration_hashes = {
            (
                row["case"],
                row["algorithm"],
                row["seed"],
            ): row["algorithm_configuration_sha256"]
            for row in configuration_manifest["runs"]
        }
        rows = []
        for case, cities in TEST_CASES:
            for seed in (10, 20):
                for algorithm, strong in (
                    (anchor, True),
                    ("baseline", False),
                ):
                    row = _row(
                        case,
                        cities,
                        algorithm,
                        seed,
                        strong=strong,
                    )
                    row["algorithm_configuration_sha256"] = (
                        configuration_hashes[
                            (case, algorithm, seed)
                        ]
                    )
                    rows.append(row)
        aggregate = directory / "aggregate.csv"
        with aggregate.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return protocol_path, aggregate, rows

    def test_complete_matched_matrix_reports_all_strict_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol_path, aggregate, _ = self._artifacts(Path(temporary))
            protocol = load_competitive_protocol(protocol_path)
            report = audit_competitive_results(
                aggregate,
                protocol,
                bootstrap_repetitions=50,
                randomization_repetitions=50,
                random_seed=7,
            )
        self.assertEqual(report["evidence_status"], "COMPLETE")
        self.assertEqual(report["contract_gate"], "PASS")
        self.assertEqual(report["overall_adoption_verdict"], "ADOPT")
        self.assertEqual(len(report["cost_ratio_results"]), 2)
        self.assertTrue(
            all(
                item["noninferiority_gate"] == "PASS"
                for item in report["cost_ratio_results"]
            )
        )
        quality_rows = report["quality_analysis"]["metric_results"]
        self.assertTrue(
            all(
                "cluster_bootstrap_ci95" in row
                and "trimmed_mean_10pct" in row
                and "winsorized_mean_10pct" in row
                and "cluster_randomization_p_two_sided" in row
                for row in quality_rows
            )
        )

    def test_metric_reference_manifest_requires_semantic_validity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, _, _ = self._artifacts(root)
            reference_payload = json.loads(
                TEST_REFERENCE_BYTES.decode("utf-8")
            )
            malformed = reference_payload["cases"]["c100a"]
            malformed["nadir"] = [0.0, 9.0]
            canonical = dict(malformed)
            canonical.pop("reference_sha256")
            malformed["reference_sha256"] = hashlib.sha256(
                json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            reference_bytes = json.dumps(
                reference_payload,
                sort_keys=True,
            ).encode("utf-8")
            (root / "reference.json").write_bytes(reference_bytes)
            protocol_payload = _protocol_payload()
            protocol_payload["reference_manifest_sha256"] = (
                hashlib.sha256(reference_bytes).hexdigest()
            )
            protocol_path.write_text(
                json.dumps(protocol_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "fails semantic validation",
            ):
                load_competitive_protocol(protocol_path)

    def test_metric_reference_manifest_rejects_forged_top_level_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, _, _ = self._artifacts(root)
            reference_payload = json.loads(
                TEST_REFERENCE_BYTES.decode("utf-8")
            )
            reference_payload["schema_version"] = 999
            reference_payload["contract"] = "WRONG"
            reference_bytes = json.dumps(
                reference_payload,
                sort_keys=True,
            ).encode("utf-8")
            (root / "reference.json").write_bytes(reference_bytes)
            protocol_payload = _protocol_payload()
            protocol_payload["reference_manifest_sha256"] = (
                hashlib.sha256(reference_bytes).hexdigest()
            )
            protocol_path.write_text(
                json.dumps(protocol_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "structural, provenance, or semantic validation",
            ):
                load_competitive_protocol(protocol_path)

    def test_information_contract_requires_exact_v2_shape_and_types(
        self,
    ) -> None:
        mutations = (
            (
                "wrong schema",
                lambda payload: payload.update(
                    {"schema": "test_information_contract_v1"}
                ),
                "wrong schema",
            ),
            (
                "missing field",
                lambda payload: payload.pop("runtime_scope"),
                "unexpected shape",
            ),
            (
                "unexpected field",
                lambda payload: payload.update({"unexpected": True}),
                "unexpected shape",
            ),
            (
                "wrong objective sense",
                lambda payload: payload.update(
                    {"objective_sense": "maximize_all"}
                ),
                "objective_sense",
            ),
            (
                "wrong list type",
                lambda payload: payload.update(
                    {"search_time_information": "complete_problem_instance"}
                ),
                "search_time_information",
            ),
            (
                "wrong list element type",
                lambda payload: payload.update(
                    {"search_time_information": ["complete_instance", 7]}
                ),
                "search_time_information",
            ),
            (
                "blank scope",
                lambda payload: payload.update({"claim_limit": ""}),
                "claim_limit",
            ),
            (
                "wrong scope type",
                lambda payload: payload.update({"claim_limit": 7}),
                "claim_limit",
            ),
            (
                "mismatched budget scope",
                lambda payload: payload.update(
                    {"budget_scope": "single_run_objective_evaluations"}
                ),
                "budget_scope does not match",
            ),
        )
        for label, mutate, expected_error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                protocol_path, _, _ = self._artifacts(root)
                information_payload = dict(TEST_INFORMATION_PAYLOAD)
                mutate(information_payload)
                information_bytes = json.dumps(
                    information_payload,
                    sort_keys=True,
                ).encode("utf-8")
                (root / "information.json").write_bytes(
                    information_bytes
                )
                protocol_payload = _protocol_payload()
                protocol_payload["information_contract_sha256"] = (
                    hashlib.sha256(information_bytes).hexdigest()
                )
                protocol_path.write_text(
                    json.dumps(protocol_payload),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    load_competitive_protocol(protocol_path)

    def test_v11_and_v12_anchors_require_their_frozen_certificate_modes(
        self,
    ) -> None:
        contracts = (
            (
                "pareto-smc-pilot-confirm-v11",
                "published",
                "v11_published",
                "regeneration",
            ),
            (
                "pareto-smc-pilot-confirm-v12",
                "regeneration",
                "v12_regeneration",
                "published",
            ),
        )
        for anchor, mode, version, wrong_mode in contracts:
            with (
                self.subTest(anchor=anchor),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                valid = self._anchor_protocol(
                    root,
                    anchor=anchor,
                    certificate_mode=mode,
                    protocol_version=version,
                )
                load_competitive_protocol(valid)
                invalid = self._anchor_protocol(
                    root,
                    anchor=anchor,
                    certificate_mode=wrong_mode,
                    protocol_version=version,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "certificate_mode",
                ):
                    load_competitive_protocol(invalid)

    def test_v12_anchor_requires_version_and_certificate_hash_bindings(
        self,
    ) -> None:
        invalid_bindings = (
            (
                "wrong protocol version",
                "v11_published",
                "e" * 64,
                "f" * 64,
                "pilot_confirm_protocol_version",
            ),
            (
                "missing certificate specification hash",
                "v12_regeneration",
                None,
                "f" * 64,
                "certificate_specification_sha256",
            ),
            (
                "malformed certificate manifest hash",
                "v12_regeneration",
                "e" * 64,
                "NOT-A-SHA256",
                "certificate_manifest_sha256",
            ),
        )
        for (
            label,
            version,
            specification_sha,
            manifest_sha,
            expected_error,
        ) in invalid_bindings:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                protocol_path = self._anchor_protocol(
                    Path(temporary),
                    anchor="pareto-smc-pilot-confirm-v12",
                    certificate_mode="regeneration",
                    protocol_version=version,
                    certificate_specification_sha256=specification_sha,
                    certificate_manifest_sha256=manifest_sha,
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    load_competitive_protocol(protocol_path)

    def test_every_v12_anchor_configuration_row_is_validated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = self._anchor_protocol(
                root,
                anchor="pareto-smc-pilot-confirm-v12",
                certificate_mode="regeneration",
                protocol_version="v12_regeneration",
            )
            manifest_path = root / "configurations.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            anchor_rows = [
                row
                for row in manifest["runs"]
                if row["algorithm"]
                == "pareto-smc-pilot-confirm-v12"
            ]
            configuration = anchor_rows[-1]["algorithm_configuration"]
            configuration["algorithm_specific"][
                "pilot_confirm_protocol_version"
            ] = "v11_published"
            anchor_rows[-1]["algorithm_configuration_sha256"] = (
                hashlib.sha256(
                    json.dumps(
                        configuration,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            manifest_bytes = json.dumps(
                manifest,
                sort_keys=True,
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            protocol_payload = json.loads(
                protocol_path.read_text(encoding="utf-8")
            )
            protocol_payload[
                "algorithm_configuration_manifest_sha256"
            ] = hashlib.sha256(manifest_bytes).hexdigest()
            protocol_path.write_text(
                json.dumps(protocol_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "pilot_confirm_protocol_version",
            ):
                load_competitive_protocol(protocol_path)

    def test_v12_publication_certificate_packet_gate_fails_closed(
        self,
    ) -> None:
        mutations = (
            (
                "missing column",
                None,
                "missing columns: publication_certificate_packet_gate",
            ),
            (
                "anchor is not pass",
                (
                    "pareto-smc-pilot-confirm-v12",
                    "FAIL",
                ),
                "v12 anchor publication_certificate_packet_gate",
            ),
            (
                "non-anchor is not not-applicable",
                ("baseline", "PASS"),
                "non-anchor publication_certificate_packet_gate",
            ),
        )
        for label, mutation, expected_reason in mutations:
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                protocol_path, aggregate, rows = (
                    self._v12_audit_artifacts(root)
                )
                protocol = load_competitive_protocol(protocol_path)
                complete = audit_competitive_results(
                    aggregate,
                    protocol,
                )
                self.assertEqual(complete["contract_gate"], "PASS")

                fieldnames = list(FIELDS)
                if mutation is None:
                    fieldnames.remove(
                        "publication_certificate_packet_gate"
                    )
                    rows = [
                        {
                            key: value
                            for key, value in row.items()
                            if key
                            != "publication_certificate_packet_gate"
                        }
                        for row in rows
                    ]
                else:
                    algorithm, gate = mutation
                    next(
                        row
                        for row in rows
                        if row["algorithm"] == algorithm
                    )["publication_certificate_packet_gate"] = gate
                with aggregate.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=fieldnames,
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                report = audit_competitive_results(
                    aggregate,
                    protocol,
                )
                self.assertEqual(report["evidence_status"], "NOT_RUN")
                self.assertEqual(report["contract_gate"], "FAIL")
                self.assertEqual(
                    report["overall_adoption_verdict"],
                    "NOT_RUN",
                )
                self.assertTrue(
                    any(
                        expected_reason in reason
                        for reason in report["reasons"]
                    )
                )

    def test_missing_or_unmatched_formal_evidence_stays_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, aggregate, rows = self._artifacts(root)
            protocol = load_competitive_protocol(protocol_path)
            missing = audit_competitive_results(root / "missing.csv", protocol)
            self.assertEqual(missing["overall_adoption_verdict"], "NOT_RUN")

            rows[0]["evaluations"] = 99
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            unmatched = audit_competitive_results(aggregate, protocol)
        self.assertEqual(unmatched["overall_adoption_verdict"], "NOT_RUN")
        self.assertTrue(
            any("evaluation budget mismatch" in reason for reason in unmatched["reasons"])
        )

    def test_blank_hashes_and_unfrozen_reference_manifest_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, aggregate, rows = self._artifacts(root)
            protocol = load_competitive_protocol(protocol_path)
            rows[0]["instance_sha256"] = ""
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            malformed = audit_competitive_results(aggregate, protocol)
            self.assertEqual(
                malformed["overall_adoption_verdict"],
                "NOT_RUN",
            )
            self.assertTrue(
                any(
                    "instance_sha256 must be a lowercase SHA-256"
                    in reason
                    for reason in malformed["reasons"]
                )
            )

            pending_payload = _protocol_payload()
            pending_payload["reference_manifest_path"] = None
            pending_payload["reference_manifest_sha256"] = None
            protocol_path.write_text(
                json.dumps(pending_payload),
                encoding="utf-8",
            )
            pending = audit_competitive_results(
                aggregate,
                load_competitive_protocol(protocol_path),
            )
        self.assertEqual(pending["overall_adoption_verdict"], "NOT_RUN")
        self.assertIn(
            "frozen metric-reference manifest hash is not yet predeclared",
            pending["reasons"],
        )

    def test_anytime_archive_cap_and_semantics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, aggregate, rows = self._artifacts(root)
            protocol = load_competitive_protocol(protocol_path)
            rows[0]["max_diagnostic_archive_size"] = 11
            rows[0]["anytime_front_semantics"] = "current_population_v1"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            report = audit_competitive_results(aggregate, protocol)
        self.assertEqual(report["overall_adoption_verdict"], "NOT_RUN")
        self.assertTrue(
            any(
                "anytime archive overflow" in reason
                for reason in report["reasons"]
            )
        )
        self.assertTrue(
            any(
                "anytime front semantics mismatch" in reason
                for reason in report["reasons"]
            )
        )

    def test_signed_additive_epsilon_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, aggregate, rows = self._artifacts(root)
            for row in rows:
                if row["algorithm"] == "smc":
                    row["additive_epsilon"] = -0.05
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            report = audit_competitive_results(
                aggregate,
                load_competitive_protocol(protocol_path),
            )
        self.assertEqual(report["evidence_status"], "COMPLETE")
        self.assertEqual(report["contract_gate"], "PASS")


if __name__ == "__main__":
    unittest.main()

