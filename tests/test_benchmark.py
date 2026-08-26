from __future__ import annotations

import csv
import json
import os
import random
import tempfile
import tracemalloc
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mo_nco.benchmark import (
    _neural_prior_path_for_backend,
    build_execution_schedule,
    run_algorithm,
    run_benchmark,
)
from mo_nco.benchmark_suite import BenchmarkCase, BenchmarkSuite, run_benchmark_suite
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.learned_move_generator import SparseMoveGenerator
from mo_nco.pcd_net import PCDResidualScalarNet
from mo_nco.pareto_smc_spec import analytic_objective_box
from mo_nco.run_freeze_metric_references import freeze_metric_references


def _torch_available() -> bool:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class BenchmarkTests(unittest.TestCase):
    def test_pilot_confirm_v11_alias_charges_both_streams(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(
            4,
            seed=410,
        )
        with patch.dict(
            os.environ,
            {
                "MO_NCO_PARETO_SMC_SPEC": str(
                    Path(
                        "benchmarks/"
                        "pareto_smc_v11_fixed_smoke_spec.json"
                    ).resolve()
                ),
                "MO_NCO_PARETO_FIXED_REFERENCE_SPEC": "",
                "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST": str(
                    Path(
                        "benchmarks/"
                        "pareto_smc_v11_fixed_reference_smoke_manifest.json"
                    ).resolve()
                ),
            },
        ):
            result = run_algorithm(
                "pareto-smc-pilot-confirm-v11",
                instance,
                seed=0,
                population=4096,
                iterations=24576,
                log_period=1024,
                archive_update_period=1024,
            )
        self.assertEqual(result.metadata["formal_packet_gate"], "PASS")
        self.assertEqual(result.metadata["pilot_evaluations"], 12288)
        self.assertEqual(result.metadata["confirm_evaluations"], 12288)
        self.assertEqual(
            result.metadata["fixed_cell_cover_archive_size"],
            2,
        )
        self.assertTrue(
            result.metadata["exact_incremental_two_opt_requested"]
        )
        self.assertFalse(
            result.metadata["local_two_opt_incremental_enabled"]
        )
        self.assertEqual(
            result.metadata["native_archive_completeness_gate"],
            "PASS",
        )
        self.assertTrue(
            any(
                diagnostic.iteration <= 12288
                for diagnostic in result.diagnostics
            )
        )
        self.assertTrue(
            any(
                diagnostic.iteration > 12288
                for diagnostic in result.diagnostics
            )
        )
        self.assertEqual(result.diagnostics[-1].iteration, 24576)

    def test_pilot_confirm_v12_alias_binds_regeneration_commitment(
        self,
    ) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(
            4,
            seed=410,
        )
        with patch.dict(
            os.environ,
            {
                "MO_NCO_PARETO_SMC_SPEC": str(
                    Path(
                        "benchmarks/"
                        "pareto_smc_v11_fixed_smoke_spec.json"
                    ).resolve()
                ),
                "MO_NCO_PARETO_FIXED_REFERENCE_SPEC": "",
                "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST": str(
                    Path(
                        "benchmarks/"
                        "pareto_smc_v11_fixed_reference_smoke_manifest.json"
                    ).resolve()
                ),
            },
        ):
            result = run_algorithm(
                "pareto-smc-pilot-confirm-v12",
                instance,
                seed=0,
                population=4096,
                iterations=24576,
                log_period=1024,
                archive_update_period=1024,
            )
        self.assertEqual(result.metadata["formal_packet_gate"], "PASS")
        self.assertEqual(
            result.metadata["pilot_confirm_protocol_version"],
            "v12_regeneration",
        )
        self.assertEqual(
            result.metadata["certificate_mode"],
            "regeneration",
        )
        self.assertEqual(
            len(result.metadata["pilot_plan_commitment_sha256"]),
            64,
        )
        certificate = result.metadata["pilot_confirm_certificate"]
        self.assertEqual(
            certificate["active_certificate_basis"],
            "regeneration",
        )
        self.assertEqual(
            certificate["pilot_plan_commitment_gate"],
            "PASS",
        )
        self.assertEqual(
            result.metadata[
                "pilot_plan_commitment_preconfirm_order_gate"
            ],
            "PASS_RUNNER_CONTROL_FLOW_ATTESTED",
        )
        self.assertFalse(
            result.metadata[
                "pilot_plan_commitment_preconfirm_timing_"
                "independently_verified"
            ]
        )
        self.assertEqual(
            certificate["publication_certificate_packet_gate"],
            "FAIL",
        )
        self.assertEqual(
            result.metadata["publication_certificate_packet_gate"],
            "FAIL",
        )
        self.assertEqual(
            certificate[
                "external_preconfirm_commitment_receipt_gate"
            ],
            "NOT_IMPLEMENTED",
        )

    def test_v12_benchmark_record_carries_publication_packet_gate(
        self,
    ) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(
            4,
            seed=410,
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "MO_NCO_PARETO_SMC_SPEC": str(
                    Path(
                        "benchmarks/"
                        "pareto_smc_v11_fixed_smoke_spec.json"
                    ).resolve()
                ),
                "MO_NCO_PARETO_FIXED_REFERENCE_SPEC": "",
                "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST": str(
                    Path(
                        "benchmarks/"
                        "pareto_smc_v11_fixed_reference_smoke_manifest.json"
                    ).resolve()
                ),
            },
        ):
            output_dir = Path(temporary)
            records, _ = run_benchmark(
                algorithms=("pareto-smc-pilot-confirm-v12",),
                seeds=(0,),
                cities=4,
                population=4096,
                iterations=24576,
                instance_seed=410,
                output_dir=output_dir,
                log_period=1024,
                archive_update_period=1024,
                instance=instance,
            )
            with (output_dir / "runs.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(
            records[0].publication_certificate_packet_gate,
            "FAIL",
        )
        self.assertEqual(
            rows[0]["publication_certificate_packet_gate"],
            "FAIL",
        )

    def test_frozen_v11_suite_binds_integer_matrix_artifacts(self) -> None:
        suite = BenchmarkSuite.from_json(
            Path("benchmarks/pareto_smc_v11_competitive_suite.json")
        )
        case = suite.cases[0]
        instance = case.load_instance()
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertTrue(instance.exact_two_opt_delta_in_binary64)
        with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
            replace(
                case,
                tsplib_sha256=("0" * 64, *case.tsplib_sha256[1:]),
            ).load_instance()

    def test_adaptive_heuristic_aliases_emit_an_explicit_claim_boundary(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=5)
        for alias, expected in (
            ("ips-heuristic-adaptive", "explicit_heuristic_descent"),
            ("ips-theory", "deprecated_theory_named_alias_for_heuristic_descent"),
        ):
            payload = SimpleNamespace(metadata={})
            with patch("mo_nco.benchmark.TheoryAlignedIPSOptimizer") as optimizer:
                optimizer.return_value.run.return_value = payload
                result = run_algorithm(alias, instance, 0, 4, 12, 4, 4)
            self.assertIs(result, payload)
            self.assertEqual(result.metadata["alias_claim_boundary"], expected)

    def test_theory_certified_alias_uses_positive_temperature_control(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=6)
        with patch.dict(
            os.environ,
            {
                "MO_NCO_CERTIFIED_TEMPERATURE": "0.07",
                "MO_NCO_CERTIFIED_CHEBYSHEV_RHO": "0.04",
                "MO_NCO_CERTIFIED_UNIFORMIZATION_RATE": "2.0",
                "MO_NCO_CERTIFIED_LAZY_PROBABILITY": "0.1",
            },
        ), patch("mo_nco.benchmark.CertifiedSingleSiteIPSOptimizer") as optimizer:
            sentinel = object()
            optimizer.return_value.run.return_value = sentinel
            result = run_algorithm(
                "ips-theory-certified",
                instance,
                seed=0,
                population=8,
                iterations=64,
                log_period=16,
                archive_update_period=8,
            )
        self.assertIs(result, sentinel)
        kwargs = optimizer.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.07)
        self.assertEqual(kwargs["chebyshev_rho"], 0.04)
        self.assertEqual(kwargs["uniformization_rate"], 2.0)
        self.assertEqual(kwargs["lazy_probability"], 0.1)

    def test_certified_env_parameters_change_configuration_signature(
        self,
    ) -> None:
        signatures = []
        with tempfile.TemporaryDirectory() as temporary:
            for index, temperature in enumerate(("0.05", "0.07")):
                with patch.dict(
                    os.environ,
                    {
                        "MO_NCO_CERTIFIED_TEMPERATURE": temperature,
                        "MO_NCO_CERTIFIED_CHEBYSHEV_RHO": "0.03",
                        "MO_NCO_CERTIFIED_UNIFORMIZATION_RATE": "1.0",
                        "MO_NCO_CERTIFIED_LAZY_PROBABILITY": "0.05",
                    },
                ):
                    records, _ = run_benchmark(
                        algorithms=["ips-theory-certified"],
                        seeds=[0],
                        cities=8,
                        population=4,
                        iterations=12,
                        instance_seed=8,
                        output_dir=Path(temporary) / str(index),
                        log_period=4,
                        archive_update_period=4,
                    )
                signatures.append(
                    records[0].algorithm_configuration_sha256
                )
        self.assertNotEqual(signatures[0], signatures[1])

    def test_pareto_smc_alias_requires_and_binds_the_external_manifest(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=66)
        root = Path(__file__).resolve().parents[1]
        manifest = root / "benchmarks" / "pareto_smc_v1_spec.json"
        payload = SimpleNamespace(
            metadata={
                "context_hash": "a" * 64,
                "reporting_context_hash": "b" * 64,
                "run_contract_hash": "c" * 64,
            }
        )
        with patch.dict(
            os.environ,
            {"MO_NCO_PARETO_SMC_SPEC": str(manifest)},
        ), patch(
            "mo_nco.benchmark.AnnealedParetoSMCOptimizer"
        ) as optimizer:
            optimizer.return_value.run.return_value = payload
            result = run_algorithm(
                "annealed-pareto-smc",
                instance,
                seed=3,
                population=32,
                iterations=512,
                log_period=64,
                archive_update_period=64,
            )

        self.assertIs(result, payload)
        kwargs = optimizer.call_args.kwargs
        self.assertEqual(kwargs["particles_per_reference"], 4)
        self.assertEqual(len(kwargs["reference_directions"]), 8)
        self.assertEqual(kwargs["beta_schedule"], (0.0, 0.5, 1.5, 3.0, 6.0, 10.0))
        self.assertEqual(kwargs["global_refresh_probability"], 0.0)
        self.assertFalse(kwargs["enable_exact_incremental_two_opt"])
        lower, upper = analytic_objective_box(instance)
        self.assertTrue(
            all(
                abs(width / (high - low) - 0.05) < 1e-15
                for width, low, high in zip(
                    kwargs["epsilon"],
                    lower,
                    upper,
                )
            )
        )
        self.assertEqual(
            result.metadata["external_specification_schema"],
            "annealed_pareto_smc_spec_v1",
        )
        self.assertEqual(len(result.metadata["external_specification_sha256"]), 64)
        self.assertEqual(len(result.metadata["specification_run_binding_sha256"]), 64)

    def test_pareto_smc_alias_fails_without_a_manifest(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=67)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MO_NCO_PARETO_SMC_SPEC", None)
            with self.assertRaisesRegex(ValueError, "MO_NCO_PARETO_SMC_SPEC"):
                run_algorithm(
                    "annealed-pareto-smc",
                    instance,
                    seed=0,
                    population=32,
                    iterations=512,
                    log_period=64,
                    archive_update_period=64,
                )

    def test_benchmark_can_emit_certified_transition_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_benchmark(
                algorithms=["ips-theory-certified"],
                seeds=[0],
                cities=8,
                population=4,
                iterations=12,
                instance_seed=7,
                output_dir=root,
                log_period=4,
                archive_update_period=4,
                certified_traces=True,
            )
            trace = root / "kernel_traces" / "ips-theory-certified_seed0.jsonl"
            self.assertTrue(trace.exists())
            metadata_lines = [
                json.loads(line)
                for line in (root / "run_metadata.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(metadata_lines[0]["metadata"]["trace_path"], str(trace))

    def test_benchmark_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, summary = run_benchmark(
                algorithms=["ips", "random2opt"],
                seeds=[0],
                cities=9,
                population=6,
                iterations=24,
                instance_seed=4,
                output_dir=Path(tmp),
                log_period=8,
                archive_update_period=4,
                measure_python_memory=True,
                output_archive_limit=1,
            )
            self.assertEqual(len(records), 2)
            self.assertIn("ips", summary)
            self.assertTrue(all(record.evaluations == 24 for record in records))
            self.assertTrue(
                all(
                    record.publication_certificate_packet_gate
                    == "NOT_APPLICABLE"
                    for record in records
                )
            )
            self.assertTrue(
                all(
                    record.python_peak_traced_memory_bytes >= 0
                    for record in records
                )
            )
            self.assertIn(
                "python_peak_traced_memory_bytes_mean",
                summary["ips"],
            )
            self.assertTrue((Path(tmp) / "runs.csv").exists())
            self.assertTrue((Path(tmp) / "anytime.csv").exists())
            self.assertTrue((Path(tmp) / "summary.csv").exists())
            self.assertTrue((Path(tmp) / "paired_comparison.md").exists())
            self.assertTrue((Path(tmp) / "pareto_fronts.svg").exists())
            self.assertTrue((Path(tmp) / "run_metadata.jsonl").exists())
            metadata_rows = [
                json.loads(line)
                for line in (Path(tmp) / "run_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(
                all(
                    row["metadata"]["memory_measurement_contract"]
                    == (
                        "python_tracemalloc_separate_replay_peak_increment_v1"
                    )
                    for row in metadata_rows
                )
            )
            fingerprint = metadata_rows[0]["metadata"][
                "runtime_environment_fingerprint"
            ]
            self.assertEqual(
                fingerprint["schema"],
                "mo_nco_runtime_environment_fingerprint_v1",
            )
            self.assertEqual(
                len(fingerprint["python_executable_sha256"]),
                64,
            )
            self.assertIn("numpy", fingerprint["distributions"])
            self.assertTrue(
                all(
                    record.max_diagnostic_archive_size <= 1
                    and record.diagnostic_archive_limit_gate == "PASS"
                    and record.anytime_front_semantics
                    == "cumulative_nondominated_best_so_far_v1"
                    for record in records
                )
            )

    def test_memory_replay_does_not_reset_caller_owned_tracemalloc(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracemalloc.start()
            retained = bytearray(1_000_000)
            peak_before = tracemalloc.get_traced_memory()[1]
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "exclusive tracemalloc ownership",
                ):
                    run_benchmark(
                        algorithms=["random2opt"],
                        seeds=[0],
                        cities=8,
                        population=4,
                        iterations=12,
                        instance_seed=9,
                        output_dir=Path(tmp),
                        log_period=4,
                        archive_update_period=4,
                        measure_python_memory=True,
                    )
                self.assertTrue(tracemalloc.is_tracing())
                self.assertGreaterEqual(
                    tracemalloc.get_traced_memory()[1],
                    peak_before,
                )
                self.assertEqual(len(retained), 1_000_000)
            finally:
                tracemalloc.stop()

    def test_seed_major_balanced_execution_rotates_algorithm_order(self) -> None:
        schedule = build_execution_schedule(
            algorithms=("a", "b", "c", "d"),
            seeds=(10, 11, 12),
            execution_order="seed-major-balanced-v1",
            rotation_offset=1,
        )
        self.assertEqual(
            schedule,
            (
                ("b", 10),
                ("c", 10),
                ("d", 10),
                ("a", 10),
                ("c", 11),
                ("d", 11),
                ("a", 11),
                ("b", 11),
                ("d", 12),
                ("a", 12),
                ("b", 12),
                ("c", 12),
            ),
        )

    def test_benchmark_uses_frozen_external_metric_reference(self) -> None:
        metric_reference = {
            "contract": "frozen_external_v1",
            "hypervolume_reference": [1000.0, 1000.0],
            "ideal": [0.0, 0.0],
            "nadir": [1000.0, 1000.0],
            "reference_front": [[100.0, 900.0], [900.0, 100.0]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_benchmark(
                algorithms=["random2opt"],
                seeds=[0],
                cities=9,
                population=6,
                iterations=24,
                instance_seed=4,
                output_dir=Path(tmp),
                log_period=8,
                archive_update_period=4,
                metric_reference=metric_reference,
                metric_reference_manifest_sha256="abc123",
            )
            metadata = json.loads(
                (Path(tmp) / "run_metadata.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
        self.assertEqual(metadata["metric_reference_contract"], "frozen_external_v1")
        self.assertEqual(metadata["metric_reference_manifest_sha256"], "abc123")
        self.assertEqual(metadata["metric_hypervolume_reference"], [1000.0, 1000.0])

    def test_metric_reference_freeze_rejects_evaluated_theory_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive.csv"
            archive.write_text(
                "tour,objective_1,objective_2\n0 1 2,1.0,2.0\n",
                encoding="utf-8",
            )
            (root / "aggregate_runs.csv").write_text(
                "case,algorithm,seed,archive_csv\n"
                f"case_a,ips-theory-heavy-no-prior,0,{archive}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "external-reference freeze refused"):
                freeze_metric_references((root,))

    def test_benchmark_writes_neural_spectral_logs(self) -> None:
        algorithm = "ips-neural-mv-jitgreedy-sprint"
        with tempfile.TemporaryDirectory() as tmp:
            run_benchmark(
                algorithms=[algorithm],
                seeds=[0],
                cities=10,
                population=6,
                iterations=32,
                instance_seed=5,
                output_dir=Path(tmp),
                log_period=8,
                archive_update_period=8,
            )
            log_path = Path(tmp) / "neural_spectral" / f"{algorithm}_seed0.jsonl"
            self.assertTrue(log_path.exists())
            with log_path.open("r", encoding="utf-8") as handle:
                payload = json.loads(handle.readline())
            self.assertIn("after", payload)
            self.assertIn("lipschitz_proxy", payload["after"])
            metadata_path = Path(tmp) / "run_metadata.jsonl"
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.loads(handle.readline())
            self.assertIn("metadata", metadata)
            self.assertIn("policy_calls", metadata["metadata"])

    def test_backend_specific_neural_prior_path_overrides_generic(self) -> None:
        old_generic = os.environ.get("MO_NCO_NEURAL_PRIOR_PATH")
        old_pcd = os.environ.get("MO_NCO_NEURAL_PRIOR_PATH_PCD")
        old_paretoflow = os.environ.get("MO_NCO_NEURAL_PRIOR_PATH_PARETOFLOW")
        try:
            os.environ["MO_NCO_NEURAL_PRIOR_PATH"] = "generic.json"
            os.environ["MO_NCO_NEURAL_PRIOR_PATH_PCD"] = "pcd.json"
            os.environ["MO_NCO_NEURAL_PRIOR_PATH_PARETOFLOW"] = "paretoflow.json"
            self.assertEqual(_neural_prior_path_for_backend("pcd"), "pcd.json")
            self.assertEqual(_neural_prior_path_for_backend("paretoflow"), "paretoflow.json")
            self.assertEqual(_neural_prior_path_for_backend("tiny"), "generic.json")
        finally:
            for key, value in {
                "MO_NCO_NEURAL_PRIOR_PATH": old_generic,
                "MO_NCO_NEURAL_PRIOR_PATH_PCD": old_pcd,
                "MO_NCO_NEURAL_PRIOR_PATH_PARETOFLOW": old_paretoflow,
            }.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_targetflow_efficient_is_prior_only_and_drops_unsupported_overheads(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            scalar_prior = Path(tmp) / "scalar.json"
            move_prior = Path(tmp) / "move.json"
            scalar_prior.write_text("{}", encoding="utf-8")
            move_prior.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "MO_NCO_NEURAL_PRIOR_PATH_PCD": str(scalar_prior),
                    "MO_NCO_LEARNED_MOVE_PRIOR_PATH": str(move_prior),
                },
            ), patch("mo_nco.benchmark.TheoryAlignedIPSOptimizer") as optimizer:
                sentinel = object()
                optimizer.return_value.run.return_value = sentinel
                result = run_algorithm(
                    "ips-neural-mv-jitgreedy-targetflow-efficient",
                    instance,
                    seed=0,
                    population=8,
                    iterations=64,
                    log_period=16,
                    archive_update_period=8,
                )
            self.assertIs(result, sentinel)
            kwargs = optimizer.call_args.kwargs
            self.assertFalse(kwargs["neural_mean_field_features"])
            self.assertFalse(kwargs["neural_online_training"])
            self.assertEqual(kwargs["neural_flow_pair_samples"], 0)
            self.assertEqual(kwargs["neural_flow_residual_weight"], 0.0)
            self.assertEqual(kwargs["neural_condition_guidance_scale"], 1.0)
            self.assertEqual(kwargs["neural_learned_move_samples"], 4)
            self.assertLess(kwargs["neural_proposal_probability"], 0.24)
            self.assertLess(kwargs["neural_learned_move_probability"], 0.90)

    def test_theory_optimized_alias_requires_endpoint_contract_at_runtime(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            scalar_prior = Path(tmp) / "scalar.json"
            move_prior = Path(tmp) / "move.json"
            scalar_prior.write_text("{}", encoding="utf-8")
            move_prior.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "MO_NCO_NEURAL_PRIOR_PATH_PCD": str(scalar_prior),
                    "MO_NCO_LEARNED_MOVE_PRIOR_PATH": str(move_prior),
                },
            ), patch("mo_nco.benchmark.TheoryAlignedIPSOptimizer") as optimizer:
                optimizer.return_value.run.return_value = object()
                run_algorithm(
                    "ips-neural-mv-jitgreedy-targetflow-theory-optimized",
                    instance,
                    seed=0,
                    population=8,
                    iterations=64,
                    log_period=16,
                    archive_update_period=8,
                )
            self.assertTrue(optimizer.call_args.kwargs["require_endpoint_only_prior"])
            self.assertTrue(optimizer.call_args.kwargs["require_target_only_move_prior"])

    def test_theory_optimized_alias_fails_fast_without_priors(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=10)
        with patch.dict(
            os.environ,
            {
                "MO_NCO_NEURAL_PRIOR_PATH_PCD": "",
                "MO_NCO_NEURAL_PRIOR_PATH": "",
                "MO_NCO_LEARNED_MOVE_PRIOR_PATH": "",
            },
        ):
            with self.assertRaises(FileNotFoundError):
                run_algorithm(
                    "ips-neural-mv-jitgreedy-targetflow-theory-optimized",
                    instance,
                    seed=0,
                    population=8,
                    iterations=64,
                    log_period=16,
                    archive_update_period=8,
                )

    def test_targetflow_efficient_safely_disables_neural_path_without_priors(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=8)
        with patch.dict(
            os.environ,
            {
                "MO_NCO_NEURAL_PRIOR_PATH_PCD": "",
                "MO_NCO_NEURAL_PRIOR_PATH": "",
                "MO_NCO_LEARNED_MOVE_PRIOR_PATH": "",
            },
        ), patch("mo_nco.benchmark.TheoryAlignedIPSOptimizer") as optimizer:
            optimizer.return_value.run.return_value = object()
            run_algorithm(
                "ips-neural-mv-jitgreedy-targetflow-efficient",
                instance,
                seed=0,
                population=8,
                iterations=64,
                log_period=16,
                archive_update_period=8,
            )
        kwargs = optimizer.call_args.kwargs
        self.assertEqual(kwargs["neural_scalar_weight"], 0.0)
        self.assertEqual(kwargs["neural_proposal_probability"], 0.0)
        self.assertEqual(kwargs["neural_learned_move_probability"], 0.0)

    def test_theory_ablation_aliases_change_only_prior_treatment(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=18)
        with tempfile.TemporaryDirectory() as tmp:
            scalar_prior = Path(tmp) / "scalar.json"
            move_prior = Path(tmp) / "move.json"
            scalar_prior.write_text("{}", encoding="utf-8")
            move_prior.write_text("{}", encoding="utf-8")
            calls = {}
            with patch.dict(
                os.environ,
                {
                    "MO_NCO_NEURAL_PRIOR_PATH_PCD": str(scalar_prior),
                    "MO_NCO_LEARNED_MOVE_PRIOR_PATH": str(move_prior),
                },
            ), patch("mo_nco.benchmark.TheoryAlignedIPSOptimizer") as optimizer:
                optimizer.return_value.run.return_value = object()
                for name in (
                    "ips-theory-heavy-no-prior",
                    "ips-theory-endpoint-only",
                    "ips-theory-move-only",
                    "ips-neural-mv-jitgreedy-targetflow-theory-optimized",
                ):
                    run_algorithm(name, instance, 0, 8, 64, 16, 8)
                    calls[name] = optimizer.call_args.kwargs

        invariant_keys = (
            "neighbor_size",
            "archive_parent_probability",
            "archive_parent_sample",
            "initial_2opt_passes",
            "initial_relocate_passes",
            "proposal_2opt_passes",
            "proposal_relocate_passes",
            "jit_polish_fraction",
            "isolate_prior_loading_rng",
            "enable_mechanism_diagnostics",
        )
        anchor = calls["ips-theory-heavy-no-prior"]
        for kwargs in calls.values():
            self.assertEqual({key: kwargs[key] for key in invariant_keys}, {key: anchor[key] for key in invariant_keys})
        self.assertFalse(anchor["enable_neural_scalar"])
        self.assertEqual(anchor["neural_learned_move_probability"], 0.0)
        self.assertTrue(anchor["isolate_prior_loading_rng"])
        self.assertEqual(anchor["ablation_contract"], "theory_search_v2:none")
        self.assertTrue(calls["ips-theory-endpoint-only"]["enable_neural_scalar"])
        self.assertEqual(calls["ips-theory-endpoint-only"]["neural_learned_move_probability"], 0.0)
        self.assertEqual(calls["ips-theory-endpoint-only"]["ablation_contract"], "theory_search_v2:scalar")
        self.assertFalse(calls["ips-theory-move-only"]["enable_neural_scalar"])
        self.assertTrue(calls["ips-theory-move-only"]["allow_move_without_scalar"])
        self.assertEqual(calls["ips-theory-move-only"]["ablation_contract"], "theory_search_v2:move")
        self.assertTrue(calls["ips-neural-mv-jitgreedy-targetflow-theory-optimized"]["enable_neural_scalar"])
        self.assertEqual(
            calls["ips-neural-mv-jitgreedy-targetflow-theory-optimized"]["ablation_contract"],
            "theory_search_v2:full",
        )

    @unittest.skipUnless(_torch_available(), "torch is required for the production PCD prior seam")
    def test_theory_ablation_real_priors_share_initial_rng_contract(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=181)
        algorithms = (
            "ips-theory-heavy-no-prior",
            "ips-theory-endpoint-only",
            "ips-theory-move-only",
            "ips-neural-mv-jitgreedy-targetflow-theory-optimized",
        )
        with tempfile.TemporaryDirectory() as tmp:
            scalar_prior = Path(tmp) / "endpoint_scalar.json"
            move_prior = Path(tmp) / "target_move.json"
            scalar_prior.write_text(
                json.dumps(
                    {
                        "feature_contract": "endpoint_state_v1",
                        "training_samples": 128,
                        "network": PCDResidualScalarNet(6, 8, random.Random(1810)).to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            move_prior.write_text(
                json.dumps(
                    {
                        "move_generator": SparseMoveGenerator(
                            input_dim=16,
                            hidden_units=8,
                            rng=random.Random(1811),
                            flow_head_weight=0.0,
                            mean_field_head_weight=0.0,
                            conductance_head_weight=0.0,
                        ).to_dict()
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "MO_NCO_NEURAL_PRIOR_PATH_PCD": str(scalar_prior),
                    "MO_NCO_LEARNED_MOVE_PRIOR_PATH": str(move_prior),
                },
            ):
                metadata = [
                    run_algorithm(
                        name,
                        instance,
                        seed=181,
                        population=8,
                        iterations=64,
                        log_period=16,
                        archive_update_period=8,
                    ).metadata
                    for name in algorithms
                ]

        self.assertEqual(len({row["initial_population_sha256"] for row in metadata}), 1)
        self.assertEqual(
            len({row["base_rng_state_after_initialization_sha256"] for row in metadata}),
            1,
        )
        self.assertTrue(all(row["prior_loading_rng_isolated"] for row in metadata))
        self.assertEqual(
            {row["ablation_contract"] for row in metadata},
            {
                "theory_search_v2:none",
                "theory_search_v2:scalar",
                "theory_search_v2:move",
                "theory_search_v2:full",
            },
        )

    def test_suite_can_override_case_evaluation_budget(self) -> None:
        suite = BenchmarkSuite(
            name="budget",
            cases=(BenchmarkCase(name="toy", evaluations=1024),),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "mo_nco.benchmark_suite.run_benchmark", return_value=([], {})
        ) as benchmark:
            run_benchmark_suite(
                suite,
                algorithms=("random2opt",),
                seeds=(0,),
                output_dir=Path(tmp),
                default_population=8,
                default_evaluations=512,
                log_period=16,
                archive_update_period=8,
                override_case_evaluations=True,
            )
        self.assertEqual(benchmark.call_args.kwargs["iterations"], 512)

    def test_suite_aggregate_carries_publication_packet_gate(self) -> None:
        suite = BenchmarkSuite(
            name="publication-gate",
            cases=(
                BenchmarkCase(
                    name="toy",
                    cities=8,
                    evaluations=12,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            run_benchmark_suite(
                suite,
                algorithms=("random2opt",),
                seeds=(0,),
                output_dir=output_dir,
                default_population=4,
                default_evaluations=12,
                log_period=4,
                archive_update_period=4,
            )
            with (output_dir / "aggregate_runs.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["publication_certificate_packet_gate"],
            "NOT_APPLICABLE",
        )


if __name__ == "__main__":
    unittest.main()

