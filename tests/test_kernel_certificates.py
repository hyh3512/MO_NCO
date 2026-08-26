from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.ips_certified import CertifiedSingleSiteIPSOptimizer
from mo_nco.kernel_trace import verify_certified_trace
from mo_nco.run_kernel_certificates import build_kernel_certificate


class KernelCertificateTests(unittest.TestCase):
    @staticmethod
    def _rewrite_hash_chain(records: list[dict]) -> str:
        previous = "0" * 64
        for record in records:
            record.pop("record_hash", None)
            record["previous_record_hash"] = previous
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            previous = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            record["record_hash"] = previous
        return previous

    @staticmethod
    def _strict_metadata() -> dict:
        return {
            "algorithm_contract": "theory_certified_single_site_v4",
            "implementation_version": "0.8.0",
            "claim_level": "certified_mh",
            "context_frozen": True,
            "context_refresh_count": 0,
            "bounds_frozen": True,
            "single_coordinate_transition": True,
            "proposal": "uniform_symmetric_two_opt",
            "proposal_symmetric": True,
            "proposal_log_ratio": 0.0,
            "temperature_schedule": "constant",
            "positive_temperature": True,
            "temperature_min": 0.05,
            "archive_feedback": False,
            "archive_role": "reporting_only_no_kernel_feedback",
            "mean_field_enabled": False,
            "neural_enabled": False,
            "compiled_polish_enabled": False,
            "crossover_enabled": False,
            "local_refinement_enabled": False,
            "acceptance_computation": "log_uniform_comparison",
            "objective_evaluation_contract": "full_tour_state_function",
            "instance_sha256": "a" * 64,
            "explicit_laziness": True,
            "aperiodicity_mechanism": "explicit_identity_mixture",
            "lazy_probability": 0.5,
            "uniformization_rate": 1.0,
            "uniformization_role": "declaration_only_not_executed",
            "db_max_abs_log_residual": 0.0,
            "evaluation_budget": 12,
            "evaluations_used": 12,
            "initial_population_evaluations": 4,
            "proposal_evaluations": 6,
            "accepted_single_site_moves": 4,
            "rejected_single_site_moves": 2,
            "transition_attempts": 8,
            "transition_evaluations": 8,
            "lazy_self_loops": 2,
            "identity_evaluations": 2,
            "evaluation_clock_kernel": "explicit_lazy_identity_mixture",
            "rng_contract": "python_random_mt19937_seed_replay_v1",
            "seed": 0,
        }

    def test_metadata_certificate_accepts_complete_strict_metadata(self) -> None:
        metadata = self._strict_metadata()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case_a"
            case_dir.mkdir()
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 0,
                        "metadata": metadata,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "certificate.md"
            self.assertTrue(build_kernel_certificate(root, "ips-theory-certified", output))
            text = output.read_text(encoding="utf-8")
            self.assertIn("NOT PROVIDED", text)
            self.assertIn("Strongest evidence level: `self_reported_metadata`", text)
            self.assertIn("Overall mechanical certificate: **PASS**", text)

    def test_require_trace_rejects_metadata_only_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case_a"
            case_dir.mkdir()
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 0,
                        "metadata": self._strict_metadata(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "certificate.md"
            self.assertFalse(
                build_kernel_certificate(root, "ips-theory-certified", output, require_trace=True)
            )
            self.assertIn("Overall mechanical certificate: **FAIL**", output.read_text(encoding="utf-8"))

    def test_certificate_rejects_alias_claim_escalation(self) -> None:
        metadata = self._strict_metadata()
        metadata["claim_level"] = "heuristic_descent"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case_a"
            case_dir.mkdir()
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 0,
                        "metadata": metadata,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "certificate.md"
            self.assertFalse(build_kernel_certificate(root, "ips-theory-certified", output))
            self.assertIn("| FAIL |", output.read_text(encoding="utf-8"))

    def test_trace_required_certificate_replays_real_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case_a"
            case_dir.mkdir()
            trace = case_dir / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=810)
            result = CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=12,
                seed=811,
                temperature=0.05,
                trace_path=trace,
            ).run()
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 811,
                        "metadata": result.metadata,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "certificate.md"
            self.assertTrue(
                build_kernel_certificate(root, "ips-theory-certified", output, require_trace=True)
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("trace replay", text)
            self.assertIn("Strongest evidence level: `internal_trace_replay`", text)
            self.assertIn("Overall mechanical certificate: **PASS**", text)

    def test_trace_rejects_rehashed_but_forged_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=820)
            CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=12,
                seed=821,
                temperature=0.05,
                trace_path=trace,
            ).run()
            records = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records[0]["context_hash"] = "f" * 64
            self._rewrite_hash_chain(records)
            trace.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            verified = verify_certified_trace(trace)
            self.assertFalse(verified.passed)
            self.assertIn("header: context hash mismatch", verified.errors)

    def test_trace_rejects_a_rehashed_rng_seed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=823)
            CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=12,
                seed=824,
                temperature=0.05,
                trace_path=trace,
            ).run()
            records = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records[0]["seed"] = 999_824
            self._rewrite_hash_chain(records)
            trace.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            verified = verify_certified_trace(trace)
            self.assertFalse(verified.passed)
            self.assertIn("header: initial population RNG replay mismatch", verified.errors)

    def test_source_replay_binds_initial_objectives_to_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=825)
            CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=12,
                seed=826,
                temperature=0.05,
                trace_path=trace,
            ).run()
            records = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records[0]["initial_objectives"][0][0] += 1.0
            self._rewrite_hash_chain(records)
            trace.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            verified = verify_certified_trace(trace, instance=instance)
            self.assertFalse(verified.passed)
            self.assertIn("header: initial objective mismatch at coordinate 0", verified.errors)

    def test_source_replay_reconstructs_the_frozen_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=827)
            CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=4,
                seed=828,
                temperature=0.05,
                trace_path=trace,
            ).run()
            records = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records[0]["scales"][0] *= 2.0
            context_payload = {
                key: records[0][key]
                for key in (
                    "ideal",
                    "nadir",
                    "scales",
                    "weights",
                    "chebyshev_rho",
                    "temperature",
                    "lazy_probability",
                )
            }
            encoded = json.dumps(
                context_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            records[0]["context_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            self._rewrite_hash_chain(records)
            trace.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            verified = verify_certified_trace(trace, instance=instance)
            self.assertFalse(verified.passed)
            self.assertIn("header: source scale context mismatch", verified.errors)

    def test_certificate_rejects_trace_metadata_cross_link_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case_a"
            case_dir.mkdir()
            trace = case_dir / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=830)
            result = CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=12,
                seed=831,
                temperature=0.05,
                trace_path=trace,
            ).run()
            metadata = dict(result.metadata)
            metadata["trace_chain_hash"] = "0" * 64
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 831,
                        "metadata": metadata,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "certificate.md"
            self.assertFalse(
                build_kernel_certificate(root, "ips-theory-certified", output, require_trace=True)
            )
            self.assertIn("final chain hash mismatch", output.read_text(encoding="utf-8"))

    def test_suite_manifest_source_binding_rejects_a_trace_from_another_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "name": "source_binding",
                        "cases": [
                            {
                                "name": "case_a",
                                "kind": "synthetic",
                                "cities": 8,
                                "instance_seed": 840,
                            },
                            {
                                "name": "case_b",
                                "kind": "synthetic",
                                "cities": 8,
                                "instance_seed": 841,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            case_dir = root / "case_b"
            case_dir.mkdir()
            trace = case_dir / "trace_from_a.jsonl"
            instance_a = MultiObjectiveTSPInstance.random_biobjective(8, seed=840)
            result = CertifiedSingleSiteIPSOptimizer(
                instance_a,
                num_particles=4,
                evaluations=12,
                seed=842,
                temperature=0.05,
                trace_path=trace,
            ).run()
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 842,
                        "metadata": result.metadata,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "certificate.md"
            self.assertFalse(
                build_kernel_certificate(
                    root,
                    "ips-theory-certified",
                    output,
                    require_trace=True,
                    suite_manifest=suite_path,
                )
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("source instance fingerprint mismatch", text)
            self.assertIn("Requested evidence mode: `source_replayed`", text)
            self.assertIn(
                "Strongest evidence level: `no_suite_wide_level_passed`",
                text,
            )

    def test_source_bound_certificate_requires_and_checks_complete_seed_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "name": "complete_source_binding",
                        "cases": [
                            {
                                "name": "case_a",
                                "kind": "synthetic",
                                "cities": 8,
                                "instance_seed": 850,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            case_dir = root / "case_a"
            case_dir.mkdir()
            trace = case_dir / "trace.jsonl"
            instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=850)
            result = CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=4,
                evaluations=12,
                seed=851,
                temperature=0.05,
                trace_path=trace,
            ).run()
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(
                    {
                        "algorithm": "ips-theory-certified",
                        "seed": 851,
                        "metadata": result.metadata,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            incomplete_output = root / "incomplete.md"
            self.assertFalse(
                build_kernel_certificate(
                    root,
                    "ips-theory-certified",
                    incomplete_output,
                    require_trace=True,
                    suite_manifest=suite_path,
                )
            )
            self.assertIn(
                "requires explicit `--expected-seeds`",
                incomplete_output.read_text(encoding="utf-8"),
            )

            complete_output = root / "complete.md"
            self.assertTrue(
                build_kernel_certificate(
                    root,
                    "ips-theory-certified",
                    complete_output,
                    require_trace=True,
                    suite_manifest=suite_path,
                    expected_seeds=(851,),
                )
            )
            complete_text = complete_output.read_text(encoding="utf-8")
            self.assertIn("Suite completeness: `PASS`", complete_text)
            self.assertIn("Strongest evidence level: `source_replayed`", complete_text)

            relabeled = json.loads(
                (case_dir / "run_metadata.jsonl").read_text(encoding="utf-8")
            )
            relabeled["seed"] = 852
            (case_dir / "run_metadata.jsonl").write_text(
                json.dumps(relabeled) + "\n",
                encoding="utf-8",
            )
            relabeled_output = root / "relabeled.md"
            self.assertFalse(
                build_kernel_certificate(
                    root,
                    "ips-theory-certified",
                    relabeled_output,
                    require_trace=True,
                    suite_manifest=suite_path,
                    expected_seeds=(852,),
                )
            )
            self.assertIn(
                "metadata seed mismatch",
                relabeled_output.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()

