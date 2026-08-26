from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from mo_nco.pareto_ijoc_analysis import build_paired_inference


_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = _ROOT / "ijoc_submission_v20" / "scripts" / name
    module_name = f"versioned_{name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ARCHIVE_BUILDER = _load_script("build_release_archive.py")
_REQUEST_BUILDER = _load_script("build_formal_freeze_request.py")


class VersionedReleaseArchiveTests(unittest.TestCase):
    def test_version_tag_derives_new_archive_manifest_and_prefix(self) -> None:
        paths = _ARCHIVE_BUILDER.resolve_build_paths(
            tag="v20_1_pls_restart",
            archive_path=None,
            manifest_path=None,
            archive_prefix=None,
        )

        self.assertEqual(
            paths.archive_path,
            _ARCHIVE_BUILDER.RELEASE_ROOT
            / "mo_nco_pareto_smc_v20_1_pls_restart_source.tar.gz",
        )
        self.assertEqual(
            paths.manifest_path,
            _ARCHIVE_BUILDER.RELEASE_ROOT
            / "source_file_manifest_v20_1_pls_restart.json",
        )
        self.assertEqual(
            paths.archive_prefix,
            "mo_nco_pareto_smc_v20_1_pls_restart",
        )

    def test_release_output_write_refuses_to_replace_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "evidence.json"
            target.write_bytes(b"immutable-old-evidence")

            with self.assertRaises(FileExistsError):
                _ARCHIVE_BUILDER.write_new_file(
                    target,
                    b"replacement-must-not-land",
                )

            self.assertEqual(target.read_bytes(), b"immutable-old-evidence")

    def test_versioned_release_build_emits_bound_outputs_only(self) -> None:
        canonical_archive_before = _ARCHIVE_BUILDER.ARCHIVE_PATH.read_bytes()
        canonical_manifest_before = _ARCHIVE_BUILDER.MANIFEST_PATH.read_bytes()
        with tempfile.TemporaryDirectory(
            dir=_ARCHIVE_BUILDER.RELEASE_ROOT
        ) as temporary_directory:
            output_root = Path(temporary_directory)
            paths = _ARCHIVE_BUILDER.resolve_build_paths(
                tag="v20_1_pls_restart",
                archive_path=output_root / "tagged.tar.gz",
                manifest_path=output_root / "tagged.json",
                archive_prefix=None,
            )

            result = _ARCHIVE_BUILDER.build_release(
                paths,
                tag="v20_1_pls_restart",
            )

            self.assertTrue(paths.archive_path.is_file())
            self.assertTrue(paths.manifest_path.is_file())
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result["archive"]["path"], paths.archive_path.relative_to(
                _ARCHIVE_BUILDER.SUBMISSION_ROOT
            ).as_posix())
            self.assertEqual(manifest["archive_prefix"], paths.archive_prefix)
            with tarfile.open(paths.archive_path, "r:gz") as archive:
                self.assertIn(
                    f"{paths.archive_prefix}/source_file_manifest.json",
                    archive.getnames(),
                )
        self.assertEqual(
            _ARCHIVE_BUILDER.ARCHIVE_PATH.read_bytes(),
            canonical_archive_before,
        )
        self.assertEqual(
            _ARCHIVE_BUILDER.MANIFEST_PATH.read_bytes(),
            canonical_manifest_before,
        )


class VersionedFreezeRequestTests(unittest.TestCase):
    def test_default_request_identity_remains_backward_compatible(self) -> None:
        options = _REQUEST_BUILDER.resolve_request_build_options(
            tag=None,
            request_path=None,
            source_archive_path=None,
            study_id=None,
            frozen_output_directory=None,
            results_directory=None,
            formal_analysis_plan_path=None,
            motsp_pls_algorithm_id=None,
            motsp_pls_version=None,
        )

        self.assertEqual(options.request_path, _REQUEST_BUILDER.REQUEST_PATH)
        self.assertEqual(
            options.source_archive_path,
            _REQUEST_BUILDER.SOURCE_ARCHIVE_PATH,
        )
        self.assertEqual(
            options.formal_analysis_plan_path,
            _REQUEST_BUILDER.ANALYSIS_PLAN_PATH,
        )
        self.assertEqual(
            options.study_id,
            "pareto_smc_v20_ijoc_motsp_mokp_30case_10seed_3budget_v1",
        )
        self.assertEqual(
            options.motsp_pls_algorithm_id,
            "motsp-pls-native-v1",
        )
        self.assertEqual(
            options.motsp_pls_version,
            "native-v1-population-40-neighborhood-40",
        )

    def test_obsolete_archive_exact_tag_does_not_select_restart_v2(self) -> None:
        options = _REQUEST_BUILDER.resolve_request_build_options(
            tag="v20_1_archive_exact",
            request_path=None,
            source_archive_path=None,
            study_id=None,
            frozen_output_directory=None,
            results_directory=None,
            formal_analysis_plan_path=None,
            motsp_pls_algorithm_id=None,
            motsp_pls_version=None,
        )

        self.assertEqual(
            options.motsp_pls_algorithm_id,
            "motsp-pls-native-v1",
        )
        self.assertEqual(
            options.motsp_pls_version,
            "native-v1-population-40-neighborhood-40",
        )

    def test_pls_restart_tag_derives_isolated_successor_identity(self) -> None:
        options = _REQUEST_BUILDER.resolve_request_build_options(
            tag="v20_1_pls_restart",
            request_path=None,
            source_archive_path=None,
            study_id=None,
            frozen_output_directory=None,
            results_directory=None,
            formal_analysis_plan_path=None,
            motsp_pls_algorithm_id=None,
            motsp_pls_version=None,
        )

        self.assertEqual(
            options.request_path,
            _REQUEST_BUILDER.SUBMISSION_ROOT
            / "freeze_request_v20_1_pls_restart.json",
        )
        self.assertEqual(
            options.source_archive_path,
            _REQUEST_BUILDER.RELEASE_ROOT
            / "mo_nco_pareto_smc_v20_1_pls_restart_source.tar.gz",
        )
        self.assertEqual(
            options.study_id,
            "pareto_smc_v20_1_pls_restart_ijoc_motsp_mokp_"
            "30case_10seed_3budget_v1",
        )
        self.assertEqual(
            options.motsp_pls_algorithm_id,
            "motsp-pls-restart-native-v2",
        )
        self.assertEqual(
            options.motsp_pls_version,
            "restart-native-v2-population-40-neighborhood-40-"
            "exact-archive-retry-64",
        )
        self.assertEqual(
            options.frozen_output_directory,
            _REQUEST_BUILDER.FORMAL_ROOT / "frozen_v20_1_pls_restart",
        )
        self.assertEqual(
            options.results_directory,
            _REQUEST_BUILDER.FORMAL_ROOT
            / "formal_results_v20_1_pls_restart",
        )
        self.assertEqual(
            options.formal_analysis_plan_path,
            _REQUEST_BUILDER.SUBMISSION_ROOT
            / "protocol"
            / "formal_analysis_plan_v20_1_pls_restart.json",
        )

    def test_successor_request_binds_identity_and_exact_archive_provenance(
        self,
    ) -> None:
        canonical_request_before = _REQUEST_BUILDER.REQUEST_PATH.read_bytes()
        with tempfile.TemporaryDirectory(
            dir=_REQUEST_BUILDER.SUBMISSION_ROOT
        ) as temporary_directory:
            request_path = Path(temporary_directory) / "request.json"
            options = _REQUEST_BUILDER.resolve_request_build_options(
                tag="v20_1_pls_restart",
                request_path=request_path,
                source_archive_path=_REQUEST_BUILDER.SOURCE_ARCHIVE_PATH,
                study_id=None,
                frozen_output_directory=None,
                results_directory=None,
                formal_analysis_plan_path=None,
                motsp_pls_algorithm_id=None,
                motsp_pls_version=None,
            )

            result = _REQUEST_BUILDER.build_freeze_request(options)

            payload = json.loads(request_path.read_text(encoding="utf-8"))
            pls = payload["algorithms"]["motsp-pls-restart-native-v2"]
            self.assertEqual(payload["study_id"], options.study_id)
            self.assertEqual(pls["version"], options.motsp_pls_version)
            self.assertNotIn("motsp-pls-native-v1", payload["algorithms"])
            self.assertEqual(pls["configuration"]["archive_tolerance"], 0.0)
            self.assertEqual(
                pls["configuration"]["stalled_expansion_policy"],
                "uniform-random-unvisited-v1",
            )
            self.assertEqual(
                pls["configuration"]["restart_random_attempts"],
                64,
            )
            self.assertEqual(
                pls["configuration"]["liveness_contract"],
                "each_nonterminal_step_adds_evaluation_or_fails_v1",
            )
            provenance = pls["configuration"]["successor_provenance"]
            self.assertEqual(
                provenance["exact_archive_equivalence_domain"],
                "integer_valued_motsp_objectives",
            )
            self.assertEqual(
                provenance["restart_trajectory_equivalence_claim"],
                "NOT_CLAIMED",
            )
            motsp = next(
                family
                for family in payload["problem_families"]
                if family["id"] == "MOTSP"
            )
            self.assertIn(
                "motsp-pls-restart-native-v2",
                motsp["required_baselines"],
            )
            self.assertNotIn(
                "motsp-pls-native-v1",
                motsp["required_baselines"],
            )
            expected_rows = (
                sum(
                    len(family["cases"]) * len(family["algorithms"])
                    for family in payload["problem_families"]
                )
                * len(payload["seeds"])
                * len(payload["budgets"])
            )
            self.assertEqual(expected_rows, 3600)
            self.assertEqual(result["expected_run_count"], 3600)
            for command in payload["reproduction_commands"]:
                self.assertIn("ijoc_submission_v20\\", command)
            self.assertEqual(result["study_id"], options.study_id)
            self.assertEqual(
                result["request_path"],
                request_path.relative_to(
                    _REQUEST_BUILDER.SUBMISSION_ROOT
                ).as_posix(),
            )
            with self.assertRaises(FileExistsError):
                _REQUEST_BUILDER.build_freeze_request(options)
        self.assertEqual(
            _REQUEST_BUILDER.REQUEST_PATH.read_bytes(),
            canonical_request_before,
        )

    def test_successor_changes_only_the_motsp_pls_baseline_arm(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=_REQUEST_BUILDER.SUBMISSION_ROOT
        ) as temporary_directory:
            root = Path(temporary_directory)
            canonical_options = _REQUEST_BUILDER.resolve_request_build_options(
                tag=None,
                request_path=root / "canonical.json",
                source_archive_path=_REQUEST_BUILDER.SOURCE_ARCHIVE_PATH,
                study_id=None,
                frozen_output_directory=root / "canonical-frozen",
                results_directory=root / "canonical-results",
                formal_analysis_plan_path=None,
                motsp_pls_algorithm_id=None,
                motsp_pls_version=None,
            )
            successor_options = _REQUEST_BUILDER.resolve_request_build_options(
                tag="v20_1_pls_restart",
                request_path=root / "successor.json",
                source_archive_path=_REQUEST_BUILDER.SOURCE_ARCHIVE_PATH,
                study_id=None,
                frozen_output_directory=root / "successor-frozen",
                results_directory=root / "successor-results",
                formal_analysis_plan_path=None,
                motsp_pls_algorithm_id=None,
                motsp_pls_version=None,
            )
            _REQUEST_BUILDER.build_freeze_request(canonical_options)
            _REQUEST_BUILDER.build_freeze_request(successor_options)
            canonical = json.loads(
                canonical_options.request_path.read_text(encoding="utf-8")
            )
            successor = json.loads(
                successor_options.request_path.read_text(encoding="utf-8")
            )

            for field in (
                "seeds",
                "budgets",
                "anytime_checkpoint_period",
                "tail_calibration_suite_receipt_path",
                "reference_calibration_precommit_path",
                "reference_calibration_completion_receipt_path",
                "tail_policy_artifact_path",
            ):
                self.assertEqual(canonical[field], successor[field])
            self.assertEqual(
                successor["formal_analysis_plan_path"],
                "protocol/formal_analysis_plan_v20_1_pls_restart.json",
            )
            canonical_algorithms = dict(canonical["algorithms"])
            successor_algorithms = dict(successor["algorithms"])
            canonical_algorithms.pop("motsp-pls-native-v1")
            successor_algorithms.pop("motsp-pls-restart-native-v2")
            self.assertEqual(canonical_algorithms, successor_algorithms)
            canonical_families = {
                family["id"]: family for family in canonical["problem_families"]
            }
            successor_families = {
                family["id"]: family for family in successor["problem_families"]
            }
            self.assertEqual(
                canonical_families["MOKP"],
                successor_families["MOKP"],
            )
            self.assertEqual(
                canonical_families["MOTSP"]["cases"],
                successor_families["MOTSP"]["cases"],
            )
            self.assertEqual(
                [
                    "pymoo-nsga2",
                    "pymoo-moead",
                    "motsp-pls-restart-native-v2",
                ],
                successor_families["MOTSP"]["required_baselines"],
            )

    def test_successor_analysis_plan_has_only_predeclared_differences(
        self,
    ) -> None:
        canonical = json.loads(
            _REQUEST_BUILDER.ANALYSIS_PLAN_PATH.read_text(encoding="utf-8")
        )
        successor = json.loads(
            _REQUEST_BUILDER.SUCCESSOR_ANALYSIS_PLAN_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            successor["required_baselines"]["MOTSP"],
            [
                "pymoo-nsga2",
                "pymoo-moead",
                "motsp-pls-restart-native-v2",
            ],
        )
        amendment = successor["missing_or_failed_rows"][
            "successor_liveness_amendment"
        ]
        self.assertEqual(
            amendment["liveness_contract"],
            "each_nonterminal_step_adds_evaluation_or_fails_v1",
        )
        self.assertEqual(
            amendment["unchanged_design"],
            "cases_seeds_budgets_metrics_statistics_holm_and_estimand",
        )
        normalized = json.loads(json.dumps(successor))
        normalized["plan_id"] = canonical["plan_id"]
        normalized["required_baselines"]["MOTSP"] = canonical[
            "required_baselines"
        ]["MOTSP"]
        normalized["missing_or_failed_rows"] = canonical[
            "missing_or_failed_rows"
        ]
        self.assertEqual(normalized, canonical)

    def test_paired_analysis_contract_accepts_restart_v2_baseline(self) -> None:
        plan = json.loads(
            _REQUEST_BUILDER.SUCCESSOR_ANALYSIS_PLAN_PATH.read_text(
                encoding="utf-8"
            )
        )
        plan["formal_seeds"] = [8100]
        plan["evaluation_budgets"] = [100]
        plan["primary_budget"] = 100
        plan["uncertainty"]["case_cluster_bootstrap_replicates"] = 20
        rows = []
        for family in plan["families"]:
            algorithms = [
                plan["treatment"],
                *plan["required_baselines"][family],
            ]
            for case_index in range(3):
                for algorithm in algorithms:
                    treatment = algorithm == plan["treatment"]
                    rows.append(
                        {
                            "family": family,
                            "case_id": f"{family.lower()}-{case_index}",
                            "algorithm": algorithm,
                            "seed": 8100,
                            "budget": 100,
                            "metrics": {
                                "normalized_left_continuous_hypervolume_auc": (
                                    0.8 if treatment else 0.7
                                ),
                                "normalized_final_hypervolume": (
                                    0.9 if treatment else 0.8
                                ),
                                "igd_plus_to_frozen_supplied_reference": (
                                    0.1 if treatment else 0.2
                                ),
                                "additive_epsilon_to_frozen_supplied_reference": (
                                    0.1 if treatment else 0.2
                                ),
                                "wall_time_seconds": (
                                    1.0 if treatment else 2.0
                                ),
                                "sampled_peak_process_tree_rss_bytes": (
                                    100 if treatment else 200
                                ),
                            },
                        }
                    )

        report = build_paired_inference(rows, plan=plan)

        self.assertEqual(len(report["primary_comparisons"]), 6)
        self.assertIn(
            "motsp-pls-restart-native-v2",
            {
                comparison["baseline"]
                for comparison in report["primary_comparisons"]
            },
        )


if __name__ == "__main__":
    unittest.main()

