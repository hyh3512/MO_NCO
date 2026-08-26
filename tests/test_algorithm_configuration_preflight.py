from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mo_nco.benchmark import run_benchmark
from mo_nco.benchmark_suite import (
    BenchmarkCase,
    BenchmarkSuite,
    build_algorithm_configuration_manifest,
    load_and_verify_algorithm_configuration_manifest,
)
from mo_nco.instance import MultiObjectiveTSPInstance


class AlgorithmConfigurationPreflightTests(unittest.TestCase):
    @staticmethod
    def _suite() -> BenchmarkSuite:
        return BenchmarkSuite(
            name="prelaunch-smoke",
            cases=(
                BenchmarkCase(
                    name="smoke",
                    cities=4,
                    instance_seed=410,
                    population=4096,
                    evaluations=24576,
                ),
            ),
        )

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "MO_NCO_PARETO_SMC_SPEC": str(
                Path(
                    "benchmarks/pareto_smc_v11_fixed_smoke_spec.json"
                ).resolve()
            ),
            "MO_NCO_PARETO_FIXED_REFERENCE_SPEC": "",
            "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST": str(
                Path(
                    "benchmarks/"
                    "pareto_smc_v11_fixed_reference_smoke_manifest.json"
                ).resolve()
            ),
            "MO_NCO_BASELINE_PYMOO_NSGA2": (
                "C:\\untrusted\\replacement.exe"
            ),
        }

    def test_complete_manifest_is_built_without_objective_evaluation(
        self,
    ) -> None:
        algorithms = (
            "pareto-smc-pilot-confirm-v11",
            "pareto-smc-pilot-confirm-v12",
            "ips-theory-certified",
            "pymoo-nsga2",
            "pymoo-moead",
            "motsp-pls",
        )
        with patch.dict(os.environ, self._environment()), patch.object(
            MultiObjectiveTSPInstance,
            "evaluate",
            side_effect=AssertionError(
                "prelaunch must not evaluate a tour"
            ),
        ):
            manifest = build_algorithm_configuration_manifest(
                suite=self._suite(),
                suite_sha256="a" * 64,
                algorithms=algorithms,
                seeds=(0,),
                default_population=4096,
                default_evaluations=24576,
                log_period=1024,
                archive_update_period=1024,
                override_case_evaluations=False,
                output_archive_limit=100,
                certified_traces=True,
            )
        self.assertEqual(len(manifest["runs"]), 6)
        self.assertEqual(
            manifest["schema"],
            "pareto_smc_algorithm_configuration_manifest_v2",
        )
        anchor = manifest["runs"][0]
        self.assertEqual(anchor["search_evaluations"], 0)
        self.assertEqual(anchor["pilot_evaluations"], 12288)
        self.assertEqual(anchor["confirm_evaluations"], 12288)
        v12 = next(
            row
            for row in manifest["runs"]
            if row["algorithm"] == "pareto-smc-pilot-confirm-v12"
        )
        self.assertEqual(
            v12["algorithm_configuration"]["algorithm_specific"][
                "certificate_mode"
            ],
            "regeneration",
        )
        self.assertEqual(
            anchor["algorithm_configuration"]["algorithm_specific"][
                "certificate_mode"
            ],
            "published",
        )
        pymoo = next(
            row
            for row in manifest["runs"]
            if row["algorithm"] == "pymoo-nsga2"
        )
        provenance = pymoo["algorithm_configuration"][
            "algorithm_specific"
        ]["external_baseline_provenance"]
        self.assertEqual(
            provenance["canonical_command"][1:],
            [
                "-m",
                "mo_nco.external_pymoo_baseline",
                "nsga2",
            ],
        )
        self.assertEqual(len(provenance["local_artifacts"]), 2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configurations.json"
            path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            index, digest = (
                load_and_verify_algorithm_configuration_manifest(
                    path,
                    expected=manifest,
                )
            )
        self.assertEqual(len(index), 6)
        self.assertEqual(len(digest), 64)

    def test_nonexact_anchor_budget_fails_before_search(self) -> None:
        with patch.dict(os.environ, self._environment()), patch.object(
            MultiObjectiveTSPInstance,
            "evaluate",
            side_effect=AssertionError,
        ):
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                build_algorithm_configuration_manifest(
                    suite=self._suite(),
                    suite_sha256="a" * 64,
                    algorithms=("pareto-smc-pilot-confirm-v11",),
                    seeds=(0,),
                    default_population=4096,
                    default_evaluations=24575,
                    log_period=1024,
                    archive_update_period=1024,
                    override_case_evaluations=True,
                    output_archive_limit=100,
                    certified_traces=False,
                )

    def test_launch_manifest_mismatch_prevents_algorithm_start(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=4)
        with tempfile.TemporaryDirectory() as directory, patch(
            "mo_nco.benchmark.run_algorithm",
            side_effect=AssertionError("algorithm must not start"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "configuration mismatch",
            ):
                run_benchmark(
                    algorithms=("random2opt",),
                    seeds=(0,),
                    cities=8,
                    population=4,
                    iterations=12,
                    instance_seed=4,
                    output_dir=Path(directory),
                    log_period=4,
                    archive_update_period=4,
                    instance=instance,
                    case_name="case-a",
                    expected_algorithm_configurations={
                        ("random2opt", 0): "0" * 64
                    },
                )


if __name__ == "__main__":
    unittest.main()

