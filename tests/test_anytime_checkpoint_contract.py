from __future__ import annotations

import unittest

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.baselines import MOTSPParetoLocalSearchOptimizer
from mo_nco.benchmark import calibrated_anytime_auc
from mo_nco.evaluation import CountingTSPInstance
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer
from mo_nco.sampler import Diagnostic, OptimizationResult


def _result(diagnostics: tuple[Diagnostic, ...]) -> OptimizationResult:
    archive = ParetoArchive(max_size=None)
    archive.update((ArchiveEntry((0, 1, 2, 3), (1.0, 1.0)),))
    return OptimizationResult(
        particles=((0, 1, 2, 3),),
        objectives=((1.0, 1.0),),
        archive=archive,
        diagnostics=diagnostics,
    )


class AnytimeCheckpointContractTests(unittest.TestCase):
    def test_left_step_auc_does_not_reward_sparse_interpolation(self) -> None:
        sparse = _result(
            (
                Diagnostic(
                    10,
                    0.0,
                    0.0,
                    1,
                    1.0,
                    0.0,
                    0.0,
                    ((1.0, 1.0),),
                    10.0,
                ),
            )
        )
        dense = _result(
            (
                Diagnostic(
                    9,
                    0.0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    (),
                    9.0,
                ),
                Diagnostic(
                    10,
                    0.0,
                    0.0,
                    1,
                    1.0,
                    0.0,
                    0.0,
                    ((1.0, 1.0),),
                    10.0,
                ),
            )
        )
        sparse_auc = calibrated_anytime_auc(
            "sparse",
            0,
            sparse,
            1.0,
            (2.0, 2.0),
            10.0,
            10,
        )
        dense_auc = calibrated_anytime_auc(
            "dense",
            0,
            dense,
            1.0,
            (2.0, 2.0),
            10.0,
            10,
        )
        self.assertEqual(sparse_auc[:2], (0.0, 0.0))
        self.assertEqual(dense_auc[:2], (0.0, 0.0))

    def test_formal_auc_rejects_a_missing_common_checkpoint(self) -> None:
        result = _result(
            (
                Diagnostic(
                    10,
                    0.0,
                    0.0,
                    1,
                    1.0,
                    0.0,
                    0.0,
                    ((1.0, 1.0),),
                    1.0,
                ),
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "missing at common evaluation checkpoints",
        ):
            calibrated_anytime_auc(
                "sparse",
                0,
                result,
                1.0,
                (2.0, 2.0),
                1.0,
                10,
                checkpoint_period=5,
            )

    def test_formal_auc_rejects_raw_hv_fallback_without_front(
        self,
    ) -> None:
        result = _result(
            (
                Diagnostic(
                    5,
                    0.0,
                    0.0,
                    1,
                    0.25,
                    0.0,
                    0.0,
                    (),
                    0.5,
                ),
                Diagnostic(
                    10,
                    0.0,
                    0.0,
                    1,
                    1.0,
                    0.0,
                    0.0,
                    ((1.0, 1.0),),
                    1.0,
                ),
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "genuine nonempty archive front",
        ):
            calibrated_anytime_auc(
                "malformed",
                0,
                result,
                1.0,
                (2.0, 2.0),
                1.0,
                10,
                checkpoint_period=5,
            )

    def test_pls_emits_exact_passive_archive_checkpoints(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=9)
        counted = CountingTSPInstance(base, max_evaluations=20)
        result = MOTSPParetoLocalSearchOptimizer(
            counted,  # type: ignore[arg-type]
            population_size=4,
            evaluations=20,
            seed=3,
            log_period=7,
            archive_max_size=None,
            neighborhood_sample=8,
            anytime_checkpoint_period=5,
        ).run()
        observed = {diagnostic.iteration for diagnostic in result.diagnostics}
        self.assertTrue({5, 10, 15, 20}.issubset(observed))
        self.assertEqual(
            result.metadata["observed_anytime_checkpoints"],
            (5, 10, 15, 20),
        )
        for checkpoint in result.metadata["checkpoint_solution_witnesses"]:
            for entry in checkpoint["entries"]:
                self.assertEqual(
                    tuple(entry["objectives"]),
                    base.evaluate(tuple(entry["tour"])),
                )

    def test_smc_emits_exact_passive_archive_checkpoints(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(6, seed=11)
        counted = CountingTSPInstance(base, max_evaluations=12)
        result = AnnealedParetoSMCOptimizer(
            counted,  # type: ignore[arg-type]
            particles_per_reference=2,
            evaluations=12,
            beta_schedule=(0.0, 0.5, 1.0),
            num_reference_types=2,
            resampling_policy="always",
            mutation_steps_by_stage=(1, 1),
            global_refresh_probability=0.1,
            archive_max_size=None,
            anytime_checkpoint_period=3,
            seed=4,
        ).run()
        observed = {diagnostic.iteration for diagnostic in result.diagnostics}
        self.assertTrue({3, 6, 9, 12}.issubset(observed))


if __name__ == "__main__":
    unittest.main()

