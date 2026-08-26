from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.ips_certified import CertifiedSingleSiteIPSOptimizer
from mo_nco.ips_efficient import EfficientIPSOptimizer
from mo_nco.kernel_trace import verify_certified_trace
from mo_nco.potential import ScalarArchivePotential


class OptimizationV5Tests(unittest.TestCase):
    def test_reference_directions_return_exact_many_objective_count(self) -> None:
        directions = ScalarArchivePotential.reference_directions(3, 17)
        self.assertEqual(len(directions), 17)
        self.assertEqual(len(set(directions)), 17)
        for direction in directions:
            self.assertEqual(len(direction), 3)
            self.assertAlmostEqual(sum(direction), 1.0)
            self.assertTrue(all(weight > 0.0 for weight in direction))

    def test_certified_trace_replays_and_detects_tampering(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(9, seed=101)
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "certified.jsonl"
            result = CertifiedSingleSiteIPSOptimizer(
                instance,
                num_particles=5,
                evaluations=20,
                seed=102,
                temperature=0.05,
                trace_path=trace,
            ).run()
            verification = verify_certified_trace(trace, instance=instance)
            self.assertTrue(verification.passed, verification.errors)
            self.assertEqual(verification.transitions, 15)
            self.assertEqual(
                verification.active_transitions + verification.identity_transitions,
                verification.transitions,
            )
            self.assertEqual(result.metadata["trace_chain_hash"], verification.final_chain_hash)

            records = trace.read_text(encoding="utf-8").splitlines()
            payload = json.loads(records[-1])
            payload["accepted"] = not bool(payload["accepted"])
            records[-1] = json.dumps(payload, sort_keys=True)
            trace.write_text("\n".join(records) + "\n", encoding="utf-8")
            tampered = verify_certified_trace(trace, instance=instance)
            self.assertFalse(tampered.passed)
            self.assertTrue(any("hash mismatch" in error or "acceptance decision" in error for error in tampered.errors))

    def test_certified_supports_more_than_two_objectives(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=103)
        matrices = base.distance_matrices + (base.distance_matrices[0],)
        instance = MultiObjectiveTSPInstance.from_distance_matrices(matrices, name="three_objectives")
        result = CertifiedSingleSiteIPSOptimizer(
            instance,
            num_particles=7,
            evaluations=14,
            seed=104,
            temperature=0.1,
        ).run()
        self.assertEqual(len(result.particles), 7)
        self.assertEqual(len(result.objectives[0]), 3)
        self.assertEqual(result.metadata["evaluations_used"], 14)

    def test_certified_rejects_instances_too_small_for_two_opt(self) -> None:
        matrix = (
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices((matrix, matrix))
        with self.assertRaisesRegex(ValueError, "at least four cities"):
            CertifiedSingleSiteIPSOptimizer(instance, num_particles=2, evaluations=4)

    def test_efficient_raw_instance_uses_total_budget_exactly(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=105)
        result = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=24,
            seed=106,
            initialization="random",
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            archive_parent_probability=0.0,
            crossover_probability=0.0,
            jit_polish_fraction=1.1,
            log_period=6,
        ).run()
        self.assertEqual(result.metadata["evaluations_used"], 24)
        self.assertEqual(result.metadata["algorithm_contract"], "fast_nonautonomous_batch_descent_v2")
        self.assertEqual(result.diagnostics[-1].iteration, 24)

    def test_archive_endpoint_geometry_is_cached_within_context(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=107)
        optimizer = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=12,
            seed=108,
            archive_conditioning_weight=1.0,
            initialization="random",
            jit_polish_fraction=1.1,
        )
        objective = optimizer.objectives[0]
        calls = 0
        original = optimizer._archive_bias_terms2

        def counted(value):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return original(value)

        optimizer._archive_bias_terms2 = counted  # type: ignore[method-assign]
        optimizer._scalar2_pairs([(objective, 0), (objective, 1)])
        optimizer._scalar2_pairs([(objective, 2), (objective, 3)])
        self.assertEqual(calls, 1)

    def test_many_objective_archive_truncates_by_crowding(self) -> None:
        archive = ParetoArchive(max_size=4)
        candidates = [
            ArchiveEntry((0, 1, 2, 3), (float(i), float(10 - i), float((i - 5) ** 2)))
            for i in range(10)
        ]
        archive.update(candidates)
        self.assertLessEqual(len(archive), 4)
        self.assertTrue(all(len(entry.objectives) == 3 for entry in archive.entries))


if __name__ == "__main__":
    unittest.main()

