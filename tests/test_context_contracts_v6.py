from __future__ import annotations

import unittest

from mo_nco.contracts import ClaimLevel
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.ips_efficient import EfficientIPSOptimizer


class ContextContractV6Tests(unittest.TestCase):
    def test_python_path_replays_every_normalization_context_jump(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(14, seed=901)
        result = EfficientIPSOptimizer(
            instance,
            num_particles=6,
            evaluations=42,
            seed=902,
            archive_update_period=2,
            archive_conditioning_weight=2.0,
            log_period=7,
            jit_polish_fraction=1.0,
        ).run()
        metadata = result.metadata

        self.assertEqual(metadata["claim_level"], ClaimLevel.HEURISTIC_DESCENT.value)
        self.assertTrue(metadata["context_jump_accounting_complete"])
        self.assertEqual(metadata["unattributed_compiled_event_count"], 0)
        self.assertEqual(
            metadata["normalization_refresh_count"],
            metadata["context_jump_event_counts"].get("normalization_refresh", 0),
        )
        self.assertAlmostEqual(
            metadata["cumulative_positive_context_jump"],
            sum(metadata["positive_context_jump_by_kind"].values()),
            places=12,
        )
        self.assertEqual(metadata["context_jump_accounting_errors"], 0)


if __name__ == "__main__":
    unittest.main()


