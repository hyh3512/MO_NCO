from __future__ import annotations

import unittest

from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.sampler import IPSMetropolisOptimizer


class SamplerTests(unittest.TestCase):
    def test_small_run_produces_archive_and_diagnostics(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(10, seed=3)
        optimizer = IPSMetropolisOptimizer(
            instance=instance,
            num_particles=8,
            iterations=40,
            seed=3,
            archive_update_period=5,
            log_period=10,
        )
        result = optimizer.run()
        self.assertGreater(len(result.archive), 0)
        self.assertGreater(len(result.diagnostics), 0)
        self.assertGreaterEqual(result.diagnostics[-1].acceptance_rate, 0.0)
        self.assertLessEqual(result.diagnostics[-1].acceptance_rate, 1.0)


if __name__ == "__main__":
    unittest.main()

