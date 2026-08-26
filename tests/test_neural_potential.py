from __future__ import annotations

import unittest

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.neural_potential import NeuralScalarPotential, TinyMLP
from mo_nco.paretoflow_net import ParetoFlowScalarNet
from mo_nco.pcd_net import PCDResidualScalarNet


def _torch_available() -> bool:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class NeuralPotentialTests(unittest.TestCase):
    def test_neural_potential_trains_and_delta_matches_recompute(self) -> None:
        objectives = [(3.0, 5.0), (4.0, 4.0), (5.0, 3.0)]
        archive = ParetoArchive()
        archive.update(
            [
                ArchiveEntry((0, 1, 2), (3.0, 5.0)),
                ArchiveEntry((0, 2, 1), (5.0, 3.0)),
            ]
        )
        potential = NeuralScalarPotential(seed=9, training_epochs=5)
        context = potential.build_context(archive, objectives)
        potential.fit(objectives, archive.entries, context)
        delta = potential.delta_replace(objectives, 1, (3.5, 3.5), context)
        updated = list(objectives)
        updated[1] = (3.5, 3.5)
        expected = potential.empirical_energy(updated, context) - potential.empirical_energy(objectives, context)
        self.assertAlmostEqual(delta, expected)

    def test_tiny_mlp_round_trip(self) -> None:
        import random

        net = TinyMLP(4, 5, random.Random(3))
        features = (0.1, 0.2, 0.3, 0.4)
        restored = TinyMLP.from_dict(net.to_dict(), random.Random(4))
        self.assertAlmostEqual(net.predict(features), restored.predict(features), places=12)

    def test_tiny_mlp_mixed_losses_preserve_bounded_weights(self) -> None:
        import random

        net = TinyMLP(4, 6, random.Random(5))
        x = (0.1, 0.2, 0.3, 0.4)
        y = (0.2, 0.1, 0.4, 0.3)
        net.fit_mixed(
            [x, y],
            [-0.1, -0.3],
            [(x, y, -0.2)],
            [(y, x, 0.01)],
            [(y, x, 0.01)],
            epochs=3,
            learning_rate=0.03,
            flow_residual_weight=0.5,
            ranking_weight=0.2,
            hypercone_weight=0.2,
            weight_norm_bound=1.5,
        )
        self.assertLessEqual(max(abs(value) for row in net.w1 for value in row), 1.5)
        self.assertLessEqual(max(abs(value) for value in net.w2), 1.5)
        self.assertEqual(len(net.predict_batch([x, y])), 2)

    def test_tiny_mlp_spectral_diagnostics_track_clipping(self) -> None:
        import random

        net = TinyMLP(4, 6, random.Random(7))
        net.w1 = [[50.0 * value for value in row] for row in net.w1]
        net.w2 = [50.0 * value for value in net.w2]
        before = net.spectral_diagnostics(bound=0.75)
        self.assertTrue(before["clip_active"])
        self.assertGreater(before["lipschitz_proxy"], 0.0)

        net.clip_weight_norms(0.75)
        after = net.spectral_diagnostics(bound=0.75)
        self.assertLessEqual(after["w1_excess_ratio"], 1.000001)
        self.assertLessEqual(after["w2_excess_ratio"], 1.000001)
        self.assertFalse(after["clip_active"])

    def test_paretoflow_scalar_net_mixed_losses_and_spectral(self) -> None:
        import random

        net = ParetoFlowScalarNet(4, 8, random.Random(11))
        x = (0.1, 0.2, 0.3, 0.4)
        y = (0.2, 0.1, 0.4, 0.3)
        before = net.predict_batch([x, y])
        net.fit_mixed(
            [x, y],
            [-0.1, -0.3],
            [(x, y, -0.2)],
            [(y, x, 0.01)],
            [(y, x, 0.01)],
            epochs=3,
            learning_rate=0.01,
            flow_residual_weight=0.5,
            ranking_weight=0.2,
            hypercone_weight=0.2,
            weight_norm_bound=2.0,
        )
        after = net.predict_batch([x, y])
        self.assertEqual(len(after), 2)
        self.assertNotEqual(before, after)
        diagnostics = net.spectral_diagnostics(bound=2.0)
        self.assertEqual(diagnostics["backend"], "paretoflow_scalar")
        self.assertLessEqual(diagnostics["max_excess_ratio"], 1.000001)

    @unittest.skipUnless(_torch_available(), "PCD residual backend requires torch")
    def test_pcd_residual_scalar_net_mixed_losses_and_spectral(self) -> None:
        import random

        net = PCDResidualScalarNet(20, 16, random.Random(13))
        x = tuple(0.05 * i for i in range(20))
        y = tuple(0.04 * (20 - i) for i in range(20))
        before = net.predict_batch([x, y])
        net.fit_mixed(
            [x, y],
            [-0.1, -0.3],
            [(x, y, -0.2)],
            [(y, x, 0.01)],
            [(y, x, 0.01)],
            epochs=2,
            learning_rate=0.003,
            flow_residual_weight=0.5,
            ranking_weight=0.2,
            hypercone_weight=0.2,
            weight_norm_bound=2.0,
        )
        after = net.predict_batch([x, y])
        self.assertEqual(len(after), 2)
        self.assertNotEqual(before, after)
        diagnostics = net.spectral_diagnostics(bound=2.0)
        self.assertEqual(diagnostics["backend"], "pcd_residual_scalar")
        self.assertLessEqual(diagnostics["max_excess_ratio"], 1.000001)


if __name__ == "__main__":
    unittest.main()

