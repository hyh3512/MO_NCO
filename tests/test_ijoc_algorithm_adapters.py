from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "ijoc_submission_v20" / "scripts" / "ijoc_algorithm_adapter.py"
REPLAY = ROOT / "ijoc_submission_v20" / "scripts" / "ijoc_replay_verifier.py"


def raw(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class IJOCAlgorithmAdapterTests(unittest.TestCase):
    @staticmethod
    def treatment_configuration() -> dict[str, object]:
        return {
            "fixed_core": {
                "reference_directions": [
                    [0.9, 0.1],
                    [0.7, 0.3],
                    [0.5, 0.5],
                    [0.3, 0.7],
                    [0.1, 0.9],
                ],
                "particles_per_reference": 2,
                "beta_schedule": [0.0, 0.5, 1.0],
                "ess_threshold": 0.5,
                "chebyshev_rho": 0.03,
                "global_refresh_probability": 0.0,
                "normalized_cell_width": 0.05,
                "deployment_archive_max_size": 100,
            },
            "frozen_tail_policy": {
                "allocation_policy": "uniform",
                "tail_fraction": 0.3,
                "quota_fraction": 0.0,
                "exp3_exploration": None,
                "reward_weights": {
                    "hypervolume": 0.75,
                    "new_cell": 0.2,
                    "scalar_improvement": 0.05,
                },
            },
        }

    def build_input(
        self,
        root: Path,
        *,
        algorithm: str,
    ) -> Path:
        instance = {
            "schema": "ijoc_mokp_integer_instance_v1",
            "case_id": "mokp-test",
            "family": "MOKP",
            "num_items": 8,
            "num_objectives": 2,
            "item_weights": [1, 2, 3, 4, 2, 1, 3, 2],
            "profits_by_objective": [
                [4, 2, 7, 8, 3, 5, 2, 9],
                [1, 8, 3, 2, 7, 4, 9, 2],
            ],
            "capacity": 8,
            "generator": {
                "name": "unit_test",
                "seed": 1,
                "weight_support": [1, 4],
                "profit_support": [1, 9],
                "capacity_fraction_floor": 0.4,
            },
        }
        instance_path = root / "instance.json"
        instance_path.write_bytes(raw(instance))
        sys.path.insert(0, str(ROOT))
        try:
            from mo_nco.pareto_ijoc_problem import (
                MultiObjectiveKnapsackInstance,
                problem_sha256,
            )

            problem = MultiObjectiveKnapsackInstance(
                tuple(instance["item_weights"]),
                tuple(tuple(row) for row in instance["profits_by_objective"]),
                int(instance["capacity"]),
                str(instance["case_id"]),
            )
            problem_digest = problem_sha256(problem)
        finally:
            sys.path.remove(str(ROOT))
        packet = {
            "schema": "ijoc_case_instance_packet_v1",
            "case_id": "mokp-test",
            "family": "MOKP",
            "problem_sha256": problem_digest,
            "artifacts": [
                {
                    "path": instance_path.name,
                    "sha256": digest(instance_path),
                }
            ],
        }
        packet_path = root / "case_packet.json"
        packet_path.write_bytes(raw(packet))
        run_key = {
            "case_id": "mokp-test",
            "algorithm": algorithm,
            "seed": 2,
            "budget": 60,
        }
        configuration = {
            **run_key,
            "families": ["MOKP"],
        }
        if algorithm == "ijoc-pareto-smc":
            configuration["treatment"] = self.treatment_configuration()
        payload = {
            "schema": "ijoc_cold_process_input_v1",
            "study_sha256": "a" * 64,
            "configuration_matrix_sha256": "b" * 64,
            "execution_plan_sha256": "c" * 64,
            "freeze_receipt_sha256": "d" * 64,
            "run_key": run_key,
            "run_key_sha256": canonical_digest(run_key),
            "configuration": configuration,
            "configuration_sha256": canonical_digest(configuration),
            "instance_artifact": {
                "path": str(packet_path.resolve()),
                "sha256": digest(packet_path),
            },
            "anytime_checkpoint_period": 20,
        }
        input_path = root / "input.json"
        input_path.write_bytes(raw(payload))
        return input_path

    def build_motsp_input(
        self,
        root: Path,
        *,
        algorithm: str,
    ) -> Path:
        objective_paths = []
        for name in ("bayg29.tsp", "bays29.tsp"):
            source = ROOT / "benchmarks" / "public_tsplib" / name
            target = root / name
            shutil.copyfile(source, target)
            objective_paths.append(target)
        sys.path.insert(0, str(ROOT))
        try:
            from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256

            problem = MultiObjectiveTSPInstance.from_tsplib_files(objective_paths)
            problem_digest = instance_sha256(problem)
        finally:
            sys.path.remove(str(ROOT))
        packet = {
            "schema": "ijoc_case_instance_packet_v1",
            "case_id": "motsp-test",
            "family": "MOTSP",
            "problem_sha256": problem_digest,
            "artifacts": [
                {"path": path.name, "sha256": digest(path)}
                for path in objective_paths
            ],
        }
        packet_path = root / "case_packet.json"
        packet_path.write_bytes(raw(packet))
        run_key = {
            "case_id": "motsp-test",
            "algorithm": algorithm,
            "seed": 2,
            "budget": 80,
        }
        configuration: dict[str, object] = {
            **run_key,
            "families": ["MOTSP"],
            "population_size": 40,
        }
        if algorithm == "ijoc-pareto-smc":
            configuration["treatment"] = self.treatment_configuration()
        if algorithm in {
            "motsp-pls-native-v1",
            "motsp-pls-restart-native-v2",
        }:
            configuration["neighborhood_sample"] = 8
        if algorithm == "motsp-pls-restart-native-v2":
            configuration.update(
                {
                    "archive_tolerance": 0.0,
                    "stalled_expansion_policy": (
                        "uniform-random-unvisited-v1"
                    ),
                    "restart_random_attempts": 64,
                    "liveness_contract": (
                        "each_nonterminal_step_adds_evaluation_or_fails_v1"
                    ),
                }
            )
        payload = {
            "schema": "ijoc_cold_process_input_v1",
            "study_sha256": "a" * 64,
            "configuration_matrix_sha256": "b" * 64,
            "execution_plan_sha256": "c" * 64,
            "freeze_receipt_sha256": "d" * 64,
            "run_key": run_key,
            "run_key_sha256": canonical_digest(run_key),
            "configuration": configuration,
            "configuration_sha256": canonical_digest(configuration),
            "instance_artifact": {
                "path": str(packet_path.resolve()),
                "sha256": digest(packet_path),
            },
            "anytime_checkpoint_period": 40,
        }
        input_path = root / "input.json"
        input_path.write_bytes(raw(payload))
        return input_path

    def run_and_replay(self, algorithm: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = self.build_input(root, algorithm=algorithm)
            result_path = root / "algorithm_result.json"
            replay_path = root / "replay.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER),
                    "--input",
                    str(input_path),
                    "--output",
                    str(result_path),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPLAY),
                    "--input",
                    str(input_path),
                    "--result",
                    str(result_path),
                    "--output",
                    str(replay_path),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["evaluations_used"], 60)
            self.assertEqual(result["observed_checkpoints"], [20, 40, 60])
            self.assertEqual(replay["status"], "PASS")
            self.assertEqual(
                replay["checkpoint_artifact_sha256"],
                result["checkpoint_artifact"]["sha256"],
            )

    def run_motsp_and_replay(self, algorithm: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = self.build_motsp_input(root, algorithm=algorithm)
            result_path = root / "algorithm_result.json"
            replay_path = root / "replay.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER),
                    "--input",
                    str(input_path),
                    "--output",
                    str(result_path),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPLAY),
                    "--input",
                    str(input_path),
                    "--result",
                    str(result_path),
                    "--output",
                    str(replay_path),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            self.assertEqual(result["evaluations_used"], 80)
            self.assertEqual(result["observed_checkpoints"], [40, 80])
            self.assertEqual(replay["status"], "PASS")

    def test_mokp_pls_adapter_and_independent_replay(self) -> None:
        self.run_and_replay("mokp-pls-native-v1")

    def test_mokp_treatment_adapter_and_independent_replay(self) -> None:
        self.run_and_replay("ijoc-pareto-smc")

    def test_motsp_treatment_adapter_and_independent_replay(self) -> None:
        self.run_motsp_and_replay("ijoc-pareto-smc")

    def test_motsp_pls_adapter_and_independent_replay(self) -> None:
        self.run_motsp_and_replay("motsp-pls-native-v1")

    def test_motsp_pls_restart_v2_adapter_and_independent_replay(self) -> None:
        self.run_motsp_and_replay("motsp-pls-restart-native-v2")

    def test_motsp_pls_adapter_fixes_exact_archive_tolerance(self) -> None:
        sys.path.insert(0, str(ROOT))
        try:
            from ijoc_submission_v20.scripts import ijoc_algorithm_adapter
            from mo_nco.instance import MultiObjectiveTSPInstance

            first = (
                (0.0, 2.0, 3.0, 4.0, 5.0),
                (2.0, 0.0, 4.0, 5.0, 3.0),
                (3.0, 4.0, 0.0, 2.0, 5.0),
                (4.0, 5.0, 2.0, 0.0, 3.0),
                (5.0, 3.0, 5.0, 3.0, 0.0),
            )
            second = tuple(tuple(reversed(row)) for row in first)
            problem = MultiObjectiveTSPInstance.from_distance_matrices(
                (first, second)
            )
            result = ijoc_algorithm_adapter.run_row(
                "MOTSP",
                problem,
                {
                    "algorithm": "motsp-pls-native-v1",
                    "population_size": 4,
                    "neighborhood_sample": 4,
                },
                seed=7,
                budget=8,
                checkpoint_period=4,
            )
        finally:
            sys.path.remove(str(ROOT))
        self.assertEqual(result.archive.tol, 0.0)
        self.assertEqual(result.metadata["archive_tolerance"], 0.0)

    def test_frozen_motsp_packets_are_in_the_exact_integer_domain(self) -> None:
        sys.path.insert(0, str(ROOT))
        try:
            from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
            from mo_nco.tsplib import parse_tsplib

            packet_root = (
                ROOT / "ijoc_submission_v20" / "formal_study" / "instances"
            )
            cases = 0
            distance_types: list[str] = []
            for packet_path in sorted(packet_root.glob("*.packet.json")):
                packet = json.loads(packet_path.read_text(encoding="utf-8"))
                if packet["family"] != "MOTSP":
                    continue
                cases += 1
                artifact_paths = []
                for binding in packet["artifacts"]:
                    artifact_path = (
                        packet_path.parent / Path(binding["path"])
                    ).resolve()
                    self.assertEqual(digest(artifact_path), binding["sha256"])
                    parsed = parse_tsplib(artifact_path)
                    distance_types.append(parsed.edge_weight_type)
                    artifact_paths.append(artifact_path)
                instance = MultiObjectiveTSPInstance.from_tsplib_files(
                    artifact_paths
                )
                self.assertEqual(instance_sha256(instance), packet["problem_sha256"])
                self.assertTrue(instance.exact_two_opt_delta_in_binary64)
                self.assertTrue(
                    all(
                        value.is_integer()
                        for matrix in instance.distance_matrices
                        for row in matrix
                        for value in row
                    )
                )
        finally:
            sys.path.remove(str(ROOT))
        self.assertEqual(cases, 15)
        self.assertEqual(
            Counter(distance_types),
            Counter({"EUC_2D": 22, "EXPLICIT": 6, "ATT": 2}),
        )

    def test_motsp_pymoo_adapters_and_independent_replay(self) -> None:
        for algorithm in ("pymoo-nsga2", "pymoo-moead"):
            with self.subTest(algorithm=algorithm):
                self.run_motsp_and_replay(algorithm)


if __name__ == "__main__":
    unittest.main()

