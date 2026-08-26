from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .archive import ArchiveEntry, ParetoArchive
from .evaluation import can_evaluate
from .instance import MultiObjectiveTSPInstance
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector, Tour


class MatureBaselineUnavailable(RuntimeError):
    """Raised when a requested external mature baseline is not configured."""


@dataclass(frozen=True)
class ExternalBaselineConfig:
    """Protocol for mature optimizer implementations.

    The command receives two paths:

        command input.json output.csv

    `input.json` contains the instance name, distance matrices, seed,
    population size, and evaluation budget. `output.csv` must contain one row
    per final solution with columns `tour`, `objective_0`, ..., where `tour` is
    a whitespace-separated permutation with city 0 fixed.
    """

    command: Sequence[str]
    name: str = "external-baseline"
    timeout_seconds: Optional[float] = None


def external_baseline_configuration_sha256(
    config: ExternalBaselineConfig,
) -> str:
    payload = {
        "schema": "external_baseline_command_configuration_v1",
        "name": config.name,
        "command": list(config.command),
        "timeout_seconds": config.timeout_seconds,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def builtin_pymoo_baseline_configuration(
    name: str,
) -> ExternalBaselineConfig:
    """Return the non-overridable in-tree pymoo command for a formal arm."""

    mapping = {
        "pymoo-nsga2": "nsga2",
        "pymoo-moead": "moead",
    }
    try:
        algorithm = mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported built-in pymoo arm: {name}") from error
    return ExternalBaselineConfig(
        command=[
            sys.executable,
            "-m",
            "mo_nco.external_pymoo_baseline",
            algorithm,
        ],
        name=name,
    )


def external_baseline_provenance(
    config: ExternalBaselineConfig,
) -> dict[str, object]:
    """Resolve auditable command tokens and local implementation bytes."""

    command = [str(token) for token in config.command]
    artifacts: list[dict[str, str]] = []
    if command:
        executable = Path(command[0]).expanduser().resolve()
        if executable.is_file():
            artifacts.append(
                {
                    "role": "command_executable",
                    "path": str(executable),
                    "sha256": hashlib.sha256(
                        executable.read_bytes()
                    ).hexdigest(),
                }
            )
    if len(command) >= 3 and command[1] == "-m":
        spec = importlib.util.find_spec(command[2])
        origin = None if spec is None else spec.origin
        if origin:
            module_path = Path(origin).resolve()
            if module_path.is_file():
                artifacts.append(
                    {
                        "role": "python_module_source",
                        "path": str(module_path),
                        "sha256": hashlib.sha256(
                            module_path.read_bytes()
                        ).hexdigest(),
                    }
                )
    else:
        for token in command[1:]:
            candidate = Path(token).expanduser()
            if candidate.is_file():
                resolved = candidate.resolve()
                artifacts.append(
                    {
                        "role": "command_file_argument",
                        "path": str(resolved),
                        "sha256": hashlib.sha256(
                            resolved.read_bytes()
                        ).hexdigest(),
                    }
                )
    return {
        "schema": "external_baseline_provenance_v1",
        "name": config.name,
        "canonical_command": command,
        "timeout_seconds": config.timeout_seconds,
        "local_artifacts": artifacts,
    }


class ExternalBaselineOptimizer:
    """Adapter for mature baselines such as pymoo, PlatEMO, jMetal, or custom C++."""

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        config: ExternalBaselineConfig,
        population_size: int,
        evaluations: int,
        seed: int,
        archive_max_size: Optional[int] = 500,
    ) -> None:
        self.instance = instance
        self.config = config
        self.population_size = population_size
        self.evaluations = evaluations
        self.seed = seed
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.archive_max_size = archive_max_size

    def run(self) -> OptimizationResult:
        if not self.config.command:
            raise MatureBaselineUnavailable("External baseline command is empty.")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.csv"
            input_path.write_text(json.dumps(self._payload(), indent=2), encoding="utf-8")
            subprocess.run(
                [*self.config.command, str(input_path), str(output_path)],
                check=True,
                timeout=self.config.timeout_seconds,
            )
            entries = self._read_output(output_path)
            diagnostics = self._read_diagnostics(output_path.with_suffix(".diagnostics.csv"))

        self.archive.update(entries)
        used_evaluations = getattr(self, "_external_evaluations", self.evaluations)
        if hasattr(self.instance, "evaluations"):
            setattr(self.instance, "evaluations", used_evaluations)
        objectives = tuple(entry.objectives for entry in self.archive.entries)
        tours = tuple(entry.tour for entry in self.archive.entries)
        if not diagnostics:
            hv = self.archive.hypervolume_2d() if self.instance.num_objectives == 2 else 0.0
            diagnostics = (
                Diagnostic(
                    used_evaluations,
                    0.0,
                    0.0,
                    len(self.archive),
                    hv,
                    0.0,
                    0.0,
                    tuple(entry.objectives for entry in self.archive.entries),
                ),
            )
        return OptimizationResult(
            tours,
            objectives,
            self.archive,
            diagnostics,
            metadata={
                "external_command_configuration_sha256": (
                    external_baseline_configuration_sha256(self.config)
                ),
                "external_baseline_provenance": (
                    external_baseline_provenance(self.config)
                ),
                "external_output_objective_equivalence_gate": "PASS",
                "external_output_objective_equivalence_contract": (
                    getattr(
                        self,
                        "_external_objective_equivalence_contract",
                        "local_full_tour_recompute_"
                        "rel1e-12_abs1e-12_v1",
                    )
                ),
                "external_output_objective_max_abs_error": getattr(
                    self,
                    "_maximum_reported_objective_error",
                    0.0,
                ),
                "external_output_verified_rows": getattr(
                    self,
                    "_verified_output_rows",
                    0,
                ),
                "external_anytime_objective_equivalence_gate": "PASS",
                "external_anytime_objective_equivalence_contract": (
                    getattr(
                        self,
                        "_external_anytime_equivalence_contract",
                        "diagnostic_tour_local_full_recompute_"
                        "rel1e-12_abs1e-12_v1",
                    )
                ),
                "external_evaluation_evidence_gate": "PASS",
                "external_evaluation_evidence_contract": (
                    "every_final_row_exact_requested_budget_and_"
                    "ordered_anytime_steps_ending_at_final_v1"
                ),
                "anytime_front_semantics": (
                    "cumulative_nondominated_best_so_far_v1"
                ),
                "native_archive_completeness_gate": (
                    "PASS"
                    if self.archive_max_size is None
                    else "FAIL"
                ),
                "native_archive_completeness_contract": (
                    "unbounded_nondominated_all_evaluated_candidates_v1"
                    if self.archive_max_size is None
                    else "bounded_native_archive"
                ),
            },
        )

    def _payload(self) -> dict:
        base = getattr(self.instance, "base", self.instance)
        matrices = getattr(base, "_distance_matrices", None)
        if matrices is None:
            raise MatureBaselineUnavailable("Instance does not expose distance matrices.")
        return {
            "name": self.instance.name,
            "num_cities": self.instance.num_cities,
            "num_objectives": self.instance.num_objectives,
            "population_size": self.population_size,
            "evaluations": self.evaluations,
            "seed": self.seed,
            "distance_matrices": matrices,
            "anytime_checkpoint_period": getattr(
                self.instance,
                "anytime_checkpoint_period",
                None,
            ),
        }

    def _read_output(self, path: Path) -> List[ArchiveEntry]:
        if not path.exists():
            raise MatureBaselineUnavailable(f"External baseline did not create {path}.")
        entries: List[ArchiveEntry] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tour = tuple(int(item) for item in row["tour"].split())
                self.instance.validate_tour(tour)
                objectives = self._verified_local_objectives(row, tour)
                if not row.get("evaluations"):
                    raise MatureBaselineUnavailable(
                        "External baseline output must report the charged "
                        "evaluation count on every row."
                    )
                parsed_evaluations = self._strict_evaluation_count(
                    row["evaluations"],
                    label="final output",
                    upper_bound=self.evaluations,
                )
                if parsed_evaluations != self.evaluations:
                    raise MatureBaselineUnavailable(
                        "External baseline final output must certify exact use "
                        "of the requested evaluation budget."
                    )
                prior = getattr(
                    self,
                    "_external_evaluations",
                    parsed_evaluations,
                )
                if prior != parsed_evaluations:
                    raise MatureBaselineUnavailable(
                        "External baseline output rows disagree on the "
                        "evaluation count."
                    )
                self._external_evaluations = parsed_evaluations
                entries.append(ArchiveEntry(tour, objectives))
        if not entries:
            raise MatureBaselineUnavailable(
                "External baseline returned no feasible output rows."
            )
        return entries

    def _read_diagnostics(self, path: Path) -> tuple[Diagnostic, ...]:
        if not path.exists():
            return ()
        by_step: dict[int, list[ArchiveEntry]] = {}
        elapsed_by_step: dict[int, float] = {}
        previous_step = 0
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("evaluations"):
                    raise MatureBaselineUnavailable(
                        "External anytime diagnostics must report an "
                        "evaluation count on every row."
                    )
                step = self._strict_evaluation_count(
                    row["evaluations"],
                    label="anytime diagnostic",
                    upper_bound=getattr(
                        self,
                        "_external_evaluations",
                        self.evaluations,
                    ),
                )
                if step < previous_step:
                    raise MatureBaselineUnavailable(
                        "External anytime diagnostics must be ordered by a "
                        "nondecreasing evaluation count."
                    )
                previous_step = step
                raw_elapsed = row.get("elapsed_seconds", "")
                if raw_elapsed:
                    try:
                        elapsed = float(raw_elapsed)
                    except ValueError as error:
                        raise MatureBaselineUnavailable(
                            "External anytime diagnostic has an invalid "
                            "elapsed_seconds value."
                        ) from error
                    if not math.isfinite(elapsed) or elapsed <= 0.0:
                        raise MatureBaselineUnavailable(
                            "External anytime diagnostic elapsed_seconds "
                            "must be positive and finite."
                        )
                    prior_elapsed = elapsed_by_step.setdefault(
                        step,
                        elapsed,
                    )
                    if prior_elapsed != elapsed:
                        raise MatureBaselineUnavailable(
                            "Rows in one external anytime snapshot disagree "
                            "on elapsed_seconds."
                        )
                tour = tuple(int(item) for item in row["tour"].split())
                self.instance.validate_tour(tour)
                objectives = self._verified_local_objectives(row, tour)
                by_step.setdefault(step, []).append(ArchiveEntry(tour, objectives))
        if by_step and max(by_step) != getattr(
            self,
            "_external_evaluations",
            self.evaluations,
        ):
            raise MatureBaselineUnavailable(
                "External anytime diagnostics must include the exact final "
                "evaluation budget."
            )
        diagnostics = []
        cumulative_archive = ParetoArchive(max_size=None)
        for step in sorted(by_step):
            cumulative_archive.update(by_step[step])
            hv = (
                cumulative_archive.hypervolume_2d()
                if self.instance.num_objectives == 2
                else 0.0
            )
            diagnostics.append(
                Diagnostic(
                    step,
                    0.0,
                    0.0,
                    len(cumulative_archive),
                    hv,
                    0.0,
                    0.0,
                    tuple(
                        entry.objectives
                        for entry in cumulative_archive.entries
                    ),
                    elapsed_seconds=elapsed_by_step.get(step, 0.0),
                )
            )
        return tuple(diagnostics)

    @staticmethod
    def _strict_evaluation_count(
        value: str,
        *,
        label: str,
        upper_bound: int,
    ) -> int:
        try:
            raw = float(value)
        except (TypeError, ValueError) as error:
            raise MatureBaselineUnavailable(
                f"External {label} has an invalid evaluation count."
            ) from error
        parsed = int(raw) if math.isfinite(raw) else 0
        if (
            not math.isfinite(raw)
            or raw != parsed
            or parsed <= 0
            or parsed > upper_bound
        ):
            raise MatureBaselineUnavailable(
                f"External {label} has an invalid evaluation count."
            )
        return parsed

    def _verified_local_objectives(
        self,
        row: dict[str, str],
        tour: Tour,
    ) -> ObjectiveVector:
        reported = tuple(
            float(row[f"objective_{idx}"])
            for idx in range(self.instance.num_objectives)
        )
        if any(not math.isfinite(value) for value in reported):
            raise MatureBaselineUnavailable(
                "External baseline reported a non-finite objective value."
            )
        base = getattr(self.instance, "base", self.instance)
        local = tuple(float(value) for value in base.evaluate(tour))
        errors = tuple(
            abs(observed - expected)
            for observed, expected in zip(reported, local)
        )
        exact_required = bool(
            getattr(
                base,
                "exact_two_opt_delta_in_binary64",
                False,
            )
        )
        self._external_objective_equivalence_contract = (
            "local_full_tour_exact_binary64_integer_v1"
            if exact_required
            else "local_full_tour_recompute_rel1e-12_abs1e-12_v1"
        )
        self._external_anytime_equivalence_contract = (
            "diagnostic_tour_local_full_recompute_exact_"
            "binary64_integer_v1"
            if exact_required
            else "diagnostic_tour_local_full_recompute_"
            "rel1e-12_abs1e-12_v1"
        )
        if any(
            (
                observed != expected
                if exact_required
                else not math.isclose(
                    observed,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            for observed, expected in zip(reported, local)
        ):
            raise MatureBaselineUnavailable(
                "External baseline objective output does not match local "
                f"full-tour evaluation: reported={reported}, local={local}."
            )
        self._maximum_reported_objective_error = max(
            getattr(self, "_maximum_reported_objective_error", 0.0),
            *errors,
        )
        self._verified_output_rows = (
            getattr(self, "_verified_output_rows", 0) + 1
        )
        return local


def load_external_baseline_from_env(name: str) -> ExternalBaselineConfig:
    external = external_solver_env_names(name)
    if external is not None:
        direct_env, bridge_env, bridge_solver = external
        command = os.environ.get(direct_env, "").strip()
        if command:
            return ExternalBaselineConfig(command=_split_command(command), name=name)
        bridge_template = os.environ.get(bridge_env, "").strip()
        if bridge_template:
            return ExternalBaselineConfig(
                command=[sys.executable, "-m", "mo_nco.external_motsp_bridge", bridge_solver],
                name=name,
            )

    normalized = name.upper().replace("-", "_")
    aliases = {
        "LKH_MOTSP": "MO_NCO_BASELINE_LKH_MOTSP",
        "EXTERNAL_LKH_MOTSP": "MO_NCO_BASELINE_LKH_MOTSP",
    }
    key = aliases.get(normalized, f"MO_NCO_BASELINE_{normalized}")
    command = os.environ.get(key, "").strip()
    if command:
        return ExternalBaselineConfig(command=_split_command(command), name=name)
    if name == "pymoo-nsga2":
        return builtin_pymoo_baseline_configuration(name)
    if name == "pymoo-moead":
        return builtin_pymoo_baseline_configuration(name)
    if name in {"lkh-scalar", "elkai-lkh", "lkh-derived"}:
        return ExternalBaselineConfig(
            command=[sys.executable, "-m", "mo_nco.external_lkh_baseline"],
            name=name,
        )
    if name in {"lkh-official", "lkh3-official", "official-lkh"}:
        return ExternalBaselineConfig(
            command=[sys.executable, "-m", "mo_nco.external_official_lkh_baseline"],
            name=name,
        )
    if name in {"lkh-2ppls", "official-lkh-2ppls", "tpls-lkh-official"}:
        return ExternalBaselineConfig(
            command=[sys.executable, "-m", "mo_nco.external_lkh_2ppls_baseline"],
            name=name,
        )
    if name in {"paquete-published-tpls", "tpls-published", "published-tpls"}:
        return ExternalBaselineConfig(
            command=[sys.executable, "-m", "mo_nco.external_paquete_published_tpls_baseline"],
            name=name,
        )
    raise MatureBaselineUnavailable(
        _missing_external_message(name, key, external)
        + " The command should write output.csv and, for fair anytime AUC, "
        "output.diagnostics.csv with evaluation-indexed archive snapshots."
    )


def external_solver_env_names(name: str) -> Optional[tuple[str, str, str]]:
    normalized = name.upper().replace("-", "_")
    mapping = {
        "PAQUETE": ("MO_NCO_BASELINE_PAQUETE", "MO_NCO_BRIDGE_PAQUETE", "paquete"),
        "EXTERNAL_PAQUETE": ("MO_NCO_BASELINE_PAQUETE", "MO_NCO_BRIDGE_PAQUETE", "paquete"),
        "TPLS_EXTERNAL": ("MO_NCO_BASELINE_TPLS", "MO_NCO_BRIDGE_TPLS", "tpls"),
        "EXTERNAL_TPLS": ("MO_NCO_BASELINE_TPLS", "MO_NCO_BRIDGE_TPLS", "tpls"),
        "MOGLS_EXTERNAL": ("MO_NCO_BASELINE_MOGLS", "MO_NCO_BRIDGE_MOGLS", "mogls"),
        "EXTERNAL_MOGLS": ("MO_NCO_BASELINE_MOGLS", "MO_NCO_BRIDGE_MOGLS", "mogls"),
    }
    return mapping.get(normalized)


def configured_external_solver(name: str) -> Optional[tuple[str, str, str]]:
    external = external_solver_env_names(name)
    if external is None:
        return None
    direct_env, bridge_env, _ = external
    direct_command = os.environ.get(direct_env, "").strip()
    if direct_command:
        return "direct", direct_env, direct_command
    bridge_template = os.environ.get(bridge_env, "").strip()
    if bridge_template:
        return "bridge", bridge_env, bridge_template
    return None


def _split_command(command: str) -> List[str]:
    parts = shlex.split(command, posix=False)
    return [_strip_balanced_quotes(part) for part in parts]


def _strip_balanced_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _missing_external_message(name: str, fallback_key: str, external: Optional[tuple[str, str, str]]) -> str:
    if external is None:
        return f"Set {fallback_key} to a mature external MOTSP command accepting input.json output.csv."
    direct_env, bridge_env, _ = external
    return (
        f"Set {direct_env} to a mature external MOTSP command accepting input.json output.csv, "
        f"or set {bridge_env} to a real solver wrapper template consumed by mo_nco.external_motsp_bridge."
    )
