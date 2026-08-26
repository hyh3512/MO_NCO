from __future__ import annotations

"""Validation and rendering for the V21e3r1 precedent–mechanism matrix."""

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

_ALLOWED = {"YES", "PARTIAL", "NO", "NOT_REPORTED"}


def load_precedent_matrix(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "v21e3r1_precedent_mechanism_matrix_v1":
        raise ValueError("Wrong precedent matrix schema.")
    components = payload.get("components")
    methods = payload.get("methods")
    if not isinstance(components, list) or not components or len(set(components)) != len(components):
        raise ValueError("The component list must be nonempty and unique.")
    if not isinstance(methods, list) or not methods:
        raise ValueError("The method list must be nonempty.")
    ids: set[str] = set()
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError("Each method row must be an object.")
        method_id = method.get("method_id")
        if not isinstance(method_id, str) or not method_id or method_id in ids:
            raise ValueError("Method IDs must be nonempty and unique.")
        ids.add(method_id)
        mechanisms = method.get("mechanisms")
        if not isinstance(mechanisms, dict) or set(mechanisms) != set(components):
            raise ValueError(f"{method_id} does not cover the exact component set.")
        for component in components[:-1]:
            if mechanisms[component] not in _ALLOWED:
                raise ValueError(f"Unsupported matrix value for {method_id}/{component}.")
        status = mechanisms[components[-1]]
        if not isinstance(status, str) or not status:
            raise ValueError(f"{method_id} needs a current evidence status.")
    if "V21E3R1_CURRENT" not in ids or "MOMAD_2014" not in ids:
        raise ValueError("The current method and MOMAD must both appear.")
    return payload


def _short(component: str) -> str:
    mapping = {
        "decomposition_or_reference_directions": "Decomp/dirs",
        "family_specific_construction": "Construction",
        "single_objective_scalar_local_search": "Scalar LS",
        "pareto_local_search": "Pareto LS",
        "external_nondominated_archive": "External archive",
        "multiple_or_cooperative_populations": "Coop populations",
        "recombination_or_path_guidance": "Recomb/path",
        "typed_neighborhood_replacement": "Typed replace",
        "duplicate_aware_first_true_evaluation_budget": "First-true budget",
        "attempt_physical_start_charge_separation": "A/P/B split",
        "durable_event_ledger": "Durable ledger",
        "objective_archive_replay": "Obj/archive replay",
        "full_algorithm_decision_replay": "Full replay",
        "prospective_adjacent_mechanism_gate": "Adjacent gate",
        "independent_confirmation": "Independent confirm",
        "current_evidence_status": "Evidence status",
    }
    return mapping.get(component, component)


def render_markdown(payload: Mapping[str, object]) -> str:
    components = list(payload["components"])
    methods = list(payload["methods"])
    visible = [
        "decomposition_or_reference_directions",
        "family_specific_construction",
        "single_objective_scalar_local_search",
        "pareto_local_search",
        "external_nondominated_archive",
        "multiple_or_cooperative_populations",
        "recombination_or_path_guidance",
        "typed_neighborhood_replacement",
        "duplicate_aware_first_true_evaluation_budget",
        "durable_event_ledger",
        "objective_archive_replay",
        "full_algorithm_decision_replay",
        "prospective_adjacent_mechanism_gate",
        "independent_confirmation",
    ]
    lines = [
        "# Precedent–Mechanism Matrix",
        "",
        "> Status: targeted primary-source matrix; not a systematic review. `NOT_REPORTED` is not evidence of absence.",
        "",
        "| Method | " + " | ".join(_short(c) for c in visible) + " |",
        "|---|" + "|".join("---" for _ in visible) + "|",
    ]
    symbol = {"YES": "Y", "PARTIAL": "P", "NO": "N", "NOT_REPORTED": "NR"}
    for method in methods:
        mech = method["mechanisms"]
        lines.append(
            "| " + str(method["citation"]) + " | "
            + " | ".join(symbol[str(mech[c])] for c in visible)
            + " |"
        )
    lines.extend(["", "Legend: Y=yes, P=partial/related, N=explicitly absent or conflicting object, NR=not established from the checked source.", "", "## Authorized novelty position", ""])
    lines.extend(f"- {item}" for item in payload["authorized_novelty_position"])
    return "\n".join(lines) + "\n"


def render_csv(payload: Mapping[str, object]) -> str:
    buffer = io.StringIO()
    components = list(payload["components"])
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["method_id", "citation", "doi", "problem_families", *components])
    for method in payload["methods"]:
        writer.writerow([
            method["method_id"],
            method["citation"],
            method.get("doi") or "",
            "; ".join(method.get("problem_families", [])),
            *(method["mechanisms"][component] for component in components),
        ])
    return buffer.getvalue()


def render_latex(payload: Mapping[str, object]) -> str:
    rows = []
    selected = [
        "decomposition_or_reference_directions",
        "family_specific_construction",
        "single_objective_scalar_local_search",
        "pareto_local_search",
        "external_nondominated_archive",
        "multiple_or_cooperative_populations",
        "duplicate_aware_first_true_evaluation_budget",
        "durable_event_ledger",
        "objective_archive_replay",
        "full_algorithm_decision_replay",
    ]
    symbol = {"YES": "Y", "PARTIAL": "P", "NO": "N", "NOT_REPORTED": "NR"}
    for method in payload["methods"]:
        citation = str(method["citation"]).replace("&", "\\&").replace("_", "\\_")
        vals = " & ".join(symbol[str(method["mechanisms"][c])] for c in selected)
        rows.append(f"{citation} & {vals} \\\\")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Targeted precedent--mechanism matrix. NR means not established from the checked source, not absence.}",
        "\\begin{tabular}{l" + "c" * len(selected) + "}",
        "\\hline",
        "Method & " + " & ".join(_short(c).replace("_", "\\_") for c in selected) + " \\\\",
        "\\hline",
        *rows,
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])


def write_rendered_matrix(input_path: str | Path, output_prefix: str | Path) -> dict[str, str]:
    payload = load_precedent_matrix(input_path)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "markdown": str(prefix.with_suffix(".md")),
        "csv": str(prefix.with_suffix(".csv")),
        "latex": str(prefix.with_suffix(".tex")),
    }
    Path(outputs["markdown"]).write_text(render_markdown(payload), encoding="utf-8")
    Path(outputs["csv"]).write_text(render_csv(payload), encoding="utf-8", newline="")
    Path(outputs["latex"]).write_text(render_latex(payload), encoding="utf-8")
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_rendered_matrix(args.input, args.output_prefix), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


