from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mo_nco.pareto_ijoc_analysis import analyze_ijoc_formal_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently recompute frozen IJOC metrics and precommitted "
            "paired inference after a PASS post-run audit."
        )
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--post-run-audit", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_ijoc_formal_results(
        args.study,
        args.execution_plan,
        args.results_directory,
        args.post_run_audit,
        args.output_directory,
    )
    print(
        json.dumps(
            {
                "output_directory": str(result.output_directory),
                "audit_path": str(result.audit_path),
                "row_count": result.row_count,
                "formal_metric_statistical_gate": (
                    result.formal_metric_statistical_gate
                ),
                "primary_superiority_gate": (
                    result.primary_superiority_gate
                ),
                "efficiency_claim_gate": result.efficiency_claim_gate,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
