from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mo_nco.pareto_ijoc_results_generation import (
    generate_ijoc_formal_result_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify completed IJOC v2 statistical outputs and deterministically "
            "generate manuscript LaTeX macros plus a machine-readable status."
        )
    )
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--post-run-audit", type=Path, required=True)
    parser.add_argument("--statistical-audit", type=Path, required=True)
    parser.add_argument("--paired-inference", type=Path, required=True)
    parser.add_argument(
        "--consumed-artifacts-manifest", type=Path, required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = generate_ijoc_formal_result_artifacts(
        args.matrix_summary,
        args.post_run_audit,
        args.statistical_audit,
        args.paired_inference,
        args.consumed_artifacts_manifest,
        args.output_directory,
    )
    print(
        json.dumps(
            {
                "output_directory": str(result.output_directory),
                "tex_path": str(result.tex_path),
                "status_path": str(result.status_path),
                "primary_superiority_gate": (
                    result.primary_superiority_gate
                ),
                "scientific_result_action": result.scientific_result_action,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
