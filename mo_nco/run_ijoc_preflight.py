from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pareto_ijoc_preflight import audit_ijoc_competitive_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen IJOC study before launch.")
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = audit_ijoc_competitive_study(args.study)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.metadata(), sort_keys=True))
    if args.require_pass and result.submission_preflight_gate != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
