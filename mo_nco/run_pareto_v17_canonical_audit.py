"""CLI for the Pareto-SMC v17 canonical theorem packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pareto_v17_canonical_packet import build_canonical_v17_packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    raw = json.loads(args.packet.read_text(encoding="utf-8"))
    result = build_canonical_v17_packet(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.require_pass and not result.overall_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
