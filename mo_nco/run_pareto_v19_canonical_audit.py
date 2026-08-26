"""CLI for the Pareto-SMC v19 canonical theorem packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pareto_v19_canonical_packet import build_canonical_v19_packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute and audit a Pareto-SMC v19 canonical packet."
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = json.loads(args.packet.read_text(encoding="utf-8"))
    result = build_canonical_v19_packet(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.require_pass and not result.overall_v19_extension_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
