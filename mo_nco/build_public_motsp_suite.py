from __future__ import annotations

import argparse
import gzip
import itertools
import json
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .tsplib import parse_tsplib


TSPLIB_RAW_BASE = "https://raw.githubusercontent.com/mastqe/tsplib/master"
PAQUETE_BOTSP_BASE = "https://eden.dei.uc.pt/~paquete/tsp/"

TSPLIB_CANDIDATES: Sequence[str] = (
    "bayg29",
    "bays29",
    "dantzig42",
    "swiss42",
    "att48",
    "gr48",
    "hk48",
    "eil76",
    "pr76",
    "kroA100",
    "kroB100",
    "kroC100",
    "kroD100",
    "kroE100",
    "rd100",
    "ch150",
    "kroA150",
    "kroB150",
    "kroA200",
    "kroB200",
    "lin318",
    "linhp318",
    "ts225",
    "tsp225",
)


@dataclass(frozen=True)
class PublicPair:
    name: str
    first_url: str
    second_url: str


def pacote_botsp_pairs(base_url: str = PAQUETE_BOTSP_BASE) -> Sequence[PublicPair]:
    pairs: List[PublicPair] = []
    for family in ("euclid", "random"):
        for size in (100, 300, 500):
            for left, right in (("A", "B"), ("C", "D"), ("E", "F")):
                pairs.append(
                    PublicPair(
                        name=f"paquete_{family}{left}{right}{size}",
                        first_url=f"{base_url}TSP/{family}{left}{size}.tsp.gz",
                        second_url=f"{base_url}TSP/{family}{right}{size}.tsp.gz",
                    )
                )
    for size in (100, 300, 500):
        for letter in ("G", "H", "I"):
            pairs.append(
                PublicPair(
                    name=f"paquete_mixed{letter}{letter}{size}",
                    first_url=f"{base_url}TSP/euclid{letter}{size}.tsp.gz",
                    second_url=f"{base_url}TSP/random{letter}{size}.tsp.gz",
                )
            )
    return tuple(pairs)


def download_public_tsplib_instances(names: Sequence[str], output_dir: Path, base_url: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name in names:
        path = output_dir / f"{name}.tsp"
        if not path.exists():
            url = f"{base_url}/{name}.tsp"
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    path.write_bytes(response.read())
            except Exception as exc:
                print(f"skip download {name}: {exc}")
                continue
        try:
            parse_tsplib(path)
        except Exception as exc:
            print(f"skip unsupported {name}: {exc}")
            continue
        paths[name] = path
    return paths


def _urlopen_bytes(url: str) -> bytes:
    context = ssl._create_unverified_context() if "eden.dei.uc.pt" in url else None
    with urllib.request.urlopen(url, timeout=45, context=context) as response:
        return response.read()


def download_public_pair(pair: PublicPair, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx, url in enumerate((pair.first_url, pair.second_url), start=1):
        suffix = ".tsp.gz" if url.endswith(".gz") else ".tsp"
        gz_path = output_dir / f"{pair.name}_obj{idx}{suffix}"
        tsp_path = output_dir / f"{pair.name}_obj{idx}.tsp"
        if not tsp_path.exists():
            data = _urlopen_bytes(url)
            if url.endswith(".gz"):
                gz_path.write_bytes(data)
                tsp_path.write_bytes(gzip.decompress(data))
            else:
                tsp_path.write_bytes(data)
        parse_tsplib(tsp_path)
        paths.append(tsp_path)
    return paths


def build_suite(
    paths: Dict[str, Path],
    suite_path: Path,
    max_cases: int,
    population: int | None,
    evaluations: int | None,
    extra_pairs: Sequence[PublicPair] = (),
    extra_output_dir: Path | None = None,
) -> None:
    by_dimension: Dict[int, List[str]] = {}
    for name, path in paths.items():
        problem = parse_tsplib(path)
        by_dimension.setdefault(problem.dimension, []).append(name)

    cases = []
    for dimension in sorted(by_dimension):
        names = sorted(by_dimension[dimension])
        for a, b in itertools.combinations(names, 2):
            cases.append(
                {
                    "name": f"public_{a}_{b}",
                    "kind": "tsplib",
                    "tsplib_files": [
                        str(paths[a]).replace("\\", "/"),
                        str(paths[b]).replace("\\", "/"),
                    ],
                    **({"population": population} if population is not None else {}),
                    **({"evaluations": evaluations} if evaluations is not None else {}),
                }
            )
            if len(cases) >= max_cases:
                break
        if len(cases) >= max_cases:
            break

    if len(cases) < max_cases and extra_pairs:
        pair_dir = extra_output_dir or (suite_path.parent / "public_botsp")
        for pair in extra_pairs:
            try:
                first, second = download_public_pair(pair, pair_dir)
                instance = [str(first).replace("\\", "/"), str(second).replace("\\", "/")]
                parse_tsplib(first)
                parse_tsplib(second)
            except Exception as exc:
                print(f"skip public pair {pair.name}: {exc}")
                continue
            cases.append(
                {
                    "name": pair.name,
                    "kind": "tsplib",
                    "tsplib_files": instance,
                    **({"population": population} if population is not None else {}),
                    **({"evaluations": evaluations} if evaluations is not None else {}),
                }
            )
            if len(cases) >= max_cases:
                break

    payload = {
        "name": "public_pair_motsp_suite",
        "source": [TSPLIB_RAW_BASE, PAQUETE_BOTSP_BASE],
        "construction": (
            "Cases first pair same-DIMENSION TSPLIB instances from the TSPLIB mirror; "
            "additional cases use Paquete-style public BOTSP objective pairs from the Biobjective TSP benchmark page."
        ),
        "cases": cases,
    }
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {suite_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public TSPLIB files and build a paired MOTSP suite.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/public_tsplib"))
    parser.add_argument("--suite-path", type=Path, default=Path("benchmarks/suite_public_motsp.json"))
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--base-url", default=TSPLIB_RAW_BASE)
    parser.add_argument("--include-paquete-botsp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paquete-base-url", default=PAQUETE_BOTSP_BASE)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--evaluations", type=int, default=None)
    args = parser.parse_args()

    paths = download_public_tsplib_instances(TSPLIB_CANDIDATES, args.output_dir, args.base_url)
    build_suite(
        paths,
        args.suite_path,
        args.max_cases,
        args.population,
        args.evaluations,
        extra_pairs=pacote_botsp_pairs(args.paquete_base_url) if args.include_paquete_botsp else (),
        extra_output_dir=args.output_dir / "paquete_botsp",
    )


if __name__ == "__main__":
    main()
