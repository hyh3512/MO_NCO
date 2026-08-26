from __future__ import annotations

"""Split-scoped prospective instance materialization for V21e3.

Development, selection, and confirmation entropy are released independently.
This module has no formal split and cannot create formal-study case bytes.
"""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from .pareto_v21_partitions import _materialize_partition


_ALLOWED_SPLITS = ("development", "selection", "confirmation")


def materialize_v21e3_partition(
    output_root: str | Path,
    *,
    split: str,
    master_seed: bytes,
    sizes: Sequence[int] = (100, 200, 500),
    cases_per_size: int = 2,
) -> Path:
    """Exclusively create one authorized V21e3 non-formal split."""

    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            "split must be development, selection, or confirmation; "
            "formal materialization is unavailable."
        )
    if not isinstance(master_seed, bytes) or not master_seed:
        raise ValueError("master_seed must be nonempty bytes.")
    parsed_sizes = tuple(int(value) for value in sizes)
    if (
        not parsed_sizes
        or len(set(parsed_sizes)) != len(parsed_sizes)
        or any(value < 3 for value in parsed_sizes)
    ):
        raise ValueError("sizes must be unique integers of at least three.")
    if (
        isinstance(cases_per_size, bool)
        or int(cases_per_size) != cases_per_size
        or int(cases_per_size) <= 0
    ):
        raise ValueError("cases_per_size must be a positive integer.")

    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Prospective output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        _materialize_partition(
            partition_root=staging,
            split=split,
            master_seed=master_seed,
            seed_commitment=hashlib.sha256(master_seed).hexdigest(),
            sizes=parsed_sizes,
            cases_per_size=int(cases_per_size),
            calibration_epoch="v21e3",
        )
        os.rename(staging, root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return root / "case_manifest.json"


__all__ = ["materialize_v21e3_partition"]
