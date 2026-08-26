from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mo_nco.pareto_v21e3_artifacts import ArtifactRoot


def test_artifact_root_resolves_one_relative_binding(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    payload_path = root / "manifests" / "metric.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(b"{}\n")

    resolved = ArtifactRoot(root).resolve_binding(
        {
            "path": "manifests/metric.json",
            "sha256": hashlib.sha256(b"{}\n").hexdigest(),
            "bytes": 3,
        }
    )

    assert resolved == payload_path.resolve()


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "C:/outside/metric.json",
        "/outside/metric.json",
        "../metric.json",
        "manifests/../../metric.json",
        r"manifests\metric.json",
        "",
        ".",
    ),
)
def test_artifact_root_rejects_absolute_escape_or_ambiguous_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    with pytest.raises(ValueError, match="relative POSIX|artifact_root"):
        ArtifactRoot(tmp_path).resolve_binding(
            {"path": unsafe_path, "sha256": "0" * 64, "bytes": 0}
        )


def test_artifact_root_rejects_malformed_digest_or_byte_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactRoot(tmp_path).resolve_binding(
            {"path": "artifact.json", "sha256": "not-a-digest", "bytes": 2}
        )
    with pytest.raises(ValueError, match="byte-count"):
        ArtifactRoot(tmp_path).resolve_binding(
            {
                "path": "artifact.json",
                "sha256": hashlib.sha256(b"{}").hexdigest(),
                "bytes": True,
            }
        )

