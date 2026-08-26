from __future__ import annotations

"""Single-root, fail-closed artifact bindings for prospective V21e3 work."""

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping


@dataclass(frozen=True)
class ArtifactRoot:
    """Resolve frozen artifact bindings under exactly one declared root."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())

    def resolve_binding(self, binding: Mapping[str, object]) -> Path:
        raw_path = binding.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("Artifact binding path must be a relative POSIX string.")
        if (
            not raw_path
            or raw_path == "."
            or "\\" in raw_path
            or PureWindowsPath(raw_path).is_absolute()
        ):
            raise ValueError("Artifact binding path must be a relative POSIX string.")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact binding must stay below artifact_root.")
        target = (self.path / Path(*relative.parts)).resolve()
        target.relative_to(self.path)
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding.get("sha256"):
            raise ValueError(f"Artifact SHA-256 mismatch: {raw_path}")
        if "bytes" in binding and len(payload) != int(binding["bytes"]):
            raise ValueError(f"Artifact byte-count mismatch: {raw_path}")
        return target


__all__ = ["ArtifactRoot"]
