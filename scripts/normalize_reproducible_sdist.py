"""Normalize a setuptools sdist into a deterministic ``.tar.gz`` archive.

The input must contain exactly one canonical top-level directory named
``mo_nco-0.21.3.14`` and a regular ``PKG-INFO`` member beneath it.  Archive
paths and member types are validated before any output is written.  File bytes
are preserved while tar metadata, PAX metadata, member order, and the gzip
header are rebuilt deterministically for an explicit SOURCE_DATE_EPOCH.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import NamedTuple, Sequence
import uuid


TOP_LEVEL = "mo_nco-0.21.3.14"
PKG_INFO = f"{TOP_LEVEL}/PKG-INFO"
_MAX_GZIP_EPOCH = (1 << 32) - 1
_REGULAR_TYPES = {tarfile.REGTYPE, tarfile.AREGTYPE}


class SdistNormalizationError(ValueError):
    """Raised when an sdist or normalization request violates the contract."""


class _Member(NamedTuple):
    name: str
    is_directory: bool
    executable: bool
    data: bytes


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validated_epoch(value: object) -> int:
    if type(value) is not int or value < 0 or value > _MAX_GZIP_EPOCH:
        raise SdistNormalizationError(
            f"SOURCE_DATE_EPOCH must be an integer in [0, {_MAX_GZIP_EPOCH}]"
        )
    return value


def _canonical_member_name(value: object, *, is_directory: bool) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise SdistNormalizationError(
            "sdist member paths must be nonempty POSIX strings"
        )
    if len(value) >= 2 and value[0].isalpha() and value[1] == ":":
        raise SdistNormalizationError(
            f"drive-absolute sdist member path prohibited: {value!r}"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SdistNormalizationError(
            f"control character prohibited in sdist member path: {value!r}"
        )
    if is_directory and value.endswith("/"):
        value = value[:-1]
    elif not is_directory and value.endswith("/"):
        raise SdistNormalizationError(
            f"regular file path must not end with a slash: {value!r}"
        )
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SdistNormalizationError(
            f"unsafe or noncanonical sdist member path: {value!r}"
        )
    return value


def _read_members(input_path: Path) -> list[_Member]:
    path = input_path.resolve(strict=True)
    if not path.is_file() or not path.name.endswith(".tar.gz"):
        raise SdistNormalizationError("input must be a regular .tar.gz file")
    try:
        archive = tarfile.open(path, mode="r:gz")
    except (tarfile.TarError, EOFError, OSError) as error:
        raise SdistNormalizationError("input is not a readable gzip tar archive") from error

    members: list[_Member] = []
    names: set[str] = set()
    folded: set[str] = set()
    root_directory_count = 0
    pkg_info_is_regular = False
    try:
        for original in archive.getmembers():
            is_directory = original.type == tarfile.DIRTYPE
            is_regular = original.type in _REGULAR_TYPES
            if not is_directory and not is_regular:
                if original.issym():
                    kind = "symlink"
                elif original.islnk():
                    kind = "hardlink"
                elif original.ischr() or original.isblk():
                    kind = "device"
                elif original.isfifo():
                    kind = "fifo"
                else:
                    kind = f"type {original.type!r}"
                raise SdistNormalizationError(
                    f"prohibited {kind} member: {original.name!r}"
                )
            name = _canonical_member_name(
                original.name,
                is_directory=is_directory,
            )
            if name == TOP_LEVEL:
                if not is_directory:
                    raise SdistNormalizationError(
                        "canonical top-level member must be a directory"
                    )
                root_directory_count += 1
            elif not name.startswith(f"{TOP_LEVEL}/"):
                raise SdistNormalizationError(
                    f"member is outside canonical top-level directory: {name!r}"
                )
            if name in names or name.casefold() in folded:
                raise SdistNormalizationError(
                    f"duplicate or case-colliding sdist member path: {name!r}"
                )
            names.add(name)
            folded.add(name.casefold())

            if is_directory:
                raw = b""
                executable = False
                if original.size != 0:
                    raise SdistNormalizationError(
                        f"directory member has nonzero size: {name!r}"
                    )
            else:
                extracted = archive.extractfile(original)
                if extracted is None:
                    raise SdistNormalizationError(
                        f"regular member is not readable: {name!r}"
                    )
                raw = extracted.read()
                if len(raw) != original.size:
                    raise SdistNormalizationError(
                        f"regular member size mismatch: {name!r}"
                    )
                executable = bool(original.mode & 0o111)
                if name == PKG_INFO:
                    pkg_info_is_regular = True
            members.append(_Member(name, is_directory, executable, raw))
    except (tarfile.TarError, EOFError, OSError) as error:
        raise SdistNormalizationError("sdist member stream is malformed") from error
    finally:
        archive.close()

    if root_directory_count != 1:
        raise SdistNormalizationError(
            "sdist must contain exactly one explicit canonical top-level directory"
        )
    if not pkg_info_is_regular:
        raise SdistNormalizationError(
            f"setuptools PKG-INFO regular member is required: {PKG_INFO}"
        )
    return sorted(members, key=lambda member: member.name)


def _normalized_tar(members: list[_Member], epoch: int) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(
        fileobj=buffer,
        mode="w",
        format=tarfile.PAX_FORMAT,
        pax_headers={},
    ) as archive:
        for member in members:
            info = tarfile.TarInfo(member.name)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = epoch
            info.pax_headers = {}
            if member.is_directory:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                archive.addfile(info)
            else:
                info.type = tarfile.REGTYPE
                info.mode = 0o755 if member.executable else 0o644
                info.size = len(member.data)
                archive.addfile(info, io.BytesIO(member.data))
    return buffer.getvalue()


def _normalized_gzip(tar_bytes: bytes, epoch: int) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=buffer,
        mtime=epoch,
    ) as compressed:
        compressed.write(tar_bytes)
    return buffer.getvalue()


def normalize_sdist(input_path: Path, *, epoch: int) -> bytes:
    """Validate and return deterministic normalized sdist bytes."""

    normalized_epoch = _validated_epoch(epoch)
    members = _read_members(input_path)
    return _normalized_gzip(_normalized_tar(members, normalized_epoch), normalized_epoch)


def write_normalized_sdist(
    input_path: Path,
    output_path: Path,
    *,
    epoch: int,
    replace: bool = False,
) -> dict[str, object]:
    """Normalize an sdist and write it exclusively or with explicit replacement."""

    normalized_epoch = _validated_epoch(epoch)
    raw = normalize_sdist(input_path, epoch=normalized_epoch)
    output = output_path.resolve()
    if not output.name.endswith(".tar.gz"):
        raise SdistNormalizationError("output must use the .tar.gz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise SdistNormalizationError(f"refusing to overwrite without --replace: {output}")

    if not replace:
        try:
            with output.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise SdistNormalizationError(
                f"refusing to overwrite without --replace: {output}"
            ) from error
    else:
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as verification:
        member_count = len(verification.getmembers())
    return {
        "schema": "mo_nco_reproducible_sdist_normalization_receipt_v1",
        "status": "PASS_REPRODUCIBLE_SDIST_NORMALIZED",
        "input": str(input_path.resolve(strict=True)),
        "output": str(output),
        "output_bytes": len(raw),
        "output_sha256": _sha256(raw),
        "source_date_epoch": normalized_epoch,
        "member_count": member_count,
        "file_bytes_preserved": True,
        "scientific_authorization_changed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = write_normalized_sdist(
            args.input,
            args.output,
            epoch=args.epoch,
            replace=args.replace,
        )
        print(_canonical_json(receipt).decode("utf-8"))
        return 0
    except (SdistNormalizationError, OSError, tarfile.TarError) as error:
        failure = {
            "schema": "mo_nco_reproducible_sdist_normalization_error_v1",
            "status": "HOLD_SDIST_NORMALIZATION",
            "error": str(error),
        }
        print(_canonical_json(failure).decode("utf-8"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PKG_INFO",
    "SdistNormalizationError",
    "TOP_LEVEL",
    "main",
    "normalize_sdist",
    "write_normalized_sdist",
]
