from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "normalize_reproducible_sdist.py"
SPEC = importlib.util.spec_from_file_location("normalize_reproducible_sdist", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
NORMALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NORMALIZER)


def _member(
    name: str,
    *,
    data: bytes | None = None,
    mode: int = 0o644,
    mtime: int = 1,
    uid: int = 1,
    gid: int = 2,
    uname: str = "user",
    gname: str = "group",
    member_type: bytes | None = None,
    linkname: str = "",
    pax_headers: dict[str, str] | None = None,
) -> tuple[tarfile.TarInfo, bytes | None]:
    info = tarfile.TarInfo(name)
    info.mode = mode
    info.mtime = mtime
    info.uid = uid
    info.gid = gid
    info.uname = uname
    info.gname = gname
    info.pax_headers = dict(pax_headers or {})
    if member_type is None:
        member_type = tarfile.DIRTYPE if data is None else tarfile.REGTYPE
    info.type = member_type
    info.linkname = linkname
    if member_type in (tarfile.REGTYPE, tarfile.AREGTYPE):
        payload = data if data is not None else b""
        info.size = len(payload)
        return info, payload
    info.size = 0
    return info, None


def _write_archive(
    path: Path,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
    *,
    gzip_mtime: int,
    gzip_filename: str,
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename=gzip_filename,
            mode="wb",
            fileobj=raw,
            mtime=gzip_mtime,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for info, data in members:
                    archive.addfile(
                        info,
                        None if data is None else io.BytesIO(data),
                    )


def _valid_members(*, drift: int, reverse: bool) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    top = NORMALIZER.TOP_LEVEL
    long_name = f"{top}/docs/{'x' * 110}.txt"
    members = [
        _member(f"{top}/", mtime=drift, uid=100 + drift, gid=200 + drift),
        _member(
            NORMALIZER.PKG_INFO,
            data=b"Metadata-Version: 2.4\nName: mo-nco\nVersion: 0.21.3.14\n",
            mtime=drift + 1,
            uid=101 + drift,
            gid=201 + drift,
            pax_headers={"comment": f"drift-{drift}"},
        ),
        _member(f"{top}/bin/", mtime=drift + 2, mode=0o700),
        _member(
            f"{top}/bin/tool.py",
            data=b"#!/usr/bin/env python\nprint('same bytes')\n",
            mode=0o711,
            mtime=drift + 3,
        ),
        _member(
            f"{top}/README.md",
            data=b"same readme bytes\n",
            mode=0o600,
            mtime=drift + 4,
        ),
        _member(f"{top}/docs/", mtime=drift + 5),
        _member(
            long_name,
            data=b"long path bytes\n",
            mtime=drift + 6,
            pax_headers={"comment": f"different-{drift}"},
        ),
    ]
    return list(reversed(members)) if reverse else members


def test_timestamp_and_metadata_drift_normalize_byte_identically(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_archive(
        first,
        _valid_members(drift=111, reverse=False),
        gzip_mtime=123,
        gzip_filename="first-source-name.tar",
    )
    _write_archive(
        second,
        _valid_members(drift=999, reverse=True),
        gzip_mtime=987,
        gzip_filename="second-source-name.tar",
    )

    epoch = 1_700_000_000
    normalized_first = NORMALIZER.normalize_sdist(first, epoch=epoch)
    normalized_second = NORMALIZER.normalize_sdist(second, epoch=epoch)
    assert normalized_first == normalized_second
    assert hashlib.sha256(normalized_first).hexdigest() == hashlib.sha256(
        normalized_second
    ).hexdigest()

    assert normalized_first[:3] == b"\x1f\x8b\x08"
    assert normalized_first[3] & 0x08 == 0  # no original gzip filename
    assert struct.unpack("<I", normalized_first[4:8])[0] == epoch
    assert normalized_first[8] == 2  # maximum-compression XFL
    assert normalized_first[9] == 255  # platform-neutral gzip OS byte

    expected_bytes = {
        NORMALIZER.PKG_INFO: (
            b"Metadata-Version: 2.4\nName: mo-nco\nVersion: 0.21.3.14\n"
        ),
        f"{NORMALIZER.TOP_LEVEL}/bin/tool.py": (
            b"#!/usr/bin/env python\nprint('same bytes')\n"
        ),
        f"{NORMALIZER.TOP_LEVEL}/README.md": b"same readme bytes\n",
        f"{NORMALIZER.TOP_LEVEL}/docs/{'x' * 110}.txt": b"long path bytes\n",
    }
    with tarfile.open(fileobj=io.BytesIO(normalized_first), mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(
            member.name for member in members
        )
        for member in members:
            assert member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            assert member.mtime == epoch
            assert "comment" not in member.pax_headers
            assert set(member.pax_headers) <= {"path"}
            if "path" in member.pax_headers:
                assert member.pax_headers["path"] == member.name
            if member.isdir():
                assert member.mode == 0o755
                assert member.size == 0
            else:
                expected_mode = (
                    0o755
                    if member.name == f"{NORMALIZER.TOP_LEVEL}/bin/tool.py"
                    else 0o644
                )
                assert member.mode == expected_mode
                extracted = archive.extractfile(member)
                assert extracted is not None
                assert extracted.read() == expected_bytes[member.name]


def test_exclusive_output_and_explicit_replace(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    output = tmp_path / "normalized.tar.gz"
    _write_archive(
        source,
        _valid_members(drift=10, reverse=False),
        gzip_mtime=20,
        gzip_filename="source.tar",
    )
    first = NORMALIZER.write_normalized_sdist(source, output, epoch=42)
    assert first["status"] == "PASS_REPRODUCIBLE_SDIST_NORMALIZED"
    before = output.read_bytes()
    with pytest.raises(NORMALIZER.SdistNormalizationError, match="overwrite"):
        NORMALIZER.write_normalized_sdist(source, output, epoch=42)
    assert output.read_bytes() == before

    replaced = NORMALIZER.write_normalized_sdist(
        source,
        output,
        epoch=43,
        replace=True,
    )
    assert replaced["source_date_epoch"] == 43
    assert output.read_bytes() != before


def test_cli_requires_explicit_input_output_epoch_and_replace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.tar.gz"
    output = tmp_path / "cli-normalized.tar.gz"
    _write_archive(
        source,
        _valid_members(drift=10, reverse=True),
        gzip_mtime=20,
        gzip_filename="source.tar",
    )
    arguments = [
        "--input",
        str(source),
        "--output",
        str(output),
        "--epoch",
        "1700000000",
    ]
    assert NORMALIZER.main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["source_date_epoch"] == 1_700_000_000
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()

    assert NORMALIZER.main(arguments) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "HOLD_SDIST_NORMALIZATION"
    assert "overwrite" in failure["error"]

    assert NORMALIZER.main([*arguments, "--replace"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == (
        "PASS_REPRODUCIBLE_SDIST_NORMALIZED"
    )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "/absolute",
        "C:/drive-absolute",
        "../escape",
        f"{NORMALIZER.TOP_LEVEL}/../escape",
        f"{NORMALIZER.TOP_LEVEL}//double",
        f"{NORMALIZER.TOP_LEVEL}\\backslash",
        f"{NORMALIZER.TOP_LEVEL}/control\nname",
        "other-top-level/file.txt",
    ],
)
def test_unsafe_or_noncanonical_paths_are_rejected(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    source = tmp_path / "unsafe.tar.gz"
    members = _valid_members(drift=1, reverse=False)
    members.append(_member(unsafe_name, data=b"unsafe"))
    _write_archive(
        source,
        members,
        gzip_mtime=1,
        gzip_filename="unsafe.tar",
    )
    with pytest.raises(NORMALIZER.SdistNormalizationError):
        NORMALIZER.normalize_sdist(source, epoch=1)


def test_duplicate_and_case_colliding_paths_are_rejected(tmp_path: Path) -> None:
    for suffix, extra in (
        ("duplicate", f"{NORMALIZER.TOP_LEVEL}/README.md"),
        ("case", f"{NORMALIZER.TOP_LEVEL}/readme.md"),
    ):
        source = tmp_path / f"{suffix}.tar.gz"
        members = _valid_members(drift=1, reverse=False)
        members.append(_member(extra, data=b"collision"))
        _write_archive(
            source,
            members,
            gzip_mtime=1,
            gzip_filename=f"{suffix}.tar",
        )
        with pytest.raises(NORMALIZER.SdistNormalizationError, match="colliding"):
            NORMALIZER.normalize_sdist(source, epoch=1)


@pytest.mark.parametrize(
    ("member_type", "label"),
    [
        (tarfile.SYMTYPE, "symlink"),
        (tarfile.LNKTYPE, "hardlink"),
        (tarfile.CHRTYPE, "device"),
        (tarfile.BLKTYPE, "device"),
        (tarfile.FIFOTYPE, "fifo"),
    ],
)
def test_links_devices_and_fifo_are_rejected(
    tmp_path: Path,
    member_type: bytes,
    label: str,
) -> None:
    source = tmp_path / f"{label}-{member_type.hex()}.tar.gz"
    members = _valid_members(drift=1, reverse=False)
    members.append(
        _member(
            f"{NORMALIZER.TOP_LEVEL}/prohibited",
            member_type=member_type,
            linkname="target",
        )
    )
    _write_archive(
        source,
        members,
        gzip_mtime=1,
        gzip_filename="prohibited.tar",
    )
    with pytest.raises(NORMALIZER.SdistNormalizationError, match=label):
        NORMALIZER.normalize_sdist(source, epoch=1)


def test_missing_explicit_root_or_pkg_info_is_rejected(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-root.tar.gz"
    _write_archive(
        missing_root,
        [_member(NORMALIZER.PKG_INFO, data=b"metadata")],
        gzip_mtime=1,
        gzip_filename="missing-root.tar",
    )
    with pytest.raises(NORMALIZER.SdistNormalizationError, match="top-level"):
        NORMALIZER.normalize_sdist(missing_root, epoch=1)

    missing_metadata = tmp_path / "missing-metadata.tar.gz"
    _write_archive(
        missing_metadata,
        [_member(f"{NORMALIZER.TOP_LEVEL}/"), _member(f"{NORMALIZER.TOP_LEVEL}/x", data=b"x")],
        gzip_mtime=1,
        gzip_filename="missing-metadata.tar",
    )
    with pytest.raises(NORMALIZER.SdistNormalizationError, match="PKG-INFO"):
        NORMALIZER.normalize_sdist(missing_metadata, epoch=1)


@pytest.mark.parametrize("epoch", [-1, 1 << 32, True, 1.5])
def test_invalid_epoch_is_rejected(epoch: object, tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    _write_archive(
        source,
        _valid_members(drift=1, reverse=False),
        gzip_mtime=1,
        gzip_filename="source.tar",
    )
    with pytest.raises(NORMALIZER.SdistNormalizationError, match="SOURCE_DATE_EPOCH"):
        NORMALIZER.normalize_sdist(source, epoch=epoch)
