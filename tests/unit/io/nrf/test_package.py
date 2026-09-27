# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The public artifact is one immutable, indexed, lossless file."""

import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from neurale.data import Recording, SignalArray
from neurale.io.nrf import NrfCorruptionError, NrfReader, diagnose_session, write_recording
from neurale.io.nrf._package import PACKAGE_MARKER, Package

from .nrf_support import NEURAL, build_writer


def test_single_file_bitwise_roundtrip_and_ranges(tmp_path):
    values = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30, 1e30, 1.0], dtype=np.float64)
    data = np.tile(values[:, None], (1025, 2))
    signal = SignalArray.from_array(data, fs=1000.0, channel_names=["a", "b"], units="V")
    path = tmp_path / "session.nrf"
    write_recording(
        path,
        Recording(signals={"neural": signal}),
        session_id="11111111-1111-4111-8111-111111111111",
        created_at="2026-09-07T00:00:00Z",
    )
    assert path.is_file()
    assert list(tmp_path.iterdir()) == [path]
    with ZipFile(path) as archive:
        assert archive.comment == PACKAGE_MARKER
        assert all(e.compress_type == ZIP_DEFLATED for e in archive.infolist())
    with NrfReader.open(path, verify_checksums=True) as reader:
        for start, stop in ((0, 1), (3, 7), (1021, 1030), (0, len(data))):
            actual = reader.read_stream("neural", start, stop)
            assert actual.tobytes() == data[start:stop].tobytes()
    assert diagnose_session(path).readable
    assert list(tmp_path.iterdir()) == [path]


def test_explicit_recovery_and_dry_run_keep_single_file(tmp_path):
    from neurale.io.nrf import read_report, recover

    signal = SignalArray.from_array(
        np.zeros((8, 2)), fs=1000.0, channel_names=["a", "b"], units="V"
    )
    path = tmp_path / "session.nrf"
    write_recording(
        path,
        Recording(signals={"neural": signal}),
        session_id="11111111-1111-4111-8111-111111111111",
        created_at="2026-09-07T00:00:00Z",
    )
    before = path.read_bytes()
    proposal = recover(path, dry_run=True)
    assert proposal.dry_run
    assert path.read_bytes() == before
    result = recover(path, recovery_id="z-first")
    assert result.report_path == path
    assert read_report(path)["recovery_id"] == result.recovery_id
    result = recover(path, recovery_id="a-second")
    assert read_report(path)["recovery_id"] == result.recovery_id
    assert list(tmp_path.iterdir()) == [path]
    with NrfReader.open(path, verify_checksums=True) as reader:
        assert reader.read_stream("neural", 0, 8).shape == (8, 2)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b", "C:/x"])
def test_invalid_member_rejected(tmp_path, name):
    path = tmp_path / "bad.nrf"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.comment = PACKAGE_MARKER
        archive.writestr("manifest.json", "{}")
        archive.writestr(name, "bad")
    if "\\" in name:
        # Windows zipfile normalizes names on creation; inject the wire form.
        path.write_bytes(path.read_bytes().replace(b"a/b", b"a\\b"))
    with pytest.raises(NrfCorruptionError):
        Package(path)
    assert not diagnose_session(path).readable


def test_directories_and_truncated_packages_are_rejected(tmp_path):
    with pytest.raises(NrfCorruptionError):
        NrfReader.open(tmp_path)
    path = tmp_path / "broken.nrf"
    path.write_bytes(b"PK\x03\x04")
    assert not diagnose_session(path).readable


def test_duplicate_objects_are_rejected(tmp_path):
    path = tmp_path / "duplicate.nrf"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.comment = PACKAGE_MARKER
        archive.writestr("manifest.json", json.dumps({}))
        with pytest.warns(UserWarning):
            archive.writestr("manifest.json", json.dumps({}))
    with pytest.raises(NrfCorruptionError):
        Package(path)


@pytest.mark.parametrize("stage", ["packing", "validation", "publication"])
def test_failed_publication_retains_workspace_and_can_retry(tmp_path, monkeypatch, stage):
    from neurale.io.nrf import _package

    target = tmp_path / "failed.nrf"
    writer = build_writer(target)
    writer.append_stream("neural", NEURAL)

    def fail(*args, **kwargs):
        raise OSError(f"injected {stage} failure")

    with monkeypatch.context() as patch:
        if stage == "packing":
            patch.setattr(ZipFile, "write", fail)
        elif stage == "validation":
            patch.setattr(NrfReader, "open", fail)
        else:
            operation = "rename" if _package.os.name == "nt" else "link"
            original = getattr(_package.os, operation)

            def fail_publication(source, destination):
                if str(source).endswith(".packing"):
                    fail()
                return original(source, destination)

            patch.setattr(_package.os, operation, fail_publication)
        with pytest.raises(OSError, match="injected"):
            writer.finalize()
    assert not target.exists()
    assert writer._root.is_dir()
    assert not list(tmp_path.glob("*.packing"))
    writer.finalize()
    writer.close()
    with NrfReader.open(target, verify_checksums=True) as reader:
        assert reader.read_stream("neural").tobytes() == NEURAL.tobytes()
    assert list(tmp_path.iterdir()) == [target]


def test_cleanup_failure_does_not_fail_published_writer(tmp_path, monkeypatch):
    from neurale.io.nrf import _writer

    target = tmp_path / "kept.nrf"
    writer = build_writer(target)
    writer.append_stream("neural", NEURAL)

    def fail(*args, **kwargs):
        raise PermissionError("workspace still held")

    with monkeypatch.context() as patch:
        patch.setattr(_writer.shutil, "rmtree", fail)
        writer.finalize()
        writer.close()
    assert target.is_file()
    assert writer.cleanup_paths == (writer._root,)
    with NrfReader.open(target, verify_checksums=True) as reader:
        assert reader.read_stream("neural").tobytes() == NEURAL.tobytes()


def test_damaged_compressed_chunk_is_rejected_on_range_read(tmp_path):
    import struct

    target = tmp_path / "crc.nrf"
    writer = build_writer(target)
    writer.append_stream("neural", NEURAL)
    writer.finalize()
    writer.close()
    with ZipFile(target) as archive:
        entry = archive.getinfo("streams/neural/data/c/0/0")
    raw = bytearray(target.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", raw, entry.header_offset + 26)
    start = entry.header_offset + 30 + name_length + extra_length
    raw[start + entry.compress_size // 2] ^= 0xFF
    target.write_bytes(raw)
    with NrfReader.open(target) as reader:
        with pytest.raises(NrfCorruptionError):
            reader.read_stream("neural", 0, 1)
    assert not diagnose_session(target).readable


def test_reader_refuses_package_modified_after_open(tmp_path):
    import os

    target = tmp_path / "changed.nrf"
    writer = build_writer(target)
    writer.append_stream("neural", NEURAL)
    writer.finalize()
    writer.close()
    with NrfReader.open(target) as reader:
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with pytest.raises(NrfCorruptionError, match="changed while open"):
            reader.read_stream("neural")


def test_direct_payload_archive_is_exclusive_and_owner_private(tmp_path, monkeypatch):
    from neurale.io.nrf import _package

    target = tmp_path / "payloads.packing"
    original_open = _package.os.open
    modes = []

    def opened(path, flags, mode=0o777, **kwargs):
        if path == target:
            modes.append(mode)
        return original_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(_package.os, "open", opened)
    archive = _package.PayloadArchive(target)
    try:
        archive.write("chunk", b"payload")
        with pytest.raises(NrfCorruptionError, match="duplicate"):
            archive.write("chunk", b"replacement")
        with pytest.raises(FileExistsError):
            _package.PayloadArchive(target)
    finally:
        archive.close()
        archive.close()
    assert modes == [0o600, 0o600]
    with ZipFile(target) as stored:
        assert stored.read("chunk") == b"payload"
