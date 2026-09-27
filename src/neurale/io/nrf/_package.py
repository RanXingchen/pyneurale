# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Single-file NRF envelope. Compression never changes journal object bytes."""

from __future__ import annotations

import io
import os
import tempfile
import time
import zlib
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from ._errors import NrfCorruptionError

PACKAGE_MARKER = b"PyNeurale NRF package 1"


class Package:
    """Indexed, read-only ZIP64 envelope; no extraction or payload cache."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise NrfCorruptionError(f"{path} is not a single-file NRF package")
        try:
            self.zip = ZipFile(path, "r")
        except (OSError, BadZipFile) as error:
            raise NrfCorruptionError(f"cannot open NRF package: {path}") from error
        try:
            self._identity = self._file_identity(os.fstat(self.zip.fp.fileno()))
            if self.zip.comment != PACKAGE_MARKER:
                raise NrfCorruptionError("unsupported NRF package envelope")
            self.entries = {}
            for entry in self.zip.infolist():
                name = entry.filename
                parts = name.split("/")
                if (
                    name in self.entries
                    or entry.orig_filename != name
                    or any(part in ("", ".", "..") for part in parts)
                    or "\\" in name
                    or ":" in name
                    or entry.is_dir()
                    or entry.flag_bits & 1
                    or entry.compress_type != ZIP_DEFLATED
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise NrfCorruptionError(f"invalid NRF package entry: {name}")
                self.entries[name] = entry
            if "manifest.json" not in self.entries:
                raise NrfCorruptionError("NRF package has no manifest.json")
        except BaseException:
            self.zip.close()
            raise

    @property
    def root(self):
        return PackagePath(self)

    @staticmethod
    def _file_identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def require_unchanged(self):
        """Never combine a cached journal with a replacement package's arrays."""
        try:
            unchanged = self._file_identity(self.path.stat()) == self._identity
        except OSError as error:
            raise NrfCorruptionError("NRF package became unavailable while open") from error
        if not unchanged:
            raise NrfCorruptionError("NRF package changed while open; reopen the reader")

    def close(self) -> None:
        self.zip.close()


class PackagePath:
    """Private object addressing used by journal and integrity readers."""

    def __init__(self, package: Package, key: str = "") -> None:
        self.package = package
        self.key = key

    def __truediv__(self, key):
        return PackagePath(self.package, f"{self.key}/{key}" if self.key else str(key))

    def __str__(self):
        return f"{self.package.path}!/{self.key}"

    def __lt__(self, other):
        return self.key < other.key

    @property
    def name(self):
        return PurePosixPath(self.key).name

    def exists(self):
        return not self.key or self.key in self.package.entries

    def is_file(self):
        return self.key in self.package.entries

    def is_dir(self):
        prefix = f"{self.key}/" if self.key else ""
        return any(key.startswith(prefix) for key in self.package.entries)

    def glob(self, pattern):
        prefix = f"{self.key}/" if self.key else ""
        return (
            PackagePath(self.package, key)
            for key in self.package.entries
            if key.startswith(prefix) and PurePosixPath(key[len(prefix) :]).match(pattern)
        )

    def stat(self):
        return SimpleNamespace(st_size=self.package.entries[self.key].file_size)

    def open(self, mode="r", encoding="utf-8"):
        if mode not in ("r", "rb"):
            raise ValueError("NRF packages are immutable")
        if not self.is_file():
            raise FileNotFoundError(str(self))
        stream = self.package.zip.open(self.key)
        return stream if mode == "rb" else io.TextIOWrapper(stream, encoding=encoding)

    def read_bytes(self):
        try:
            with self.open("rb") as stream:
                return stream.read()
        except (BadZipFile, zlib.error, EOFError) as error:
            raise NrfCorruptionError(f"damaged NRF package entry: {self.key}") from error

    def read_text(self, encoding="utf-8"):
        return self.read_bytes().decode(encoding)

    def relative_to(self, root):
        return PurePosixPath(self.key).relative_to(PurePosixPath(root.key))

    def rglob(self, pattern):
        if pattern != "*":
            raise ValueError("only object enumeration is supported")
        return (PackagePath(self.package, key) for key in self.package.entries)


class PayloadArchive:
    """Private finalizer sink: encode one immutable object directly into the package.

    Only spool conversion uses this sink. The spool remains authoritative until
    the completed package passes the finalizer's full integrity cross-check.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
        )
        self._file = os.fdopen(descriptor, "w+b")
        try:
            self.archive = ZipFile(
                self._file, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True
            )
        except BaseException:
            self._file.close()
            raise
        self.archive.comment = PACKAGE_MARKER
        self.byte_count = 0

    def write(self, key: str, raw: bytes) -> None:
        if key in self.archive.NameToInfo:
            raise NrfCorruptionError(f"duplicate NRF payload object: {key}")
        self.archive.writestr(key, raw)
        self.byte_count += len(raw)

    def close(self) -> None:
        try:
            self.archive.close()
        finally:
            self._file.close()


def publish_package(
    workspace: Path,
    target: Path,
    progress=None,
    *,
    require_termination=True,
    include_staging=False,
    payload_archive: PayloadArchive | None = None,
) -> Path | None:
    """Stream, validate, then publish without overwriting an existing target."""
    from ._reader import NrfReader

    if payload_archive is None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".packing", dir=target.parent
        )
        temporary = Path(temporary)
    else:
        payload_archive.close()
        temporary = payload_archive.path
        descriptor = os.open(temporary, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        paths = [
            p
            for p in sorted(workspace.rglob("*"))
            if p != temporary
            and p.is_file()
            and (include_staging or p.relative_to(workspace).parts[0] != ".staging")
        ]
        done = 0 if payload_archive is None else payload_archive.byte_count
        total = done + sum(p.stat().st_size for p in paths)
        reported = time.monotonic()
        if progress is not None:
            progress("packing", done, total)
        with os.fdopen(descriptor, "w+b" if payload_archive is None else "r+b") as file:
            descriptor = None
            with ZipFile(
                file,
                "w" if payload_archive is None else "a",
                compression=ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as archive:
                archive.comment = PACKAGE_MARKER
                for path in paths:
                    if path.is_symlink():
                        raise NrfCorruptionError(f"symbolic link in NRF workspace: {path}")
                    if path.relative_to(workspace).as_posix() in archive.NameToInfo:
                        raise NrfCorruptionError(f"duplicate NRF package object: {path}")
                    archive.write(path, path.relative_to(workspace).as_posix())
                    done += path.stat().st_size
                    if progress is not None and time.monotonic() - reported >= 1:
                        progress("packing", done, total)
                        reported = time.monotonic()
            file.flush()
            os.fsync(file.fileno())
        if progress is not None:
            progress("verifying_package", done, total)
        # Spool conversion performs its one full payload verification in
        # _cross_check, before public publication or any spool deletion.
        with NrfReader.open(temporary, verify_checksums=payload_archive is None) as reader:
            if require_termination and not reader._state.terminated:
                raise NrfCorruptionError("cannot publish an unterminated NRF package")
        # Both forms refuse overwrite. Windows rename also supports local
        # filesystems without hard links; POSIX rename would overwrite.
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Preserve the primary conversion error, or report the leftover
            # after success. Failed cleanup cannot undo published data.
            pass
    return temporary if temporary.exists() else None
