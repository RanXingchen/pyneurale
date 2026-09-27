# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Test-only object editing: damage archives without restoring directory support."""

import io
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

from neurale.io.nrf._package import PACKAGE_MARKER


class Objects:
    def __init__(self, path, key=""):
        self.path = Path(path)
        self.key = key

    def __fspath__(self):
        if self.key:
            return str(self._working() / self.key)
        return str(self.path)

    def __str__(self):
        return str(self.path) if not self.key else f"{self.path}!/{self.key}"

    def __truediv__(self, key):
        return Objects(self.path, f"{self.key}/{key}" if self.key else str(key))

    def __lt__(self, other):
        return str(self) < str(other)

    def _working(self):
        if self.path.is_dir():
            return self.path
        candidates = list(self.path.parent.glob(f".{self.path.name}.*.working"))
        if len(candidates) != 1:
            raise FileNotFoundError(str(self))
        return candidates[0]

    def _keys(self):
        if self.path.is_file():
            with ZipFile(self.path) as archive:
                return archive.namelist()
        return [
            p.relative_to(self._working()).as_posix()
            for p in self._working().rglob("*")
            if p.is_file()
        ]

    @property
    def name(self):
        return PurePosixPath(self.key).name if self.key else self.path.name

    @property
    def suffix(self):
        return PurePosixPath(self.name).suffix

    @property
    def parent(self):
        return (
            Objects(self.path, str(PurePosixPath(self.key).parent).removeprefix("."))
            if self.key
            else self.path.parent
        )

    def relative_to(self, other):
        return PurePosixPath(self.key).relative_to(PurePosixPath(other.key))

    def exists(self):
        return self.path.exists() if not self.key else self.is_file() or self.is_dir()

    def is_file(self):
        return self.path.is_file() if not self.key else self.key in self._keys()

    def is_dir(self):
        return (
            self.path.is_dir()
            if not self.key
            else any(k.startswith(self.key + "/") for k in self._keys())
        )

    def rglob(self, pattern):
        prefix = self.key + "/" if self.key else ""
        return (
            Objects(self.path, k)
            for k in self._keys()
            if k.startswith(prefix) and PurePosixPath(k).match(pattern)
        )

    def glob(self, pattern):
        return self.rglob(pattern)

    def read_bytes(self):
        if not self.key:
            return self.path.read_bytes()
        if not self.path.is_file():
            return (self._working() / self.key).read_bytes()
        with ZipFile(self.path) as archive:
            return archive.read(self.key)

    def read_text(self, encoding="utf-8"):
        return self.read_bytes().decode(encoding)

    def stat(self):
        return SimpleNamespace(st_size=len(self.read_bytes()))

    def mkdir(self, **kwargs):
        if not self.key:
            self.path.mkdir(**kwargs)
        elif not self.path.is_file():
            (self._working() / self.key).mkdir(**kwargs)

    def _replace(self, value):
        temporary = self.path.with_suffix(".edited")
        with (
            ZipFile(self.path) as source,
            ZipFile(temporary, "w", compression=ZIP_DEFLATED) as dest,
        ):
            dest.comment = source.comment
            for entry in source.infolist():
                if entry.filename != self.key:
                    dest.writestr(entry.filename, source.read(entry))
            if value is not None:
                dest.writestr(self.key, value)
        temporary.replace(self.path)

    def write_bytes(self, data):
        if not self.path.is_file():
            return (self._working() / self.key).write_bytes(data)
        self._replace(data)
        return len(data)

    def write_text(self, data, encoding="utf-8"):
        self.write_bytes(data.encode(encoding))
        return len(data)

    def unlink(self, missing_ok=False):
        if not self.key:
            return self.path.unlink(missing_ok=missing_ok)
        if not self.path.is_file():
            return (self._working() / self.key).unlink(missing_ok=missing_ok)
        if not self.is_file() and not missing_ok:
            raise FileNotFoundError(str(self))
        self._replace(None)

    @contextmanager
    def open(self, mode="r", encoding="utf-8"):
        binary = "b" in mode
        initial = b"" if "w" in mode else self.read_bytes()
        buffer = io.BytesIO(initial) if binary else io.StringIO(initial.decode(encoding))
        if "a" in mode:
            buffer.seek(0, 2)
        try:
            yield buffer
        finally:
            if "w" in mode or "a" in mode:
                value = buffer.getvalue()
                self.write_bytes(value if binary else value.encode(encoding))
            buffer.close()


def snapshot(writer):
    """Create an intentionally unsealed package fixture from private transaction state."""
    with ZipFile(writer.root, "w", compression=ZIP_DEFLATED) as archive:
        archive.comment = PACKAGE_MARKER
        for item in writer._root.rglob("*"):
            if item.is_file():
                archive.write(item, item.relative_to(writer._root).as_posix())
