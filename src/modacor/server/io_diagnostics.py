# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path

from .runtime_policy import _FILE_SINK_TYPES, _FILE_SOURCE_TYPES


def check_file_location(*, kind: str, io_type: str, location: str) -> None:
    """Check local access without opening files or creating output directories.

    Registration treats these errors as advisory; runtime builders check again.
    Other backends retain their own availability/connection semantics.
    """
    file_types = _FILE_SOURCE_TYPES if kind == "source" else _FILE_SINK_TYPES
    if io_type.strip().lower() not in file_types:
        return
    path = Path(location).expanduser()
    if kind == "source":
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        if not path.is_file():
            raise OSError(f"Source location is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Source file is not readable: {path}")
    elif path.exists():
        if not path.is_file():
            raise OSError(f"Sink location is not a file: {path}")
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Sink file is not writable: {path}")
    else:
        ancestor = path.parent
        while not ancestor.exists() and not ancestor.is_symlink():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise NotADirectoryError(f"Sink ancestor is not a directory: {ancestor}")
        if not os.access(ancestor, os.W_OK | os.X_OK):
            raise PermissionError(f"Sink ancestor directory is not writable and searchable: {ancestor}")
