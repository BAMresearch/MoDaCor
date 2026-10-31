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
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Sink parent directory does not exist: {path.parent}")
        if not os.access(path.parent, os.W_OK):
            raise PermissionError(f"Sink parent directory is not writable: {path.parent}")
