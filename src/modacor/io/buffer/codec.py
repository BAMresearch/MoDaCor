# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from io import BytesIO

import numpy as np

__all__ = ["decode_npy", "encode_npy"]


def encode_npy(array: np.ndarray) -> bytes:
    """Encode one array as NumPy .npy bytes."""

    stream = BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def decode_npy(payload: bytes) -> np.ndarray:
    """Decode NumPy .npy bytes into an array."""

    stream = BytesIO(payload)
    array = np.load(stream, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("Decoded .npy payload did not contain a NumPy array.")
    return array
