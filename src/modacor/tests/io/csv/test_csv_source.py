# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "12/12/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path

import numpy as np
import pytest

# Adjust this import to match your actual module layout:
# here we assume: modacor/io/csv/csv_source.py
from modacor.io.csv.csv_source import CSVSource


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_csvsource_genfromtxt_names_true(tmp_path):
    """
    CSVSource should work with np.genfromtxt(names=True) where the first row
    contains column names.
    """
    csv_content = "q,I,I_sigma\n1.0,2.0,0.1\n3.0,4.0,0.2\n"
    file_path = tmp_path / "data_names_true.csv"
    write_text_file(file_path, csv_content)

    src = CSVSource(
        source_reference="test_genfromtxt_names_true",
        resource_location=file_path,
        source_format_options={"delimiter": ",", "names": True},
        method=np.genfromtxt,
    )

    # dtype.names should be populated
    assert src._data_cache.dtype.names == ("q", "I", "I_sigma")

    # Data access
    q = src.get_data("q")
    I = src.get_data("I")  # noqa: E741
    I_sigma = src.get_data("I_sigma")

    assert np.allclose(q, [1.0, 3.0])
    assert np.allclose(I, [2.0, 4.0])
    assert np.allclose(I_sigma, [0.1, 0.2])

    # Shape / dtype helpers
    assert src.get_data_shape("q") == (2,)
    assert src.get_data_dtype("q") == src._data_cache["q"].dtype


def test_csvsource_genfromtxt_explicit_names(tmp_path):
    """
    CSVSource should work with np.genfromtxt(names=[...]) on data without a header row.
    """
    csv_content = "1.0,2.0,0.1\n3.0,4.0,0.2\n"
    file_path = tmp_path / "data_explicit_names.csv"
    write_text_file(file_path, csv_content)

    src = CSVSource(
        source_reference="test_genfromtxt_explicit_names",
        resource_location=file_path,
        source_format_options={
            "delimiter": ",",
            "names": ["q", "I", "I_sigma"],
        },
        method=np.genfromtxt,
    )

    assert src._data_cache.dtype.names == ("q", "I", "I_sigma")

    q = src.get_data("q")
    I = src.get_data("I")  # noqa: E741
    I_sigma = src.get_data("I_sigma")

    assert np.allclose(q, [1.0, 3.0])
    assert np.allclose(I, [2.0, 4.0])
    assert np.allclose(I_sigma, [0.1, 0.2])


def test_csvsource_loadtxt_structured_dtype(tmp_path):
    """
    CSVSource should support np.loadtxt when a structured dtype with field names
    is provided.
    """
    csv_content = "1.0 2.0 0.1\n3.0 4.0 0.2\n"
    file_path = tmp_path / "data_loadtxt_structured.csv"
    write_text_file(file_path, csv_content)

    structured_dtype = [("q", float), ("I", float), ("I_sigma", float)]

    src = CSVSource(
        source_reference="test_loadtxt_structured",
        resource_location=file_path,
        source_format_options={
            "delimiter": " ",
            "dtype": structured_dtype,
        },
        method=np.loadtxt,
    )

    assert src._data_cache.dtype.names == ("q", "I", "I_sigma")

    q = src.get_data("q")
    I = src.get_data("I")  # noqa: E741
    I_sigma = src.get_data("I_sigma")

    assert np.allclose(q, [1.0, 3.0])
    assert np.allclose(I, [2.0, 4.0])
    assert np.allclose(I_sigma, [0.1, 0.2])


def test_csvsource_raises_if_no_dtype_names(tmp_path):
    """
    CSVSource should raise a ValueError if the resulting array has no dtype.names,
    e.g. when using np.loadtxt with a plain dtype.
    """
    csv_content = "1.0,2.0,3.0\n4.0,5.0,6.0\n"
    file_path = tmp_path / "data_no_names.csv"
    write_text_file(file_path, csv_content)

    with pytest.raises(ValueError, match="dtype.names is None"):
        CSVSource(
            source_reference="test_no_names",
            resource_location=file_path,
            source_format_options={"delimiter": ","},  # no structured dtype, no names
            method=np.loadtxt,
        )


def test_csvsource_get_data_invalid_key_raises(tmp_path):
    """
    Requesting a non-existent data_key should raise a KeyError with a helpful message.
    """
    csv_content = "q,I\n1.0,2.0\n3.0,4.0\n"
    file_path = tmp_path / "data_invalid_key.csv"
    write_text_file(file_path, csv_content)

    src = CSVSource(
        source_reference="test_invalid_key",
        resource_location=file_path,
        source_format_options={"delimiter": ",", "names": True},
        method=np.genfromtxt,
    )

    with pytest.raises(KeyError, match="Data key 'nonexistent' not found"):
        src.get_data("nonexistent")
