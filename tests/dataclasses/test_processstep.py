# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Jérome Kieffer", "Anja Hörmann", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io import IoSources

TEST_IO_SOURCES = IoSources()

_TEST_KEYS = {
    "test_str": {
        "type": str,
        "allow_iterable": False,
        "allow_none": False,
        "default": "",
    },
    "test_str_allow_list": {
        "type": str,
        "allow_iterable": True,
        "allow_none": False,
        "default": "",
    },
    "test_str_allow_none": {
        "type": str,
        "allow_iterable": False,
        "allow_none": True,
        "default": None,
    },
    "test_str_allow_list_none": {
        "type": str,
        "allow_iterable": True,
        "allow_none": True,
        "default": None,
    },
    "test_int": {
        "type": int,
        "allow_iterable": False,
        "allow_none": False,
        "default": 0,
    },
    "test_int_allow_list": {
        "type": int,
        "allow_iterable": True,
        "allow_none": False,
        "default": 0,
    },
    "test_int_allow_none": {
        "type": int,
        "allow_iterable": False,
        "allow_none": True,
        "default": None,
    },
    "test_int_allow_list_none": {
        "type": int,
        "allow_iterable": True,
        "allow_none": True,
        "default": None,
    },
}

_TEST_VALUES = ["", "test", 1, 42, None]
_TEST_LISTS = [["", "2"], ["", 12], [12, 42], [12, None], ["b", None]]
_TEST_TUPLES = [("12", "b"), ("", 12), (12, 42), ("a", None), (12, None)]


class TESTProcessingStep(ProcessStep):
    CONFIG_KEYS = {_k: _v for _k, _v in _TEST_KEYS.items()}

    def calculate(self) -> dict[str, DataBundle]:
        _data = self.processing_data.get("dummy_key", DataBundle())
        _data["new_key"] = BaseData(signal=np.arange(100).reshape(10, 10), uncertainties={"sem": 0.0}, units=ureg.meter)
        self.processing_data["dummy_key"] = _data
        _data2 = self.processing_data.get("bundle2", DataBundle())
        _data2["new_key"] = BaseData(signal=np.zeros(20), uncertainties={"sem": 0.0}, units=ureg.meter)
        self.processing_data["bundle2"] = _data2
        return {"dummy_key": _data, "bundle2": _data2}


class DocumentedConfigStep(ProcessStep):
    documentation = ProcessStepDescriber(
        calling_name="Documented config step",
        calling_id="DocumentedConfigStep",
        calling_module_path=Path(__file__),
        calling_version="0",
        arguments={
            "alpha": {"type": int, "default": 1},
            "nested": {"type": dict, "default": {"values": []}},
            "optional_label": {"type": (str, type(None)), "default": None},
            "vector": {"type": tuple, "default": (1.0, 0.0, 0.0)},
            "source_key": {
                "type": str,
                "default": "signal",
                "dependency_role": "processing_read_basedata_key",
            },
            "target_key": {
                "type": str,
                "default": "mask",
                "dependency_role": "processing_write_basedata_key",
            },
            "read_write_keys": {
                "type": list,
                "default": ["signal"],
                "dependency_role": "processing_read_write_basedata_key_list",
            },
        },
    )

    def calculate(self) -> dict[str, DataBundle]:
        return {}


class ReturnOnlyStep(ProcessStep):
    def calculate(self) -> dict[str, DataBundle]:
        bundle = DataBundle()
        bundle["signal"] = BaseData(signal=np.array([1.0]), units=ureg.dimensionless)
        return {"return_only": bundle}


@pytest.fixture
def class_with_config_keys(request):
    _keys = request.param

    class TestClass(TESTProcessingStep):
        CONFIG_KEYS = {_key: _TEST_KEYS[_key] for _key in _keys}

    return _keys, TestClass


@pytest.fixture
def processing_data():
    data = ProcessingData()
    data["bundle1"] = DataBundle()
    data["bundle2"] = DataBundle()
    data["bundle1"]["key1"] = BaseData(signal=np.arange(50), uncertainties={"sem": 0.0}, units=ureg.meter)
    data["bundle2"]["key2"] = BaseData(signal=np.ones((10, 10)), uncertainties={"sem": 0.0}, units=ureg.meter)
    return data


def test_process_step_default_config__generic():
    _defaults = ProcessStep.default_config()
    assert isinstance(_defaults, dict)
    assert all(key in _defaults for key in ProcessStep.CONFIG_KEYS.keys())


@pytest.mark.parametrize(
    "class_with_config_keys", [["test_str"], ["test_int"], ["test_str", "test_int"]], indirect=True
)
def test_process_step_default_config__specific(class_with_config_keys):
    _keys, _class = class_with_config_keys
    _defaults = _class.default_config()
    assert isinstance(_defaults, dict)
    assert all(key in _defaults for key in _keys)


@pytest.mark.parametrize("item", _TEST_VALUES + _TEST_LISTS + _TEST_TUPLES)
@pytest.mark.parametrize("class_with_config_keys", [[_k] for _k in _TEST_KEYS.keys()], indirect=True)
def test_is_process_step_dict__w_correct_key(class_with_config_keys, item):
    _keys, _class = class_with_config_keys
    _config = _class.CONFIG_KEYS[_keys[0]]
    _test_dict = {_keys[0]: item}
    if item is None:
        assert _class.is_process_step_dict(None, None, _test_dict) == _config["allow_none"]
    elif not _config["allow_iterable"]:
        assert _class.is_process_step_dict(None, None, _test_dict) == isinstance(item, _config["type"])
    elif _config["allow_iterable"]:
        assert _class.is_process_step_dict(None, None, _test_dict) == (
            (isinstance(item, Iterable) and not isinstance(item, str))
            and all(isinstance(i, _config["type"]) for i in item)
            or isinstance(item, _config["type"])
        )
    else:
        assert False


def test_is_process_step_dict__w_wrong_key():
    test_dict = ProcessStep.default_config() | {"wrong_key": "value"}
    assert not ProcessStep.is_process_step_dict(None, None, test_dict)


def test_minimal_instantiation():
    ps = ProcessStep(io_sources=TEST_IO_SOURCES)
    assert isinstance(ps, ProcessStep)


def test_instantiation_of_subclass():
    instance = TESTProcessingStep(io_sources=TEST_IO_SOURCES)
    assert all(k in instance.configuration for k in TESTProcessingStep.CONFIG_KEYS)
    assert isinstance(instance, TESTProcessingStep)


def test_constructor_configuration_overrides_documented_defaults():
    instance = DocumentedConfigStep(configuration={"alpha": 7, "optional_label": "sample"})
    assert instance.configuration["alpha"] == 7
    assert instance.configuration["optional_label"] == "sample"
    assert instance.configuration["with_processing_keys"] is None


def test_documented_argument_defaults_are_isolated_between_instances():
    first = DocumentedConfigStep()
    second = DocumentedConfigStep()

    first.configuration["nested"]["values"].append("changed")

    assert second.configuration["nested"]["values"] == []


def test_documented_arguments_are_validated_on_init_and_modify():
    with pytest.raises(TypeError, match="alpha"):
        DocumentedConfigStep(configuration={"alpha": "bad"})

    instance = DocumentedConfigStep()
    with pytest.raises(TypeError, match="alpha"):
        instance.modify_config_by_kwargs(alpha="bad")


def test_documented_arguments_are_in_process_step_dict_validation():
    assert DocumentedConfigStep.is_process_step_dict(None, None, {"alpha": 3})
    assert not DocumentedConfigStep.is_process_step_dict(None, None, {"alpha": "bad"})


def test_tuple_config_accepts_yaml_style_list_and_stores_tuple():
    instance = DocumentedConfigStep(configuration={"vector": [0.0, 1.0, 0.0]})

    assert instance.configuration["vector"] == (0.0, 1.0, 0.0)
    assert isinstance(instance.configuration["vector"], tuple)
    assert DocumentedConfigStep.is_process_step_dict(None, None, {"vector": [0.0, 1.0, 0.0]})


def test_dependency_contract_uses_process_step_describer_dependency_roles():
    instance = DocumentedConfigStep(
        configuration={
            "with_processing_keys": ["sample"],
            "source_key": "flatfield",
            "target_key": "flatfield_mask",
            "read_write_keys": ["signal", "weights"],
            "optional_label": "file::/entry/value",
        }
    )

    contract = instance.dependency_contract()

    assert contract.source_refs == frozenset({"file"})
    assert contract.processing_reads == frozenset({"sample.flatfield", "sample.signal", "sample.weights"})
    assert contract.processing_writes == frozenset({"sample.flatfield_mask", "sample.signal", "sample.weights"})


def test_dependency_contract_with_dependency_roles_falls_back_to_wildcard_without_processing_keys():
    instance = DocumentedConfigStep()

    contract = instance.dependency_contract()

    assert contract.source_refs == frozenset()
    assert contract.processing_reads == frozenset({"*"})
    assert contract.processing_writes == frozenset({"*"})


def test_process_step__reset():
    ps = ProcessStep(io_sources=TEST_IO_SOURCES)
    ps.produced_outputs = {"a": 1}
    ps._ProcessStep__prepared = True
    ps.executed = True
    ps.reset()
    assert ps.produced_outputs == {}
    assert ps._ProcessStep__prepared is False
    assert ps.executed is False


def test_normalised_processing_keys_with_single_entry():
    data = ProcessingData()
    data["only"] = DataBundle()
    ps = ProcessStep(io_sources=TEST_IO_SOURCES, processing_data=data)
    ps.configuration["with_processing_keys"] = None
    assert ps._normalised_processing_keys() == ["only"]


def test_normalised_processing_keys_with_multiple_entries_requires_explicit_keys():
    data = ProcessingData()
    data["a"] = DataBundle()
    data["b"] = DataBundle()
    ps = ProcessStep(io_sources=TEST_IO_SOURCES, processing_data=data)
    ps.configuration["with_processing_keys"] = None
    with pytest.raises(ValueError):
        ps._normalised_processing_keys()


def test_normalised_processing_keys_accepts_string_or_list():
    data = ProcessingData()
    data["a"] = DataBundle()
    ps = ProcessStep(io_sources=TEST_IO_SOURCES, processing_data=data)
    ps.configuration["with_processing_keys"] = "a"
    assert ps._normalised_processing_keys() == ["a"]

    ps.configuration["with_processing_keys"] = ["a", "b"]
    assert ps._normalised_processing_keys() == ["a", "b"]


def test_normalised_processing_keys_rejects_empty_list():
    data = ProcessingData()
    data["a"] = DataBundle()
    ps = ProcessStep(io_sources=TEST_IO_SOURCES, processing_data=data)
    ps.configuration["with_processing_keys"] = []
    with pytest.raises(ValueError):
        ps._normalised_processing_keys()


def test_normalised_processing_keys_requires_processing_data():
    ps = ProcessStep(io_sources=TEST_IO_SOURCES, processing_data=None)
    ps.configuration["with_processing_keys"] = ["a"]
    with pytest.raises(RuntimeError):
        ps._normalised_processing_keys()


@pytest.mark.parametrize("class_with_config_keys", [["test_str"]], indirect=True)
def test_modify_config__valid_key(class_with_config_keys):
    instance = class_with_config_keys[1](io_sources=TEST_IO_SOURCES)
    instance.modify_config_by_kwargs(test_str="new_value")
    assert instance.configuration["test_str"] == "new_value"
    assert not instance._ProcessStep__prepared


@pytest.mark.parametrize("class_with_config_keys", [["test_str"]], indirect=True)
def test_modify_config__by_dict(class_with_config_keys):
    instance = class_with_config_keys[1](io_sources=TEST_IO_SOURCES)
    instance.modify_config_by_dict({"test_str": "new_value"})
    assert instance.configuration["test_str"] == "new_value"
    assert not instance._ProcessStep__prepared


@pytest.mark.parametrize("class_with_config_keys", [["test_str"]], indirect=True)
def test_modify_config__invalid_key(class_with_config_keys):
    instance = class_with_config_keys[1](io_sources=TEST_IO_SOURCES)
    with pytest.raises(KeyError):
        instance.modify_config_by_kwargs(silly_key="new_value")


def test_calculate():
    ps = TESTProcessingStep(io_sources=TEST_IO_SOURCES)
    ps.processing_data = ProcessingData()
    _return = ps.calculate()
    assert isinstance(_return, dict)


def test_calculate__abstract():
    ps = ProcessStep(io_sources=TEST_IO_SOURCES)
    with pytest.raises(NotImplementedError):
        ps.calculate()


def test_execute(processing_data):
    ps = TESTProcessingStep(io_sources=TEST_IO_SOURCES)
    ps.execute(processing_data)
    assert ps.executed is True
    assert ps._ProcessStep__prepared is True
    assert isinstance(ps.produced_outputs, dict)
    assert isinstance(ps.produced_outputs["dummy_key"], DataBundle)
    assert isinstance(ps.produced_outputs["bundle2"], DataBundle)
    assert "dummy_key" in processing_data
    assert isinstance(processing_data["bundle2"]["key2"], BaseData)
    assert isinstance(processing_data["bundle2"]["new_key"], BaseData)


def test_execute_does_not_merge_return_only_outputs():
    data = ProcessingData()
    ps = ReturnOnlyStep(io_sources=TEST_IO_SOURCES)

    ps.execute(data)

    assert ps.executed is True
    assert "return_only" in ps.produced_outputs
    assert "return_only" not in data


def test_execute_accepts_none_output_bookkeeping():
    class NoneOutputStep(ProcessStep):
        def calculate(self):
            return None

    data = ProcessingData()
    ps = NoneOutputStep(io_sources=TEST_IO_SOURCES)

    ps.execute(data)

    assert ps.produced_outputs == {}


def test_execute_rejects_non_dict_output_bookkeeping():
    class BadOutputStep(ProcessStep):
        def calculate(self):
            return []

    data = ProcessingData()
    ps = BadOutputStep(io_sources=TEST_IO_SOURCES)

    with pytest.raises(TypeError, match="must return a dict"):
        ps.execute(data)


def test_call(processing_data):
    ps = TESTProcessingStep(io_sources=TEST_IO_SOURCES)
    ps(processing_data)
    assert ps.executed is True
    assert ps._ProcessStep__prepared is True
    assert isinstance(ps.produced_outputs, dict)
    assert isinstance(ps.produced_outputs["dummy_key"], DataBundle)
    assert isinstance(ps.produced_outputs["bundle2"], DataBundle)
    assert "dummy_key" in processing_data
    assert isinstance(processing_data["bundle2"]["key2"], BaseData)
    assert isinstance(processing_data["bundle2"]["new_key"], BaseData)
