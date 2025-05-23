from typing import Iterable

import numpy as np
import pint
import pytest

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
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


class TestProcessingStep(ProcessStep):
    CONFIG_KEYS = {_k: _v for _k, _v in _TEST_KEYS.items()}

    def calculate(self) -> dict[str, DataBundle]:
        _data = self.processing_data.get("dummy_key", DataBundle())
        _data["new_key"] = BaseData(signal=np.arange(100).reshape(10, 10))
        _data2 = self.processing_data.get("bundle2", DataBundle())
        _data2["new_key"] = BaseData(signal=np.zeros(20))
        return {"dummy_key": _data, "bundle2": _data2}


@pytest.fixture
def class_with_config_keys(request):
    _keys = request.param

    class TestClass(TestProcessingStep):
        CONFIG_KEYS = {_key: _TEST_KEYS[_key] for _key in _keys}

    return _keys, TestClass


@pytest.fixture
def processing_data():
    data = ProcessingData()
    data["bundle1"] = DataBundle()
    data["bundle2"] = DataBundle()
    data["bundle1"]["key1"] = BaseData(signal=np.arange(50))
    data["bundle2"]["key2"] = BaseData(signal=np.ones((10, 10)))
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
@pytest.mark.parametrize(
    "class_with_config_keys", [[_k] for _k in _TEST_KEYS.keys()], indirect=True
)
def test_is_process_step_dict__w_correct_key(class_with_config_keys, item):
    _keys, _class = class_with_config_keys
    _config = _class.CONFIG_KEYS[_keys[0]]
    _test_dict = {_keys[0]: item}
    if item is None:
        assert _class.is_process_step_dict(None, None, _test_dict) == _config["allow_none"]
    elif not _config["allow_iterable"]:
        assert _class.is_process_step_dict(None, None, _test_dict) == isinstance(
            item, _config["type"]
        )
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
    ps = ProcessStep(TEST_IO_SOURCES)
    assert isinstance(ps, ProcessStep)


def test_instantiation_of_subclass():
    instance = TestProcessingStep(TEST_IO_SOURCES)
    assert all(k in instance.configuration for k in TestProcessingStep.CONFIG_KEYS)
    assert isinstance(instance, TestProcessingStep)


def test_process_step__reset():
    ps = ProcessStep(TEST_IO_SOURCES)
    ps.produced_outputs = {"a": 1}
    ps._ProcessStep__prepared = True
    ps.executed = True
    ps.reset()
    assert ps.produced_outputs == {}
    assert ps._ProcessStep__prepared is False
    assert ps.executed is False


@pytest.mark.parametrize("class_with_config_keys", [["test_str"]], indirect=True)
def test_modify_config__valid_key(class_with_config_keys):
    instance = class_with_config_keys[1](TEST_IO_SOURCES)
    instance.modify_config("test_str", "new_value")
    assert instance.configuration["test_str"] == "new_value"
    assert not instance._ProcessStep__prepared


@pytest.mark.parametrize("class_with_config_keys", [["test_str"]], indirect=True)
def test_modify_config__invalid_key(class_with_config_keys):
    instance = class_with_config_keys[1](TEST_IO_SOURCES)
    with pytest.raises(KeyError):
        instance.modify_config("silly_key", "new_value")


def test_calculate():
    ps = TestProcessingStep(TEST_IO_SOURCES)
    ps.processing_data = ProcessingData()
    _return = ps.calculate()
    assert isinstance(_return, dict)


def test_calculate__abstract():
    ps = ProcessStep(TEST_IO_SOURCES)
    with pytest.raises(NotImplementedError):
        ps.calculate()


def test_execute(processing_data):
    ps = TestProcessingStep(TEST_IO_SOURCES)
    ps.execute(processing_data)
    assert ps.executed is True
    assert ps._ProcessStep__prepared is True
    assert isinstance(ps.produced_outputs, dict)
    assert isinstance(ps.produced_outputs["dummy_key"], DataBundle)
    assert isinstance(ps.produced_outputs["bundle2"], DataBundle)
    assert "dummy_key" in processing_data
    assert isinstance(processing_data["bundle2"]["key2"], BaseData)
    assert isinstance(processing_data["bundle2"]["new_key"], BaseData)
