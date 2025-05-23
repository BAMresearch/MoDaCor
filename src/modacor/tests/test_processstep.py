import pytest

from modacor.dataclasses.process_step import ProcessStep
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


@pytest.fixture
def class_with_config_key(request):
    class TestClass(ProcessStep):
        CONFIG_KEYS = {_key: _TEST_KEYS[_key] for _key in request.param}

    return TestClass


def test_process_step_default_config__generic():
    _defaults = ProcessStep.default_config()
    assert isinstance(_defaults, dict)
    assert all(key in _defaults for key in ProcessStep.CONFIG_KEYS.keys())


#
# @pytest.mark.parametrize("item", _TEST_VALUES + _TEST_LISTS + _TEST_TUPLES)
# @pytest.mark.parametrize("class_with_config_key", _TEST_KEYS.keys(), indirect=True)
# def test_is_process_step_dict__w_dict(class_with_config_key, item):
#     if item is None:
#         assert is_process_step_dict(None, None, _test_dict) == _config["allow_none"]
#     elif not _config["allow_iterable"]:
#         assert is_process_step_dict(None, None, _test_dict) == isinstance(item, _config["type"])
#     elif _config["allow_iterable"]:
#         assert is_process_step_dict(None, None, _test_dict) == (
#             (isinstance(item, Iterable) and not isinstance(item, str))
#             and all(isinstance(i, _config["type"]) for i in item)
#             or isinstance(item, _config["type"])
#         )
#     else:
#         assert False
#
#
# def test_minimal_instantiation():
#     """
#     Test that a ProcessStep can be instantiated with only the minimal arguments.
#     """
#     ps = ProcessStep(TEST_IO_SOURCES)
#     assert isinstance(ps, ProcessStep)
#
#
# def test_process_step__reset():
#     """
#     Test that the reset method clears the processing data.
#     """
#     ps = ProcessStep(TEST_IO_SOURCES)
#     ps.produced_outputs = {"a": 1}
#     ps._ProcessStep__prepared = True
#     ps.executed = True
#     ps.reset()
#     assert ps.produced_outputs == {}
#     assert ps._ProcessStep__prepared is False
#     assert ps.executed is False
#

# @pytest.mark.parametrize("item", PROCESS_STEP_CONFIGURATION_KEYS.keys())
# def test_modify_config__valid_key(item):
#     ps = ProcessStep(TEST_IO_SOURCES)
#     _type = PROCESS_STEP_CONFIGURATION_KEYS[item]["type"]
#     if _type == str:
#     new_config = {
#         "with_processing_keys": ["key1", "key2"],
#         "output_processing_key": "output_key",
#     }
#     ps.modify_config(new_config)
#     assert ps.config["with_processing_keys"] == new_config["with_processing_keys"]
#     assert ps.config["output_processing_key"] == new_config["output_processing_key"]
