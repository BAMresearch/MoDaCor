"""Exercise tracing through real steps, including classes defined after installation."""

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData


def _value():
    return BaseData(np.array([1.0]), ureg.count, rank_of_data=1)


def _data():
    return ProcessingData(sample=DataBundle(signal=_value(), hidden=_value()))


def _step(action, patterns=(), *, prepare=None):
    class ToyStep(ProcessStep):
        def dependency_contract(self):
            return ProcessStepDependencies(processing_reads=patterns)

        def calculate(self):
            action(self.processing_data)

        def prepare_execution(self):
            if prepare is not None:
                prepare(self.processing_data)

    return ToyStep(processing_data=_data())


@pytest.mark.parametrize("entrypoint", ["execute", "prepare_execution", "calculate"])
def test_omitted_read_fails_for_every_entrypoint(entrypoint):
    def read(data):
        data["sample"]["hidden"]

    step = _step(read, {"sample.signal"}, prepare=read)
    with pytest.raises(AssertionError, match=r"ToyStep: undeclared read access 'sample.hidden'; contract="):
        if entrypoint == "execute":
            step.execute(step.processing_data)
        else:
            getattr(step, entrypoint)()


@pytest.mark.parametrize("style", ["mapping", "kwargs", "pairs"])
def test_update_reports_individual_write_keys(style):
    def update(data):
        bundle = data["sample"]
        if style == "mapping":
            bundle.update({"extra": _value()})
        elif style == "kwargs":
            bundle.update(extra=_value())
        else:
            bundle.update(iter([("extra", _value())]))

    with pytest.raises(AssertionError, match=r"undeclared write access 'sample.extra'"):
        _step(update, {"sample.signal"}).calculate()
    _step(update, {"sample.extra"}).calculate()


@pytest.mark.parametrize("operation", ["__getitem__", "get", "__contains__"])
def test_bundle_key_reads(operation):
    def read(data):
        getattr(data["sample"], operation)("hidden")

    with pytest.raises(AssertionError, match=r"undeclared read access 'sample.hidden'"):
        _step(read, {"sample.signal"}).calculate()
    _step(read, {"sample.hidden"}).calculate()


@pytest.mark.parametrize("operation", ["__setitem__", "__delitem__", "pop", "setdefault"])
def test_bundle_key_writes(operation):
    def write(data):
        args = ("hidden", _value()) if operation in {"__setitem__", "setdefault"} else ("hidden",)
        getattr(data["sample"], operation)(*args)

    with pytest.raises(AssertionError, match=r"undeclared write access 'sample.hidden'"):
        _step(write, {"sample.signal"}).calculate()
    _step(write, {"sample.hidden"}).calculate()


@pytest.mark.parametrize("operation", ["keys", "items", "values", "__iter__", "copy"])
def test_whole_bundle_access_requires_wildcard(operation):
    def read(data):
        getattr(data["sample"], operation)()

    with pytest.raises(AssertionError, match=r"undeclared read access 'sample.\*'"):
        _step(read, {"sample.signal", "sample.hidden"}).calculate()
    _step(read, {"sample.*"}).calculate()
    _step(read, {"*"}).calculate()


def test_correct_step_passes_using_union_of_reads_and_writes():
    class CorrectStep(ProcessStep):
        def dependency_contract(self):
            return ProcessStepDependencies(processing_writes={"sample.signal"})

        def calculate(self):
            self.processing_data["sample"]["signal"] = self.processing_data["sample"]["signal"]

    CorrectStep().execute(_data())


def test_insertion_records_only_keys_present_at_insertion():
    def insert(data):
        bundle = DataBundle(signal=_value())
        bundle.keys()  # A temporary bundle is outside the identity map.
        data["new"] = bundle

    _step(insert, {"new.signal"}).calculate()
    with pytest.raises(AssertionError, match=r"undeclared write access 'new.signal'"):
        _step(insert).calculate()


def test_inserted_bundle_is_tracked_after_insertion():
    def insert_and_write(data):
        bundle = DataBundle(signal=_value())
        data["new"] = bundle
        bundle["extra"] = _value()

    with pytest.raises(AssertionError, match=r"undeclared write access 'new.extra'"):
        _step(insert_and_write, {"new.signal"}).calculate()


def test_restoring_same_bundle_is_ignored():
    def restore(data):
        data["sample"] = data["sample"]

    _step(restore).calculate()


def test_replacing_bundle_requires_whole_bundle_contract():
    def replace(data):
        data["sample"] = DataBundle(signal=_value())

    with pytest.raises(AssertionError, match=r"undeclared write access 'sample.\*'"):
        _step(replace, {"sample.signal"}).calculate()
    _step(replace, {"sample.*"}).calculate()


@pytest.mark.parametrize("operation", ["__delitem__", "pop"])
def test_processing_data_removal_is_whole_bundle_write(operation):
    def remove(data):
        getattr(data, operation)("sample")

    with pytest.raises(AssertionError, match=r"undeclared write access 'sample.\*'"):
        _step(remove, {"sample.signal", "sample.hidden"}).calculate()
    _step(remove, {"sample.*"}).calculate()


def test_processing_data_navigation_is_ignored():
    def navigate(data):
        data["sample"]
        data.get("sample")
        "sample" in data
        list(data)
        list(data.items())
        list(data.values())

    _step(navigate).calculate()


def test_temporary_bundles_and_other_processing_data_are_ignored():
    def temporary(data):
        bundle = DataBundle(hidden=_value())
        bundle["hidden"]
        bundle.keys()
        ProcessingData(other=bundle)["other"] = DataBundle(signal=_value())

    _step(temporary).calculate()


def test_shared_bundle_is_attributed_to_each_processing_key():
    data = _data()
    data["alias"] = data["sample"]
    step = _step(lambda data: data["sample"]["signal"], {"sample.signal"})
    step.processing_data = data
    with pytest.raises(AssertionError, match=r"undeclared read access 'alias.signal'"):
        step.calculate()


def test_removed_bundle_is_no_longer_tracked():
    def remove(data):
        bundle = data.pop("sample")
        bundle["hidden"]

    _step(remove, {"sample.*"}).calculate()


def test_only_outermost_successful_call_is_checked():
    class OuterStep(ProcessStep):
        checks = 0

        def dependency_contract(self):
            self.checks += 1
            return ProcessStepDependencies(processing_reads={"sample.hidden"})

        def prepare_execution(self):
            self.processing_data["sample"]["hidden"]

        def calculate(self):
            self.processing_data["sample"]["hidden"]

    step = OuterStep()
    step.execute(_data())
    assert step.checks == 1


def test_failed_executions_are_not_checked_and_trace_is_reset():
    def fail(data):
        data["sample"]["hidden"]
        raise ValueError("deliberate failure")

    with pytest.raises(ValueError, match="deliberate failure"):
        _step(fail).calculate()
    with pytest.raises(AssertionError, match="undeclared read"):
        _step(lambda data: data["sample"]["hidden"]).calculate()


def test_inherited_methods_on_future_subclass_are_traced():
    parent = type(_step(lambda data: data["sample"]["hidden"]))

    class ChildStep(parent):
        pass

    with pytest.raises(AssertionError, match=r"ChildStep: undeclared read access 'sample.hidden'"):
        ChildStep(processing_data=_data()).calculate()
