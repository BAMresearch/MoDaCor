import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from modacor.dataclasses.process_step import ProcessStep
from modacor.io.io_sources import IoSources


# Dummy BaseData for testing execute() since ProcessStep.execute() requires it.
class DummyBaseData:
    pass


def test_minimal_instantiation():
    """
    Test that a ProcessStep can be instantiated with only the minimal arguments.
    """
    ps = ProcessStep(io_sources=IoSources())
    assert ps is not None


def test_full_instantiation():
    """
    Test that a ProcessStep can be instantiated with all arguments.

    Some like step_id and executed will be managed by the pipeline
    """
    ps = ProcessStep(
        io_sources=IoSources(),
        configuration={"mask": "IOdata::/path/to/mask"},
        requires_steps=[1, 2, 4],
        step_id=3,
    )
    assert ps is not None


def test_missing_required_input():
    """
    Test that a ProcessStep successfully requires the io_sources argument.
    """
    with pytest.raises(ValueError, match="Missing required argument: 'io_sources'"):
        ProcessStep()
