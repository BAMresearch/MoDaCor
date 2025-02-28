import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from modacor.dataclasses.processstep import ProcessStep

# Dummy BaseData for testing execute() since ProcessStep.execute() requires it.
class DummyBaseData:
    pass

def test_can_execute_success():
    """
    Test that can_execute returns True when all required keys are present.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_001",
        calling_module=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
        required_data_keys=["key1", "key2"],
        calling_data={"key1": 100, "key2": 200},
        step_keywords=["test"],
        note="A test process",
        produced_values={},
        use_cache=[],
        status="pending",
        start_time=None,
        end_time=None,
        execution_messages=[]
    )
    assert ps.can_execute() is True

def test_can_execute_failure():
    """
    Test that can_execute returns False when a required key is missing.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_002",
        calling_module=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
        required_data_keys=["key1", "key2"],
        calling_data={"key1": 100},  # missing "key2"
        step_keywords=["test"],
        note="A test process",
        produced_values={},
        use_cache=[],
        status="pending",
        start_time=None,
        end_time=None,
        execution_messages=[]
    )
    assert ps.can_execute() is False

def test_duration_calculation():
    """
    Test that the duration property correctly computes elapsed time.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_003",
        calling_module=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
        required_data_keys=[],
        calling_data={},
        step_keywords=[],
        note=None,
        produced_values={},
        use_cache=[],
        status="pending",
        start_time=None,
        end_time=None,
        execution_messages=[]
    )
    # Manually set start_time 5 seconds before now.
    ps.start_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    ps.finish()  # Sets end_time to now.
    duration = ps.duration
    assert duration is not None and duration >= 5

def test_execute_raises_error():
    """
    Test that execute() raises a ValueError when can_execute() returns False.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_004",
        calling_module=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
        required_data_keys=["key1"],
        calling_data={},  # "key1" is missing
        step_keywords=["test"],
        note="Test execute failure",
        produced_values={},
        use_cache=[],
        status="pending",
        start_time=None,
        end_time=None,
        execution_messages=[]
    )
    with pytest.raises(ValueError, match="Process step cannot be executed"):
        ps.execute(DummyBaseData())