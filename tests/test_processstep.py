import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from modacor.dataclasses.process_step import ProcessStep


# Dummy BaseData for testing execute() since ProcessStep.execute() requires it.
class DummyBaseData:
    pass


def test_minimal_instantiation():
    """
    Test that a ProcessStep can be instantiated with only the minimal arguments.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_001",
        calling_module_path=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
    )
    assert ps is not None


def test_full_instantiation():
    """
    Test that a ProcessStep can be instantiated with all arguments.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_002",
        calling_module_path=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
        required_data_keys=['fish', 'cake'],
        calling_arguments={'fish': 'salmon', 'cake': 'chocolate'},
        step_keywords=['test', 'full'],
        step_doc="Test full instantiation",
        step_reference="doi: 10.1234/5678",
        step_note="I am a note.",
        # produced_values={},  # this is internally generated
        use_frames_cache=['fish'],
        use_overall_cache=['cake'],
        saved={'cake': '/path/to/cake'},
    )
    ps.start()
    ps.stop()
    assert ps is not None


def test_missing_required_input():
    """
    Test that a ProcessStep can be instantiated with all arguments.
    """
    with pytest.raises(ValueError, match="Missing required data keys in calling_arguments: \['bread'\]"):
        ProcessStep(
            calling_name="Test Process",
            calling_id="proc_002",
            calling_module_path=Path("src/modacor/modules/some_module.py"),
            calling_version="1.0",
            required_data_keys=['fish', 'cake', 'bread'],
            calling_arguments={'fish': 'salmon', 'cake': 'chocolate'},
            step_keywords=['test', 'full'],
            step_doc="Test full instantiation",
            step_reference="doi: 10.1234/5678",
            step_note="I am a note.",
            # produced_values={},  # this is internally generated
            use_frames_cache=['fish'],
            use_overall_cache=['cake'],
            saved={'cake': '/path/to/cake'},
        )


def test_duration_calculation():
    """
    Test that the duration property correctly computes elapsed time.
    """
    ps = ProcessStep(
        calling_name="Test Process",
        calling_id="proc_003",
        calling_module_path=Path("src/modacor/modules/some_module.py"),
        calling_version="1.0",
        required_data_keys=[],
        calling_arguments={},
        step_keywords=['test', 'duration'],
        step_doc="Test duration calculation",
        step_reference="",
        step_note=None,
        produced_values={},
        use_frames_cache=[],
    )
    # Manually set start_time 5 seconds before now.
    ps.start_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    ps.stop()  # Sets stop_time to now.
    duration = ps.duration
    assert duration is not None and duration >= 5
