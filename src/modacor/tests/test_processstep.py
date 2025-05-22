from modacor.dataclasses.process_step import ProcessStep
from modacor.io import IoSources

TEST_IO_SOURCES = IoSources()


def test_minimal_instantiation():
    """
    Test that a ProcessStep can be instantiated with only the minimal arguments.
    """
    ps = ProcessStep(TEST_IO_SOURCES)
    assert isinstance(ps, ProcessStep)
