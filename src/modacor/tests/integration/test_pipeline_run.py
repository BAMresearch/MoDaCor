# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Anja Hörmann, Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "18/06/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np
import pytest
import tempfile
import unittest
from os import unlink
from pathlib import Path
import h5py

from modacor import ureg

from ...dataclasses.basedata import BaseData
from ...dataclasses.databundle import DataBundle
from ...dataclasses.process_step import ProcessStep
from ...dataclasses.processing_data import ProcessingData
from ...io.io_sources import IoSources
from ...io.yaml.yaml_loader import YAMLLoader
from ...io.hdf.hdf_loader import HDFLoader
from ...modules.base_modules.poisson_uncertainties import PoissonUncertainties
from ...runner.pipeline import Pipeline

TEST_IO_SOURCES = IoSources()
TEST_DATA = ProcessingData()
TEST_DATA["data"] = DataBundle()


@pytest.fixture
def flat_data():
    data = ProcessingData()
    data["bundle1"] = DataBundle()
    data["bundle2"] = DataBundle()
    data["bundle1"]["signal"] = BaseData(signal=np.arange(50), units=ureg.Unit("count"))
    data["bundle2"]["signal"] = BaseData(signal=np.ones((10, 10)), units=ureg.Unit("count"))
    return data


class DummyProcessStep(ProcessStep):
    def calculate(self):
        return {"test": DataBundle()}


def test_processstep_pipeline(flat_data):
    "tests execution of a linear processstep pipeline (not actually doing anything)"
    steps = [DummyProcessStep(TEST_IO_SOURCES, step_id=i) for i in range(3)]
    graph = {steps[i]: {steps[i + 1]} for i in range(len(steps) - 1)}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    sequence = []
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.processing_data = flat_data
            sequence.append(node)
            node.execute(flat_data)
            pipeline.done(node)
    assert pipeline._nfinished == len(steps)


def test_actual_processstep(flat_data):
    "test running the PoissonUncertainties Process step"
    step = PoissonUncertainties(TEST_IO_SOURCES)
    # we need to supply a list of values here
    step.modify_config_by_kwargs(with_processing_keys=["bundle2"])
    graph = {step: {}}

    pipeline = Pipeline(graph=graph)
    pipeline.prepare()
    while pipeline.is_active():
        for node in pipeline.get_ready():
            node.processing_data = flat_data
            node.execute(flat_data)
            pipeline.done(node)
    assert node.produced_outputs["bundle2"]["signal"].variances["Poisson"].mean().astype(int) == 1


class TestRealisticPipeline(unittest.TestCase):
    """Tests using yaml files as input for both data and pipeline.

    the yaml loader needs an actual path to load, can't take a string.

    """

    def setUp(self):
        # setup yaml static input file
        self.temp_file_handle = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
        self.temp_file_path = self.temp_file_handle.name
        self.temp_file_handle.close()
        with open(self.temp_file_path, "w") as yaml_file:
            yaml_file.write("""
            instrument:
              name: "SAXSess I"
              type: "X-ray scattering"
              manufacturer: "Anton Paar"
              model: "SAXSess"
            wavelength:
              value: 0.1542
              units: "nm"
              uncertainty: 0.0005
            detector:
              name: "Mythen2"
              type: "1D strip detector"
              manufacturer: "Dectris"
              n_pixels: 1280
              darkcurrent:
                value: 1e-5
                units: "counts/second"
                uncertainty: 0.1e-5
              """)
        # setup two small hdf5 files for sample and background
        self.temp_dataset_shape = (10, 2)
        self.temp_time_handle = "frame_exposure_time"
        self.temp_hdf_file_sample = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
        self.temp_hdf_path_sample = self.temp_hdf_file_sample.name
        self.temp_hdf_file_sample.close()
        self.temp_hdf_file_background = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
        self.temp_hdf_path_background = self.temp_hdf_file_background.name
        self.temp_hdf_file_background.close()

        self.temp_data = {
            "sample": 10 * np.ones(self.temp_dataset_shape),
            "sample_background": np.ones(self.temp_dataset_shape),
        }
        self.temp_hdf_paths = {"sample": self.temp_hdf_path_sample, "sample_background": self.temp_hdf_path_background}
        for key, path in self.temp_hdf_paths.items():
            with h5py.File(path, "w") as hdf_file:
                data = hdf_file.create_dataset("data", data=self.temp_data[key], dtype="float64", compression="gzip")
                data.attrs["units"] = "counts"
                detector = hdf_file.create_group("detector")
                time = detector.create_dataset(self.temp_time_handle, data=10.0)
                time.attrs["units"] = "s"
                time.attrs["uncertainties"] = f"{self.temp_time_handle}_uncertainty"
                time_uncertainty = detector.create_dataset(f"{self.temp_time_handle}_uncertainty", data=0.1)
                time_uncertainty.attrs["units"] = "s"

        self.yaml_semirealistic_linear_pipeline = """
        name: freestanding_solid
        steps:
          poisson:
            module: PoissonUncertainties
            step_id: 1
            requires_steps: []
            configuration:
              with_processing_keys:
                - sample
                - sample_background
          normalize_by_time:
            module: Divide
            step_id: 2
            requires_steps: [1]
            configuration:
              divisor_source: sample::detector/frame_exposure_time
              divisor_uncertainties_sources:
                propagate_to_all: sample::detector/frame_exposure_time_uncertainty
              divisor_units_source: sample::detector/frame_exposure_time@units
              with_processing_keys:
                - sample
                - sample_background
          subtract dark current:
            module: Subtract
            step_id: 3
            requires_steps: [2]
            configuration:
              subtrahend_source: yaml::detector/darkcurrent/value
              subtrahend_uncertainties_sources:
                propagate_to_all: yaml::detector/darkcurrent/uncertainty
              subtrahend_units_source: yaml::detector/darkcurrent/units
              with_processing_keys:
              - sample
              - sample_background
          subtract background:
            module: SubtractDatabundles
            step_id: 4
            requires_steps: [3]
            configuration:
              with_processing_keys:
                - sample
                - sample_background
        """

    def tearDown(self):
        unlink(self.temp_file_path)
        unlink(self.temp_hdf_path_sample)
        unlink(self.temp_hdf_path_background)

    def test_semirealistic_pipeline(self):
        metadata_source = YAMLLoader(source_reference="yaml", resource_location=self.temp_file_path)
        sources = IoSources()
        sources.register_source(source=metadata_source)
        sources.register_source(HDFLoader(source_reference="sample", resource_location=self.temp_hdf_path_sample))
        sources.register_source(
            HDFLoader(source_reference="sample_background", resource_location=self.temp_hdf_path_background)
        )

        source = 'sample'
        _=[print(f"{source}::{key} with shape {val}") for key, val in sources.get_source(source)._file_datasets_shapes.items()]

        processingdata = ProcessingData()
        for key in ["sample", "sample_background"]:
            processingdata[key] = DataBundle()
            processingdata[key]["signal"] = BaseData(
                sources.get_data(f"{key}::data"), units=sources.get_data_attributes(f"{key}::data")["units"]
            )

        pipeline = Pipeline.from_yaml(self.yaml_semirealistic_linear_pipeline)
        pipeline.prepare()
        sequence = []
        while pipeline.is_active():
            for node in pipeline.get_ready():
                node.processing_data = processingdata
                node.io_sources = sources
                sequence.append(node)
                node.execute(processingdata)
                pipeline.done(node)

        assert np.isclose(np.mean(processingdata["sample"]["signal"].signal), 0.9)
        # rough estimate for poisson error
        expected_error = np.sqrt((np.sqrt(10) / 10.0) ** 2 + (np.sqrt(1) / 10.0) ** 2)
        assert np.isclose(
            np.mean(processingdata["sample"]["signal"].uncertainties["Poisson"]), expected_error, atol=2e-4
        )
