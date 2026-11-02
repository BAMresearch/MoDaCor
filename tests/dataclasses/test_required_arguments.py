# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
import yaml

import modacor.modules as modules
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber, argument_value_is_missing
from modacor.dataclasses.processing_data import ProcessingData
from modacor.runner.pipeline import Pipeline


class RequiredStep(ProcessStep):
    documentation = ProcessStepDescriber(
        calling_name="Required step",
        calling_id="RequiredStep",
        calling_module_path=Path(__file__),
        calling_version="1",
        arguments={"first": {"required": True}, "second": {"required": True}},
    )

    def prepare_execution(self):
        self.prepared = True

    def calculate(self):
        self.calculated = True


@pytest.mark.parametrize("name", modules.__all__)
def test_public_required_arguments_have_placeholder_defaults(name):
    step_class = getattr(modules, name)
    for spec in step_class.documentation.arguments.values():
        if spec["required"]:
            assert argument_value_is_missing(spec["default"])


@pytest.mark.parametrize("value", [None, "", "  ", {}, [], (), set()])
def test_required_arguments_rejected_before_prepare_or_calculate(value):
    step = RequiredStep(step_id="missing-input", configuration={"first": value})
    with pytest.raises(ValueError, match=r"RequiredStep \(missing-input\).*first.*second"):
        step.execute(ProcessingData())
    assert not getattr(step, "prepared", False)
    assert not getattr(step, "calculated", False)


@pytest.mark.parametrize("value", [0, False])
def test_zero_and_false_are_supplied(value):
    step = RequiredStep(configuration={"first": value, "second": value})
    step.execute(ProcessingData())
    assert step.prepared and step.calculated


@pytest.mark.parametrize("value", [0, False, "signal", ["sample"], {"signal": "count"}])
def test_required_usable_default_rejected_at_class_definition(value):
    with pytest.raises(ValueError, match="InvalidStep.*input.*placeholder"):

        class InvalidStep(ProcessStep):
            documentation = ProcessStepDescriber(
                calling_name="Invalid step",
                calling_id="InvalidStep",
                calling_module_path=Path(__file__),
                calling_version="1",
                arguments={"input": {"required": True, "default": value}},
            )


@pytest.mark.parametrize("loader", ["yaml", "spec"])
def test_pipeline_load_names_step_id(loader):
    with pytest.raises(ValueError, match=r"UnitsLabelUpdate \(units\).*update_pairs"):
        if loader == "yaml":
            Pipeline.from_yaml("steps: {units: {module: UnitsLabelUpdate}}")
        else:
            Pipeline.from_spec({"nodes": [{"id": "units", "module": "UnitsLabelUpdate"}]})


def test_units_label_update_no_longer_silently_does_nothing():
    with pytest.raises(ValueError, match="UnitsLabelUpdate.*update_pairs"):
        modules.UnitsLabelUpdate().execute(ProcessingData())


@pytest.mark.parametrize("value", [None, "", "  ", {}, [], (), set()])
@pytest.mark.parametrize("route", ["execute", "yaml", "spec"])
def test_typed_required_placeholders_use_shared_missing_input_error(value, route):
    config = {"update_pairs": value}
    with pytest.raises(ValueError, match=r"UnitsLabelUpdate \(units\).*update_pairs"):
        if route == "execute":
            modules.UnitsLabelUpdate(step_id="units", configuration=config).execute(ProcessingData())
        elif route == "yaml":
            Pipeline.from_yaml(
                yaml.safe_dump({"steps": {"units": {"module": "UnitsLabelUpdate", "configuration": config}}})
            )
        else:
            Pipeline.from_spec({"nodes": [{"id": "units", "module": "UnitsLabelUpdate", "config": config}]})


@pytest.mark.parametrize("override", [0, False, {"value": 0, "units": "m"}])
def test_zero_sample_z_override_is_supplied(override):
    step = modules.XSGeometryFromPixelCoordinates(
        configuration={
            "detector_frame": {"type": "nexus"},
            "sample_z_override": override,
            "photon_source": "sample::/energy",
            "photon_units_source": "sample::/energy@units",
        }
    )
    assert step.missing_required_arguments() == []


GEOMETRY_SOURCES = {
    "PixelCoordinates3D": (
        "det_coord_x_source",
        "det_coord_y_source",
        "det_coord_z_source",
        "pixel_pitch_slow_source",
        "pixel_pitch_fast_source",
    ),
    "XSGeometryFromPixelCoordinates": (
        "pixel_pitch_slow_source",
        "pixel_pitch_fast_source",
        "sample_z_source",
    ),
}


@pytest.mark.parametrize("name", GEOMETRY_SOURCES)
@pytest.mark.parametrize("form", ["adapter", "sources", "neither", "partial"])
@pytest.mark.parametrize("route", ["execute", "yaml", "spec"])
def test_geometry_required_alternatives(name, form, route, monkeypatch):
    step_class = getattr(modules, name)
    config = {"with_processing_keys": ["sample"]}
    if name == "XSGeometryFromPixelCoordinates":
        config.update(photon_source="sample::/energy", photon_units_source="sample::/energy@units")
    if form == "adapter":
        config["detector_frame"] = {"type": "nexus", "source": "sample", "detector_path": "/detector"}
        if name == "XSGeometryFromPixelCoordinates":
            config["sample_z_override"] = {"value": 0.0, "units": "m"}
    if form in {"sources", "partial"}:
        for source in GEOMETRY_SOURCES[name]:
            config[source] = "sample::/" + source
        if form == "partial":
            del config["pixel_pitch_fast_source"]

    # Exercise the validation boundary without loading detector files or changing numbers.
    calls = []
    monkeypatch.setattr(step_class, "prepare_execution", lambda self: None)
    monkeypatch.setattr(step_class, "calculate", lambda self: calls.append(self.step_id))

    def run():
        if route == "execute":
            step_class(step_id="geometry", configuration=config).execute(ProcessingData())
        elif route == "yaml":
            Pipeline.from_yaml(yaml.safe_dump({"steps": {"geometry": {"module": name, "configuration": config}}}))
        else:
            Pipeline.from_spec({"nodes": [{"id": "geometry", "module": name, "config": config}]})

    if form in {"adapter", "sources"}:
        run()
        assert calls == (["geometry"] if route == "execute" else [])
    else:
        with pytest.raises(ValueError) as error:
            run()
        message = str(error.value)
        assert f"{name} (geometry)" in message
        assert "detector_frame" in message
        assert "pixel_pitch_fast_source" in message
        if form == "neither":
            assert all(source in message for source in GEOMETRY_SOURCES[name])
            if name == "XSGeometryFromPixelCoordinates":
                assert "sample_z_override" in message
        else:
            assert "pixel_pitch_slow_source" not in message
        assert not calls
