# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

__all__ = ["SOURCE_PROFILES", "get_source_profile", "list_source_profiles"]


SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "mouse": {
        "name": "MOUSE basic profile",
        "description": "Typical sample/background pair for MOUSE correction pipelines.",
        "required_sources": [
            {"ref": "sample", "type": "hdf", "description": "Primary sample data file."},
            {"ref": "background", "type": "hdf", "description": "Background data file."},
        ],
        "optional_sources": [
            {"ref": "defaults", "type": "yaml", "description": "Static metadata / defaults file."},
            {"ref": "intensity_calibration", "type": "hdf", "description": "Optional calibration dataset."},
            {
                "ref": "intensity_calibration_background",
                "type": "hdf",
                "description": "Optional calibration background.",
            },
        ],
    },
    "saxsess": {
        "name": "SAXSess freestanding profile",
        "description": "Common multi-source set for SAXSess freestanding solids pipelines.",
        "required_sources": [
            {"ref": "sample", "type": "hdf", "description": "Primary sample measurement."},
            {"ref": "sample_background", "type": "hdf", "description": "Background measurement."},
            {"ref": "defaults", "type": "yaml", "description": "Static defaults / instrument metadata."},
        ],
        "optional_sources": [
            {"ref": "intensity_calibration", "type": "hdf", "description": "Calibration sample measurement."},
            {
                "ref": "intensity_calibration_background",
                "type": "hdf",
                "description": "Calibration background measurement.",
            },
            {"ref": "gc_calibration", "type": "csv", "description": "Glassy carbon reference curve."},
        ],
    },
}


def get_source_profile(profile_name: str) -> dict[str, Any] | None:
    return SOURCE_PROFILES.get(str(profile_name).strip().lower())


def list_source_profiles() -> dict[str, dict[str, Any]]:
    return SOURCE_PROFILES
