"""Guard exported steps' schema reads and single source of configuration defaults.

Follow only self.configuration and aliases/parameters receiving that object.
Shared helpers are followed for key declarations, not their reusable fallbacks.
"""

import ast
import inspect
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

import modacor.modules as modules
from modacor.dataclasses.process_step import ProcessStep

_CONFIG = object()
_UNKNOWN = object()
# R03a baseline: exact occurrences, removed by R03b. New/stale entries fail.
ALLOWED_VIOLATIONS = {
    "AngleToQ: angle_convention: literal fallback 'scattering_angle'; schema default 'scattering_angle' (angle_to_q.py:178)",
    "AngleToQ: angle_key: literal fallback 'angle'; schema default 'angle' (angle_to_q.py:132)",
    "AngleToQ: angle_key: literal fallback 'angle'; schema default 'angle' (angle_to_q.py:173)",
    "AngleToQ: center_key: literal fallback 'beam_center'; schema default 'beam_center' (angle_to_q.py:135)",
    "AngleToQ: center_key: literal fallback 'beam_center'; schema default 'beam_center' (angle_to_q.py:152)",
    "AngleToQ: output_key: literal fallback 'Q'; schema default 'Q' (angle_to_q.py:138)",
    "AngleToQ: output_key: literal fallback 'Q'; schema default 'Q' (angle_to_q.py:174)",
    "AngleToQ: output_units: literal fallback '1/nm'; schema default '1/nm' (angle_to_q.py:177)",
    "AngleToQ: photon_uncertainties_sources: literal fallback {}; schema default {} (angle_to_q.py:166)",
    "AppendProcessingData: databundle_output_key: literal fallback 'signal'; schema default 'signal' (append_processing_data.py:232)",
    "AppendProcessingData: uncertainties_sources: literal fallback {}; schema default {} (append_processing_data.py:248)",
    "AppendSink: iosink_method_kwargs: literal fallback {}; schema default {} (append_sink.py:105)",
    "AppendSource: iosource_method_kwargs: literal fallback {}; schema default {} (append_source.py:121)",
    "ApplyMask: basedata_to_mask: literal fallback ['signal']; schema default ['signal'] (apply_mask.py:138)",
    "ApplyMask: mask_key: literal fallback 'mask'; schema default 'mask' (apply_mask.py:137)",
    "ApplyMask: masked_value: literal fallback 'nan'; schema default 'nan' (apply_mask.py:139)",
    "AttenuatorPlateCorrection: apply_as: literal fallback 'divide'; schema default 'divide' (attenuator_plate_correction.py:226)",
    "AttenuatorPlateCorrection: correction_key: literal fallback 'attenuator_transmission'; schema default 'attenuator_transmission' (attenuator_plate_correction.py:212)",
    "AttenuatorPlateCorrection: correction_key: literal fallback 'attenuator_transmission'; schema default 'attenuator_transmission' (attenuator_plate_correction.py:230)",
    "AttenuatorPlateCorrection: cos_alpha_key: literal fallback 'CosAlpha'; schema default 'CosAlpha' (attenuator_plate_correction.py:209)",
    "AttenuatorPlateCorrection: cos_alpha_key: literal fallback 'CosAlpha'; schema default 'CosAlpha' (attenuator_plate_correction.py:229)",
    "AttenuatorPlateCorrection: minimum_cos_alpha: literal fallback 1e-12; schema default 1e-12 (attenuator_plate_correction.py:224)",
    "AttenuatorPlateCorrection: normalize_to_normal_incidence: literal fallback False; schema default False (attenuator_plate_correction.py:225)",
    "BitwiseOrMasks: target_mask_key: literal fallback 'mask'; schema default 'mask' (bitwise_or_masks.py:90)",
    "CapillarySampleContainerCorrection: absolute_tolerance: literal fallback 1e-12; schema default 1e-12 (capillary_sample_container_correction.py:271)",
    "CapillarySampleContainerCorrection: chord_order: literal fallback 12; schema default 12 (capillary_sample_container_correction.py:360)",
    "CapillarySampleContainerCorrection: coord_x_key: literal fallback 'coord_x'; schema default 'coord_x' (capillary_sample_container_correction.py:193)",
    "CapillarySampleContainerCorrection: coord_x_key: literal fallback 'coord_x'; schema default 'coord_x' (capillary_sample_container_correction.py:382)",
    "CapillarySampleContainerCorrection: coord_y_key: literal fallback 'coord_y'; schema default 'coord_y' (capillary_sample_container_correction.py:194)",
    "CapillarySampleContainerCorrection: coord_y_key: literal fallback 'coord_y'; schema default 'coord_y' (capillary_sample_container_correction.py:383)",
    "CapillarySampleContainerCorrection: coord_z_key: literal fallback 'coord_z'; schema default 'coord_z' (capillary_sample_container_correction.py:195)",
    "CapillarySampleContainerCorrection: coord_z_key: literal fallback 'coord_z'; schema default 'coord_z' (capillary_sample_container_correction.py:384)",
    "CapillarySampleContainerCorrection: detector_chunk_size: literal fallback 256; schema default 256 (capillary_sample_container_correction.py:273)",
    "CapillarySampleContainerCorrection: detector_chunk_size: literal fallback 256; schema default 256 (capillary_sample_container_correction.py:287)",
    "CapillarySampleContainerCorrection: effective_sample_mu_key: literal fallback 'capillary_effective_sample_mu'; schema default 'capillary_effective_sample_mu' (capillary_sample_container_correction.py:216)",
    "CapillarySampleContainerCorrection: effective_sample_mu_key: literal fallback 'capillary_effective_sample_mu'; schema default 'capillary_effective_sample_mu' (capillary_sample_container_correction.py:567)",
    "CapillarySampleContainerCorrection: empty_calculated_transmission_key: literal fallback 'capillary_empty_calculated_transmission'; schema default 'capillary_empty_calculated_transmission' (capillary_sample_container_correction.py:215)",
    "CapillarySampleContainerCorrection: empty_calculated_transmission_key: literal fallback 'capillary_empty_calculated_transmission'; schema default 'capillary_empty_calculated_transmission' (capillary_sample_container_correction.py:552)",
    "CapillarySampleContainerCorrection: empty_processing_key: literal fallback 'background'; schema default 'background' (capillary_sample_container_correction.py:191)",
    "CapillarySampleContainerCorrection: empty_processing_key: literal fallback 'background'; schema default 'background' (capillary_sample_container_correction.py:334)",
    "CapillarySampleContainerCorrection: evaluation_mode: literal fallback 'adaptive'; schema default 'adaptive' (capillary_sample_container_correction.py:260)",
    "CapillarySampleContainerCorrection: filled_calculated_transmission_key: literal fallback 'capillary_filled_calculated_transmission'; schema default 'capillary_filled_calculated_transmission' (capillary_sample_container_correction.py:214)",
    "CapillarySampleContainerCorrection: filled_calculated_transmission_key: literal fallback 'capillary_filled_calculated_transmission'; schema default 'capillary_filled_calculated_transmission' (capillary_sample_container_correction.py:544)",
    "CapillarySampleContainerCorrection: filled_processing_key: literal fallback 'sample'; schema default 'sample' (capillary_sample_container_correction.py:190)",
    "CapillarySampleContainerCorrection: filled_processing_key: literal fallback 'sample'; schema default 'sample' (capillary_sample_container_correction.py:333)",
    "CapillarySampleContainerCorrection: incident_direction: literal fallback (0.0, 0.0, 1.0); schema default (0.0, 0.0, 1.0) (capillary_sample_container_correction.py:358)",
    "CapillarySampleContainerCorrection: max_depth: literal fallback 10; schema default 10 (capillary_sample_container_correction.py:272)",
    "CapillarySampleContainerCorrection: minimum_attenuation_factor: literal fallback 1e-12; schema default 1e-12 (capillary_sample_container_correction.py:446)",
    "CapillarySampleContainerCorrection: profile_retained_fraction_key: literal fallback 'capillary_beam_profile_retained_fraction'; schema default 'capillary_beam_profile_retained_fraction' (capillary_sample_container_correction.py:217)",
    "CapillarySampleContainerCorrection: profile_retained_fraction_key: literal fallback 'capillary_beam_profile_retained_fraction'; schema default 'capillary_beam_profile_retained_fraction' (capillary_sample_container_correction.py:575)",
    "CapillarySampleContainerCorrection: relative_tolerance: literal fallback 0.001; schema default 0.001 (capillary_sample_container_correction.py:270)",
    "CapillarySampleContainerCorrection: sample_attenuation_key: literal fallback 'capillary_sample_attenuation'; schema default 'capillary_sample_attenuation' (capillary_sample_container_correction.py:210)",
    "CapillarySampleContainerCorrection: sample_attenuation_key: literal fallback 'capillary_sample_attenuation'; schema default 'capillary_sample_attenuation' (capillary_sample_container_correction.py:529)",
    "CapillarySampleContainerCorrection: sample_evaluated_mask_key: literal fallback 'capillary_sample_attenuation_evaluated'; schema default 'capillary_sample_attenuation_evaluated' (capillary_sample_container_correction.py:218)",
    "CapillarySampleContainerCorrection: sample_evaluated_mask_key: literal fallback 'capillary_sample_attenuation_evaluated'; schema default 'capillary_sample_attenuation_evaluated' (capillary_sample_container_correction.py:535)",
    "CapillarySampleContainerCorrection: wall_empty_attenuation_key: literal fallback 'capillary_wall_attenuation_empty'; schema default 'capillary_wall_attenuation_empty' (capillary_sample_container_correction.py:212)",
    "CapillarySampleContainerCorrection: wall_empty_attenuation_key: literal fallback 'capillary_wall_attenuation_empty'; schema default 'capillary_wall_attenuation_empty' (capillary_sample_container_correction.py:533)",
    "CapillarySampleContainerCorrection: wall_empty_evaluated_mask_key: literal fallback 'capillary_wall_empty_attenuation_evaluated'; schema default 'capillary_wall_empty_attenuation_evaluated' (capillary_sample_container_correction.py:220)",
    "CapillarySampleContainerCorrection: wall_empty_evaluated_mask_key: literal fallback 'capillary_wall_empty_attenuation_evaluated'; schema default 'capillary_wall_empty_attenuation_evaluated' (capillary_sample_container_correction.py:541)",
    "CapillarySampleContainerCorrection: wall_filled_attenuation_key: literal fallback 'capillary_wall_attenuation_filled'; schema default 'capillary_wall_attenuation_filled' (capillary_sample_container_correction.py:211)",
    "CapillarySampleContainerCorrection: wall_filled_attenuation_key: literal fallback 'capillary_wall_attenuation_filled'; schema default 'capillary_wall_attenuation_filled' (capillary_sample_container_correction.py:530)",
    "CapillarySampleContainerCorrection: wall_filled_evaluated_mask_key: literal fallback 'capillary_wall_filled_attenuation_evaluated'; schema default 'capillary_wall_filled_attenuation_evaluated' (capillary_sample_container_correction.py:219)",
    "CapillarySampleContainerCorrection: wall_filled_evaluated_mask_key: literal fallback 'capillary_wall_filled_attenuation_evaluated'; schema default 'capillary_wall_filled_attenuation_evaluated' (capillary_sample_container_correction.py:538)",
    "CapillarySampleContainerCorrection: wall_subtraction_scale_key: literal fallback 'capillary_wall_subtraction_scale'; schema default 'capillary_wall_subtraction_scale' (capillary_sample_container_correction.py:213)",
    "CapillarySampleContainerCorrection: wall_subtraction_scale_key: literal fallback 'capillary_wall_subtraction_scale'; schema default 'capillary_wall_subtraction_scale' (capillary_sample_container_correction.py:534)",
    "CapillarySelfAbsorptionCorrection: absolute_tolerance: literal fallback 1e-12; schema default 1e-12 (capillary_self_absorption_correction.py:693)",
    "CapillarySelfAbsorptionCorrection: attenuation_key: literal fallback 'capillary_sample_attenuation'; schema default 'capillary_sample_attenuation' (capillary_self_absorption_correction.py:346)",
    "CapillarySelfAbsorptionCorrection: attenuation_key: literal fallback 'capillary_sample_attenuation'; schema default 'capillary_sample_attenuation' (capillary_self_absorption_correction.py:666)",
    "CapillarySelfAbsorptionCorrection: calculated_transmission_key: literal fallback 'capillary_calculated_transmission'; schema default 'capillary_calculated_transmission' (capillary_self_absorption_correction.py:348)",
    "CapillarySelfAbsorptionCorrection: calculated_transmission_key: literal fallback 'capillary_calculated_transmission'; schema default 'capillary_calculated_transmission' (capillary_self_absorption_correction.py:668)",
    "CapillarySelfAbsorptionCorrection: capillary_axis: literal fallback (1.0, 0.0, 0.0); schema default (1.0, 0.0, 0.0) (capillary_self_absorption_correction.py:472)",
    "CapillarySelfAbsorptionCorrection: capillary_centre: literal fallback (0.0, 0.0, 0.0); schema default (0.0, 0.0, 0.0) (capillary_self_absorption_correction.py:465)",
    "CapillarySelfAbsorptionCorrection: capillary_centre_units: literal fallback 'm'; schema default 'm' (capillary_self_absorption_correction.py:466)",
    "CapillarySelfAbsorptionCorrection: chord_order: literal fallback 12; schema default 12 (capillary_self_absorption_correction.py:627)",
    "CapillarySelfAbsorptionCorrection: coord_x_key: literal fallback 'coord_x'; schema default 'coord_x' (capillary_self_absorption_correction.py:337)",
    "CapillarySelfAbsorptionCorrection: coord_x_key: literal fallback 'coord_x'; schema default 'coord_x' (capillary_self_absorption_correction.py:662)",
    "CapillarySelfAbsorptionCorrection: coord_y_key: literal fallback 'coord_y'; schema default 'coord_y' (capillary_self_absorption_correction.py:338)",
    "CapillarySelfAbsorptionCorrection: coord_y_key: literal fallback 'coord_y'; schema default 'coord_y' (capillary_self_absorption_correction.py:663)",
    "CapillarySelfAbsorptionCorrection: coord_z_key: literal fallback 'coord_z'; schema default 'coord_z' (capillary_self_absorption_correction.py:339)",
    "CapillarySelfAbsorptionCorrection: coord_z_key: literal fallback 'coord_z'; schema default 'coord_z' (capillary_self_absorption_correction.py:664)",
    "CapillarySelfAbsorptionCorrection: correction_key: literal fallback 'capillary_self_absorption'; schema default 'capillary_self_absorption' (capillary_self_absorption_correction.py:347)",
    "CapillarySelfAbsorptionCorrection: correction_key: literal fallback 'capillary_self_absorption'; schema default 'capillary_self_absorption' (capillary_self_absorption_correction.py:667)",
    "CapillarySelfAbsorptionCorrection: detector_chunk_size: literal fallback 256; schema default 256 (capillary_self_absorption_correction.py:695)",
    "CapillarySelfAbsorptionCorrection: detector_chunk_size: literal fallback 256; schema default 256 (capillary_self_absorption_correction.py:707)",
    "CapillarySelfAbsorptionCorrection: effective_sample_mu_key: literal fallback 'capillary_effective_sample_mu'; schema default 'capillary_effective_sample_mu' (capillary_self_absorption_correction.py:349)",
    "CapillarySelfAbsorptionCorrection: effective_sample_mu_key: literal fallback 'capillary_effective_sample_mu'; schema default 'capillary_effective_sample_mu' (capillary_self_absorption_correction.py:669)",
    "CapillarySelfAbsorptionCorrection: evaluated_mask_key: literal fallback 'capillary_attenuation_evaluated'; schema default 'capillary_attenuation_evaluated' (capillary_self_absorption_correction.py:351)",
    "CapillarySelfAbsorptionCorrection: evaluated_mask_key: literal fallback 'capillary_attenuation_evaluated'; schema default 'capillary_attenuation_evaluated' (capillary_self_absorption_correction.py:670)",
    "CapillarySelfAbsorptionCorrection: evaluation_mode: literal fallback 'adaptive'; schema default 'adaptive' (capillary_self_absorption_correction.py:657)",
    "CapillarySelfAbsorptionCorrection: incident_direction: literal fallback (0.0, 0.0, 1.0); schema default (0.0, 0.0, 1.0) (capillary_self_absorption_correction.py:619)",
    "CapillarySelfAbsorptionCorrection: input_state: literal fallback 'transmission_normalized'; schema default 'transmission_normalized' (capillary_self_absorption_correction.py:595)",
    "CapillarySelfAbsorptionCorrection: max_depth: literal fallback 10; schema default 10 (capillary_self_absorption_correction.py:694)",
    "CapillarySelfAbsorptionCorrection: minimum_attenuation_factor: literal fallback 1e-12; schema default 1e-12 (capillary_self_absorption_correction.py:674)",
    "CapillarySelfAbsorptionCorrection: profile_retained_fraction_key: literal fallback 'capillary_beam_profile_retained_fraction'; schema default 'capillary_beam_profile_retained_fraction' (capillary_self_absorption_correction.py:350)",
    "CapillarySelfAbsorptionCorrection: profile_retained_fraction_key: literal fallback 'capillary_beam_profile_retained_fraction'; schema default 'capillary_beam_profile_retained_fraction' (capillary_self_absorption_correction.py:671)",
    "CapillarySelfAbsorptionCorrection: relative_tolerance: literal fallback 0.001; schema default 0.001 (capillary_self_absorption_correction.py:692)",
    "CapillarySelfAbsorptionCorrection: sample_mu_sensitivity_relative_step: literal fallback 0.01; schema default 0.01 (capillary_self_absorption_correction.py:485)",
    "CapillarySelfAbsorptionCorrection: sample_mu_uncertainties_sources: literal fallback {}; schema default '<undeclared>' (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: sample_mu_uncertainties_sources: undeclared configuration key (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: sample_phase_absorption_uncertainties_sources: literal fallback {}; schema default {} (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: sample_phase_absorption_units: undeclared configuration key (material_attenuation.py:73)",
    "CapillarySelfAbsorptionCorrection: sample_phase_absorption_units_source: undeclared configuration key (material_attenuation.py:66)",
    "CapillarySelfAbsorptionCorrection: sample_phase_thickness_uncertainties_sources: literal fallback {}; schema default {} (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: sample_phase_transmission_uncertainties_sources: literal fallback {}; schema default {} (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: sample_phase_transmission_units: undeclared configuration key (material_attenuation.py:73)",
    "CapillarySelfAbsorptionCorrection: sample_phase_transmission_units_source: undeclared configuration key (material_attenuation.py:66)",
    "CapillarySelfAbsorptionCorrection: sample_radius_uncertainties_sources: literal fallback {}; schema default '<undeclared>' (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: sample_radius_uncertainties_sources: undeclared configuration key (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: transmission_uncertainties_sources: literal fallback {}; schema default {} (capillary_self_absorption_correction.py:607)",
    "CapillarySelfAbsorptionCorrection: wall_mu_uncertainties_sources: literal fallback {}; schema default '<undeclared>' (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: wall_mu_uncertainties_sources: undeclared configuration key (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: wall_thickness_uncertainties_sources: literal fallback {}; schema default '<undeclared>' (capillary_self_absorption_correction.py:374)",
    "CapillarySelfAbsorptionCorrection: wall_thickness_uncertainties_sources: undeclared configuration key (capillary_self_absorption_correction.py:374)",
    "CombineUncertainties: combinations: literal fallback {}; schema default {} (combine_uncertainties.py:93)",
    "CombineUncertainties: drop_source_keys: literal fallback False; schema default False (combine_uncertainties.py:99)",
    "CombineUncertainties: ignore_missing: literal fallback False; schema default False (combine_uncertainties.py:100)",
    "CombineUncertainties: target_basedata_key: literal fallback 'signal'; schema default 'signal' (combine_uncertainties.py:98)",
    "CombineUncertaintiesMax: combinations: literal fallback {}; schema default {} (combine_uncertainties_max.py:76)",
    "CombineUncertaintiesMax: drop_source_keys: literal fallback False; schema default False (combine_uncertainties_max.py:84)",
    "CombineUncertaintiesMax: ignore_missing: literal fallback False; schema default False (combine_uncertainties_max.py:85)",
    "CombineUncertaintiesMax: target_basedata_key: literal fallback 'signal'; schema default 'signal' (combine_uncertainties_max.py:83)",
    "ConcatenateDatabundles: data_keys: literal fallback ['signal', 'Q']; schema default ['signal', 'Q'] (concatenate_databundles.py:105)",
    "ConcatenateDatabundles: descending: literal fallback False; schema default False (concatenate_databundles.py:280)",
    "ConcatenateDatabundles: output_processing_key: literal fallback 'concatenated'; schema default 'concatenated' (concatenate_databundles.py:115)",
    "ConcatenateDatabundles: source_index_key: literal fallback 'source_index'; schema default 'source_index' (concatenate_databundles.py:150)",
    "ConcatenateDatabundles: uncertainty_key_policy: literal fallback 'require_matching'; schema default 'require_matching' (concatenate_databundles.py:211)",
    "CopyDataBundleKeys: copy: literal fallback True; schema default True (copy_databundle_keys.py:105)",
    "CopyDataBundleKeys: copy_axes: literal fallback True; schema default True (copy_databundle_keys.py:110)",
    "CopyDataBundleKeys: data_keys: literal fallback None; schema default None (copy_databundle_keys.py:79)",
    "CopyDataBundleKeys: key_map: literal fallback None; schema default None (copy_databundle_keys.py:80)",
    "DetectorEfficiencyCorrection: correction_key: literal fallback 'detector_efficiency'; schema default 'detector_efficiency' (detector_efficiency_correction.py:202)",
    "DetectorEfficiencyCorrection: correction_key: literal fallback 'detector_efficiency'; schema default 'detector_efficiency' (detector_efficiency_correction.py:216)",
    "DetectorEfficiencyCorrection: cos_alpha_key: literal fallback 'CosAlpha'; schema default 'CosAlpha' (detector_efficiency_correction.py:200)",
    "DetectorEfficiencyCorrection: cos_alpha_key: literal fallback 'CosAlpha'; schema default 'CosAlpha' (detector_efficiency_correction.py:215)",
    "DetectorEfficiencyCorrection: minimum_cos_alpha: literal fallback 1e-12; schema default 1e-12 (detector_efficiency_correction.py:213)",
    "DetectorEfficiencyCorrection: normalize_to_normal_incidence: literal fallback True; schema default True (detector_efficiency_correction.py:214)",
    "DilateMask: backend: literal fallback 'auto'; schema default 'auto' (dilate_mask.py:194)",
    "DilateMask: footprint_shape: literal fallback 'square'; schema default 'square' (dilate_mask.py:193)",
    "DilateMask: radius: literal fallback 1; schema default 1 (dilate_mask.py:117)",
    "DilateMask: source_mask_key: literal fallback 'mask'; schema default 'mask' (dilate_mask.py:176)",
    "DilateMask: source_mask_key: literal fallback 'mask'; schema default 'mask' (dilate_mask.py:190)",
    "DilateMask: target_mask_key: literal fallback 'mask'; schema default 'mask' (dilate_mask.py:177)",
    "DilateMask: target_mask_key: literal fallback 'mask'; schema default 'mask' (dilate_mask.py:191)",
    "Divide: divisor_uncertainties_sources: literal fallback {}; schema default {} (divide.py:84)",
    "Divide: divisor_units_source: literal fallback None; schema default None (divide.py:83)",
    "FindCenterOfMass1D: axis_key: literal fallback 'yaw'; schema default 'yaw' (find_center_of_mass_1d.py:120)",
    "FindCenterOfMass1D: axis_key: literal fallback 'yaw'; schema default 'yaw' (find_center_of_mass_1d.py:166)",
    "FindCenterOfMass1D: baseline: literal fallback 0.0; schema default 0.0 (find_center_of_mass_1d.py:203)",
    "FindCenterOfMass1D: convergence_tolerance: literal fallback 0.0; schema default 0.0 (find_center_of_mass_1d.py:199)",
    "FindCenterOfMass1D: diagnostic_prefix: literal fallback 'centroid'; schema default 'centroid' (find_center_of_mass_1d.py:110)",
    "FindCenterOfMass1D: maximum_iterations: literal fallback 5; schema default 5 (find_center_of_mass_1d.py:170)",
    "FindCenterOfMass1D: output_key: literal fallback 'beam_center'; schema default 'beam_center' (find_center_of_mass_1d.py:109)",
    "FindCenterOfMass1D: signal_key: literal fallback 'signal'; schema default 'signal' (find_center_of_mass_1d.py:119)",
    "FindCenterOfMass1D: signal_key: literal fallback 'signal'; schema default 'signal' (find_center_of_mass_1d.py:165)",
    "FindScaleFactor1D: background_output_key: literal fallback 'scale_background'; schema default 'scale_background' (find_scale_factor1d.py:225)",
    "FindScaleFactor1D: background_output_key: literal fallback 'scale_background'; schema default 'scale_background' (find_scale_factor1d.py:340)",
    "FindScaleFactor1D: background_uncertainty_key: literal fallback 'propagate_to_all'; schema default 'propagate_to_all' (find_scale_factor1d.py:341)",
    "FindScaleFactor1D: fit_background: literal fallback False; schema default False (find_scale_factor1d.py:224)",
    "FindScaleFactor1D: fit_background: literal fallback False; schema default False (find_scale_factor1d.py:315)",
    "FindScaleFactor1D: fit_model: literal fallback 'normal'; schema default 'normal' (find_scale_factor1d.py:277)",
    "FindScaleFactor1D: independent_axis_key: literal fallback 'Q'; schema default 'Q' (find_scale_factor1d.py:221)",
    "FindScaleFactor1D: independent_axis_key: literal fallback 'Q'; schema default 'Q' (find_scale_factor1d.py:257)",
    "FindScaleFactor1D: interpolation_kind: literal fallback 'linear'; schema default 'linear' (find_scale_factor1d.py:309)",
    "FindScaleFactor1D: require_overlap: literal fallback True; schema default True (find_scale_factor1d.py:308)",
    "FindScaleFactor1D: robust_fscale: literal fallback 1.0; schema default 1.0 (find_scale_factor1d.py:325)",
    "FindScaleFactor1D: robust_loss: literal fallback 'huber'; schema default 'huber' (find_scale_factor1d.py:324)",
    "FindScaleFactor1D: scale_output_key: literal fallback 'scale_factor'; schema default 'scale_factor' (find_scale_factor1d.py:222)",
    "FindScaleFactor1D: scale_output_key: literal fallback 'scale_factor'; schema default 'scale_factor' (find_scale_factor1d.py:328)",
    "FindScaleFactor1D: scale_uncertainty_key: literal fallback 'propagate_to_all'; schema default 'propagate_to_all' (find_scale_factor1d.py:329)",
    "FindScaleFactor1D: signal_key: literal fallback 'signal'; schema default 'signal' (find_scale_factor1d.py:220)",
    "FindScaleFactor1D: signal_key: literal fallback 'signal'; schema default 'signal' (find_scale_factor1d.py:256)",
    "FindScaleFactor1D: use_basedata_weights: literal fallback True; schema default True (find_scale_factor1d.py:312)",
    "FlatPlateSelfAbsorptionCorrection: correction_key: literal fallback 'flat_plate_self_absorption'; schema default 'flat_plate_self_absorption' (flat_plate_self_absorption_correction.py:134)",
    "FlatPlateSelfAbsorptionCorrection: correction_key: literal fallback 'flat_plate_self_absorption'; schema default 'flat_plate_self_absorption' (flat_plate_self_absorption_correction.py:99)",
    "FlatPlateSelfAbsorptionCorrection: cos_alpha_key: literal fallback 'CosAlpha'; schema default 'CosAlpha' (flat_plate_self_absorption_correction.py:133)",
    "FlatPlateSelfAbsorptionCorrection: cos_alpha_key: literal fallback 'CosAlpha'; schema default 'CosAlpha' (flat_plate_self_absorption_correction.py:98)",
    "FlatPlateSelfAbsorptionCorrection: minimum_attenuation_factor: literal fallback 1e-12; schema default 1e-12 (flat_plate_self_absorption_correction.py:130)",
    "FlatPlateSelfAbsorptionCorrection: minimum_cos_alpha: literal fallback 1e-12; schema default 1e-12 (flat_plate_self_absorption_correction.py:129)",
    "FlatPlateSelfAbsorptionCorrection: transmission_uncertainties_sources: literal fallback {}; schema default {} (flat_plate_self_absorption_correction.py:103)",
    "FlatPlateSelfAbsorptionCorrection: transmission_uncertainties_sources: literal fallback {}; schema default {} (flat_plate_self_absorption_correction.py:123)",
    "IndexedAverager: use_value_uncertainty_weights: literal fallback False; schema default False (indexed_averager.py:324)",
    "IndexedAverager: use_value_weights: literal fallback True; schema default True (indexed_averager.py:321)",
    "Integrate1D: axis_key: literal fallback 'q'; schema default 'q' (integrate_1d.py:108)",
    "Integrate1D: axis_key: literal fallback 'q'; schema default 'q' (integrate_1d.py:285)",
    "Integrate1D: duplicate_axis: literal fallback 'error'; schema default 'error' (integrate_1d.py:288)",
    "Integrate1D: method: literal fallback 'trapezoid'; schema default 'trapezoid' (integrate_1d.py:286)",
    "Integrate1D: output_key: literal fallback 'integral'; schema default 'integral' (integrate_1d.py:117)",
    "Integrate1D: output_key: literal fallback 'integral'; schema default 'integral' (integrate_1d.py:309)",
    "Integrate1D: signal_key: literal fallback 'signal'; schema default 'signal' (integrate_1d.py:107)",
    "Integrate1D: signal_key: literal fallback 'signal'; schema default 'signal' (integrate_1d.py:284)",
    "Integrate1D: sort_axis: literal fallback False; schema default False (integrate_1d.py:300)",
    "Multiply: multiplier_uncertainties_sources: literal fallback {}; schema default {} (multiply.py:81)",
    "Multiply: multiplier_units_source: literal fallback None; schema default None (multiply.py:80)",
    "Negate: data_key: literal fallback 'signal'; schema default 'signal' (negate.py:46)",
    "Negate: data_key: literal fallback 'signal'; schema default 'signal' (negate.py:51)",
    "PixelCoordinates3D: basis_fast: literal fallback (1.0, 0.0, 0.0); schema default (1.0, 0.0, 0.0) (pixel_coordinates_3d.py:305)",
    "PixelCoordinates3D: basis_normal: literal fallback (0.0, 0.0, 1.0); schema default (0.0, 0.0, 1.0) (pixel_coordinates_3d.py:307)",
    "PixelCoordinates3D: basis_slow: literal fallback (0.0, 1.0, 0.0); schema default (0.0, 1.0, 0.0) (pixel_coordinates_3d.py:306)",
    "PixelCoordinates3D: det_coord_x_uncertainties_sources: literal fallback {}; schema default {} (pixel_coordinates_3d.py:235)",
    "PixelCoordinates3D: det_coord_x_units_source: literal fallback None; schema default None (pixel_coordinates_3d.py:234)",
    "PixelCoordinates3D: det_coord_y_uncertainties_sources: literal fallback {}; schema default {} (pixel_coordinates_3d.py:235)",
    "PixelCoordinates3D: det_coord_y_units_source: literal fallback None; schema default None (pixel_coordinates_3d.py:234)",
    "PixelCoordinates3D: det_coord_z_uncertainties_sources: literal fallback {}; schema default {} (pixel_coordinates_3d.py:235)",
    "PixelCoordinates3D: det_coord_z_units_source: literal fallback None; schema default None (pixel_coordinates_3d.py:234)",
    "PixelCoordinates3D: pixel_pitch_fast_uncertainties_sources: literal fallback {}; schema default {} (pixel_coordinates_3d.py:235)",
    "PixelCoordinates3D: pixel_pitch_fast_units_source: literal fallback None; schema default None (pixel_coordinates_3d.py:234)",
    "PixelCoordinates3D: pixel_pitch_slow_uncertainties_sources: literal fallback {}; schema default {} (pixel_coordinates_3d.py:235)",
    "PixelCoordinates3D: pixel_pitch_slow_units_source: literal fallback None; schema default None (pixel_coordinates_3d.py:234)",
    "PixelCoordinates3D: with_processing_keys: literal fallback None; schema default None (pixel_coordinates_3d.py:420)",
    "PixelCoordinates3D: with_processing_keys: literal fallback None; schema default None (pixel_coordinates_3d.py:442)",
    "Plot1DVisualization: auto_log_x: literal fallback True; schema default True (plot_1d_visualization.py:116)",
    "Plot1DVisualization: auto_log_y: literal fallback True; schema default True (plot_1d_visualization.py:122)",
    "Plot2DVisualization: auto_zmax_percentile: literal fallback 99.0; schema default 99.0 (plot_2d_visualization.py:254)",
    "Plot2DVisualization: auto_zmin_percentile: literal fallback 5.0; schema default 5.0 (plot_2d_visualization.py:253)",
    "Plot2DVisualization: reverse_y: literal fallback True; schema default True (plot_2d_visualization.py:292)",
    "Plot2DVisualization: transpose: literal fallback False; schema default False (plot_2d_visualization.py:247)",
    "PolarizationCorrection: correction_key: literal fallback 'polarization_factor_map'; schema default 'polarization_factor_map' (polarization_correction.py:129)",
    "PolarizationCorrection: correction_key: literal fallback 'polarization_factor_map'; schema default 'polarization_factor_map' (polarization_correction.py:161)",
    "PolarizationCorrection: minimum_polarization_factor: literal fallback 1e-12; schema default 1e-12 (polarization_correction.py:155)",
    "PolarizationCorrection: mode: literal fallback 'linear_fraction'; schema default 'linear_fraction' (polarization_correction.py:145)",
    "PolarizationCorrection: polarization_angular_offset_units: literal fallback 'degree'; schema default 'degree' (polarization_correction.py:121)",
    "PolarizationCorrection: psi_key: literal fallback 'Psi'; schema default 'Psi' (polarization_correction.py:128)",
    "PolarizationCorrection: psi_key: literal fallback 'Psi'; schema default 'Psi' (polarization_correction.py:160)",
    "PolarizationCorrection: two_theta_key: literal fallback 'TwoTheta'; schema default 'TwoTheta' (polarization_correction.py:127)",
    "PolarizationCorrection: two_theta_key: literal fallback 'TwoTheta'; schema default 'TwoTheta' (polarization_correction.py:159)",
    "ReduceDimensionality: nan_policy: literal fallback 'omit'; schema default 'omit' (reduce_dimensionality.py:716)",
    "ReduceDimensionality: reduction: literal fallback 'mean'; schema default 'mean' (reduce_dimensionality.py:701)",
    "ReduceDimensionality: reduction: literal fallback 'mean'; schema default 'mean' (reduce_dimensionality.py:717)",
    "ReduceDimensionality: use_weights: literal fallback True; schema default True (reduce_dimensionality.py:715)",
    "ReduceMask: reduction: literal fallback 'any'; schema default 'any' (reduce_mask.py:144)",
    "ReduceMask: source_mask_key: literal fallback 'mask'; schema default 'mask' (reduce_mask.py:128)",
    "ReduceMask: source_mask_key: literal fallback 'mask'; schema default 'mask' (reduce_mask.py:142)",
    "ReduceMask: target_mask_key: literal fallback 'mask'; schema default 'mask' (reduce_mask.py:129)",
    "ReduceMask: target_mask_key: literal fallback 'mask'; schema default 'mask' (reduce_mask.py:143)",
    "SinkProcessingData: data_paths: literal fallback []; schema default [] (sink_processing_data.py:66)",
    "Subtract: subtrahend_uncertainties_sources: literal fallback {}; schema default {} (subtract.py:81)",
    "Subtract: subtrahend_units_source: literal fallback None; schema default None (subtract.py:80)",
    "SubtractInterpolated1D: axis_key: literal fallback 'Q'; schema default 'Q' (subtract_interpolated_1d.py:109)",
    "SubtractInterpolated1D: axis_key: literal fallback 'Q'; schema default 'Q' (subtract_interpolated_1d.py:163)",
    "SubtractInterpolated1D: mode: literal fallback 'nearest'; schema default 'nearest' (subtract_interpolated_1d.py:167)",
    "SubtractInterpolated1D: output_mask_key: literal fallback 'remap_mask'; schema default 'remap_mask' (subtract_interpolated_1d.py:238)",
    "SubtractInterpolated1D: output_signal_key: literal fallback 'signal'; schema default 'signal' (subtract_interpolated_1d.py:119)",
    "SubtractInterpolated1D: output_signal_key: literal fallback 'signal'; schema default 'signal' (subtract_interpolated_1d.py:164)",
    "SubtractInterpolated1D: signal_key: literal fallback 'signal'; schema default 'signal' (subtract_interpolated_1d.py:108)",
    "SubtractInterpolated1D: signal_key: literal fallback 'signal'; schema default 'signal' (subtract_interpolated_1d.py:162)",
    "ThresholdMask: mask_mode: literal fallback 'outside'; schema default 'outside' (threshold_mask.py:153)",
    "ThresholdMask: source_basedata_key: literal fallback 'signal'; schema default 'signal' (threshold_mask.py:151)",
    "ThresholdMask: target_mask_key: literal fallback 'threshold_mask'; schema default 'threshold_mask' (threshold_mask.py:150)",
    "UnitsLabelUpdate: update_pairs: literal fallback {}; schema default {} (units_label_update.py:52)",
    "XSGeometryFromPixelCoordinates: detector_normal: literal fallback (0.0, 0.0, 1.0); schema default (0.0, 0.0, 1.0) (xs_geometry_from_pixel_coordinates.py:414)",
    "XSGeometryFromPixelCoordinates: photon_uncertainties_sources: literal fallback {}; schema default {} (xs_geometry_from_pixel_coordinates.py:195)",
    "XSGeometryFromPixelCoordinates: photon_units_source: literal fallback None; schema default None (xs_geometry_from_pixel_coordinates.py:194)",
    "XSGeometryFromPixelCoordinates: pixel_pitch_fast_uncertainties_sources: literal fallback {}; schema default {} (xs_geometry_from_pixel_coordinates.py:195)",
    "XSGeometryFromPixelCoordinates: pixel_pitch_fast_units_source: literal fallback None; schema default None (xs_geometry_from_pixel_coordinates.py:194)",
    "XSGeometryFromPixelCoordinates: pixel_pitch_slow_uncertainties_sources: literal fallback {}; schema default {} (xs_geometry_from_pixel_coordinates.py:195)",
    "XSGeometryFromPixelCoordinates: pixel_pitch_slow_units_source: literal fallback None; schema default None (xs_geometry_from_pixel_coordinates.py:194)",
    "XSGeometryFromPixelCoordinates: sample_z_uncertainties_sources: literal fallback {}; schema default {} (xs_geometry_from_pixel_coordinates.py:195)",
    "XSGeometryFromPixelCoordinates: sample_z_units_source: literal fallback None; schema default None (xs_geometry_from_pixel_coordinates.py:194)",
    "XSGeometryFromPixelCoordinates: with_processing_keys: literal fallback None; schema default None (xs_geometry_from_pixel_coordinates.py:373)",
    "XSGeometryFromPixelCoordinates: with_processing_keys: literal fallback None; schema default None (xs_geometry_from_pixel_coordinates.py:448)",
}


def _value(node, bindings):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self" and node.attr == "configuration":
            return _CONFIG
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # Common spelling: cfg = self.configuration or {}.
        return _value(node.values[0], bindings)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            value = _value(part.value, bindings) if isinstance(part, ast.FormattedValue) else part.value
            if not isinstance(value, str):
                return _UNKNOWN
            parts.append(value)
        return "".join(parts)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _UNKNOWN


@lru_cache(None)
def _function_source(function):
    lines, start = inspect.getsourcelines(function)
    source = textwrap.dedent("".join(lines))
    return ast.parse(source).body[0], source.splitlines(), start


def _aliases(nodes, initial):
    bindings = dict(initial)
    # Fixed point follows alias chains, never nested mapping values.
    for _ in range(len(nodes)):
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            if _value(node.value, bindings) is not _CONFIG:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and bindings.get(target.id) is not _CONFIG:
                    bindings[target.id] = _CONFIG
                    changed = True
        if not changed:
            break
    return bindings


def _read(node, bindings):
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        if _value(node.value, bindings) is _CONFIG:
            return _value(node.slice, bindings)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        if _value(node.func.value, bindings) is _CONFIG and node.args:
            return _value(node.args[0], bindings)
    return _UNKNOWN


def _literal_fallback(node, key, schema, lines):
    if not isinstance(node, ast.Call) or len(node.args) != 2:
        return None
    try:
        fallback = ast.literal_eval(node.args[1])
    except (ValueError, TypeError):
        return None
    # Exceptions require a None schema default and an inline reason.
    comment = lines[node.lineno - 1].partition("#")[2].strip()
    if key in schema and schema[key]["default"] is None and comment.startswith("config-default: "):
        return None
    return f"literal fallback {fallback!r}; schema default {schema.get(key, {}).get('default', '<undeclared>')!r}"


def _callee(node, function, methods):
    if not isinstance(node, ast.Call):
        return None
    target = None
    if isinstance(node.func, ast.Name):
        target = function.__globals__.get(node.func.id)
    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id in {"self", "cls"}:
            target = methods.get(node.func.attr)
    if (inspect.isfunction(target) or inspect.ismethod(target)) and target.__module__.startswith("modacor.modules."):
        return target
    return None


def _passed_arguments(node, target, bindings):
    parameters = list(inspect.signature(target).parameters)
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    arguments = dict(zip(parameters, node.args))
    arguments.update({kw.arg: kw.value for kw in node.keywords if kw.arg is not None})
    passed = {}
    for name, argument in arguments.items():
        value = _value(argument, bindings)
        if value is _CONFIG or isinstance(value, str):
            passed[name] = value
    return passed


def _configuration_violations(step):
    """Return one stable diagnostic per source occurrence and resolved key."""
    schema = step.effective_config_schema()
    found = set()
    visited = set()
    methods = {
        name: getattr(step, name)
        for name in step.__dict__
        if inspect.isfunction(getattr(step, name)) or inspect.ismethod(getattr(step, name))
    }

    def scan(function, initial):
        function = inspect.unwrap(getattr(function, "__func__", function))
        context = (function, tuple(sorted((key, repr(value)) for key, value in initial.items())))
        if context in visited:
            return
        visited.add(context)
        tree, lines, start = _function_source(function)
        nodes = list(ast.walk(tree))
        bindings = _aliases(nodes, initial)
        shared = ".helpers." in function.__module__
        for node in nodes:
            key = _read(node, bindings)
            problems = []
            if isinstance(key, str):
                if key not in schema:
                    problems.append("undeclared configuration key")
                if not shared and (problem := _literal_fallback(node, key, schema, lines)):
                    problems.append(problem)
            for problem in problems:
                location = f"{Path(inspect.getfile(function)).name}:{start + node.lineno - 1}"
                found.add(f"{step.__name__}: {key}: {problem} ({location})")
            target = _callee(node, function, methods)
            if target is not None:
                passed = _passed_arguments(node, target, bindings)
                if any(value is _CONFIG for value in passed.values()) or target in methods.values():
                    scan(target, passed)

    for method in methods.values():
        scan(method, {})
    return found


def _exported_steps():
    return [
        step
        for name in modules.__all__
        if inspect.isclass(step := getattr(modules, name))
        and issubclass(step, ProcessStep)
        and "deprecated" not in step.__module__.split(".")
    ]


def test_configuration_defaults():
    actual = set().union(*(_configuration_violations(step) for step in _exported_steps()))
    unexpected = actual - ALLOWED_VIOLATIONS
    stale = ALLOWED_VIOLATIONS - actual
    if unexpected or stale:
        pytest.fail(
            "\n".join(sorted(unexpected) + [f"Stale allowlist: {item}" for item in sorted(stale)]), pytrace=False
        )


@pytest.mark.parametrize(
    "key",
    [
        "chemical_formula",
        "chemical_formula_source",
        "density",
        "density_source",
        "density_units",
        "density_units_source",
    ],
)
def test_guard_follows_shared_helper_reads(monkeypatch, key):
    step = modules.AttenuatorPlateCorrection
    reduced_schema = dict(step.effective_config_schema())
    del reduced_schema[key]

    def effective_schema(cls):
        return reduced_schema

    monkeypatch.setattr(step, "effective_config_schema", classmethod(effective_schema))
    violations = _configuration_violations(step)
    assert any(
        f"{step.__name__}: {key}: undeclared configuration key (material_attenuation.py:" in item for item in violations
    )


def test_guard_excludes_nested_dictionaries_and_unrelated_cfg_names():
    tree = ast.parse("""
def calculate(self):
    cfg = self.configuration
    alias = cfg
    nested = alias["uncertainty_estimation"]
    nested.get("method", "std")
    unrelated = {}
    unrelated.get("anything", 1)
    alias.get("reduction", "mean")
""")
    nodes = list(ast.walk(tree))
    bindings = _aliases(nodes, {})
    reads = {_read(node, bindings) for node in nodes} - {_UNKNOWN}
    assert reads == {"uncertainty_estimation", "reduction"}


@pytest.mark.parametrize(
    ("default", "comment", "allowed"),
    [
        (None, "", False),
        (None, "# config-default: None means use the computed value below.", True),
        (1, "# config-default: Invalid exception for a non-None default.", False),
    ],
)
def test_guard_requires_none_schema_default_and_reason(default, comment, allowed):
    source = f'cfg.get("key", None) {comment}'
    call = ast.parse(source).body[0].value
    problem = _literal_fallback(call, "key", {"key": {"default": default}}, [source])
    assert (problem is None) == allowed
