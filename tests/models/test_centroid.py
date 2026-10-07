from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from modacor.models.centroid import intensity_centroid_1d


def test_intensity_centroid_clips_baseline_subtracted_negative_weights():
    result = intensity_centroid_1d(
        np.array([-1.0, 0.0, 1.0, 2.0]),
        np.array([2.0, 3.0, 2.0, 0.0]),
        baseline=1.0,
    )

    assert_allclose(result.center, 0.0)
    assert_allclose(result.intensity_sum, 4.0)
    assert result.contributor_count == 3
    assert_array_equal(result.contributors, [True, True, True, False])
    assert_allclose(result.signal_sensitivity, [-0.25, 0.0, 0.25, 0.0])
    assert_allclose(result.axis_sensitivity, [0.25, 0.5, 0.25, 0.0])
