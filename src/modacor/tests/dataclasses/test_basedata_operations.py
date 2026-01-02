# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"

import unittest

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData


class TestBinaryOpsWithUncertainties(unittest.TestCase):
    """
    Tests binary operations (+, -, *, /) between BaseData objects,
    including units and uncertainty propagation with propagate_to_all.
    """

    def setUp(self):
        # 10x10 integer signal
        self.signal = np.arange(1, 101, dtype=float).reshape((10, 10))
        self.base = BaseData(signal=self.signal, units=ureg.Unit("count"))
        # Poisson variance (≥1 to avoid zero)
        poisson_variance = self.signal.copy()
        self.base.variances["Poisson"] = poisson_variance

        # Multiplier: 2.0 ± 0.2 s, stored as absolute uncertainty via propagate_to_all
        self.mult = BaseData(
            signal=2.0,
            units=ureg.Unit("second"),
            uncertainties={"propagate_to_all": 0.1 * 2.0},
        )

    def test_multiply_basedata(self):
        result = self.base * self.mult

        # Expected nominal values
        expected_val = (self.signal * self.base.units) * (self.mult.signal * self.mult.units)

        # Expected uncertainties using standard propagation:
        #   (σ_M / M)^2 = (σ_A / A)^2 + (σ_B / B)^2
        sigma_A = np.sqrt(self.base.variances["Poisson"])  # absolute σ_A
        A = self.signal * self.base.units
        # B = self.mult.signal * self.mult.units
        sigma_B = self.mult.uncertainties["propagate_to_all"]  # absolute σ_B
        rel_sigma_A = sigma_A / A.magnitude  # unitless
        rel_sigma_B = sigma_B / self.mult.signal  # unitless
        sigma_M = np.sqrt(rel_sigma_A**2 + rel_sigma_B**2) * expected_val.magnitude

        # Build expected BaseData
        expected = BaseData(
            signal=expected_val.magnitude,
            units=expected_val.units,
            uncertainties={"Poisson": sigma_M},
        )

        self.assertEqual(result.units, expected.units)
        np.testing.assert_allclose(result.signal, expected.signal)
        np.testing.assert_allclose(result.uncertainties["Poisson"], expected.uncertainties["Poisson"])

    def test_divide_basedata(self):
        result = self.base / self.mult

        expected_val = (self.signal * self.base.units) / (self.mult.signal * self.mult.units)

        sigma_A = np.sqrt(self.base.variances["Poisson"])
        A = self.signal * self.base.units
        # B = self.mult.signal * self.mult.units
        sigma_B = self.mult.uncertainties["propagate_to_all"]
        rel_sigma_A = sigma_A / A.magnitude
        rel_sigma_B = sigma_B / self.mult.signal
        sigma_Q = np.sqrt(rel_sigma_A**2 + rel_sigma_B**2) * expected_val.magnitude

        expected = BaseData(
            signal=expected_val.magnitude,
            units=expected_val.units,
            uncertainties={"Poisson": sigma_Q},
        )

        self.assertEqual(result.units, expected.units)
        np.testing.assert_allclose(result.signal, expected.signal)
        np.testing.assert_allclose(result.uncertainties["Poisson"], expected.uncertainties["Poisson"])

    def test_add_basedata(self):
        result = self.base + self.base

        expected_val = (self.signal * self.base.units) + (self.signal * self.base.units)
        # σ_sum^2 = σ_A^2 + σ_B^2 = 2 * variance
        sigma_sum = np.sqrt(2.0 * self.base.variances["Poisson"])

        expected = BaseData(
            signal=expected_val.magnitude,
            units=expected_val.units,
            uncertainties={"Poisson": sigma_sum},
        )

        self.assertEqual(result.units, expected.units)
        np.testing.assert_allclose(result.signal, expected.signal)
        np.testing.assert_allclose(result.uncertainties["Poisson"], expected.uncertainties["Poisson"])

    def test_subtract_basedata(self):
        result = self.base - self.base

        expected_val = (self.signal * self.base.units) - (self.signal * self.base.units)
        sigma_diff = np.sqrt(2.0 * self.base.variances["Poisson"])  # same as sum for uncorrelated

        expected = BaseData(
            signal=expected_val.magnitude,
            units=expected_val.units,
            uncertainties={"Poisson": sigma_diff},
        )

        self.assertEqual(result.units, expected.units)
        np.testing.assert_allclose(result.signal, expected.signal)
        np.testing.assert_allclose(result.uncertainties["Poisson"], expected.uncertainties["Poisson"])

    def test_nonmatching_uncertainties_transfer_and_propagate_independently(self):
        # Non-matching non-global keys: result should contain the union of keys.
        a = BaseData(signal=5.0, uncertainties={"Poisson": 0.1}, units=ureg.Unit("m"))
        b = BaseData(signal=3.0, uncertainties={"apply_to_all": 0.2}, units=ureg.Unit("s"))

        result = a * b

        self.assertIn("Poisson", result.uncertainties)
        self.assertIn("apply_to_all", result.uncertainties)

        # Mul: σR² = (B σA)² + (A σB)², but per-key independent:
        # - "Poisson" comes only from a: σ = |B| σA
        # - "apply_to_all" comes only from b: σ = |A| σB
        self.assertAlmostEqual(float(result.uncertainties["Poisson"]), 3.0 * 0.1)
        self.assertAlmostEqual(float(result.uncertainties["apply_to_all"]), 5.0 * 0.2)

    def test_propagate_to_all_fallback(self):
        # other has only propagate_to_all → used for all keys of left
        a = BaseData(signal=np.array([1.0, 2.0]), uncertainties={"u": np.array([0.1, 0.2])}, units=ureg.m)
        b = BaseData(signal=2.0, uncertainties={"propagate_to_all": 0.3}, units=ureg.dimensionless)

        result = a * b
        # simple scalar check: check first element manually
        A1, σA1 = 1.0, 0.1
        B, σB = 2.0, 0.3
        M1 = A1 * B
        σM1 = np.sqrt((σA1 / A1) ** 2 + (σB / B) ** 2) * M1
        self.assertAlmostEqual(result.signal[0], M1)
        self.assertAlmostEqual(result.uncertainties["u"][0], σM1)

    def test_both_propagate_to_all_results_in_propagate_to_all_only(self):
        a = BaseData(
            signal=np.array([2.0, 3.0]), units=ureg.m, uncertainties={"propagate_to_all": np.array([0.2, 0.3])}
        )
        b = BaseData(signal=4.0, units=ureg.s, uncertainties={"propagate_to_all": 0.4})

        result = a * b

        self.assertEqual(set(result.uncertainties.keys()), {"propagate_to_all"})

        A = a.signal
        B = 4.0
        sigma_A = np.array([0.2, 0.3])
        sigma_B = 0.4

        expected = np.sqrt((B * sigma_A) ** 2 + (A * sigma_B) ** 2)
        np.testing.assert_allclose(result.uncertainties["propagate_to_all"], expected)

    def test_nonmatching_keys_union_for_addition(self):
        a = BaseData(signal=np.array([1.0, 2.0]), units=ureg.m, uncertainties={"u": np.array([0.1, 0.2])})
        b = BaseData(signal=np.array([3.0, 4.0]), units=ureg.m, uncertainties={"v": np.array([0.3, 0.4])})

        result = a + b

        self.assertEqual(set(result.uncertainties.keys()), {"u", "v"})
        # Add: each key propagates independently; "u" comes only from a, "v" only from b
        np.testing.assert_allclose(result.uncertainties["u"], np.array([0.1, 0.2]))
        np.testing.assert_allclose(result.uncertainties["v"], np.array([0.3, 0.4]))

    def test_nonmatching_keys_union_for_division(self):
        a = BaseData(signal=np.array([2.0, 4.0]), units=ureg.m, uncertainties={"u": np.array([0.2, 0.4])})
        b = BaseData(signal=np.array([5.0, 10.0]), units=ureg.s, uncertainties={"v": np.array([0.5, 1.0])})

        result = a / b

        self.assertEqual(set(result.uncertainties.keys()), {"u", "v"})

        A = np.array([2.0, 4.0])
        B = np.array([5.0, 10.0])
        sigma_u = np.array([0.2, 0.4])
        sigma_v = np.array([0.5, 1.0])

        # Div per-key independent:
        # u: σ = σA / B
        expected_u = sigma_u / B
        # v: σ = A σB / B^2
        expected_v = (A * sigma_v) / (B**2)

        np.testing.assert_allclose(result.uncertainties["u"], expected_u)
        np.testing.assert_allclose(result.uncertainties["v"], expected_v)


class TestScalarAndQuantityCoercion(unittest.TestCase):
    """
    Tests that scalars and pint.Quantity operands are correctly coerced into BaseData
    with zero uncertainties.
    """

    def setUp(self):
        self.signal = np.array([1.0, 2.0, 3.0])
        self.base = BaseData(
            signal=self.signal,
            units=ureg.m,
            uncertainties={"u": np.array([0.1, 0.2, 0.3])},
        )

    def test_multiply_by_scalar(self):
        result = self.base * 2.0

        np.testing.assert_allclose(result.signal, self.signal * 2.0)
        # uncertainties should scale by the same factor
        np.testing.assert_allclose(result.uncertainties["u"], np.array([0.1, 0.2, 0.3]) * 2.0)
        self.assertEqual(result.units, self.base.units)

    def test_right_multiply_by_scalar(self):
        result = 2.0 * self.base
        np.testing.assert_allclose(result.signal, self.signal * 2.0)
        np.testing.assert_allclose(result.uncertainties["u"], np.array([0.1, 0.2, 0.3]) * 2.0)
        self.assertEqual(result.units, self.base.units)

    def test_add_scalar(self):
        result = self.base + 1.0
        np.testing.assert_allclose(result.signal, self.signal + 1.0)
        # uncertainties unchanged if scalar has zero uncertainty
        np.testing.assert_allclose(result.uncertainties["u"], np.array([0.1, 0.2, 0.3]))
        self.assertEqual(result.units, self.base.units)

    def test_add_pint_quantity_same_units(self):
        result = self.base + 2.0 * ureg.m
        np.testing.assert_allclose(result.signal, self.signal + 2.0)
        np.testing.assert_allclose(result.uncertainties["u"], np.array([0.1, 0.2, 0.3]))
        self.assertEqual(result.units, self.base.units)


class TestUnaryOps(unittest.TestCase):
    """
    Tests unary transformations: sqrt, square, power, log, exp, trig functions, reciprocal.
    Checks both signal and uncertainty propagation, plus domain masking behavior.
    """

    def setUp(self):
        self.signal = np.array([1.0, 4.0, 9.0])
        self.unc = np.array([0.1, 0.2, 0.3])
        self.base = BaseData(
            signal=self.signal,
            units=ureg.m,
            uncertainties={"u": self.unc},
        )

    def test_sqrt(self):
        result = self.base.sqrt()
        expected_signal = np.sqrt(self.signal)
        expected_sigma = np.abs(0.5 / np.sqrt(self.signal) * self.unc)

        np.testing.assert_allclose(result.signal, expected_signal)
        np.testing.assert_allclose(result.uncertainties["u"], expected_sigma)
        self.assertEqual(result.units, self.base.units**0.5)

    def test_square(self):
        result = self.base.square()
        expected_signal = self.signal**2
        expected_sigma = np.abs(2.0 * self.signal * self.unc)

        np.testing.assert_allclose(result.signal, expected_signal)
        np.testing.assert_allclose(result.uncertainties["u"], expected_sigma)
        self.assertEqual(result.units, self.base.units**2)

    def test_power(self):
        exponent = 3.0
        result = self.base**exponent
        expected_signal = self.signal**exponent
        expected_sigma = np.abs(exponent * self.signal ** (exponent - 1.0) * self.unc)

        np.testing.assert_allclose(result.signal, expected_signal)
        np.testing.assert_allclose(result.uncertainties["u"], expected_sigma)
        self.assertEqual(result.units, self.base.units**exponent)

    def test_log(self):
        # all positive
        base = BaseData(
            signal=np.array([1.0, 2.0, 4.0]), units=ureg.dimensionless, uncertainties={"u": np.array([0.1, 0.2, 0.4])}
        )
        result = base.log()
        expected_signal = np.log(base.signal)
        expected_sigma = np.abs((1.0 / base.signal) * base.uncertainties["u"])

        np.testing.assert_allclose(result.signal, expected_signal)
        np.testing.assert_allclose(result.uncertainties["u"], expected_sigma)
        self.assertEqual(result.units, ureg.dimensionless)

    def test_log_domain_masking(self):
        base = BaseData(
            signal=np.array([-1.0, 0.0, 1.0]),
            units=ureg.dimensionless,
            uncertainties={"u": np.array([0.1, 0.1, 0.1])},
        )
        result = base.log()

        # invalid at -1 and 0 → NaN
        self.assertTrue(np.isnan(result.signal[0]))
        self.assertTrue(np.isnan(result.signal[1]))
        self.assertTrue(np.isnan(result.uncertainties["u"][0]))
        self.assertTrue(np.isnan(result.uncertainties["u"][1]))
        # valid at 1
        self.assertFalse(np.isnan(result.signal[2]))
        self.assertFalse(np.isnan(result.uncertainties["u"][2]))

    def test_reciprocal(self):
        base = BaseData(
            signal=np.array([1.0, 2.0, 4.0]),
            units=ureg.s,
            uncertainties={"u": np.array([0.1, 0.2, 0.4])},
        )
        result = base.reciprocal()
        expected_signal = 1.0 / base.signal
        expected_sigma = np.abs(1.0 / (base.signal**2) * base.uncertainties["u"])

        np.testing.assert_allclose(result.signal, expected_signal)
        np.testing.assert_allclose(result.uncertainties["u"], expected_sigma)
        self.assertEqual(result.units, 1 / base.units)

    def test_trig_functions(self):
        # small angles in radians
        base = BaseData(
            signal=np.array([0.0, np.pi / 4]),
            units=ureg.radian,
            uncertainties={"u": np.array([0.01, 0.02])},
        )

        sin_res = base.sin()
        cos_res = base.cos()
        tan_res = base.tan()

        # sin: σ ≈ |cos(x)| σ_x
        expected_sin_sigma = np.abs(np.cos(base.signal) * base.uncertainties["u"])
        np.testing.assert_allclose(sin_res.signal, np.sin(base.signal))
        np.testing.assert_allclose(sin_res.uncertainties["u"], expected_sin_sigma)
        self.assertEqual(sin_res.units, ureg.dimensionless)

        # cos: σ ≈ |sin(x)| σ_x
        expected_cos_sigma = np.abs(np.sin(base.signal) * base.uncertainties["u"])
        np.testing.assert_allclose(cos_res.signal, np.cos(base.signal))
        np.testing.assert_allclose(cos_res.uncertainties["u"], expected_cos_sigma)
        self.assertEqual(cos_res.units, ureg.dimensionless)

        # tan: σ ≈ |1/cos^2(x)| σ_x
        expected_tan_sigma = np.abs(1.0 / (np.cos(base.signal) ** 2) * base.uncertainties["u"])
        np.testing.assert_allclose(tan_res.signal, np.tan(base.signal))
        np.testing.assert_allclose(tan_res.uncertainties["u"], expected_tan_sigma)
        self.assertEqual(tan_res.units, ureg.dimensionless)

    def test_inverse_trig_domain_masking(self):
        base = BaseData(
            signal=np.array([-1.5, -1.0, 0.0, 1.0, 1.5]),
            units=ureg.dimensionless,
            uncertainties={"u": np.array([0.1, 0.1, 0.1, 0.1, 0.1])},
        )
        asin_res = base.arcsin()
        acos_res = base.arccos()

        # |x| > 1 → NaN
        for idx in (0, 4):
            self.assertTrue(np.isnan(asin_res.signal[idx]))
            self.assertTrue(np.isnan(asin_res.uncertainties["u"][idx]))
            self.assertTrue(np.isnan(acos_res.signal[idx]))
            self.assertTrue(np.isnan(acos_res.uncertainties["u"][idx]))

        # |x| <= 1 → finite
        for idx in (1, 2, 3):
            self.assertFalse(np.isnan(asin_res.signal[idx]))
            self.assertFalse(np.isnan(asin_res.uncertainties["u"][idx]))
            self.assertFalse(np.isnan(acos_res.signal[idx]))
            self.assertFalse(np.isnan(acos_res.uncertainties["u"][idx]))


class TestNegationAndCopySafety(unittest.TestCase):
    """
    Tests that negation preserves uncertainty magnitudes and that uncertainties
    are deep-copied (no aliasing between original and result).
    """

    def test_negation_copies_uncertainties(self):
        base = BaseData(
            signal=np.array([1.0, 2.0]),
            units=ureg.m,
            uncertainties={"u": np.array([0.1, 0.2])},
        )
        neg = -base

        # values & units
        np.testing.assert_allclose(neg.signal, -base.signal)
        self.assertEqual(neg.units, base.units)
        np.testing.assert_allclose(neg.uncertainties["u"], base.uncertainties["u"])

        # modify neg; base must not change
        neg.uncertainties["u"][0] = 999.0
        self.assertNotEqual(neg.uncertainties["u"][0], base.uncertainties["u"][0])


class TestBroadcastValidation(unittest.TestCase):
    """
    Tests that validate_broadcast is effectively enforced for uncertainties.
    """

    def test_invalid_uncertainty_shape_raises(self):
        signal = np.zeros((2, 2))
        # Shape (3,) cannot broadcast to (2,2)
        with self.assertRaises(ValueError):
            BaseData(
                signal=signal,
                units=ureg.count,
                uncertainties={"u": np.array([1.0, 2.0, 3.0])},
            )

    def test_scalar_uncertainty_broadcasts(self):
        signal = np.zeros((2, 2))
        bd = BaseData(
            signal=signal,
            units=ureg.count,
            uncertainties={"u": 0.1},
        )
        self.assertEqual(bd.uncertainties["u"].shape, ())
        # and unary op should broadcast this fine
        res = bd.square()
        self.assertEqual(res.uncertainties["u"].shape, signal.shape)
