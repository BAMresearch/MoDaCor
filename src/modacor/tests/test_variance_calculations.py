"""
Test suite for variance propagation functions in modacor.math.variance_calculations.

This file verifies the correctness of add, subtract, multiply, and divide operations
with variance propagation by comparing against reference results from the
`uncertainties` package. Randomly generated arrays simulate real-world numeric data
with associated variances.
"""

import numpy as np
from uncertainties.unumpy import nominal_values, std_devs, uarray

import modacor.math.variance_calculations as varc

samples = 1000


def generate_samples(size, low=1, high=1.0e9):
    return np.random.uniform(low, high, size)


def generate_error(values, add_zero_errors=False):
    s = np.sqrt(values)
    if add_zero_errors:  # to check if scalars will work
        s[-10:] = 0
    return s


def test_add():
    x = generate_samples(samples)
    dx = generate_error(x)
    y = generate_samples(samples)
    dy = generate_error(y, add_zero_errors=True)

    u_x = uarray(x, dx)
    u_y = uarray(y, dy)
    expected = u_x + u_y

    result, variances = varc.add(x, dx**2, y, dy**2)

    assert np.allclose(result, nominal_values(expected))
    assert np.allclose(variances, std_devs(expected) ** 2)


def test_subtract():
    x = generate_samples(samples)
    dx = generate_error(x)
    y = generate_samples(samples)
    dy = generate_error(y, add_zero_errors=True)

    u_x = uarray(x, dx)
    u_y = uarray(y, dy)
    expected = u_x - u_y

    result, variances = varc.subtract(x, dx**2, y, dy**2)

    assert np.allclose(result, nominal_values(expected))
    assert np.allclose(variances, std_devs(expected) ** 2)


def test_multiply():
    x = generate_samples(samples)
    dx = generate_error(x)
    y = generate_samples(samples)
    dy = generate_error(y, add_zero_errors=True)

    u_x = uarray(x, dx)
    u_y = uarray(y, dy)
    expected = u_x * u_y

    result, variances = varc.multiply(x, dx**2, y, dy**2)

    assert np.allclose(result, nominal_values(expected))
    assert np.allclose(variances, std_devs(expected) ** 2)


def test_divide():
    x = generate_samples(samples)
    dx = generate_error(x)
    y = generate_samples(samples, low=1e3)  # avoid divide by very small
    dy = generate_error(y, add_zero_errors=True)

    u_x = uarray(x, dx)
    u_y = uarray(y, dy)
    expected = u_x / u_y

    result, variances = varc.divide(x, dx**2, y, dy**2)

    assert np.allclose(result, nominal_values(expected))
    assert np.allclose(variances, std_devs(expected) ** 2)
