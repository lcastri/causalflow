"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

import unittest

import numpy

from dgp.src.structural_equations import monotonic_function, piecewise_linear_function, trigonometric_function


class TestStructuralEquations(unittest.TestCase):
    def setUp(self):
        # Instantiate Random Generator.
        self.rng = numpy.random.default_rng(seed=1)

        # Generate data.
        self.num_samples = 100
        self.x = numpy.linspace(-10, 10, self.num_samples)

    def test_monotonic(self):
        # Generate random parameters.
        num_sigmoids = 5
        weight_ranges = [(-0.99, -0.7), (0.7, 0.99)]
        out_weights = self.rng.uniform(*weight_ranges[1], size=num_sigmoids)
        in_weights = self.rng.uniform(low=0.5, high=10.0, size=num_sigmoids)
        offsets = self.rng.uniform(low=-5.0, high=5.0, size=num_sigmoids)
        bias = self.rng.uniform(low=-0.1, high=0.1)
        is_decreasing = False

        # Calculate function output.
        y = numpy.full_like(self.x, fill_value=numpy.nan)
        for i in range(self.num_samples):
            y[i] = monotonic_function(self.x[i], out_weights, in_weights, offsets, bias, is_decreasing)

        # Confirm all finite and monotonically increasing.
        self.assertTrue(numpy.all(numpy.isfinite(y)))
        self.assertTrue(numpy.all(numpy.diff(y) >= 0.0))

        # Check monotonically decreasing.
        is_decreasing = True
        y = numpy.full_like(self.x, fill_value=numpy.nan)
        for i in range(self.num_samples):
            y[i] = monotonic_function(self.x[i], out_weights, in_weights, offsets, bias, is_decreasing)
        self.assertTrue(numpy.all(numpy.isfinite(y)))
        self.assertTrue(numpy.all(numpy.diff(y) <= 0.0))

    def test_piecewise_linear(self):
        # Naive calculation for 3 piecewise linear terms in all in_weights set to 1.
        num_knots = 3
        offsets = numpy.array([-5.0, 0.0, 5.0])
        weight_ranges = [(-0.99, -0.7), (0.7, 0.99)]
        out_weights = numpy.array([self.rng.uniform(*self.rng.choice(weight_ranges)) for _ in range(num_knots)])
        in_weights = numpy.ones(num_knots)
        bias = 0.5

        y_naive = bias * numpy.ones_like(self.x)
        for i in range(num_knots):
            mask = self.x >= offsets[i]
            y_naive[mask] += out_weights[i] * (self.x[mask] - offsets[i])

        # Calculate function output.
        y = numpy.full_like(self.x, fill_value=numpy.nan)
        for i in range(self.num_samples):
            y[i] = piecewise_linear_function(self.x[i], out_weights, in_weights, offsets, bias)

        # Confirm all finite.
        self.assertTrue(numpy.all(numpy.isfinite(y)))

        # Confirm both calculations are close.
        numpy.testing.assert_allclose(y_naive, y)

    def test_trigonometric(self):
        # Naive calculation with amplitude = frequency = 1 and phase = 0.
        bias = 0.5
        y_naive = numpy.sin(2.0 * numpy.pi * self.x) + bias

        # Calculate function output.
        y = numpy.full_like(self.x, fill_value=numpy.nan)
        for i in range(self.num_samples):
            y[i] = trigonometric_function(self.x[i], 1.0, 1.0, 0.0, bias)

        # Confirm all finite.
        self.assertTrue(numpy.all(numpy.isfinite(y)))

        # Confirm both calculations are close.
        numpy.testing.assert_allclose(y_naive, y)


if __name__ == '__main__':
    unittest.main()
