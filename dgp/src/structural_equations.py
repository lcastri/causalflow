"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

import numpy
from scipy.special import expit


def monotonic_function(x: float, output_weights: numpy.ndarray, input_weights: numpy.ndarray, offsets: numpy.ndarray,
                       bias: float = 0.0, is_decreasing: bool = False) -> float:
    """
    This function implements a monotonic function using the provided parameters.

    .. math::

            f(x) = \\text{bias} \\pm \\sum_{i=1}^{n} |wo_i| * \\sigma(|wi_i| * (x - o_i))

        where :math:`wi_i, wo_i` are the weights, :math:`o_i` are the offsets, and :math:`n` are the number of sigmoids.

    :param x: The input value.
    :param output_weights: The output weights as a numpy array with num_sigmoids elements.
    :param input_weights: The input weights as a numpy array with num_sigmoids elements.
    :param offsets: The offsets as a numpy array with num_sigmoids elements.
    :param bias: The bias term.
    :param is_decreasing: A boolean to determine if the function is monotonic increasing or decreasing. This sets the
        sign in front of the summation.
    :return: The output of the monotonic function.
    """

    # Both sets of weights and the offsets must be same size. For speed as this is called often, do not validate input
    # to save time.

    sign = -1.0 if is_decreasing else 1.0

    return sign * numpy.sum(numpy.abs(output_weights) * expit(numpy.abs(input_weights) * (x - offsets))) + bias


def piecewise_linear_function(x: float, output_weights: numpy.ndarray, input_weights: numpy.ndarray,
                              offsets: numpy.ndarray, bias: float = 0.0) -> float:
    """
    This function implements a piecewise linear function using the provided parameters.

    .. math::

            f(x) = \\sum_{i=1}^{n} wo_i * \\max(0, wi_i * (x - o_i)) + \\text{bias}

        where :math:`wi_i, wo_i` are the weights, :math:`o_i` are the offsets, and :math:`n` are the number of knots.

    :param x: The input value.
    :param output_weights: The output weights as a numpy array with num_knots elements.
    :param input_weights: The input weights as a numpy array with num_knots elements.
    :param offsets: The offsets as a numpy array with num_knots elements.
    :param bias: The bias term.
    :return: The output of the piecewise linear function.
    """

    # Both sets of weights and the offsets must be same size. For speed as this is called often, do not validate input
    # to save time.

    return numpy.sum(output_weights * numpy.maximum(0.0, input_weights * (x - offsets))) + bias


def trigonometric_function(x: float, amplitude: float, frequency: float, phase: float, bias: float = 0.0) -> float:
    """
    This function implements a trigonometric function using the provided parameters.

    .. math:
        f(x) = \\text{amplitude} * \\sin(2 * \\pi * \\text{freq} * x + \\text{phase}) + \\text{bias}

    :param x: The input value.
    :param amplitude: The amplitude of the sinusoid.
    :param frequency: The frequency of the sinusoid.
    :param phase: The phase of the sinusoid.
    :param bias: The bias term.
    :return: The output of the trigonometric function.
    """

    # As this is called often, for speed, do not validate inputs.

    return amplitude * numpy.sin(2.0 * numpy.pi * frequency * x + phase) + bias
