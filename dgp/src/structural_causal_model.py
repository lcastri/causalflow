"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import numpy
import pandas

from dgp.src.data_generation_configs import FunctionConfig, NoiseConfig
from dgp.src.data_generation_exceptions import DataGenerationException
from dgp.src.structural_equations import monotonic_function, piecewise_linear_function, trigonometric_function
from dgp.src.time_series_causal_graph import TimeSeriesCausalGraph

CallableOrPartial = Union[Callable[[float], float], partial]


_DEFAULT_FUNCTION_CONFIG_VALUES = {
        0: {
            'functions': ['linear'],
            'prob_functions': [1.0]
        },
        10: {
            'functions': ['linear', 'monotonic'],
            'prob_functions': [0.7, 0.3]
        },
        20: {
            'functions': ['linear', 'piecewise_linear', 'monotonic'],
            'prob_functions': [0.5, 0.2, 0.3]
        },
        30: {
            'functions': ['linear', 'piecewise_linear', 'monotonic', 'trigonometric'],
            'prob_functions': [0.2, 0.3, 0.3, 0.2]
        }
}


class StructuralCausalModel:
    """
    This class generates a structural causal model (SCM), in particular a Causal Additive Model (CAM), from the
    provided TimeSeriesCausalGraph object and FunctionConfig object.
    """

    def __init__(self,
                 function_config: FunctionConfig,
                 causal_graph: TimeSeriesCausalGraph,
                 rng: numpy.random.Generator):
        """
        A structural causal model (SCM) capturing the functional dependencies on each edge of the provided time series
        causal graph.

        :param function_config: The FunctionConfig object used to parameterise the process to randomly generate
            the functional dependencies on each edge of the time series causal graph.
        :param causal_graph: A time series causal graph used as the basis for the SCM.
        :param rng: A random number Generator.
        """

        self.function_config = deepcopy(function_config)
        self._complete_function_config()

        self.rng = rng
        self.causal_graph = causal_graph
        self.structural_equations = self._generate_structural_equations()  # Dictionary whose keys are nodes at time t.

    def generate_dataset(self, num_samples: int,
                         noise_config: NoiseConfig,
                         return_observed_data_only: bool = True,
                         normalize_data: bool = False,
                         percent_missing: float = 0.0) -> pandas.DataFrame:
        """
        This method generates a data set from the structural causal model using the specified noise configuration.

        :param num_samples: The number of samples to be generated. Must be provided as a positive integer.
        :param noise_config: A complete NoiseConfig object.
        :param return_observed_data_only: Boolean to control if latent and noise variables are also returned.
        :param normalize_data: Boolean to control if data is normalised to zero mean and unit variance.
        :param percent_missing: A float to set the percentage of missing data (i.e., NaNs) in the final data set.
        :return: A pandas DataFrame with the generated data.
        """

        # Validate input.
        assert isinstance(num_samples, int) and num_samples > 0, \
            f'num_samples must be a positive integer. Got {num_samples}.'
        # No need to validate the other parameters as they are validated/completed by the TimeSeriesGenerator,
        # which calls this method.

        # Increment number of samples by twice max lag as we will not return the first 2 * max_lag samples.
        num_samples += 2 * self.causal_graph.graph_config.max_lag

        # Initialise data for non-noise nodes with zeroes.
        data = dict.fromkeys(self.structural_equations.keys())
        data.update({k: numpy.zeros(num_samples, dtype=numpy.float_) for k in
                     self.causal_graph.get_unique_variables(include_noise=False)})

        # Sample IID noise.
        data.update({k: self._sample_noise(noise_config=noise_config, num_samples=num_samples) for k in
                     self.causal_graph.get_unique_noise_variables()})

        # Sanity check.
        assert all(v is not None for v in data.values()), 'Data for some nodes has not been initialised.'

        # Iterate through noise nodes first as they may have autoregressive component.
        for t in range(1, num_samples):
            for node in self.causal_graph.get_unique_noise_variables():
                # Add any autoregressive component.
                for parent, func in self.structural_equations[node].items():
                    # Convert causal graph node to specific variable and its lag.
                    var, lag = self.causal_graph.get_var_and_lag(parent)
                    data[node][t] += func(data[var][t-lag])

        # Get topological order of nodes at time t to ensure we set each value in order.
        # This is essential when instantaneous links are allowed.
        topological_order = self.causal_graph.get_topological_order_of_t_nodes()
        var_topological_order = [node.split('_')[0] for node in topological_order]

        # Iterate over time.
        for t in range(self.causal_graph.graph_config.max_lag, num_samples):
            for node in var_topological_order:
                for parent, func in self.structural_equations[node].items():
                    # Convert causal graph node to specific variable and its lag.
                    var, lag = self.causal_graph.get_var_and_lag(parent)
                    data[node][t] += func(data[var][t - lag])

        # Create DataFrame from data dict.
        if return_observed_data_only:
            df = pandas.DataFrame.from_dict({k: v for k, v in data.items() if k.startswith('X') or k.startswith('Y')},
                                            dtype=numpy.float_)
        else:
            df = pandas.DataFrame.from_dict(data, dtype=numpy.float_)

        # Remove first 2 * max_lag samples as buffer was added to beginning.
        df = df.iloc[2 * self.causal_graph.graph_config.max_lag:]

        # Confirm process has produced finite values and no series exploded to infinity.
        if not numpy.all(numpy.isfinite(df)):
            # Raise DataGenerationException. This is handled correctly in the TimeSeriesGenerator so the entire
            # data generating process is not killed in case the user is generating multiple data sets.
            raise DataGenerationException('Data generating process produced non-finite data!')

        if percent_missing > 0.0:
            # Replace (approximately) the specified percentage of values with NaNs.
            df.mask(self.rng.random(df.shape) < percent_missing, other=numpy.nan, inplace=True)

        if normalize_data:
            # Normalise each column to zero mean and unit variance.
            df = df.apply(lambda column: (column - column.mean(skipna=True)) / column.std(skipna=True, ddof=1), axis=0)

        # Return DataFrame.
        return df

    def _generate_structural_equations(self) -> Dict[str, Dict[str, CallableOrPartial]]:
        """
        This method builds the structural equations on each edge of the given time series causal graph, the completed
        FunctionConfig object, and the random number Generator.

        :return: Returns a dictionary whose keys are each node at time t and whose values is another dictionary
            capturing the parent nodes and the corresponding functional dependencies between the parent and child nodes.
        """

        # Define keys for dictionary of structural equations as nodes at time t.
        nodes = self.causal_graph.get_unique_variables(include_noise=True)
        structural_equations = dict.fromkeys(nodes)

        # Define ranges for weights to ensure values do not explode over time.
        initial_weight_ranges = [(-0.99, -0.7), (0.7, 0.99)]

        # Iterate through each node and generate a function to define it.
        for node in nodes:
            parents = self.causal_graph.get_parents(node + '_t')

            # Set weight ranges based on number of parents. Therefore sum of weights is within the initial range.
            num_parents = len(parents)
            if num_parents > 1:
                weight_ranges = [
                    (low / num_parents, high / num_parents) for low, high in initial_weight_ranges
                ]
            else:
                weight_ranges = initial_weight_ranges

            # Define empty dictionary for each function of each parent.
            structural_equations[node] = dict.fromkeys(parents)

            # Iterate through each parent and determine functional relationship to node.
            for parent in parents:
                if self.causal_graph.is_noise_node(node):
                    # Note: Currently only supports linear autoregressive noise component.
                    weight = self._sample_weights(ranges=weight_ranges)
                    structural_equations[node][parent] = (lambda x, w=weight: w * x)

                elif not self.causal_graph.is_noise_node(parent):
                    # Sample function based on config.
                    function_type = self.rng.choice(a=self.function_config.functions,
                                                    p=self.function_config.prob_functions)

                    # Sample parameters depending on function type.
                    if function_type == 'linear':
                        weight = self._sample_weights(ranges=weight_ranges)
                        bias = self.rng.uniform(low=-0.1, high=0.1)
                        structural_equations[node][parent] = (lambda x, w=weight, b=bias: w * x + b)
                    elif function_type == 'piecewise_linear':
                        # Sample number of knots used to define piecewise linear function.
                        num_knots = self.rng.choice(10) + 1
                        out_weights = self._sample_weights(ranges=weight_ranges, num_samples=num_knots)
                        in_weights = self._sample_weights(ranges=weight_ranges, num_samples=num_knots)
                        offsets = self.rng.uniform(low=-5.0, high=5.0, size=num_knots)
                        bias = self.rng.uniform(low=-0.1, high=0.1)
                        structural_equations[node][parent] = partial(
                            piecewise_linear_function, output_weights=out_weights, input_weights=in_weights,
                            offsets=offsets, bias=bias
                        )
                    elif function_type == 'monotonic':
                        # Sample number of sigmoids used to define monotonic function.
                        num_sigmoids = self.rng.choice(10) + 1
                        # Only need positive weights so just look at positive side of weight ranges.
                        out_weights = self.rng.uniform(*weight_ranges[1], size=num_sigmoids)
                        # These weights are inside sigmoid so can be larger values as sigmoid is bounded between 0, 1.
                        in_weights = self.rng.uniform(low=0.5, high=10.0, size=num_sigmoids)
                        offsets = self.rng.uniform(low=-5.0, high=5.0, size=num_sigmoids)
                        bias = self.rng.uniform(low=-0.1, high=0.1)
                        is_decreasing = self.rng.random() < 0.5
                        structural_equations[node][parent] = partial(
                            monotonic_function, output_weights=out_weights, input_weights=in_weights, offsets=offsets,
                            bias=bias, is_decreasing=is_decreasing
                        )
                    elif function_type == 'trigonometric':
                        # Sample parameters for function.
                        weight = self._sample_weights(ranges=weight_ranges)
                        freq = self.rng.uniform(low=0.5, high=5.0)
                        phase = self.rng.uniform(low=-numpy.pi, high=numpy.pi)
                        bias = self.rng.uniform(low=-0.1, high=0.1)
                        structural_equations[node][parent] = partial(
                            trigonometric_function, amplitude=weight, frequency=freq, phase=phase, bias=bias
                        )
                    else:
                        raise ValueError(f'Chosen function type does not match any expected type. Got {function_type}.')

                else:
                    # Use identity function for noise as we simply add true noise value.
                    structural_equations[node][parent] = lambda x: x

        return structural_equations

    def _sample_weights(self, ranges: List[Tuple[float, float]], num_samples: int = 1) -> numpy.ndarray:
        """
        This helper method samples weights from multiple uniform ranges. It first chooses which range to use, then
        samples from a uniform distribution with the chosen range.

        :param ranges: A list of ranges (as a tuple of floats) to choose from.
        :param num_samples: The number of samples to make. If more than 1, it will randomly choose the range to use for
            each sample. Therefore, not all the samples will necessarily be drawn from the same range.
        :return: A numpy array with the sampled weights.
        """

        if num_samples == 1:
            return self.rng.uniform(*self.rng.choice(ranges))
        else:
            return numpy.array([self.rng.uniform(*self.rng.choice(ranges)) for _ in range(num_samples)])

    def _sample_noise(self, noise_config: NoiseConfig, num_samples: int) -> numpy.ndarray:
        """
        This helper method samples IID data using the specified noise configuration.

        :param noise_config: A complete NoiseConfig object.
        :param num_samples: The number of samples to be generated. Must be provided as a positive integer.
        :return: A number array with the sampled noise.
        """

        # Sample variance from uniform distribution if range is provided.
        if isinstance(noise_config.noise_variance, list):
            noise_var = self.rng.uniform(*noise_config.noise_variance)
        else:
            noise_var = noise_config.noise_variance

        # Sample noise distribution.
        noise_dist = self.rng.choice(a=noise_config.distributions, p=noise_config.prob_distributions)

        # Sample noise based on the chosen distribution.
        if noise_dist == 'gaussian':
            # Calculate scale multiplier.
            scale_multiplier = numpy.sqrt(noise_var)

            # Sample values and return.
            return scale_multiplier * self.rng.standard_normal(size=num_samples, dtype=numpy.float_)
        elif noise_dist == 'laplace':
            # Variance of Laplace distribution is 2(b ** 2), where b is the scale parameter.
            scale = numpy.sqrt(0.5 * noise_var)

            # Sample values and return.
            return self.rng.laplace(loc=0.0, scale=scale, size=num_samples)
        elif noise_dist == 'students_t':
            # Sample degrees of freedom for Student's t distribution based on complexity setting for noise.
            lower_bounds = {0: 10.0, 10: 5.0, 20: 3.0, 30: 2.1}
            dof = self.rng.uniform(lower_bounds[noise_config.noise_complexity], 22.0)

            # Calculate scale multiplier. Variance is defined as dof / (dof - 2). Therefore, must scale samples
            # accordingly to produce to specified noise variance.
            scale_multiplier = numpy.sqrt(noise_var * (dof - 2.0) / dof)

            # Sample values and return.
            return scale_multiplier * self.rng.standard_t(dof, size=num_samples)
        elif noise_dist == 'uniform':
            # Variance of uniform distribution is defined as ((b - a) ** 2) / 12. We want to have symmetric noise so
            # we can explicitly solve for the upper and lower bounds of the uniform range.
            bound = numpy.sqrt(3.0 * noise_var)

            # Sample values and return.
            return self.rng.uniform(-bound, bound, size=num_samples)
        else:
            raise ValueError(f'Chosen noise distribution does not match any expected type. Got {noise_dist}.')

    def _complete_function_config(self):
        """
        This method updates the function configuration with default values for any unspecified parameters based off
        the specified complexity parameter value.
        """

        # Validate complexity and set default values if parameters are not provided.
        assert self.function_config.function_complexity in [0, 10, 20, 30], \
            f'function_complexity must be one of 0, 10, 20, or 30. Got {self.function_config.function_complexity}.'

        complexity = self.function_config.function_complexity

        #  Set default values if parameters are not provided.
        if self.function_config.functions is None:
            self.function_config.functions = _DEFAULT_FUNCTION_CONFIG_VALUES[complexity]['functions']

        if self.function_config.prob_functions is None:
            self.function_config.prob_functions = _DEFAULT_FUNCTION_CONFIG_VALUES[complexity]['prob_functions']

        # Confirm list of functions and their corresponding likelihoods are the same length.
        assert len(self.function_config.prob_functions) == len(self.function_config.functions), (
            'The length of the list of function types and the length of the list of probabilities of each '
            'function type must be the same.')
