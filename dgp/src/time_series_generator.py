"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

import logging
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy
import pandas

from dgp.src.data_generation_configs import DataGenerationConfig, NoiseConfig, RuntimeConfig
from dgp.src.data_generation_exceptions import DataGenerationException
from dgp.src.structural_causal_model import StructuralCausalModel
from dgp.src.time_series_causal_graph import TimeSeriesCausalGraph

logger = logging.getLogger(__name__)


_DEFAULT_NOISE_CONFIG_VALUES = {
        0: {
            'noise_variance': 0.01,
            'distributions': ['gaussian'],
            'prob_distributions': [1.0],
        },
        10: {
            'noise_variance': 0.1,
            'distributions': ['gaussian'],
            'prob_distributions': [1.0],
        },
        20: {
            'noise_variance': 0.1,
            'distributions': ['gaussian', 'uniform'],
            'prob_distributions': [0.75, 0.25],
        },
        30: {
            'noise_variance': 0.1,
            'distributions': ['gaussian', 'laplace', 'students_t', 'uniform'],
            'prob_distributions': [0.4, 0.2, 0.2, 0.2],
        }
    }


_DEFAULT_RUNTIME_CONFIG_VALUES = {
    'num_samples': [100],
    'data_generating_seed': [42]
}


class TimeSeriesGenerator:
    """
    The top-level class for the structural causal model (SCM)-based synthetic data generating process.
    """

    def __init__(self, config: DataGenerationConfig):
        """
        The top-level generator for building a structural causal model and sampling data from it.

        :param config: The DataGenerationConfig object used to parameterise the entire process.
        """

        # Set initial config.
        self.initial_config = deepcopy(config)
        self.noise_config = deepcopy(config.noise_config)
        self.runtime_config = deepcopy(config.runtime_config)

        # Validate configs and fill in defaults if necessary.
        self._complete_noise_config()
        self._complete_runtime_config()

        # Validate input.
        assert isinstance(self.initial_config.random_seed, int) and self.initial_config.random_seed >= 0, \
            f'The random seed must be provided as a non-negative integer. Got {self.initial_config.random_seed}.'

        # Instantiate Random Generator.
        self._rng = numpy.random.default_rng(seed=self.initial_config.random_seed)

        # Randomly generate a time series causal graph.
        # Then, given the graph, randomly generate a structural causal model.
        self.causal_graph = TimeSeriesCausalGraph(graph_config=config.causal_graph_config, rng=self._rng)
        self.structural_causal_model = StructuralCausalModel(function_config=config.function_config,
                                                             causal_graph=self.causal_graph,
                                                             rng=self._rng)

    def get_full_config(self) -> DataGenerationConfig:
        """ Returns the completed DataGenerationConfig to see final values for any initial unspecified ones. """
        full_config = DataGenerationConfig(random_seed=self.initial_config.random_seed,
                                           complexity=self.initial_config.complexity,
                                           percent_missing=self.initial_config.percent_missing,
                                           causal_graph_config=self.causal_graph.graph_config,
                                           function_config=self.structural_causal_model.function_config,
                                           noise_config=self.noise_config,
                                           runtime_config=self.runtime_config)

        return full_config

    def generate_datasets(self, noise_config: Optional[NoiseConfig] = None,
                          runtime_config: Optional[RuntimeConfig] = None) -> Tuple[List[pandas.DataFrame],
                                                                                         TimeSeriesCausalGraph]:
        """
        This method generates data set(s) from the structural causal model. The initial NoiseConfig and/or RuntimeConfig
        can be overridden to change the underlying structure/distributions of the return data, but the structural
        causal model and its corresponding time series causal graph remain the same.

        :param noise_config: An optional, complete NoiseConfig object.
        :param runtime_config: An optional, complete RuntimeConfig object.
        :return: A tuple whose first element is a list of pandas DataFrames each capturing a single data set generated
            from the SCM. For example, if runtime_config.num_samples = [100, 200], then the first DataFrame will have
            100 samples while the second will have 200. The second element of the tuple is the TimeSeriesCausalGraph.
            Therefore, the ground truth causal structure is known for the synthetic data.
        """

        ret_val = ([], self.causal_graph)

        # Use provided noise config or original one if none is provided.
        if noise_config is None:
            noise_config = self.noise_config
        else:
            # Validate noise_config. Do not complete with defaults as user should provide complete configuration
            # if they are looking to override the initial one.
            assert (noise_config.noise_variance is not None and
                    noise_config.distributions is not None and
                    noise_config.prob_distributions is not None), (
                f'A valid and fully specified NoiseConfig should be provided when overriding the original. '
                f'Got {noise_config.noise_variance} for noise variance, '
                f'{noise_config.distributions} for probability distributions, and '
                f'{noise_config.prob_distributions} for likelihoods of each probability distribution.')

        # Use provided runtime config or original one if none is provided.
        if runtime_config is None:
            runtime_config = self.runtime_config
        else:
            # Validate runtime_config. Do not complete with defaults as user should provide complete configuration
            # if they are looking to override the initial one.
            assert runtime_config.num_samples is not None and runtime_config.data_generating_seed is not None, (
                f'A valid and fully specified RuntimeConfig should be provided when overriding the original. '
                f'Got {runtime_config.num_samples} for number of samples and {runtime_config.data_generating_seed} '
                f'for data generating seed(s).')

            if isinstance(runtime_config.num_samples, int):
                runtime_config.num_samples = [runtime_config.num_samples]
            if isinstance(runtime_config.data_generating_seed, int):
                runtime_config.data_generating_seed = [runtime_config.data_generating_seed]

        # Iterate over number of samples and seeds in RuntimeConfig.
        for idx in range(max(len(runtime_config.num_samples), len(runtime_config.data_generating_seed))):
            num_samples = runtime_config.num_samples[idx if len(runtime_config.num_samples) > 1 else 0]
            seed_val = runtime_config.data_generating_seed[idx if len(runtime_config.data_generating_seed) > 1 else 0]

            # Reset random generator in SCM.
            self.structural_causal_model.rng = numpy.random.default_rng(seed=seed_val)

            # Generate data for current config and append to list.
            try:
                ret_val[0].append(
                    self.structural_causal_model.generate_dataset(
                        num_samples=num_samples,
                        noise_config=noise_config,
                        return_observed_data_only=runtime_config.return_observed_data_only,
                        normalize_data=runtime_config.normalize_data,
                        percent_missing=self.initial_config.percent_missing
                    )
                )
            except DataGenerationException as e:
                logger.warning('Data generating process failed at index %d: %s. '
                               'Ignoring this configuration and continuing.', idx, str(e))

        return ret_val

    def _complete_noise_config(self):
        """
        This method updates the noise configuration with default values for any unspecified parameters based off
        the specified complexity parameter value.
        """

        # Validate complexity and set default values if parameters are not provided.
        assert self.noise_config.noise_complexity in [0, 10, 20, 30], \
            f'noise_complexity must be one of 0, 10, 20, or 30. Got {self.noise_config.noise_complexity}.'

        complexity = self.noise_config.noise_complexity

        # Set default values if parameters are not provided.
        if self.noise_config.noise_variance is None:
            self.noise_config.noise_variance = _DEFAULT_NOISE_CONFIG_VALUES[complexity]['noise_variance']

        if self.noise_config.distributions is None:
            self.noise_config.distributions = _DEFAULT_NOISE_CONFIG_VALUES[complexity]['distributions']

        if self.noise_config.prob_distributions is None:
            self.noise_config.prob_distributions = _DEFAULT_NOISE_CONFIG_VALUES[complexity]['prob_distributions']

        # Confirm lists of distributions and their corresponding probabilities are the same length.
        assert len(self.noise_config.prob_distributions) == len(self.noise_config.distributions), (
            'The length of the list of probability distributions and the length of the list of probabilities of '
            'each probability distribution must be the same.')

    def _complete_runtime_config(self):
        """
        This method updates the runtime configuration with default values for any unspecified parameters.
        """

        if self.runtime_config.num_samples is None:
            self.runtime_config.num_samples = _DEFAULT_RUNTIME_CONFIG_VALUES['num_samples']

        if self.runtime_config.data_generating_seed is None:
            self.runtime_config.data_generating_seed = _DEFAULT_RUNTIME_CONFIG_VALUES['data_generating_seed']

        # Confirm lists of number of samples and data generating seeds are compatible.
        if len(self.runtime_config.num_samples) > 1 and len(self.runtime_config.data_generating_seed) > 1:
            assert len(self.runtime_config.num_samples) == len(self.runtime_config.data_generating_seed), (
                'The length of the list of number of samples and the length of the list of data generating random '
                'seeds must be the same if both lists have more than one element.')
