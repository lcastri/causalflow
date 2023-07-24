"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

from typing import List, Optional, Union


_VALID_FUNCTIONS = ['linear', 'piecewise_linear', 'monotonic', 'trigonometric']
_VALID_NOISE_DISTRIBUTIONS = ['gaussian', 'laplace', 'students_t', 'uniform']


class CausalGraphConfig:
    """
    CausalGraphConfig provides the capability to set the parameters for generating the full time causal graph.
    """

    def __init__(self,
                 graph_complexity: int = 10,
                 include_noise: bool = True,
                 max_lag: Optional[int] = None,
                 min_lag: Optional[int] = None,
                 num_targets: Optional[int] = None,
                 num_features: Optional[int] = None,
                 num_latent: Optional[int] = None,
                 prob_edge: Optional[float] = None,
                 max_parents_per_variable: Optional[int] = None,
                 max_target_parents: Optional[int] = None,
                 max_target_children: Optional[int] = None,
                 prob_target_parent: Optional[float] = None,
                 max_feature_parents: Optional[int] = None,
                 max_feature_children: Optional[int] = None,
                 prob_feature_parent: Optional[float] = None,
                 max_latent_parents: Optional[int] = None,
                 max_latent_children: Optional[int] = None,
                 prob_latent_parent: Optional[float] = None,
                 allow_latent_direct_target_cause: Optional[bool] = None,
                 allow_target_direct_target_cause: Optional[bool] = None,
                 prob_target_autoregressive: Optional[float] = None,
                 prob_feature_autoregressive: Optional[float] = None,
                 prob_latent_autoregressive: Optional[float] = None,
                 prob_noise_autoregressive: Optional[float] = None):
        """
        A lower-level configuration object, which provides the capability to set the parameters for generating the full
        time causal graph.

        :param graph_complexity: Used to automatically fill other configuration attributes if they are not specified.
            Can be [0, 10, 20, 30] with larger values leading to more complex causal graphs. Default is 10.
        :param include_noise: Boolean to control if noise is included in the system. Default is True and should always
            be set to True.
        :param max_lag: The maximum possible lag between a parent and child node. If zero, then process generates
            independent and identically distributed (IID) data.
        :param min_lag: The minimum possible lag between a parent and child node. Must be less than or equal to max_lag.
            If zero, this allows for instantaneous effects.
        :param num_targets: The number of target variables.
        :param num_features: The number of feature variables.
        :param num_latent: The number of latent (i.e., unobserved) variables.
        :param prob_edge: The likelihood of an edge. Must be a float in range [0, 1] and is used to control graph
            sparsity. This is superseded by prob_<var_type>_parent if defined for a specific <var_type>.
        :param max_parents_per_variable: The maximum number of lagged nodes of a variable that can be a parent of
            another node. For example, if this is 1, then it prevents X(t-1) -> Y(t) and X(t-2) -> Y(t) as a max number
            of 1 connection from X to Y is allowed. Therefore, this parameter also helps to control graph sparsity.
        :param max_target_parents: The maximum number of parents for a target node.
        :param max_target_children: The maximum number of children for a target node.
        :param prob_target_parent: The likelihood of an edge between a possible parent and a target node.
            Must be a float in range [0, 1].
        :param max_feature_parents: The maximum number of parents for a feature node.
        :param max_feature_children: The maximum number of children for a feature node.
        :param prob_feature_parent: The likelihood of an edge between a possible parent and a feature node.
            Must be a float in range [0, 1].
        :param max_latent_parents: The maximum number of parents for a latent node.
        :param max_latent_children: The maximum number of children for a latent node.
        :param prob_latent_parent: The likelihood of an edge between a possible parent and a latent node.
            Must be a float in range [0, 1].
        :param allow_latent_direct_target_cause: Boolean to control if a latent variable can be a direct cause of a
            target variable.
        :param allow_target_direct_target_cause: Boolean to control if a target variable can be a direct cause of
            another target variable.
        :param prob_target_autoregressive: The likelihood of an autoregressive relationship for a target variable.
            Must be a float in range [0, 1].
        :param prob_feature_autoregressive: The likelihood of an autoregressive relationship for a feature variable.
            Must be a float in range [0, 1].
        :param prob_latent_autoregressive: The likelihood of an autoregressive relationship for a latent variable.
            Must be a float in range [0, 1].
        :param prob_noise_autoregressive: The likelihood of an autoregressive relationship for a noise variable.
            Must be a float in range [0, 1].
        """

        # Validate input.
        assert graph_complexity in [0, 10, 20, 30], \
            f'graph_complexity must be one of 0, 10, 20, or 30. Got {graph_complexity}.'
        assert isinstance(include_noise, bool), \
            f'include_noise must be provided as a boolean. Got {type(include_noise)}.'

        if max_lag is not None:
            assert isinstance(max_lag, int) and max_lag >= 0, f'max_lag must be a non-negative integer. Got {max_lag}.'
        if min_lag is not None:
            assert isinstance(min_lag, int) and min_lag >= 0, f'min_lag must be a non-negative integer. Got {min_lag}.'
        if max_lag is not None and min_lag is not None:
            assert min_lag <= max_lag, (
                f'min_lag must be less than or equal to max_lag. Currently, min_lag is {min_lag} and '
                f'max_lag is {max_lag}.')

        if num_targets is not None:
            assert isinstance(num_targets, int) and num_targets >= 0, \
                f'num_targets must be a non-negative integer. Got {num_targets}.'
        if num_features is not None:
            assert isinstance(num_features, int) and num_features >= 0, \
                f'num_features must be a non-negative integer. Got {num_features}.'
        if num_latent is not None:
            assert isinstance(num_latent, int) and num_latent >= 0, \
                f'num_latent must be a non-negative integer. Got {num_latent}.'

        if prob_edge is not None:
            assert isinstance(prob_edge, float) and 0.0 <= prob_edge <= 1.0, \
                f'prob_edge must be a float between [0.0, 1.0], inclusive. Got {prob_edge}.'
        if prob_target_parent is not None:
            assert isinstance(prob_target_parent, float) and 0.0 <= prob_target_parent <= 1.0, \
                f'prob_target_parent must be a float between [0.0, 1.0], inclusive. Got {prob_target_parent}.'
        if prob_feature_parent is not None:
            assert isinstance(prob_feature_parent, float) and 0.0 <= prob_feature_parent <= 1.0, \
                f'prob_feature_parent must be a float between [0.0, 1.0], inclusive. Got {prob_feature_parent}.'
        if prob_latent_parent is not None:
            assert isinstance(prob_latent_parent, float) and 0.0 <= prob_latent_parent <= 1.0, \
                f'prob_latent_parent must be a float between [0.0, 1.0], inclusive. Got {prob_latent_parent}.'

        if max_parents_per_variable is not None:
            assert isinstance(max_parents_per_variable, int) and max_parents_per_variable > 0, \
                f'max_parents_per_variable must be a positive integer. Got {max_parents_per_variable}.'

        if max_target_parents is not None:
            assert isinstance(max_target_parents, int) and max_target_parents >= 0, \
                f'max_target_parents must be a non-negative integer. Got {max_target_parents}.'
        if max_target_children is not None:
            assert isinstance(max_target_children, int) and max_target_children >= 0, \
                f'max_target_children must be a non-negative integer. Got {max_target_children}.'

        if max_feature_parents is not None:
            assert isinstance(max_feature_parents, int) and max_feature_parents >= 0, \
                f'max_feature_parents must be a non-negative integer. Got {max_feature_parents}.'
        if max_feature_children is not None:
            assert isinstance(max_feature_children, int) and max_feature_children >= 0, \
                f'max_feature_children must be a non-negative integer. Got {max_feature_children}.'

        if max_latent_parents is not None:
            assert isinstance(max_latent_parents, int) and max_latent_parents >= 0, \
                f'max_latent_parents must be a non-negative integer. Got {max_latent_parents}.'
        if max_latent_children is not None:
            assert isinstance(max_latent_children, int) and max_latent_children >= 0, \
                f'max_latent_children must be a non-negative integer. Got {max_latent_children}.'

        if allow_latent_direct_target_cause is not None:
            assert isinstance(allow_latent_direct_target_cause, bool), (
                f'allow_latent_direct_target_cause must be provided as a boolean. '
                f'Got {type(allow_latent_direct_target_cause)}.')
        if allow_target_direct_target_cause is not None:
            assert isinstance(allow_target_direct_target_cause, bool), (
                f'allow_target_direct_target_cause must be provided as a boolean. '
                f'Got {type(allow_target_direct_target_cause)}.')

        if prob_target_autoregressive is not None:
            assert isinstance(prob_target_autoregressive, float) and 0.0 <= prob_target_autoregressive <= 1.0, (
                f'prob_target_autoregressive must be a float between [0.0, 1.0], inclusive. '
                f'Got {prob_target_autoregressive}.')
        if prob_feature_autoregressive is not None:
            assert isinstance(prob_feature_autoregressive, float) and 0.0 <= prob_feature_autoregressive <= 1.0, (
                f'prob_feature_autoregressive must be a float between [0.0, 1.0], inclusive. '
                f'Got {prob_feature_autoregressive}.')
        if prob_latent_autoregressive is not None:
            assert isinstance(prob_latent_autoregressive, float) and 0.0 <= prob_latent_autoregressive <= 1.0, (
                f'prob_latent_autoregressive must be a float between [0.0, 1.0], inclusive. '
                f'Got {prob_latent_autoregressive}.')
        if prob_noise_autoregressive is not None:
            assert isinstance(prob_noise_autoregressive, float) and 0.0 <= prob_noise_autoregressive <= 1.0, (
                f'prob_noise_autoregressive must be a float between [0.0, 1.0], inclusive. '
                f'Got {prob_noise_autoregressive}.')

        # Set attributes.
        self.graph_complexity = graph_complexity
        self.include_noise = include_noise
        self.max_lag = max_lag
        self.min_lag = min_lag

        self.num_targets = num_targets
        self.num_features = num_features
        self.num_latent = num_latent
        self.prob_edge = prob_edge
        self.max_parents_per_variable = max_parents_per_variable

        self.max_target_parents = max_target_parents
        self.max_target_children = max_target_children
        self.prob_target_parent = prob_target_parent

        self.max_feature_parents = max_feature_parents
        self.max_feature_children = max_feature_children
        self.prob_feature_parent = prob_feature_parent

        self.max_latent_parents = max_latent_parents
        self.max_latent_children = max_latent_children
        self.prob_latent_parent = prob_latent_parent

        self.allow_latent_direct_target_cause = allow_latent_direct_target_cause
        self.allow_target_direct_target_cause = allow_target_direct_target_cause

        self.prob_target_autoregressive = prob_target_autoregressive
        self.prob_feature_autoregressive = prob_feature_autoregressive
        self.prob_latent_autoregressive = prob_latent_autoregressive
        self.prob_noise_autoregressive = prob_noise_autoregressive


class FunctionConfig:
    """
    FunctionConfig provides the capability to set the parameters for the structural equations in the structural causal
    model.
    """

    def __init__(self,
                 function_complexity: int = 10,
                 functions: Optional[List[str]] = None,
                 prob_functions: Optional[List[float]] = None):
        """
        A lower-level configuration object, which provides the capability to set the parameters for the structural
        equations in the generated structural causal model.

        :param function_complexity: Used to automatically fill other configuration attributes if they are not specified.
            Can be [0, 10, 20, 30] with larger values leading to more complex structural equations. Default is 10.
        :param functions: A list of possible functional dependencies for each edge in the causal graph.
        :param prob_functions: A list of floats capturing the likelihood of selecting the provided functional
            dependency. This list must be the same length as functions and sum to 1.
        """

        # Validate input.
        assert function_complexity in [0, 10, 20, 30], \
            f'function_complexity must be one of 0, 10, 20, or 30. Got {function_complexity}.'

        # Functions and their likelihood should be provided together.
        if functions is not None or prob_functions is not None:
            assert isinstance(functions, list) and all(f in _VALID_FUNCTIONS for f in functions), \
                f'The function types must be provided as a list of supported functions. ' \
                f'Got {functions} and supported options are {_VALID_FUNCTIONS}.'

            assert (isinstance(prob_functions, list) and
                    all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in prob_functions)), \
                f'Probability of function types must be float between [0, 1], inclusive. Got {prob_functions}.'
            assert sum(prob_functions) == 1.0, \
                f'Probability of function types must sum to 1. Got {sum(prob_functions)}.'

            # Confirm lists of functions and their likelihoods are compatible.
            assert len(functions) == len(prob_functions), (
                'The length of the list of function types and the length of the list of probabilities of '
                'each function type must be the same.')

        # Set attributes.
        self.function_complexity = function_complexity
        self.functions = functions
        self.prob_functions = prob_functions


class NoiseConfig:
    """
    NoiseConfig provides the capability to set the parameters for the noise when sampling data from the generated
    structural causal model.
    """

    def __init__(self,
                 noise_complexity: int = 10,
                 noise_variance: Optional[Union[float, List[float]]] = None,
                 distributions: Optional[List[str]] = None,
                 prob_distributions: Optional[List[float]] = None):
        """
        A lower-level configuration object, which provides the capability to set the parameters for the noise when
        sampling data from the generated structural causal model.

        :param noise_complexity: Used to automatically fill other configuration attributes if they are not specified.
            Can be [0, 10, 20, 30] with larger values leading to more complex noise. Default is 10.
        :param noise_variance: A float or list of 2 floats to specify the variance of each noise node. If provided, as
            a list, the variance for each node will be drawn from a uniform distribution with the given range.
        :param distributions: A list of possible probability distributions for each noise node.
        :param prob_distributions: A list of floats capturing the likelihood of selecting the provided distributions.
            This list must be the same length as distributions and sum to 1.
        """

        # Validate input.
        assert noise_complexity in [0, 10, 20, 30], \
            f'noise_complexity must be one of 0, 10, 20, or 30. Got {noise_complexity}.'

        if noise_variance is not None:
            if isinstance(noise_variance, list):
                assert (len(noise_variance) == 2 and
                        all(isinstance(nv, float) and nv > 0.0 for nv in noise_variance) and
                        noise_variance[0] < noise_variance[1]), (
                    'When provided as a list, noise variance can only have two positive float elements and the '
                    'first element must be less than the second.')
            elif isinstance(noise_variance, float):
                assert noise_variance > 0.0, \
                    f'Noise variance must be provided as a positive float. Got {noise_variance}.'
            else:
                raise TypeError(f'Noise variance did not match expected type of List[float] with 2 elements or float. '
                                f'Got {noise_variance}.')

        # Distributions and their likelihood should be provided together.
        if distributions is not None or prob_distributions is not None:
            assert isinstance(distributions, list) and all(d in _VALID_NOISE_DISTRIBUTIONS for d in distributions), \
                f'The probability distributions must be provided as a list of supported distributions. ' \
                f'Got {distributions} and supported options are {_VALID_NOISE_DISTRIBUTIONS}.'

            assert (isinstance(prob_distributions, list) and
                    all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in prob_distributions)), (
                f'Probability of probability distributions must be float between [0, 1], inclusive. '
                f'Got {prob_distributions}.')
            assert sum(prob_distributions) == 1.0, \
                f'Probability of probability distributions must sum to 1. Got {sum(prob_distributions)}.'

            # Confirm lists of distributions and their likelihoods are compatible.
            assert len(distributions) == len(prob_distributions), (
                'The length of the list of probability distributions and the length of the list of probabilities of '
                'each probability distribution must be the same.')

        # Set attributes.
        self.noise_complexity = noise_complexity
        self.noise_variance = noise_variance
        self.distributions = distributions
        self.prob_distributions = prob_distributions


class RuntimeConfig:
    """
    RuntimeConfig provides the capability to set the parameters for the sampling of data from the generated structural
    causal model.
    """

    def __init__(self,
                 num_samples: Optional[Union[int, List[int]]] = None,
                 data_generating_seed: Optional[Union[int, List[int]]] = None,
                 return_observed_data_only: bool = True,
                 normalize_data: bool = False):
        """
        A lower-level configuration object, which provides the capability to set the parameters for sampling data from
        the generated structural causal model.

        :param num_samples: The number of samples to be generated. Can be provided as a positive integer or a list of
            positive integers. If both, num_samples and data_generating_seed are provided as lists then they must be
            the same length.
        :param data_generating_seed: The random seeds to control random behaviour and ensure reproducibility. If both,
            num_samples and data_generating_seed are provided as lists then they must be the same length.
        :param return_observed_data_only: Boolean to control if latent and noise variables are also returned.
            Default is True so only observed (i.e., target and feature variables) are returned by default.
        :param normalize_data: Boolean to control if data is normalised to zero mean and unit variance.
            Default is False.
        """

        # Validate input.
        assert isinstance(return_observed_data_only, bool), \
            f'return_observed_data_only must be provided as a boolean. Got {type(return_observed_data_only)}.'
        assert isinstance(normalize_data, bool), \
            f'normalize_data must be provided as a boolean. Got {type(normalize_data)}.'

        if num_samples is not None:
            if isinstance(num_samples, list):
                # Assert all elements of list are positive integers.
                assert all(isinstance(ns, int) and ns > 0 for ns in num_samples), \
                    f'Every number of samples must be provided as a positive integer. Got {num_samples}.'
            elif isinstance(num_samples, int):
                assert num_samples > 0, f'Number of samples must be provided as a positive integer. Got {num_samples}.'
                num_samples = [num_samples]
            else:
                raise TypeError(f'Number of samples did not match expected type of List[int] or int. '
                                f'Got {num_samples}.')

        if data_generating_seed is not None:
            if isinstance(data_generating_seed, list):
                # Assert all elements of list are non-negative integers.
                assert all(isinstance(rs, int) and rs >= 0 for rs in data_generating_seed), (
                    f'Every data generating random seed must be provided as a non-negative integer. '
                    f'Got {data_generating_seed}.')
            elif isinstance(data_generating_seed, int):
                assert data_generating_seed >= 0, (
                    f'The data generating random seed must be provided as a non-negative integer. '
                    f'Got {data_generating_seed}.')
                data_generating_seed = [data_generating_seed]
            else:
                raise TypeError(f'The data generating random seed did not match expected type of List[int] or int. '
                                f'Got {data_generating_seed}.')

        # Confirm lists of number of samples and data generating seeds are compatible.
        if (num_samples is not None and data_generating_seed is not None and
                len(num_samples) > 1 and len(data_generating_seed) > 1):

            assert len(num_samples) == len(data_generating_seed), (
                'The length of the list of number of samples and the length of the list of data generating random '
                'seeds must be the same if both lists have more than one element.')

        # Set attributes.
        self.num_samples = num_samples
        self.data_generating_seed = data_generating_seed
        self.return_observed_data_only = return_observed_data_only
        self.normalize_data = normalize_data


class DataGenerationConfig:
    """
    The top-level configuration class for the data generation process and provides the capability to specify all
    necessary parameters to generate a structural causal model and sample time series data.
    """

    def __init__(self,
                 random_seed: int,
                 complexity: int = 10,
                 percent_missing: float = 0.0,
                 causal_graph_config: Optional[CausalGraphConfig] = None,
                 function_config: Optional[FunctionConfig] = None,
                 noise_config: Optional[NoiseConfig] = None,
                 runtime_config: Optional[RuntimeConfig] = None):
        """
        The top-level configuration object, which provides the capability to specify all the necessary parameters to
        generate time series data using a structural causal model.

        :param random_seed: Controls random behaviour and ensures reproducibility. Must be provided as a non-negative
            integer.
        :param complexity: Used to initialise any unspecified configurations. Can be [0, 10, 20, 30] with larger values
            leading to more complex time series data. Default is 10.
        :param percent_missing: A float to set the percentage of missing data (i.e., NaNs) in the final data set(s).
            Default is 0.0.
        :param causal_graph_config: CausalGraphConfig provides the capability to set the parameters for the full time
            causal graph.
        :param function_config: FunctionConfig provides the capability to set the parameters for the structural
            equations in the structural causal model.
        :param noise_config: NoiseConfig provides the capability to set the parameters for the noise distributions in
            the data generating process.
        :param runtime_config: RuntimeConfig provides the capability to set the parameters for sampling data from the
            generated structural causal model.
        """

        # Validate input.
        assert isinstance(random_seed, int) and random_seed >= 0, \
            f'The random seed must be provided as a non-negative integer. Got {random_seed}.'

        assert complexity in [0, 10, 20, 30], f'complexity must be one of 0, 10, 20, or 30. Got {complexity}.'
        assert isinstance(percent_missing, float) and 0.0 <= percent_missing <= 1.0, \
            f'percent_missing must be a float in the range [0.0, 1.0], inclusive. Got {percent_missing}.'

        if causal_graph_config is None:
            causal_graph_config = CausalGraphConfig(graph_complexity=complexity)
        else:
            assert isinstance(causal_graph_config, CausalGraphConfig), (
                f'causal_graph_config must be provided as an instance of CausalGraphConfig. '
                f'Got {type(causal_graph_config)}.')

        if function_config is None:
            function_config = FunctionConfig(function_complexity=complexity)
        else:
            assert isinstance(function_config, FunctionConfig), (
                f'function_config must be provided as an instance of FunctionConfig. '
                f'Got {type(function_config)}.')

        if noise_config is None:
            noise_config = NoiseConfig(noise_complexity=complexity)
        else:
            assert isinstance(noise_config, NoiseConfig), (
                f'noise_config must be provided as an instance of NoiseConfig. '
                f'Got {type(noise_config)}.')

        if runtime_config is None:
            runtime_config = RuntimeConfig()
        assert isinstance(runtime_config, RuntimeConfig), (
            f'runtime_config must be provided as an instance of RuntimeConfig. '
            f'Got {type(runtime_config)}.')

        # Set attributes.
        self.random_seed = random_seed
        self.complexity = complexity
        self.percent_missing = percent_missing
        self.causal_graph_config = causal_graph_config
        self.function_config = function_config
        self.noise_config = noise_config
        self.runtime_config = runtime_config
