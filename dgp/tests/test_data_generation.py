"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

import unittest

import networkx
import numpy
import pandas

from dgp.src.data_generation_configs import (
    CausalGraphConfig, DataGenerationConfig, FunctionConfig, NoiseConfig, RuntimeConfig
)
from dgp.src.structural_causal_model import StructuralCausalModel
from dgp.src.time_series_causal_graph import TimeSeriesCausalGraph
from dgp.src.time_series_generator import TimeSeriesGenerator


class TestDataGeneration(unittest.TestCase):
    def test_configs(self):
        # Valid configs.
        _ = DataGenerationConfig(random_seed=1)
        _ = CausalGraphConfig()
        _ = FunctionConfig()
        _ = NoiseConfig()
        _ = RuntimeConfig()

        # Invalid configs.
        with self.assertRaises(AssertionError):
            _ = DataGenerationConfig(random_seed=-1)
        with self.assertRaises(AssertionError):
            _ = CausalGraphConfig(min_lag=3, max_lag=2)
        with self.assertRaises(AssertionError):
            _ = FunctionConfig(functions=['linear', 'monotonic'], prob_functions=None)
        with self.assertRaises(AssertionError):
            _ = NoiseConfig(distributions=['gaussian', 'students_x'], prob_distributions=[0.5, 0.5])
        with self.assertRaises(AssertionError):
            _ = RuntimeConfig(num_samples=[10, 20, 30], data_generating_seed=[1, 2])

    def test_causal_graph(self):
        rng = numpy.random.default_rng(seed=1)
        config = CausalGraphConfig(graph_complexity=20, max_lag=3, num_targets=1)
        causal_graph = TimeSeriesCausalGraph(graph_config=config, rng=rng)

        # Confirm graph is a valid DAG.
        self.assertTrue(networkx.is_directed_acyclic_graph(causal_graph.causal_graph))
        self.assertEquals(len(causal_graph.get_target_nodes()), 4)  # 1 target variable with nodes at t, t-1, t-2, t-3

    def test_structural_causal_model(self):
        rng = numpy.random.default_rng(seed=1)
        graph_config = CausalGraphConfig(graph_complexity=10, num_latent=1)
        causal_graph = TimeSeriesCausalGraph(graph_config=graph_config, rng=rng)

        function_config = FunctionConfig(function_complexity=0)
        scm = StructuralCausalModel(function_config=function_config, causal_graph=causal_graph, rng=rng)

        noise_config = NoiseConfig(noise_variance=0.01, distributions=['gaussian'], prob_distributions=[1.0])
        num_samples = 50
        df_default = scm.generate_dataset(num_samples, noise_config, return_observed_data_only=True,
                                          normalize_data=False, percent_missing=0.0)
        df_norm = scm.generate_dataset(num_samples, noise_config, return_observed_data_only=True,
                                       normalize_data=True, percent_missing=0.0)
        df_latent = scm.generate_dataset(num_samples, noise_config, return_observed_data_only=False,
                                         normalize_data=False, percent_missing=0.0)
        df_missing = scm.generate_dataset(num_samples, noise_config, return_observed_data_only=True,
                                          normalize_data=False, percent_missing=0.5)

        # Confirm no latent or noise nodes in default output.
        self.assertTrue(all(not (c.startswith('U') or c.startswith('S')) for c in df_default.columns))

        # Confirm mean and variance are 0 and 1 respectively for each series in data frame.
        self.assertTrue(all(df_norm.mean().abs() < 1.0e-10))
        self.assertTrue(all((df_norm.var() - 1.0).abs() < 1.0e-10))

        # Confirm one latent node in latent output and confirm noise is also included.
        self.assertEquals(1, len([c for c in df_latent.columns if c.startswith('U')]))
        self.assertTrue(any(c.startswith('S') for c in df_latent.columns))

        # Confirm number of missing values roughly matches expected.
        num_nan = df_missing.size - sum(df_missing.count())
        percent_missing = 1.0 * num_nan / df_missing.size
        self.assertTrue(0.45 < percent_missing < 0.55)

    def test_time_series_generator(self):
        config = DataGenerationConfig(random_seed=1, complexity=10)
        data_generator = TimeSeriesGenerator(config=config)

        # Confirm configuration is completed by spot checking some fields in each lower-level config.
        full_config = data_generator.get_full_config()
        self.assertIsNotNone(full_config.causal_graph_config.max_target_parents)
        self.assertIsNotNone(full_config.causal_graph_config.allow_latent_direct_target_cause)
        self.assertIsNotNone(full_config.causal_graph_config.prob_noise_autoregressive)
        self.assertIsNotNone(full_config.function_config.functions)
        self.assertIsNotNone(full_config.function_config.prob_functions)
        self.assertIsNotNone(full_config.noise_config.noise_variance)
        self.assertIsNotNone(full_config.noise_config.distributions)
        self.assertIsNotNone(full_config.noise_config.prob_distributions)
        self.assertIsNotNone(full_config.runtime_config.num_samples)
        self.assertIsNotNone(full_config.runtime_config.data_generating_seed)

        # Generate data.
        output = data_generator.generate_datasets()
        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(isinstance(output[0], list))
        self.assertTrue(all(isinstance(df, pandas.DataFrame) for df in output[0]))
        self.assertTrue(isinstance(output[1], TimeSeriesCausalGraph))

        # Confirm we can override NoiseConfig and RuntimeConfig.
        noise_config = NoiseConfig(noise_variance=0.01, distributions=['gaussian'], prob_distributions=[1.0])
        runtime_config = RuntimeConfig(num_samples=[50, 25], data_generating_seed=[10, 20])
        output = data_generator.generate_datasets(noise_config=noise_config, runtime_config=runtime_config)
        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(isinstance(output[0], list))
        self.assertEquals(len(output[0]), 2)
        self.assertTrue(all(isinstance(df, pandas.DataFrame) for df in output[0]))
        self.assertEquals(len(output[0][0].index), 50)
        self.assertEquals(len(output[0][1].index), 25)
        self.assertTrue(isinstance(output[1], TimeSeriesCausalGraph))

        # Check overriding with just list of seeds also works as expected.
        runtime_config = RuntimeConfig(num_samples=10, data_generating_seed=[10, 20, 30])
        output = data_generator.generate_datasets(noise_config=noise_config, runtime_config=runtime_config)
        self.assertTrue(isinstance(output, tuple))
        self.assertTrue(isinstance(output[0], list))
        self.assertEquals(len(output[0]), 3)
        self.assertTrue(all(isinstance(df, pandas.DataFrame) for df in output[0]))
        self.assertEquals(len(output[0][0].index), 10)
        self.assertEquals(len(output[0][1].index), 10)
        self.assertEquals(len(output[0][2].index), 10)
        self.assertTrue(isinstance(output[1], TimeSeriesCausalGraph))

    def test_complex_generation(self):
        config = DataGenerationConfig(random_seed=11, complexity=30, runtime_config=RuntimeConfig(num_samples=500))
        data_generator = TimeSeriesGenerator(config=config)

        # Confirm all generated values are finite.
        output = data_generator.generate_datasets()
        df = output[0][0]
        self.assertEquals(len(df.index), 500)
        self.assertTrue(numpy.all(numpy.isfinite(df)))


if __name__ == '__main__':
    unittest.main()
