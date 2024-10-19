"""
This module provides various classes for Mutual Information-based feature selection analysis.

Classes:
    MIestimator: support class for handling different Mutual Information estimators.
    MI: Mutual Information class.
"""

from enum import Enum
from causalflow.selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from idtxl.multivariate_mi import MultivariateMI
from idtxl.data import Data
from causalflow.CPrinter import CP
from scipy.stats import shapiro, kstest
import importlib

class MIestimator(Enum):
    """MIestimator Enumerator."""

    Auto = 'Auto'
    Gaussian = 'JidtGaussianCMI'
    Kraskov = 'JidtKraskovCMI'
    OpenCLKraskov = 'OpenCLKraskovCMI'


class MI(SelectionMethod):
    """Feature selection method based on Mutual Information analysis."""
    
    def __init__(self, estimator: MIestimator):
        """
        Class contructor.

        Args:
            estimator (MIestimator): Gaussian/Kraskov
        """
        super().__init__(CTest.MI)
        self.estimator = estimator
        
    @property
    def isOpenCLinstalled(self) -> bool:
        """
        Check whether the pyopencl pkg is installed.

        Returns:
            bool: True if pyopencl is installed.
        """
        try:
            importlib.import_module('pyopencl')
            return True
        except ImportError:
            return False
        
    def _select_estimator(self):
        """Select the MI estimator."""
        CP.info("\n##")
        CP.info("## MI Estimator selection")
        CP.info("##")

        isGaussian = True

        for f in self.data.features:
            # Perform Shapiro-Wilk test
            shapiro_stat, shapiro_p_value = shapiro(self.data.d[f])
            # Perform Kolmogorov-Smirnov test
            ks_stat, ks_p_value = kstest(self.data.d[f], 'norm')
                
            # Print results
            CP.debug("\n")
            CP.debug(f"Feature '{f}':")
            CP.debug(f"\t- Shapiro-Wilk test: val={round(shapiro_stat, 2)}, p-val={round(shapiro_p_value, 2)}")
            CP.debug(f"\t- Kolmogorov-Smirnov test: val={round(ks_stat, 2)}, p-val={round(ks_p_value, 2)}")
                
            # Check if p-values are less than significance level (e.g., 0.05) for normality
            if shapiro_p_value < 0.05 or ks_p_value < 0.05:
                CP.debug("\tNot normally distributed")
                isGaussian = False
                # break
            else:
                CP.debug("\tNormally distributed")
                
        if isGaussian:
            self.estimator = MIestimator.Gaussian
        else:
            self.estimator = MIestimator.OpenCLKraskov if self.isOpenCLinstalled else MIestimator.Kraskov
        CP.info("\n## MI Estimator: " + self.estimator.value)

    def compute_dependencies(self):
        """
        Compute list of dependencies for each target by mutual information analysis.

        Returns:
            (DAG): dependency dag
        """
        if self.estimator is MIestimator.Auto: self._select_estimator()

        with _suppress_stdout():
            data = Data(self.d.values, dim_order='sp') # sp = samples(row) x processes(col)

            network_analysis = MultivariateMI()
            settings = {'cmi_estimator': self.estimator.value,
                        'max_lag_sources': self.max_lag,
                        'min_lag_sources': self.min_lag,
                        'alpha_max_stats': self.alpha,
                        'alpha_min_stats': self.alpha,
                        'alpha_omnibus': self.alpha,
                        'alpha_max_seq': self.alpha,
                        'verbose': False}
            results = network_analysis.analyse_network(settings=settings, data=data)

        for t in results._single_target.keys():
            sel_sources = [s[0] for s in results._single_target[t]['selected_vars_sources']]
            if sel_sources:
                sel_sources_lag = [s[1] for s in results._single_target[t]['selected_vars_sources']]
                sel_sources_score = results._single_target[t]['selected_sources_mi']
                sel_sources_pval = results._single_target[t]['selected_sources_pval']
                for s, score, pval, lag in zip(sel_sources, sel_sources_score, sel_sources_pval, sel_sources_lag):
                    self._add_dependency(self.features[t], self.features[s], score, pval, lag)

        return self.result