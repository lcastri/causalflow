"""
This module provides various classes for Transfer Entropy-based feature selection analysis.

Classes:
    TEestimator: support class for handling different Transfer Entropy estimators.
    TE: Transfer Entropy class.
"""

import copy
from enum import Enum
import numpy as np
from causalflow.selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_mi import BivariateMI
from idtxl.data import Data
from causalflow.CPrinter import CP
from scipy.stats import shapiro, kstest
import importlib


class TEestimator(Enum):
    """TEestimator Enumerator."""
    
    Auto = 'Auto'
    Gaussian = 'JidtGaussianCMI'
    Kraskov = 'JidtKraskovCMI'
    OpenCLKraskov = 'OpenCLKraskovCMI'


class TE(SelectionMethod):
    """Feature selection method based on Trasfer Entropy analysis."""
    
    def __init__(self, estimator: TEestimator):
        """
        Class contructor.

        Args:
            estimator (TEestimator): Gaussian/Kraskov.
        """
        super().__init__(CTest.TE)
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
        """Select the TE estimator."""
        CP.info("\n##")
        CP.info("## TE Estimator selection")
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
            self.estimator = TEestimator.Gaussian
        else:
            self.estimator = TEestimator.OpenCLKraskov if self.isOpenCLinstalled else TEestimator.Kraskov
        CP.info("\n## TE Estimator: " + self.estimator.value)


    def compute_dependencies(self):
        """
        Compute list of dependencies for each target by transfer entropy analysis.

        Returns:
            (DAG): dependency dag.
        """
        if self.estimator is TEestimator.Auto: self._select_estimator()

        multi_network_analysis = MultivariateTE()
        bi_network_analysis = BivariateMI()
        cross_settings = {'cmi_estimator': self.estimator.value,
                    'max_lag_sources': self.max_lag,
                    'min_lag_sources': self.min_lag,
                    'max_lag_target': self.max_lag,
                    'min_lag_target': self.min_lag,
                    'alpha_max_stats': self.alpha,
                    'alpha_min_stats': self.alpha,
                    'alpha_omnibus': self.alpha,
                    'alpha_max_seq': self.alpha,
                    'verbose': False}
        autodep_settings = copy.deepcopy(cross_settings)
        if self.min_lag == 0:
            autodep_settings['min_lag_sources'] = 1
        
        CP.info("\n##")
        CP.info("## " + self.name + " analysis")
        CP.info("##")
        for target in self.data.features:
            CP.info("\n## Target variable: " + target)
            with _suppress_stdout():
                t = self.data.features.index(target)
                
                # Check auto-dependency
                tmp_d = np.c_[self.data.d.values[:, t], self.data.d.values[:, t]]
                data = Data(tmp_d, dim_order='sp') # sp = samples(row) x processes(col)
                res_auto = bi_network_analysis.analyse_single_target(settings = autodep_settings, data = data, target = 0, sources = 1)
                
                # Check cross-dependencies
                data = Data(self.data.d.values, dim_order='sp') # sp = samples(row) x processes(col)
                res_cross = multi_network_analysis.analyse_single_target(settings = cross_settings, data = data, target = t)
            
            # Auto-dependency handling
            auto_lag = [s[1] for s in res_auto._single_target[0]['selected_vars_sources']]
            auto_score = res_auto._single_target[0]['selected_sources_mi']
            auto_pval = res_auto._single_target[0]['selected_sources_pval']
            if auto_score is not None:
                for score, pval, lag in zip(auto_score, auto_pval, auto_lag):
                    self._add_dependency(self.data.features[t], self.data.features[t], score, pval, lag)
            
            # Cross-dependencies handling    
            sel_sources = [s[0] for s in res_cross._single_target[t]['selected_vars_sources']]
            if sel_sources:
                sel_sources_lag = [s[1] for s in res_cross._single_target[t]['selected_vars_sources']]
                sel_sources_score = res_cross._single_target[t]['selected_sources_te']
                sel_sources_pval = res_cross._single_target[t]['selected_sources_pval']
                for s, score, pval, lag in zip(sel_sources, sel_sources_score, sel_sources_pval, sel_sources_lag):
                    self._add_dependency(self.data.features[t], self.data.features[s], score, pval, lag)
            
            if auto_score is None and not sel_sources:
                CP.info("\tno sources selected")

        return self.result