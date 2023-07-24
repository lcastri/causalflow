from enum import Enum
import numpy as np
from fpcmci.selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_mi import BivariateMI
from idtxl.data import Data
from fpcmci.CPrinter import CP


class TEestimator(Enum):
    Gaussian = 'JidtGaussianCMI'
    Kraskov = 'JidtKraskovCMI'


class TE(SelectionMethod):
    """
    Feature selection method based on Trasfer Entropy analysis
    """
    def __init__(self, estimator: TEestimator):
        """
        TE class contructor

        Args:
            estimator (TEestimator): Gaussian/Kraskov
        """
        super().__init__(CTest.TE)
        self.estimator = estimator


    def compute_dependencies(self):
        """
        compute list of dependencies for each target by transfer entropy analysis

        Returns:
            (dict): dictonary(TARGET: list SOURCES)
        """
        multi_network_analysis = MultivariateTE()
        bi_network_analysis = BivariateMI()
        settings = {'cmi_estimator': self.estimator.value,
                    'max_lag_sources': self.max_lag,
                    'min_lag_sources': self.min_lag,
                    'max_lag_target': self.max_lag,
                    'min_lag_target': self.min_lag,
                    'alpha_max_stats': self.alpha,
                    'alpha_min_stats': self.alpha,
                    'alpha_omnibus': self.alpha,
                    'alpha_max_seq': self.alpha,
                    'verbose': False}
        
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
                res_auto = bi_network_analysis.analyse_single_target(settings = settings, data = data, target = 0, sources = 1)
                
                # Check cross-dependencies
                data = Data(self.data.d.values, dim_order='sp') # sp = samples(row) x processes(col)
                res_cross = multi_network_analysis.analyse_single_target(settings = settings, data = data, target = t)
            
            # Auto-dependency handling
            auto_lag = [s[1] for s in res_auto._single_target[0]['selected_vars_sources']]
            auto_score = res_auto._single_target[0]['selected_sources_mi']
            auto_pval = res_auto._single_target[0]['selected_sources_pval']
            if auto_score is not None:
                for score, pval, lag in zip(auto_score, auto_pval, auto_lag):
                    self._add_dependecy(self.data.features[t], self.data.features[t], score, pval, lag)
            
            # Cross-dependencies handling    
            sel_sources = [s[0] for s in res_cross._single_target[t]['selected_vars_sources']]
            if sel_sources:
                sel_sources_lag = [s[1] for s in res_cross._single_target[t]['selected_vars_sources']]
                sel_sources_score = res_cross._single_target[t]['selected_sources_te']
                sel_sources_pval = res_cross._single_target[t]['selected_sources_pval']
                for s, score, pval, lag in zip(sel_sources, sel_sources_score, sel_sources_pval, sel_sources_lag):
                    self._add_dependecy(self.data.features[t], self.data.features[s], score, pval, lag)
            
            if auto_score is None and not sel_sources:
                CP.info("\tno sources selected")

        return self.result