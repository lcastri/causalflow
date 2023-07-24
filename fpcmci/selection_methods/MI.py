from enum import Enum
from fpcmci.selection_methods.SelectionMethod import SelectionMethod, CTest, _suppress_stdout
from idtxl.multivariate_mi import MultivariateMI
from idtxl.data import Data

class MIestimator(Enum):
    Gaussian = 'JidtGaussianCMI'
    Kraskov = 'JidtKraskovCMI'


class MI(SelectionMethod):
    """
    Feature selection method based on Mutual Information analysis
    """
    def __init__(self, estimator: MIestimator):
        """
        MI class contructor

        Args:
            estimator (MIestimator): Gaussian/Kraskov
        """
        super().__init__(CTest.MI)
        self.estimator = estimator

    def compute_dependencies(self):
        """
        compute list of dependencies for each target by mutual information analysis

        Returns:
            (dict): dictonary(TARGET: list SOURCES)
        """
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
                    self._add_dependecy(self.features[t], self.features[s], score, pval, lag)

        return self.result