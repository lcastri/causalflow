from fpcmci.selection_methods.SelectionMethod import SelectionMethod, CTest
from sklearn.feature_selection import f_regression
from fpcmci.CPrinter import CP

class Corr(SelectionMethod):
    """
    Feature selection method based on Correlation analysis
    """
    def __init__(self):
        """
        Corr contructor class
        """
        super().__init__(CTest.Corr)


    def compute_dependencies(self):
        """
        compute list of dependencies for each target by correlation analysis

        Returns:
            (dict): dictonary(TARGET: list SOURCES)
        """
        CP.info("\n##")
        CP.info("## " + self.name + " analysis")
        CP.info("##")

        for lag in range(self.min_lag, self.max_lag + 1):
            for target in self.data.features:
                CP.info("\n## Target variable: " + target)

                X, Y = self._prepare_ts(target, lag)
                scores, pval = f_regression(X, Y)
                
                # Filter on pvalue
                f = pval < self.alpha

                # Result of the selection
                sel_sources, sel_sources_score, sel_sources_pval = X.columns[f].tolist(), scores[f].tolist(), pval[f].tolist()

                for s, score, pval in zip(sel_sources, sel_sources_score, sel_sources_pval):
                    self._add_dependecy(target, s, score, pval, lag)

        return self.result