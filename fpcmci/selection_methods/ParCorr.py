from fpcmci.selection_methods.SelectionMethod import SelectionMethod, CTest
from fpcmci.CPrinter import CP
from scipy import stats, linalg
import numpy as np

class ParCorr(SelectionMethod):
    """
    Feature selection method based on Partial Correlation analysis
    """
    def __init__(self):
        """
        ParCorr class contructor
        """
        super().__init__(CTest.Corr)


    def get_residual(self, covar, target):
        """
        Calculate residual of the target variable obtaining conditioning on the covar variables

        Args:
            covar (np.array): conditioning variables
            target (np.array): target variable

        Returns:
            (np.array): residual
        """
        beta = np.linalg.lstsq(covar, target, rcond=None)[0]
        return target - np.dot(covar, beta)


    def partial_corr(self, X, Y, Z):
        """
        Calculate Partial correlation between X and Y conditioning on Z

        Args:
            X (np.array): source candidate variable
            Y (np.array): target variable
            Z (np.array): conditioning variable

        Returns:
            (float, float): partial correlation, p-value
        """

        pcorr, pval = stats.pearsonr(self.get_residual(Z, X), self.get_residual(Z, Y))

        return pcorr, pval

    def compute_dependencies(self):
        """
        compute list of dependencies for each target by partial correlation analysis

        Returns:
            (dict): dictonary(TARGET: list SOURCES)
        """
        CP.info("\n##")
        CP.info("## " + self.name + " analysis")
        CP.info("##")

        for lag in range(self.min_lag, self.max_lag + 1):
            for target in self.data.features:
                CP.info("\n## Target variable: " + target)
                candidates = self.data.features

                Y = np.array(self.data.d[target][lag:])

                while candidates:
                    tmp_res = None
                    covars = self._get_sources(target)
                    Z = np.array(self.data.d[covars][:-lag])

                    for candidate in candidates:
                        X = np.array(self.data.d[candidate][:-lag])
                        score, pval = self.partial_corr(X, Y, Z)
                        if pval < self.alpha and (tmp_res is None or abs(tmp_res[1]) < abs(score)):
                            tmp_res = (candidate, score, pval)

                    if tmp_res is not None: 
                        self._add_dependecy(target, tmp_res[0], tmp_res[1], tmp_res[2], lag)
                        candidates.remove(tmp_res[0])
                    else:
                        break
        return self.result