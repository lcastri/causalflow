from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager
import sys, os
from ts_causal_discovery.preprocessing.data import Data
from ts_causal_discovery.basics.constants import *
from ts_causal_discovery.CPrinter import CP
from ts_causal_discovery.graph.DAG import DAG


class CTest(Enum):
    Corr = "Correlation"
    MI = "Mutual Information"
    TE = "Transfer Entropy"


@contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class SelectionMethod(ABC):
    """
    SelectionMethod abstract class
    """
    def __init__(self, ctest):
        self.ctest = ctest
        self.data = None
        self.alpha = None
        self.min_lag = None
        self.max_lag = None
        self.result = None


    @property
    def name(self):
        """
        Returns Selection Method name

        Returns:
            (str): Selection Method name
        """
        return self.ctest.value


    def initialise(self, data: Data, alpha, min_lag, max_lag, graph):
        """
        Initialises the selection method

        Args:
            data (Data): Data
            alpha (float): significance threshold
            min_lag (int): min lag time
            max_lag (int): max lag time
        """
        self.data = data
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = graph


    @abstractmethod
    def compute_dependencies(self) -> DAG:
        """
        abstract method
        """
        pass
    

    def _prepare_ts(self, target, lag, apply_lag = True, consider_autodep = True):
        """
        prepare the dataframe to the analysis

        Args:
            target (str): name target var
            lag (int): lag time to apply
            apply_lag (bool, optional): True if you want to apply the lag, False otherwise. Defaults to True.

        Returns:
            tuple(DataFrame, DataFrame): source and target dataframe
        """
        if not consider_autodep:
            if apply_lag:
                Y = self.data.d[target][lag:]
                X = self.data.d.loc[:, self.data.d.columns != target][:-lag]
            else:
                Y = self.data.d[target]
                X = self.data.d.loc[:, self.data.d.columns != target]
        else:
            if apply_lag:
                Y = self.data.d[target][lag:]
                X = self.data.d[:-lag]
            else:
                Y = self.data.d[target]
                X = self.data.d
        return X, Y


    def _add_dependecy(self, t, s, score, pval, lag):
        """
        Adds found dependency from source (s) to target (t) specifying the 
        score, pval and the lag

        Args:
            t (str): target feature name
            s (str): source feature name
            score (float): selection method score
            pval (float): pval associated to the dependency
            lag (int): lag time of the dependency
        """
        self.result.add_source(t, s, score, pval, lag)
        
        str_s = "(" + s + " -" + str(lag) + ")"
        str_t = "(" + t + ")"

        CP.info("\tlink: " + str_s + " -?> " + str_t)
        CP.info("\t|val = " + str(round(score,3)) + " |pval = " + str(str(round(pval,3))))
        # CP.info("\n")


