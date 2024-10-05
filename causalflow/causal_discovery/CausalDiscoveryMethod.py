"""
This module provides the CausalDiscoveryMethod class.

Classes:
    CausalDiscoveryMethod: abstract class used by all the causal discovery algorithms.
"""

from abc import ABC, abstractmethod
import sys
import copy
import pickle
from causalflow.graph.DAG import DAG
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.basics.logger import Logger
import causalflow.basics.utils as utils
from causalflow.preprocessing.data import Data


class CausalDiscoveryMethod(ABC):
    """
    CausalDiscoveryMethod class.

    CausalDiscoveryMethod is an abstract causal discovery method for 
    large-scale time series datasets.
    """

    def __init__(self, 
                 data: Data, 
                 min_lag, max_lag, 
                 verbosity: CPLevel, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,
                 clean_cls = True):
        """
        Class contructor.

        Args:
            data (Data): data to analyse.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            verbosity (CPLevel): verbosity level.
            alpha (float, optional): significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.

        """
        self.data = data
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.CM = DAG(self.data.features, min_lag, max_lag, neglect_only_autodep)
        self.neglect_only_autodep = neglect_only_autodep

        self.resfolder = resfolder
        self.respath, self.dag_path, self.ts_dag_path = None, None, None
        if resfolder is not None:
            logpath, self.respath, self.dag_path, self.ts_dag_path = utils.get_selectorpath(resfolder)  
            self.logger = Logger(logpath, clean_cls)
            sys.stdout = self.logger
        
        CP.set_verbosity(verbosity)


    @abstractmethod
    def run(self) -> DAG:
        """
        Run causal discovery method.
        
        Returns:
            DAG: causal model.
        """
        pass

    
    def load(self, res_path):
        """
        Load previously estimated result .

        Args:
            res_path (str): pickle file path.
        """
        with open(res_path, 'rb') as f:
            r = pickle.load(f)
            self.CM = r['causal_model']
            self.alpha = r['alpha']
            self.dag_path = r['dag_path']
            self.ts_dag_path = r['ts_dag_path']
            
            
    def save(self):
        """Save causal discovery result as pickle file if resfolder is set."""
        if self.respath is not None:
            if self.CM:
                res = dict()
                res['causal_model'] = copy.deepcopy(self.CM)
                res['alpha'] = self.alpha
                res['dag_path'] = self.dag_path
                res['ts_dag_path'] = self.ts_dag_path
                with open(self.respath, 'wb') as resfile:
                    pickle.dump(res, resfile)
            else:
                CP.warning("Causal model impossible to save")