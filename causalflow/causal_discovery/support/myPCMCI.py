"""
This module provides the myPCMCI class.

Classes:
    myPCMCI: support class for F-PCMCI.
"""

from tigramite.pcmci import PCMCI as VAL
from tigramite.independence_tests.independence_tests_base import CondIndTest
import tigramite.data_processing as pp
import numpy as np
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data


class myPCMCI():
    """
    myPCMCI class.
    
    It wraps the PCMCI causal disocvery method and augments it wth some functionalities for F-PCMCI.
    """
    
    def __init__(self, alpha, min_lag, max_lag, val_condtest: CondIndTest, verbosity: CPLevel, sys_context = dict(), neglect_only_autodep = False):
        """
        Class constructor.

        Args:
            alpha (float): significance level.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            val_condtest (CondIndTest): validation method.
            verbosity (CPLevel): verbosity level.
        """
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = None
        self.CM = None
        self.val_method = None
        self.val_condtest = val_condtest
        self.verbosity = verbosity.value
        self.sys_context = sys_context
        self.neglect_only_autodep = neglect_only_autodep
        

    def run(self, data: Data, link_assumptions = None):
        """
        Run causal discovery algorithm.

        Args:
            data (Data): Data obj to analyse.
            link_assumptions (dict, optional): prior knowledge on causal model links. Defaults to None.

        Returns:
            (DAG): estimated causal model.
        """       
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = self.verbosity)

        self.result = self.val_method.run_pcmci(link_assumptions = link_assumptions,
                                                tau_max = self.max_lag,
                                                tau_min = self.min_lag,
                                                alpha_level = self.alpha,
                                                # pc_alpha = self.alpha
                                                )
        
        self.result['var_names'] = data.features
        self.result['pretty_var_names'] = data.pretty_features
        
        self.CM = self._to_DAG()
        return self.CM
    
    
    def run_plus(self, data: Data, link_assumptions = None):
        """
        Run causal discovery algorithm.

        Args:
            data (Data): Data obj to analyse.
            link_assumptions (dict, optional): prior knowledge on causal model links. Defaults to None.

        Returns:
            (DAG): estimated causal model.
        """
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        dataframe = pp.DataFrame(data = d, var_names = data.features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                              cond_ind_test = self.val_condtest,
                              verbosity = self.verbosity)

        self.result = self.val_method.run_pcmciplus(link_assumptions = link_assumptions,
                                                    tau_max = self.max_lag,
                                                    tau_min = 0,
                                                    pc_alpha = self.alpha,
                                                    )
        
        self.result['var_names'] = data.features
        self.result['pretty_var_names'] = data.pretty_features
        
        self.CM = self._to_DAG()
        return self.CM
    
           
    def _to_DAG(self):
        """
        Re-elaborate the PCMCI result in a new dictionary.

        Returns:
            (DAG): pcmci result re-elaborated.
        """
        vars = self.result['var_names']
        tmp_dag = DAG(vars, self.min_lag, self.max_lag)
        tmp_dag.sys_context = dict()
        N, lags = self.result['graph'][0].shape
        for s in range(len(self.result['graph'])):
            for t in range(N):
                for lag in range(lags):
                    if self.result['graph'][s][t,lag] != '':
                        arrowtype = self.result['graph'][s][t,lag]
                        
                        if arrowtype == LinkType.Bidirected.value:
                            if ((vars[s], abs(lag)) in tmp_dag.g[vars[t]].sources and 
                                tmp_dag.g[t].sources[(vars[s], abs(lag))][TYPE] == LinkType.Bidirected.value):
                                continue
                            else:
                                tmp_dag.add_source(vars[t], 
                                                vars[s],
                                                self.result['val_matrix'][s][t,lag],
                                                self.result['p_matrix'][s][t,lag],
                                                lag,
                                                arrowtype)
                                
                        
                        elif arrowtype == LinkType.Uncertain.value:
                            if ((vars[t], abs(lag)) in tmp_dag.g[vars[s]].sources and 
                                tmp_dag.g[vars[s]].sources[(vars[t], abs(lag))][TYPE] == LinkType.Uncertain.value):
                                continue
                            else:
                                tmp_dag.add_source(vars[t], 
                                                vars[s],
                                                self.result['val_matrix'][s][t,lag],
                                                self.result['p_matrix'][s][t,lag],
                                                lag,
                                                arrowtype)
                                
                        
                        elif (arrowtype == LinkType.Directed.value or
                              arrowtype == LinkType.HalfUncertain.value):
                            tmp_dag.add_source(vars[t], 
                                            vars[s],
                                            self.result['val_matrix'][s][t,lag],
                                            self.result['p_matrix'][s][t,lag],
                                            lag,
                                            arrowtype)
        return tmp_dag 