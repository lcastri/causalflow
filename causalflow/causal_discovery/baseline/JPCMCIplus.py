"""
This module provides the J-PCMCI+ class.

Classes:
    JPCMCIplus: class containing the J-PCMCI+ causal discovery algorithm.
"""

from tigramite.jpcmciplus import JPCMCIplus as jpcmci
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.regressionCI import RegressionCI
import tigramite.data_processing as pp
import numpy as np
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 
from collections.abc import Iterable

class JPCMCIplus(CausalDiscoveryMethod):
    """J-PCMCI+ causal discovery method."""
    
    def __init__(self, 
                 data: Iterable,
                 min_lag, max_lag, 
                 val_condtest: CondIndTest,
                 node_classification: dict,
                 data_type : dict,
                 verbosity: CPLevel,
                 alpha = 0.05, 
                 resfolder = None, 
                 neglect_only_autodep = False,
                 clean_cls = True):
        """
        Class constructor.

        Args:
            data (Iterable[Data]): list/tuple/dict of data to analyse.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            val_condtest (CondIndTest): validation method.
            node_classification (dict): {node index: "system" | "space_context" | "time_context"}.
            data_type (dict): {node index: DataType.Continuos | DataType.Discrete}.
            verbosity (CPLevel): verbosity level.
            alpha (float, optional): significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.

        Raises:
            ValueError: data_type field needed if Conditional Independece Test == RegressionCI
        """
        if isinstance(val_condtest, RegressionCI) and data_type is None:
            raise ValueError("data_type field needed if Conditional Independece Test == RegressionCI")
        
        super().__init__(data[0], min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)
        DATA_DICT = {}
        DATA_TYPE = {}
        for d in data.values():
            idx = len(DATA_DICT)
            DATA_DICT[idx] = d.d.values
            DATA_TYPE[idx] = np.zeros(d.d.values.shape, dtype='int')
            for v in d.features:
                DATA_TYPE[idx][:, d.features.index(v)] = data_type[v].value
                
        dataframe = pp.DataFrame(
            data = DATA_DICT,
            data_type = DATA_TYPE if isinstance(val_condtest, RegressionCI) else None,
            analysis_mode = 'multiple',
            var_names = data[0].features
            )

        
        # init pcmci
        self.jpcmci = jpcmci(dataframe = dataframe,
                             cond_ind_test = val_condtest,
                             node_classification = node_classification,
                             verbosity = verbosity.value)
        

    def run(self, link_assumptions=None) -> DAG:
        """
        Run causal discovery algorithm.
        
        Args:
            link_assumptions (dict, optional): prior knowledge on causal model links. Defaults to None.

        Returns:
            (DAG): estimated causal model.
        """
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        self.result = self.jpcmci.run_jpcmciplus(link_assumptions=link_assumptions,
                                                 tau_max = self.max_lag,
                                                 tau_min = self.min_lag,
                                                 pc_alpha = self.alpha)
                
        self.CM = self._to_DAG()
        
        if self.resfolder is not None: self.logger.close()
        return self.CM
    
    
    def _to_DAG(self):
        """
        Re-elaborates the PCMCI result in a new dictionary.

        Returns:
            (DAG): pcmci result re-elaborated.
        """
        vars = self.data.features
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