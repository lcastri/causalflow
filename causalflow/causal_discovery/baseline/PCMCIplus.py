from tigramite.pcmci import PCMCI as pcmci
from tigramite.independence_tests.independence_tests_base import CondIndTest
import tigramite.data_processing as pp
import numpy as np
from causalflow.CPrinter import CPLevel, CP
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 

class PCMCIplus(CausalDiscoveryMethod):
    """
    PCMCI+ causal discovery method.
    """
    def __init__(self, 
                 data: Data, 
                 min_lag, max_lag, 
                 val_condtest: CondIndTest, 
                 verbosity: CPLevel,
                 alpha = 0.05, 
                 resfolder = None, 
                 neglect_only_autodep = False,
                 clean_cls = True):
        """
        PCMCI+ class constructor

        Args:
            data (Data): data to analyse
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            val_condtest (CondIndTest): validation method
            verbosity (CPLevel): verbosity level
            alpha (float, optional): significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.
        """
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)
        
        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        
        # init pcmci
        self.pcmci = pcmci(dataframe = pp.DataFrame(data = d, var_names = data.features),
                           cond_ind_test = val_condtest,
                           verbosity = verbosity.value)
        

    def run(self, link_assumptions=None) -> DAG:
        """
        Run causal discovery algorithm

        Returns:
            (DAG): estimated causal model
        """
        
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        self.result = self.pcmci.run_pcmciplus(link_assumptions=link_assumptions,
                                               tau_max = self.max_lag,
                                               tau_min = 0,
                                               pc_alpha = self.alpha)
                
        self.CM = self._to_DAG()
        
        if self.resfolder is not None: self.logger.close()
        return self.CM
    
    
    def _to_DAG(self):
        """
        Re-elaborates the PCMCI result in a new dictionary

        Returns:
            (DAG): pcmci result re-elaborated
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
                        # print(f'({self.data.features[s]}, -{lag}) {arrowtype} {self.data.features[t]}')
                        
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