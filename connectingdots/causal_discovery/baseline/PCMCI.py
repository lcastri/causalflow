from tigramite.pcmci import PCMCI as pcmci
from tigramite.independence_tests.independence_tests_base import CondIndTest
import tigramite.data_processing as pp
import numpy as np
from connectingdots.CPrinter import CPLevel, CP
from connectingdots.basics.constants import *
from connectingdots.graph.DAG import DAG
from connectingdots.preprocessing.data import Data
from connectingdots.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 


class PCMCI(CausalDiscoveryMethod):
    """
    PCMCI causal discovery method.
    """
    def __init__(self, 
                 data: Data, 
                 min_lag, max_lag, 
                 val_condtest: CondIndTest, 
                 verbosity: CPLevel,
                 pc_alpha = 0.05, 
                 alpha = 0.05, 
                 resfolder = None, 
                 neglect_only_autodep = False):
        """
        PCMCI class constructor

        Args:
            data (Data): data to analyse
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            val_condtest (CondIndTest): validation method
            verbosity (CPLevel): verbosity level
            pc_alpha (float, optional): PC significance level. Defaults to 0.05.
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
        """
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep)
        self.pc_alpha = pc_alpha
        
        # build tigramite dataset
        vector = np.vectorize(float)
        d = vector(data.d)
        
        # init pcmci
        self.pcmci = pcmci(dataframe = pp.DataFrame(data = d, var_names = data.features),
                           cond_ind_test = val_condtest,
                           verbosity = verbosity.value)
        

    def run(self) -> DAG:
        """
        Run causal discovery algorithm

        Args:
            data (Data): Data obj to analyse
            link_assumptions (dict, optional): prior assumptions on causal model links. Defaults to None.

        Returns:
            (DAG): estimated causal model
        """
        
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")
        self.result = self.pcmci.run_pcmci(tau_max = self.max_lag,
                                           tau_min = self.min_lag,
                                           alpha_level = self.alpha,
                                           pc_alpha = self.pc_alpha)
        
        self.CM = self._to_DAG()
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
                    if self.result['graph'][s][t,lag] == LinkType.Directed.value:
                        tmp_dag.add_source(vars[t], 
                                           vars[s],
                                           self.result['val_matrix'][s][t,lag],
                                           self.result['p_matrix'][s][t,lag],
                                           lag)
                    elif self.result['graph'][s][t,lag] == LinkType.Undirected.value:
                        tmp_dag.add_source(vars[t], 
                                           vars[s],
                                           self.result['val_matrix'][s][t,lag],
                                           self.result['p_matrix'][s][t,lag],
                                           lag,
                                           LinkType.Undirected.value)
        return tmp_dag