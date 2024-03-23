from subprocess import Popen, PIPE
import os
import glob
import pandas as pd
from causalflow.causal_discovery.baseline.pkgs import utils
from causalflow.graph.DAG import DAG
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 
import networkx as nx


class TiMINo(CausalDiscoveryMethod):
    """
    TiMINO causal discovery method.
    """
    def __init__(self, 
                 data, 
                 min_lag,
                 max_lag, 
                 verbosity, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,):
        """
        TiMINO class constructor

        Args:
            data (Data): data to analyse
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            verbosity (CPLevel): verbosity level
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
        """
        
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep)
               
    
    def run(self) -> DAG:
        """
        Run causal discovery algorithm

        Returns:
            (DAG): estimated causal model
        """
        df = utils.runTiMINo(self.data.d, self.max_lag, self.alpha)
        a = utils.dataframe_to_graph(self.data.features, df)
        print(a)
    
    
    def _to_DAG(self, graph):
        """
        Re-elaborates the result in a DAG

        Returns:
            (DAG): result re-elaborated
        """
        vars = list(graph.keys())
        tmp_dag = DAG(vars, self.min_lag, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                lag = abs(s[1])
                if lag >= self.min_lag and lag <= self.max_lag:
                    tmp_dag.add_source(t, s[0], utils.DSCORE, 0, s[1])
        return tmp_dag


    