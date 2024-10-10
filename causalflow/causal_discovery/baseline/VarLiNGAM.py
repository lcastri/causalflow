"""
This module provides the VarLiNGAM class.

Classes:
    VarLiNGAM: class containing the VarLiNGAM causal discovery algorithm.
"""

from lingam.var_lingam import VARLiNGAM
import numpy as np
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod
from causalflow.graph.DAG import DAG
from causalflow.causal_discovery.baseline.pkgs import utils


class VarLiNGAM(CausalDiscoveryMethod):
    """VarLiNGAM causal discovery method."""
    
    def __init__(self, 
                 data, 
                 min_lag,
                 max_lag, 
                 verbosity, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,
                 clean_cls = True):
        """
        Class constructor.

        Args:
            data (Data): data to analyse.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            verbosity (CPLevel): verbosity level.
            alpha (float, optional): PCMCI significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
            clean_cls (bool): Clean console bit. Default to True.
        """
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep, clean_cls)


    def run(self) -> DAG:
        """
        Run causal discovery algorithm.

        Returns:
            (DAG): estimated causal model.
        """
        split_by_causal_effect_sign = True

        model = VARLiNGAM(lags = self.max_lag, criterion='bic', prune=True)
        model.fit(self.data.d)

        m = model._adjacency_matrices
        am = np.concatenate([*m], axis=1)

        dag = np.abs(am) > self.alpha

        if split_by_causal_effect_sign:
            direction = np.array(np.where(dag))
            signs = np.zeros_like(dag).astype('int64')
            for i, j in direction.T:
                signs[i][j] = np.sign(am[i][j]).astype('int64')
            dag = signs

        dag = np.abs(dag)
        names = self.data.features
        res_dict = dict()
        for e in range(dag.shape[0]):
            res_dict[names[e]] = []
        for c in range(dag.shape[0]):
            for te in range(dag.shape[1]):
                if dag[c][te] == 1:
                    e = te%dag.shape[0]
                    t = te//dag.shape[0]
                    res_dict[names[e]].append((names[c], -t))
        self.CM = self._to_DAG(res_dict)
        
        if self.resfolder is not None: self.logger.close()
        return self.CM
    
    def _to_DAG(self, graph):
        """
        Re-elaborates the result in a DAG.
        
        Args:
            graph (dict): graph to convert into a DAG
            
        Returns:
            (DAG): result re-elaborated.
        """
        tmp_dag = DAG(self.data.features, self.min_lag, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                lag = abs(s[1])
                if lag >= self.min_lag and lag <= self.max_lag:
                    tmp_dag.add_source(t, s[0], utils.DSCORE, 0, s[1])
        return tmp_dag