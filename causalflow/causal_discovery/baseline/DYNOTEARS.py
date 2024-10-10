"""
This module provides the DYNOTEARS class.

Classes:
    DYNOTEARS: class containing the DYNOTEARS causal discovery algorithm.
"""

from causalnex.structure.dynotears import from_pandas_dynamic
from causalflow.graph.DAG import DAG
from causalflow.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 


class DYNOTEARS(CausalDiscoveryMethod):
    """DYNOTEARS causal discovery method."""
    
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
        Run DYNOTEARS algorithm.

        Returns:
            DAG: causal discovery result.
        """
        graph_dict = dict()
        for name in self.data.features:
            graph_dict[name] = []
        sm = from_pandas_dynamic(self.data.d, p=self.max_lag)

        tname_to_name_dict = dict()
        count_lag = 0
        idx_name = 0
        for tname in sm.nodes:
            tname_to_name_dict[tname] = self.data.features[idx_name]
            if count_lag == self.max_lag:
                idx_name = idx_name +1
                count_lag = -1
            count_lag = count_lag +1

        for ce in sm.edges:
            c = ce[0]
            e = ce[1]
            w = sm.adj[c][e]["weight"]
            tc = int(c.partition("lag")[2])
            te = int(e.partition("lag")[2])
            t = tc - te
            if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
                graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], w, -t))

        self.CM = self._to_DAG(graph_dict)
        
        if self.resfolder is not None: self.logger.close()
        return self.CM
    
    
    def _to_DAG(self, graph):
        """
        Re-elaborate the result in a DAG.            

        Args:
            graph (dict): graph to convert into a DAG

        Returns:
            (DAG): result re-elaborated.
        """
        tmp_dag = DAG(self.data.features, self.min_lag, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                lag = abs(s[2])
                if lag >= self.min_lag and lag <= self.max_lag:
                    tmp_dag.add_source(t, s[0], abs(s[1]), 0, s[2])
        return tmp_dag