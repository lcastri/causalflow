from causalnex.structure.dynotears import from_pandas_dynamic
from connectingdots.graph.DAG import DAG
from connectingdots.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod 


class DYNOTEARS(CausalDiscoveryMethod):
    """
    DYNOTEARS causal discovery method.
    """
    def __init__(self, 
                 data, 
                 min_lag,
                 max_lag, 
                 verbosity, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,):
        
        super().__init__(data, min_lag, max_lag, verbosity, alpha, resfolder, neglect_only_autodep)
        
    def run(self) -> DAG:
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
        return self.CM
    
    
    def _to_DAG(self, graph):
        """
        Re-elaborates the result in a DAG

        Returns:
            (DAG): result re-elaborated
        """
        tmp_dag = DAG(self.data.features, self.min_lag, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                lag = abs(s[2])
                if lag >= self.min_lag and lag <= self.max_lag:
                    tmp_dag.add_source(t, s[0], abs(s[1]), 0, s[2])
        # tmp_dag.remove_unneeded_features()
        return tmp_dag