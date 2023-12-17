import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic
from connectingdots.graph.DAG import DAG


class DYNOTEARS():
    
    def __init__(self, alpha, max_lag):
        self.alpha = alpha
        self.max_lag = max_lag
        
    def run(self, data) -> DAG:
        graph_dict = dict()
        for name in data.columns:
            graph_dict[name] = []
        sm = from_pandas_dynamic(data, p=self.max_lag, w_threshold=0.01, lambda_w=0.05, lambda_a=0.05)

        tname_to_name_dict = dict()
        count_lag = 0
        idx_name = 0
        for tname in sm.nodes:
            tname_to_name_dict[tname] = data.columns[idx_name]
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

        return self._to_DAG(graph_dict)
    
    
    def _to_DAG(self, graph):
        """
        Re-elaborates the result in a DAG

        Returns:
            (DAG): result re-elaborated
        """
        vars = list(graph.keys())
        tmp_dag = DAG(vars, 0, self.max_lag)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                tmp_dag.add_source(t, s[0], abs(s[1]), 0, s[2])
        return tmp_dag