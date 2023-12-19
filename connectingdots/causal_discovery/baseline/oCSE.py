import numpy as np
import pandas as pd
from connectingdots.causal_discovery.CausalDiscoveryMethod import CausalDiscoveryMethod
from tigramite.independence_tests.cmiknn import CMIknn
from connectingdots.CPrinter import CP
from connectingdots.graph.DAG import DAG

class oCSE(CausalDiscoveryMethod):
    def __init__(self, 
                 data, 
                 max_lag, 
                 verbosity, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False,):
        
        super().__init__(data, 0, max_lag, verbosity, alpha, resfolder, neglect_only_autodep)
        

    def _fit(self, x, y, z=None):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        dim_x = x.shape[1]
        dim_y = y.shape[1]
        if z is not None:
            dim_z = z.shape[1]
            X = np.concatenate((x, y, z), axis=1)
            xyz = np.array([0] * dim_x + [1] * dim_y+ [2] * dim_z)
        else:
            X = np.concatenate((x, y), axis=1)
            xyz = np.array([0] * dim_x + [1] * dim_y)
        value = self.cid.get_dependence_measure(X.T, xyz)
        pvalue = self.cid.get_shuffle_significance(X.T, xyz, value)
        return pvalue, value


    def _causation_entropy(self, p, q, r=None):
        qt = q.iloc[1:].values
        pt_1 = p.iloc[:-1].values
        if r is not None:
            rt_1 = r.iloc[:-1].values
        else:
            rt_1 = None

        self.cid = CMIknn(mask_type=None, significance='shuffle_test', fixed_thres=None, sig_samples=10000,
                sig_blocklength=3, knn=10, confidence='bootstrap', conf_lev=0.9, conf_samples=10000,
                conf_blocklength=1, verbosity=0)
        pval, val = self._fit(qt, pt_1, rt_1)
        return pval


    def run(self):
        parents = dict()
        for q in range(self.data.d.shape[1]):
            name_q = self.data.features[q]
            CP.info("Select " + name_q)
            CP.info("Aggregative Discovery of Causal Nodes")
            pval = 0
            # parents[name_q] = [name_q]
            parents[name_q] = []
            series_q = self.data.d[name_q]
            # not_parents_q = list(set(data.columns) - {name_q})
            not_parents_q = list(self.data.features)
            while (pval <= self.alpha) and (len(not_parents_q) > 0):
                pval_list = []
                for name_p in not_parents_q:
                    # print(not_parents_q)
                    # print(name_p)
                    # print(parents[name_q])
                    series_p = self.data.d[name_p]
                    series_cond = self.data.d[parents[name_q]]
                    pval = self._causation_entropy(series_p, series_q, series_cond)
                    pval_list.append(pval)
                # print(pval_list)
                pval = np.min(pval_list)
                # p = np.argmin(pval_list)
                p_list = list(np.argwhere(pval_list == pval)[:, 0])
                names_p = []
                for p in p_list:
                    names_p.append(not_parents_q[p])
                CP.info('test indeps :' + str(pval_list))
                CP.info('CE('+str(names_p)+'->'+name_q+'|'+str(parents[name_q])+') = '+str(pval))
                if pval <= self.alpha:
                    CP.info(str(names_p)+'->'+name_q)
                    for name_p in names_p:
                        parents[self.data.features[q]].append(name_p)
                        not_parents_q.remove(name_p)

            CP.info("Progressive Removal of Non-Causal Nodes")
            # parents_q = list(set(parents[name_q]) - {name_q})
            parents_q = parents[name_q].copy()
            for name_p in parents_q:
                parents_q_without_p = list(set(parents[self.data.features[q]]) - {name_p})
                series_p = self.data.d[name_p]
                series_cond = self.data.d[parents_q_without_p]
                pval = self._causation_entropy(series_p, series_q, series_cond)
                CP.info('CE('+name_p+'->'+name_q+'|'+str(parents_q_without_p)+') = '+str(pval))
                if pval > self.alpha:
                    CP.info('Remove '+name_p+' from parents of '+name_q)
                    parents[self.data.features[q]].remove(name_p)

        parents_df = pd.DataFrame(np.zeros([self.data.d.shape[1], self.data.d.shape[1]], dtype=np.int8), columns=self.data.features,
                                index=self.data.features)
        for name_q in parents.keys():
            parents_df[name_q].loc[name_q] = 1
            for name_p in parents[name_q]:
                if name_q == name_p:
                    parents_df[name_q].loc[name_p] = 1
                else:
                    parents_df[name_q].loc[name_p] = 2
                    if parents_df[name_p].loc[name_q] == 0:
                        parents_df[name_p].loc[name_q] = 1
        self.CM = self._to_DAG(parents_df)
        return self.CM
    
    
    def _to_DAG(self, graph):
        """
        Re-elaborates the result in a DAG

        Returns:
            (DAG): result re-elaborated
        """
        vars = list(graph.keys())
        tmp_dag = DAG(vars, 0, self.max_lag, self.neglect_only_autodep)
        tmp_dag.sys_context = dict()
        for t in graph.keys():
            for s in graph[t]:
                tmp_dag.add_source(t, s[0], abs(s[1]), 0, s[2])
        return tmp_dag