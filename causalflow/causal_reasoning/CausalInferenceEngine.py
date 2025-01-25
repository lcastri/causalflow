import os
import pickle
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from causalflow.CPrinter import CP
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Utils import *
from causalflow.causal_reasoning.DynamicBayesianNetwork import DynamicBayesianNetwork
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.CPrinter import CPLevel, CP
import copy
import networkx as nx


class CausalInferenceEngine():
    def __init__(self, 
                 dag: DAG, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType],
                 max_components = 50,
                 model_path: str = '', 
                 verbosity = CPLevel.INFO):
        """
        CausalEngine constructor.

        Args:
            dag (DAG): observational dataset extracted from a causal discovery method.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}
            verbosity (CPLevel, optional): Verbosity level. Defaults to DEBUG.
        """
        os.makedirs(model_path, exist_ok=True)
        CP.set_verbosity(verbosity)
        CP.set_logpath(os.path.join(model_path, 'log.txt'))
        CP.info("\n##")
        CP.info("## Causal Inference Engine")
        CP.info(f"## - Data Type:")
        for k, v in data_type.items(): CP.info(f"##    {k} : {v.name}")
        CP.info(f"## - Node Type:")
        for k, v in node_type.items(): CP.info(f"##    {k} : {v.name}")
        CP.info("##")
        
        self.Q = {}
        self.data_type = data_type
        self.node_type = node_type
        self.max_components = max_components
        self.DAG = {'complete': dag, 'system': self.remove_context(dag)}
        self.model_path = model_path
        
        self.contexts = []
        self.obs_id = -1
        self.int_id = -1
        
        self.Ds = {}
        self.DBNs = {}
        
        
    def remove_context(self, dag: DAG):
        tmp = copy.deepcopy(dag)
        for t in dag.g:
            if self.node_type[t] is NodeType.Context: continue
            for s in dag.g[t].sources:
                if self.node_type[s[0]] is NodeType.Context:
                    tmp.del_source(t, s[0], s[1])
        for t in dag.g:
            if self.node_type[t] is NodeType.Context: 
                tmp.g.pop(t)
        return tmp 
    
    
    @staticmethod
    def remove_intVarParents(dag: DAG, target):
        if target not in dag.g: return dag
        tmp = copy.deepcopy(dag)
        for s in dag.g[target].sources:
            tmp.del_source(target, s[0], s[1])
        return tmp


    def addObsData(self, data: Data):
        """
        Add new observational dataset.

        Args:
            data (Data): new observational dataset.
        """
        self.obs_id += 1
        id = ('obs', self.obs_id)
        CP.info(f"\n## Building DBN for DAG ID {str(id)}")
        
        recycle = {}
        for existing_id, d in self.Ds.items():
            full_d = d["complete"]
            for node in full_d.features:
                if node in recycle: continue
                if np.array_equal(full_d.d[node].values, data.d[node].values):
                    CP.info(f"Recycling node: {node} from DBN ID {str(existing_id)}")
                    recycle[node] = {
                        'dbn': self.DBNs[existing_id].dbn[node],
                        'data': self.DBNs[existing_id].data[node],
                        'do': self.DBNs[existing_id].DO[node]
                    }
                    continue
        if not recycle: recycle = None
    
        self.DBNs[id] = DynamicBayesianNetwork(self.DAG['complete'], data, self.data_type, self.node_type, recycle, max_components = self.max_components)
        self.Ds[id] = {"complete": data, "specific": self.DBNs[id].data}
        return id
           
    
    def save(self, respath):
        """
        Save a CausalInferenceEngine object from a pickle file.

        Args:
            respath (str): pickle save path.
        """
        pkl = dict()
        pkl['DAG'] = self.DAG
        pkl['Ds'] = self.Ds
        pkl['DBNs'] = self.DBNs
        pkl['data_type'] = self.data_type 
        pkl['node_type'] = self.node_type 
        pkl['max_components'] = self.max_components
        pkl['model_path'] = self.model_path
        pkl['verbosity'] = CP.verbosity
        pkl['contexts'] = self.contexts
        pkl['obs_id'] = self.obs_id
        pkl['int_id'] = self.int_id

        with open(respath, 'wb') as resfile:
            pickle.dump(pkl, resfile)    
    
    @classmethod
    def load(cls, pklpath):
        """
        Load a CausalInferenceEngine object from a pickle file.

        Args:
            pklpath (str): pickle filepath.

        Returns:
            CausalInferenceEngine: loaded CausalInferenceEngine object.
        """
        with open(pklpath, 'rb') as f:
            pkl = pickle.load(f)
            cie = cls(pkl['DAG']['complete'], pkl['data_type'], pkl['node_type'], pkl['max_components'], pkl['model_path'], pkl['verbosity'])
            cie.contexts = pkl['contexts']
            cie.obs_id = pkl['obs_id']
            cie.int_id = pkl['int_id']
            cie.Ds = pkl['Ds']
            cie.DBNs = pkl['DBNs']
            return cie
           
    
    def whatIf(self, 
               treatment: str, 
               values: np.array, 
               data: np.array, 
               prior_knowledge: Dict[str, np.array] = None):
        """
        Predict system behaviours in response to a certain time-series intervention.

        Args:
            treatment (str): treatment variable.
            values (np.array): treatment values.
            data (np.array): data where to start to predict from.
            prior_knowledge (dict[str: np.array], optional): future prior knowledge about variables different from the treatment one. Defaults to None.

        Returns:
            np.array: prediction.
        """
        if prior_knowledge is not None and len(values) != len(list(prior_knowledge.values())[0]):
            raise ValueError("prior_knowledge items must have the same length of the treatment values")
        intT = len(values)
        res = np.full((intT + self.DAG['complete'].max_lag, len(self.DAG['complete'].features)), np.nan)
        res[:self.DAG['complete'].max_lag, :] = data[-self.DAG['complete'].max_lag:, :]
        
        if prior_knowledge is not None:
            for f, f_data in prior_knowledge.items():
                res[self.DAG['complete'].max_lag:, self.DAG['complete'].features.index(f)] = f_data
        
        # Build an interventional DAG where the treatment variable has no parents
        dag = self.remove_intVarParents(self.DAG['complete'], treatment)
        g = self._DAG2NX(dag)
        calculation_order = list(nx.topological_sort(g))
        for t in range(self.DAG['complete'].max_lag, intT + self.DAG['complete'].max_lag):
            self.Q[TREATMENT] = treatment
            self.Q[VALUE] = values[t-self.DAG['complete'].max_lag]
            
            for f, lag in calculation_order:
                if np.isnan(res[t - abs(lag), f]):
                    if self.DAG['complete'].features[f] == self.Q[TREATMENT]:
                        res[t, f] = self.Q[VALUE]
                    else:
                        var = self.DAG['system'].features.index(self.DAG['complete'].features[f])
                        system_p = {}
                        context_p = {}
                        pID = None

                        for s in self.DAG['system'].g[self.DAG['system'].features[var]].sources:
                            system_p[s[0]] = res[t - abs(s[1]), self.DAG['complete'].features.index(s[0])]
                        anchestors = self.DAG['complete'].get_anchestors(self.DAG['complete'].features[f])
                        context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                        for a in context_anchestors:
                            context_p[a] = int(res[t, self.DAG['complete'].features.index(a)])

                        # pID, occ = self._findSource_intersection(system_p)
                        pID, pContext, pSegment, occ = self._findSource_intersection(self.DAG['complete'].features[f], system_p, context_p)

                        res[t, f] = self.DBNs[pID].dbn[self.DAG['system'].features[var]][pContext]['full'].predict(system_p) if pID is not None else np.nan
                        
        return res[self.DAG['complete'].max_lag:, :]
    
    
    def _DAG2NX(self, dag: DAG) -> nx.DiGraph:
        G = nx.DiGraph()

        # 1. Nodes definition
        for i in range(len(dag.features)):
            for j in range(dag.max_lag, -1, -1):
                G.add_node((i, -j))

        # 2. Edges definition
        edges = list()
        for t in dag.g:
            for s in dag.g[t].sources:
                s_index = dag.features.index(s[0])
                t_index = dag.features.index(t)
                
                if s[1] == 0:
                    for j in range(dag.max_lag, -1, -1):
                        s_node = (s_index, -j)
                        t_node = (t_index, -j)
                        edges.append((s_node, t_node))
                        
                else:
                    s_lag = -s[1]
                    t_lag = 0
                    while s_lag >= -dag.max_lag:
                        s_node = (s_index, s_lag)
                        t_node = (t_index, t_lag)
                        edges.append((s_node, t_node))
                        s_lag -= 1
                        t_lag -= 1
                    
        G.add_edges_from(edges)
                
        return G
    
        
    def _findSourceQ(self, Ds):
        """
        finds source population with maximum number of occurrences treatment = value

        Args:
            Ds (dict): dataset dictionary {id (str): d (Data)}

        Returns:
            tuple: number of occurrences, source dataset
        """
        occurrences = 0
        sourceP = None
        for id, d in Ds.items():
            indexes = np.where(np.isclose(d.d[self.Q[TREATMENT]], self.Q[VALUE], atol = self.resolutions[TREATMENT]))[0]
            if len(indexes) > occurrences: 
                occurrences = len(indexes)
                sourceP = id
                
        return occurrences, sourceP
    
    
    def _findSource_intersection(self, target, parents, context=None):
        """
        Find the source population with the maximum number of occurrences 
        of a given intersection of parent values.

        Args:
            parents_values (dict): Dictionary where keys are parent variable names 
                                and values are the desired values for those parents.

        Returns:
            tuple: number of occurrences, source dataset
        """
        max_occurrences = 0
        pID = None
        pContext = None
        pSegment = None
        
        context = format_combo(tuple(context.items()))
        
        def _find_occurrences(d, parents, atol):
            mask = np.ones(len(d.d), dtype=bool)
            for parent, value in parents.items():
                # mask &= np.isclose(d.d[parent], value, atol = atol)
                mask &= np.isclose(d.d[parent], value, atol = atol)
                # mask &= np.isclose(d.d[parent], value, atol = atol, rtol=atol*10)
                    
            # Count the number of occurrences where all parents match
            occurrences = np.sum(mask)
            return occurrences
        
        def compute_adaptive_atol(parent_values, data, scale_factor=0.05):
            """
            Compute an adaptive atol based on the variability of the parent values.

            Args:
                parent_values (dict): Dictionary of parent variable values.
                data (pd.DataFrame): Data containing the parent variables.
                scale_factor (float): Factor to scale the variability.

            Returns:
                float: Adaptive atol.
            """
            variances = []
            for parent in parent_values.keys():
                parent_data = data[parent].values
                variances.append(np.std(parent_data))  # Or use np.ptp(parent_data), np.median_absolute_deviation, etc.

            return scale_factor * np.mean(variances)  # Average variability scaled by the factor      
                        
        # return pID, pContext, pSegment, max_occurrences
        for id in self.Ds:
            if context not in self.Ds[id]['specific'][target]: continue
            d = self.Ds[id]['specific'][target][context]['full']
            adaptive_atol = compute_adaptive_atol(parents, d.d)
            occurrences = _find_occurrences(d, parents, adaptive_atol)
                
            # Update the best source if this dataset has more occurrences
            if occurrences > max_occurrences:
                max_occurrences = occurrences
                pID = id
                pContext = context
                pSegment = 'full'
                        
        if pID is None:
            max_samples = 0
            for id in self.Ds:
                if context not in self.Ds[id]['specific'][target]: continue
                d = self.Ds[id]['specific'][target][context]['full']
                num_samples = d.T
                if num_samples > max_samples:
                    max_samples = num_samples
                    pID = id
                    pContext = context
                    pSegment = 'full'
                        
        return pID, pContext, pSegment, max_occurrences
        
            
    def plot_pE(self, y, parent, density, expectation = None, show = False, path = None):
        plt.figure(figsize=(10, 6))
        plt.plot(y.aligndata, density, label='Density')
        if expectation is not None: plt.axvline(expectation, color='r', linestyle='--', label=f'Expectation = {expectation:.2f}')
        plt.xlabel(f'${y.varname}$')
        pa = []
        for p in parent:
            if p == self.Q[TREATMENT]: 
                pa.append(f'do(${self.Q[TREATMENT]}$ = {self.Q[VALUE]:.2f})')
            else:
                pa.append(f'${p}$')
        pa = ','.join(pa)
        plt.ylabel(f'p(${y.varname}$|{pa})')
        plt.legend()
        if show: 
            plt.show()
        else:
            plt.savefig(os.path.join(path, f'p({self.Q[OUTCOME]}|do({self.Q[TREATMENT]} = {str(self.Q[VALUE])})).png'))