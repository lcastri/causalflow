import ast
import copy
import gc
import itertools
import json
import numpy as np
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Utils import *
from causalflow.basics.constants import SampleMode
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.basics.constants import *
from typing import Dict
from scipy.stats import entropy
import h5py

class DynamicBayesianNetwork():
    def __init__(self, 
                 filename, id, 
                 dag: DAG, 
                 data: Data, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType], 
                 dbn = None):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}.
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}.
            dbn (DynamicBayesianNetwork): DBN to load. Default None.
        """

        self.data_type = data_type
        self.node_type = node_type
        
        if dbn is None:
            self.dbn = {node: None for node in dag.g}
            for node in dag.g:
                if self.node_type[node] == NodeType.Context: continue
                anchestors = dag.get_node_anchestors(node)
                context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                contexts = self._extract_contexts(data, context_anchestors)
                for context in contexts:
                    # Retrieve data for each node
                    d = self.get_context_specific_data(data, context, node, dag.g[node].sourcelist)      
                    
                    Y = Process(d[node].to_numpy(), node, 0, self.data_type[node], self.node_type[node])
                    X = self._extract_parents(node, d, dag, self.data_type, self.node_type)

                    parents_str = f" - parents {', '.join(list(X.keys()))}" if X else ""
                    CP.info(f"\n    ### Target variable: {node}{parents_str}")
                    CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
                    if self.dbn[node] is None: self.dbn[node] = {}
                    self.dbn[node][context] = Density(Y, X)
                    # self.save(filename, id, node, context)
                
        else:
            self.dbn = dbn          
              
        del dag, data
        gc.collect()
        
        
    def save(self, filename: str, id, node, context = None):
        """Save the DBN object to an HDF5 file."""
        with h5py.File(filename, 'a') as f:
            group_name = f"DBNs/{id}/{node}/{context}" if context is not None else f"DBNs/{id}/{node}"
                
            if group_name in f: del f[group_name]
            node_group = f.require_group(group_name)
            density = self.dbn[node][context] if context is not None else self.dbn[node]
                        
            node_group.create_dataset("PriorDensity", data=json.dumps(density.PriorDensity), dtype=h5py.special_dtype(vlen=str))
            node_group.create_dataset("JointDensity", data=json.dumps(density.JointDensity), dtype=h5py.special_dtype(vlen=str)) #density.JointDensity)
            if density.parents is not None:
                node_group.create_dataset("ParentJointDensity", data=json.dumps(density.ParentJointDensity), dtype=h5py.special_dtype(vlen=str)) #density.ParentJointDensity)
            else:
                node_group.create_dataset("ParentJointDensity", data=[])
            # node_group.create_dataset("CondDensity", data=density.CondDensity)
            # node_group.create_dataset("MarginalDensity", data=density.MarginalDensity)
            
    @property
    def isThereContext(self):
        return any(n is NodeType.Context for n in self.node_type.values())

    @staticmethod
    def estimate_optimal_samples(data, mode=SampleMode.Entropy):
        """
        Allocate samples based on the entropy or variance of each variable.
        
        Parameters:
            data (DataFrame): Data obj.
            mode (Sample mode): 'entropy': Allocate samples based on the entropy of each variable.
                                'variance': Allocate samples based on the variance of each variable.
                                'full': Allocate all samples to each variable.
        
        Returns:
            optimal_samples : dict
                Sample sizes for each variable based on the selected mode.
        """
        optimal_samples = {}
        tolerance = 0.3
        
        for node in data.columns:
            series = data[node].dropna()
            if mode is SampleMode.Entropy:
                _, counts = np.unique(series, return_counts=True)
                prob_dist = counts / len(series)
                series_entropy = entropy(prob_dist) 

                # Estimate sample size based on entropy
                sample_size = int(np.ceil(series_entropy / tolerance ** 2))  
            
            elif mode is SampleMode.Variance:
                variance = np.var(series)  # Calculate the variance of the series
                std_dev = np.sqrt(variance)  # Standard deviation

                # Estimate sample size based on variance (rule of thumb)
                sample_size = int(np.ceil(std_dev ** 2 / tolerance ** 2))  # Adjust based on the desired tolerance
            
            elif mode is SampleMode.Full:
                return {feature: data.shape[0] for feature in data.columns}
                
            sample_size = min(max(sample_size, 10), len(data))
            optimal_samples[node] = sample_size
        
        return optimal_samples
        
        
    def _extract_parents(self, node, data, dag: DAG, data_type, node_type):
        """
        Extract the parents of a specified node.

        Args:
            node (str): Node belonging to the dag.

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)].
        """
        parents = {s[0]: Process(data[s[0]].to_numpy(), s[0], s[1], data_type[s[0]], node_type[s[0]]) 
                   for s in dag.g[node].sources if self.node_type[s[0]] is not NodeType.Context}
        if not parents: return None
        return parents
    
    
    def get_context_specific_data(self, data, context, node, parents):
        context_dict = dict(context)

        # Filter the dataframe based on the context dictionary
        filtered_data = data.d
        for key, value in context_dict.items():
            filtered_data = filtered_data[filtered_data[key] == value]

        system_parents = list(set([node] + [p for p in parents]))
        system_parents = [p for p in system_parents if self.node_type[p] is not NodeType.Context]
        # Check if the filtered data is non-empty (i.e., if the context combination exists)
        return filtered_data[system_parents]
        
        
    def _extract_contexts(self, data, contexts):
        res = {}
        for n, ntype in self.node_type.items():
            if n in contexts and ntype is NodeType.Context:
                res[n] = np.array(np.unique(data.d[n]), dtype=int if self.data_type[n] is DataType.Discrete else float)
        
        # Generate the combinations
        tmp = list(itertools.product(*[[(k, float(v) if self.data_type[k] is DataType.Continuous else int(v)) for v in values] for k, values in res.items()]))
        return [format_combo(c) for c in tmp]
    
    
    def get_lag(self, treatment: str, outcome: str, dag: DAG):
        """
        Output the lag-time associated to the treatment -> outcome link.

        Args:
            treatment (str): treatment variable.
            outcome (str): outcome variable.

        Returns:
            int: treatment -> outcome link's lag-time.
        """
        matching_keys = [key[1] for key in dag.g[outcome].sources.keys() if key[0] == treatment]
        # if multiple, here it is returned only the minimum lag (closest to 0)
        return min(matching_keys)

    
    def get_adjset(self, treatment: str, outcome: str, dag: DAG):
        """
        Output the optimal adjustment set associated to the treatment-outcome intervention.
        
        The adjustment set is calculated through the TIGRAMITE pkg based on [1]
        
        [1] Runge, Jakob. "Necessary and sufficient graphical conditions for optimal adjustment 
            sets in causal graphical models with hidden variables." Advances in Neural Information 
            Processing Systems 34 (2021): 15762-15773.  

        Args:
            treatment (str): treatment variable.
            outcome (str): outcome variable.

        Returns:
            tuple: optimal adjustment set for the treatment -> outcome link.
        """
        lag = self.get_lag(treatment, outcome, dag)
        
        graph = CausalEffects.get_graph_from_dict(dag.get_Adj(indexed=True), tau_max = dag.max_lag)
        opt_adj_set = CausalEffects(graph, graph_type='stationary_dag', 
                                    X = [(dag.features.index(treatment), -abs(lag))], 
                                    Y = [(dag.features.index(outcome), 0)]).get_optimal_set()
        del graph
        return opt_adj_set
    

    def compute_do_density(self, outcome: str, features, dag: DAG):
        """
        Compute the p(outcome|do(treatment)) density.

        Args:
            outcome (str): outcome variable.
        """
        CP.info(f"\n### DO Densities - Outcome {outcome}")
        if self.dbn[outcome].parents is None: return
        for treatment in self.dbn[outcome].parents:
            CP.info(f"- Treatment {treatment}")
                    
            # Select the adjustment set
            adjset = self.get_adjset(treatment, outcome, dag)
                        
            # Compute the adjustment density
            p_adj = np.ones((self.nsample, 1)).squeeze()
            
            for node in adjset: p_adj = p_adj * self.dbn[features[node[0]]].CondDensity # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_adj = normalise(p_adj)
            
            # Compute the p(outcome|treatment,adjustment) density
            p_yxadj = normalise(self.dbn[outcome].CondDensity * self.dbn[treatment].CondDensity * p_adj) # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_xadj = normalise(self.dbn[treatment].CondDensity * p_adj) # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_y_given_xadj = normalise(p_yxadj / p_xadj)
            
            # Compute the p(outcome|do(treatment)) and p(outcome|do(treatment),adjustment)*p(adjustment) densities
            if len(p_y_given_xadj.shape) > 2: 
                # Sum over the adjustment set
                p_y_do_x_adj = normalise(p_y_given_xadj * p_adj)
                p_y_do_x = normalise(np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape))))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
            else:
                p_y_do_x_adj = p_y_given_xadj
                p_y_do_x = p_y_given_xadj
            
            self.dbn[outcome].DO[treatment][ADJ] = np.array(adjset, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ] = np.array(p_y_do_x_adj, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX] = np.array(p_y_do_x, dtype=np.float16)
            
            del adjset, p_y_do_x_adj, p_y_do_x, p_adj, p_yxadj, p_xadj, p_y_given_xadj
            gc.collect()