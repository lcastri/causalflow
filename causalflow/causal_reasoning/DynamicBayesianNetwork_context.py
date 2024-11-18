import itertools
import numpy as np
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Density_utils import normalise
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density_CPU import Density as Density_CPU
from causalflow.causal_reasoning.Density_GPU import Density as Density_GPU
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.basics.constants import *
from typing import Dict

class DynamicBayesianNetwork():
    def __init__(self, dag: DAG, data: Data, nsample: int, data_type: Dict[str, DataType], node_type: Dict[str, NodeType], use_gpu: bool = False):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            nsample (int): Number of samples used for density estimation.
            atol (float): Absolute tolerance used to check if a specific intervention has been already observed.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}
            use_gpu (bool): If True, use GPU for density estimation; otherwise, use CPU.
        """
        self.dag = dag
        self.data = data
        self.nsample = nsample
        self.data_type = data_type
        self.node_type = node_type
        self.use_gpu = use_gpu
        Density = Density_GPU if self.use_gpu else Density_CPU
        
        self.dbn = {node: None for node in dag.g}
        for node in self.dbn:
            if self.node_type[node] is NodeType.Context: 
                CP.info(f"\n### Context variable: {node} -- skipped")
                continue
            if self.dbn[node] is None: self.dbn[node] = {}
            self.dbn[node]['contexts'] = []
            if self.dag.g[node].sources is None:
                CP.info(f"\n### Target variable: {node}")
                Y = Process(data.d[node].to_numpy(), node, 0, self.nsample, self.data_type[node], self.node_type[node])
                self.dbn[node] = Density(Y)
            else:
                c_parents = self._extract_cparents(node)
                if c_parents:
                    contexts = [(c, float(v) if c_parents[c].data_type is DataType.Continuous else int(v)) 
                                for c in c_parents for v in c_parents[c].sorted_samples]
                    combinations = itertools.combinations(contexts, len(c_parents))

                    for combo in combinations:
                        if len(set([c[0] for c in combo])) != len(combo): continue
                        if len(combo) == 1: 
                            combo = combo[0]
                            node_data, s_parents = self._extract_sparents(node, [combo])
                            CP.info(f"\n### Target variable: {node} - System parents {', '.join(list(s_parents.keys()))} - Context {', '.join([str(c[0]) + '=' + str(c[1]) for c in [combo]])}")
                        else:
                            combo = DynamicBayesianNetwork.get_combo(combo)
                            node_data, s_parents = self._extract_sparents(node, combo)
                            CP.info(f"\n### Target variable: {node} - System parents {', '.join(list(s_parents.keys()))} - Context {', '.join([str(c[0]) + '=' + str(c[1]) for c in combo])}")
                        Y = Process(node_data, node, 0, self.nsample, self.data_type[node], self.node_type[node])
                        self.dbn[node][combo] = Density(Y, s_parents)
                        self.dbn[node]['contexts'].append(combo)
                else:
                    s_parents = self._extract_sparents(node)
                    CP.info(f"\n### Target variable: {node} - System parents {', '.join(list(s_parents.keys()))}")
                    Y = Process(data.d[node].to_numpy(), node, 0, self.nsample, self.data_type[node], self.node_type[node])
                    self.dbn[node] = Density(Y, s_parents)
        
        # for node in self.dbn:
        #     if self.node_type[node] is NodeType.Context: 
        #         CP.info(f"\n### DO Densities - Context {node} -- skipped")
        #         continue
        #     CP.info(f"\n### DO Densities - Outcome {node}")
        #     self.computeDoDensity(node)
    
    @staticmethod
    def get_combo(combo):
        return tuple(sorted(combo))
        
        
    def _extract_cparents(self, node):
        """
        Extract the parents of a specified node.

        Args:
            node (str): Node belonging to the dag.

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)].
        """
        return {s[0]: Process(self.data.d[s[0]].to_numpy(), s[0], s[1], self.nsample, self.data_type[s[0]], self.node_type[s[0]]) 
                for s in self.dag.g[node].sources if self.node_type[s[0]] is NodeType.Context}
        
        
    def _extract_sparents(self, node, contexts = None):
        """
        Extract the parents of a specified node.

        Args:
            node (str): Node belonging to the dag.

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)].
        """
        if contexts is None:
            return {s[0]: Process(self.data.d[s[0]].to_numpy(), s[0], s[1], self.nsample, self.data_type[s[0]], self.node_type[s[0]]) 
                    for s in self.dag.g[node].sources if self.node_type[s[0]] is NodeType.System}
        
        sparents = {}

        parent_data = self.data.d.to_numpy()
        mask = np.ones(len(parent_data), dtype=bool)  # Start with all True (keep all rows)
        for c in contexts:
            context_column_index = self.data.features.index(c[0])
            mask &= (parent_data[:, context_column_index] == c[1])

        filtered_data = parent_data[mask]
        for s in self.dag.g[node].sources:
            if self.node_type[s[0]] is NodeType.System:
                sparents[s[0]] = Process(filtered_data[:, self.data.features.index(s[0])], s[0], s[1], self.nsample, self.data_type[s[0]], self.node_type[s[0]])

        return filtered_data[:, self.data.features.index(node)], sparents
    
    
    # def _extract_sparents(self, node):
    #     """
    #     Extract the parents of a specified node.

    #     Args:
    #         node (str): Node belonging to the dag.

    #     Returns:
    #         dict: parents express as dict[parent name (str), parent process (Process)].
    #     """
    #     parents = {s[0]: Process(self.data.d[s[0]].to_numpy(), s[0], s[1], self.nsample, self.data_type[s[0]], self.node_type[s[0]]) for s in self.dag.g[node].sources}
    #     if not parents: return None
    #     s_parents = {s: p for s, p in parents.items() if p.node_type is NodeType.System}
    #     c_parents = {c: p for c, p in parents.items() if p.node_type is NodeType.Context}
    #     return parents, s_parents, c_parents
    
    
    def get_lag(self, treatment: str, outcome: str):
        """
        Output the lag-time associated to the treatment -> outcome link.

        Args:
            treatment (str): treatment variable.
            outcome (str): outcome variable.

        Returns:
            int: treatment -> outcome link's lag-time.
        """
        matching_keys = [key[1] for key in self.dag.g[outcome].sources.keys() if key[0] == treatment]
        # if multiple, here it is returned only the minimum lag (closest to 0)
        return min(matching_keys)

    
    def get_adjset(self, treatment: str, outcome: str):
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
        lag = self.get_lag(treatment, outcome)
        
        graph = CausalEffects.get_graph_from_dict(self.dag.get_Adj(indexed=True), tau_max = self.dag.max_lag)
        opt_adj_set = CausalEffects(graph, graph_type='stationary_dag', 
                                    X = [(self.dag.features.index(treatment), -abs(lag))], 
                                    Y = [(self.dag.features.index(outcome), 0)]).get_optimal_set()
        return opt_adj_set
    
    
    def computeDoDensity(self, outcome: str):
        """
        Compute the p(outcome|do(treatment)) density.

        Args:
            outcome (str): outcome variable.
        """
        if len(self.dbn[outcome]['contexts']) == 0:
            if self.dbn[outcome].parents is None: return
            for treatment in self.dbn[outcome].parents:
                CP.info(f"- Treatment {treatment}")
                        
                # Select the adjustment set
                adjset = self.get_adjset(treatment, outcome)
                            
                # Compute the adjustment density
                p_adj = np.ones((self.nsample, 1)).squeeze()
                
                for node in adjset: p_adj = p_adj * self.dbn[self.data.features[node[0]]].CondDensity # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
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
                
                self.dbn[outcome].DO[treatment][ADJ] = adjset
                self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ] = p_y_do_x_adj
                self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX] = p_y_do_x
        else:
            for context in self.dbn[outcome]['contexts']:
                if self.dbn[outcome][context].parents is None: return
                for treatment in self.dbn[outcome][context].parents:
                    if len([context]) > 1:
                        CP.info(f"- Treatment {treatment} - Context {', '.join([str(c[0]) + '=' + str(c[1]) for c in context])}")
                    else:
                        CP.info(f"- Treatment {treatment} - Context {', '.join([str(c[0]) + '=' + str(c[1]) for c in [context]])}")
        
                    # Select the adjustment set
                    adjset = self.get_adjset(treatment, outcome)
                                
                    # Compute the adjustment density
                    p_adj = np.ones((self.nsample, 1)).squeeze()
                    
                    for node in adjset:
                        if self.node_type[self.data.features[node[0]]] is NodeType.Context: continue
                        if len(self.dbn[self.data.features[node[0]]]['contexts']) == 0:
                            p_adj = p_adj * self.dbn[self.data.features[node[0]]].CondDensity
                        else:
                            adj_context = self.dbn[self.data.features[node[0]]]['contexts']
                            common_context = [c for c in adj_context if c in context] if len([context]) > 1 else [c for c in adj_context if c in [context]]
                            common_context = DynamicBayesianNetwork.get_combo(common_context) if len(common_context) > 1 else common_context[0]
                            p_adj = p_adj * self.dbn[self.data.features[node[0]]][common_context].CondDensity
                    p_adj = normalise(p_adj)
                    
                    # Compute the p(outcome|treatment,adjustment) density
                    p_yxadj = normalise(self.dbn[outcome][context].CondDensity * self.dbn[treatment][context].CondDensity * p_adj) # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
                    p_xadj = normalise(self.dbn[treatment][context].CondDensity * p_adj) # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
                    p_y_given_xadj = normalise(p_yxadj / p_xadj)
                    
                    # Compute the p(outcome|do(treatment)) and p(outcome|do(treatment),adjustment)*p(adjustment) densities
                    if len(p_y_given_xadj.shape) > 2: 
                        # Sum over the adjustment set
                        p_y_do_x_adj = normalise(p_y_given_xadj * p_adj)
                        p_y_do_x = normalise(np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape))))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
                    else:
                        p_y_do_x_adj = p_y_given_xadj
                        p_y_do_x = p_y_given_xadj
                    
                    self.dbn[outcome][context].DO[treatment][ADJ] = adjset
                    self.dbn[outcome][context].DO[treatment][P_Y_GIVEN_DOX_ADJ] = p_y_do_x_adj
                    self.dbn[outcome][context].DO[treatment][P_Y_GIVEN_DOX] = p_y_do_x
            
        
