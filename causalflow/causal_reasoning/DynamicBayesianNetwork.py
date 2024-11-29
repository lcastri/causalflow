import copy
import gc
import itertools
import numpy as np
import pandas as pd
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Utils import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.basics.constants import *
from typing import Dict

class DynamicBayesianNetwork():
    def __init__(self, 
                 dag: DAG, 
                 data: Data, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType]):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}.
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}.
        """
        self.data_type = data_type
        self.node_type = node_type
        
        self.dbn = {node: None for node in dag.g}
        for node in dag.g:
            Y = Process(data.d[node].to_numpy(), node, 0, self.data_type[node], self.node_type[node])
                                                    
            # Parent(s) X process
            X = {s[0]: Process(data.d[s[0]].to_numpy(), s[0], s[1], self.data_type[s[0]], self.node_type[s[0]])
                for s in dag.g[node].sources}
                    
            # Density params estimation
            parents_str = f" - parents {', '.join(list(X.keys()))}" if X else ""
            CP.info(f"\n    ### Target variable: {node}{parents_str}")

            self.dbn[node] = Density(Y, X if X else None)

        del dag, data
        gc.collect()
                    
    
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
        lag = min([key[1] for key in dag.g[outcome].sources.keys() if key[0] == treatment])
        
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
                
            # Compute the p(adjustment) density            
            for node in adjset: p_adj = p_adj * self.dbn[features[node[0]]].CondDensity
            
            # Compute the p(outcome|treatment,adjustment) density
            p_yxadj = self.dbn[outcome].CondDensity * self.dbn[treatment].CondDensity * p_adj
            p_xadj = self.dbn[treatment].CondDensity * p_adj
            p_y_given_xadj = p_yxadj / p_xadj
            
            # Compute the p(outcome|do(treatment)) and p(outcome|do(treatment),adjustment)*p(adjustment) densities
            if len(p_y_given_xadj.shape) > 2: 
                # Sum over the adjustment set
                p_y_do_x_adj = p_y_given_xadj * p_adj
                p_y_do_x = normalise(np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape))))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
            else:
                p_y_do_x_adj = p_y_given_xadj
                p_y_do_x = p_y_given_xadj
            
            self.dbn[outcome].DO[treatment][ADJ] = np.array(adjset, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ] = np.array(p_y_do_x_adj, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX] = np.array(p_y_do_x, dtype=np.float16)
            
            del adjset, p_y_do_x_adj, p_y_do_x, p_adj, p_yxadj, p_xadj, p_y_given_xadj
            gc.collect()