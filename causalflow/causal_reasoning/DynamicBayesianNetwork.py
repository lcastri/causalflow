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
    def __init__(self, dag: DAG, data: Data, nsample: int, atol: float, data_type: Dict[str, DataType], use_gpu: bool = False):
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
        self.use_gpu = use_gpu
        
        self.dbn = {node: None for node in dag.g}
        for node in self.dbn:
            Y = Process(data.d[node].to_numpy(), node, 0, self.nsample, self.data_type[node])
            parents = self._extract_parents(node)
            if parents is None:
                CP.info(f"\n### Target variable: {node}")
            else:
                CP.info(f"\n### Target variable: {node} - parents {', '.join(list(parents.keys()))}")
            Density = Density_GPU if self.use_gpu else Density_CPU
            self.dbn[node] = Density(Y, parents, atol)
        
        for node in self.dbn: self.computeDoDensity(node)
        
        
    def _extract_parents(self, node):
        """
        Extract the parents of a specified node.

        Args:
            node (str): Node belonging to the dag.

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)].
        """
        parents = {s[0]: Process(self.data.d[s[0]].to_numpy(), s[0], s[1], self.nsample, self.data_type[s[0]]) for s in self.dag.g[node].sources}
        if not parents: return None
        return parents
    
    
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
        CP.info(f"\n### DO Densities - Outcome {outcome}")
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
        
        
