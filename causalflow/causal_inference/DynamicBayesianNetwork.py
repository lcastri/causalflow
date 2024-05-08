import numpy as np
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_inference.Density import Density
from causalflow.causal_inference.Process import Process
import copy
from tigramite.causal_effects import CausalEffects



class DynamicBayesianNetwork():
    def __init__(self, dag: DAG, data: Data, nsample = 100):
        """
        DynamicBayesianNetwork contructer
        """
        self.dag = dag
        self.data = data
        self.nsample = nsample
        
        self.dbn = {node: None for node in dag.g}
        for node in self.dbn:
            Y = Process(data.d[node].to_numpy(), node, 0, self.nsample)
            parents = self._extract_parents(node)
            self.dbn[node] = Density(Y, parents)
            
        for node in self.dbn: self.computeDoDensity(node)
        
        
    def _extract_parents(self, node):
        """
        Extracts the parents of a specified node

        Args:
            node (str): Node belonging to the dag

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)]
        """
        parents = {s[0]: Process(self.data.d[s[0]].to_numpy(), s[0], s[1], self.nsample) for s in self.dag.g[node].sources}
        if not parents: return None
        return parents
    
    
    def get_lag(self, treatment: str, outcome: str):
        """
        Outputs the lag-time associated to the treatment -> outcome link

        Args:
            treatment (str): treatment variable
            outcome (str): outcome variable

        Returns:
            int: treatment -> outcome link's lag-time 
        """
        matching_keys = [key[1] for key in self.dag.g[outcome].sources.keys() if key[0] == treatment]
        # if multiple, here it is returned only the minimum lag (closest to 0)
        return min(matching_keys)

    
    def get_adjset(self, treatment: str, outcome: str, lag: int):
        """
        Outputs the optimal adjustment set associated to the treatment-outcome intervention.
        The adjustment set is calculated through the TIGRAMITE pkg based on [1]
        
        [1] Runge, Jakob. "Necessary and sufficient graphical conditions for optimal adjustment 
            sets in causal graphical models with hidden variables." Advances in Neural Information 
            Processing Systems 34 (2021): 15762-15773.  

        Args:
            treatment (str): treatment variable
            outcome (str): outcome variable
            lag (int): lag time where the intervention is performed.

        Returns:
            tuple: optimal adjustment set for the treatment->outcome link 
        """
        graph = CausalEffects.get_graph_from_dict(self.dag.get_parents(), tau_max = self.dag.max_lag)
        opt_adj_set = CausalEffects(graph, graph_type='stationary_dag', 
                                    X = [(self.dag.features.index(treatment), -abs(lag))], 
                                    Y = [(self.dag.features.index(outcome), 0)]).get_optimal_set()
        return opt_adj_set
    
    
    # def computeDoDensity(self, outcome: str):
    #     """
    #     Computes the p(outcome|do(treatment)) density

    #     Args:
    #         outcome (str): outcome variable
    #     """
    #     if self.dbn[outcome].parents is None: return
    #     for treatment in self.dbn[outcome].parents:
                    
    #         # Select the adjustment set
    #         adjset = self.get_adjset(treatment, outcome, self.get_lag(treatment, outcome))
                        
    #         # Compute the adjustment density
    #         p_adj = np.ones((self.nsample, 1)).squeeze()
            
    #         for node in adjset: p_adj = p_adj * self.dbn[self.data.features[node[0]]].CondDensity # TODO: to verify if computed like this is equal to compute the joint density directly through KDE

    #         # Compute the p(outcome|treatment,adjustment) density
    #         p_yxadj = self.dbn[outcome].CondDensity * self.dbn[treatment].CondDensity * p_adj # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
    #         p_xadj = self.dbn[treatment].CondDensity * p_adj # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
    #         p_y_given_xadj = p_yxadj / p_xadj
            
    #         # Compute the p(outcome|do(treatment)) density
    #         if len(p_y_given_xadj.shape) > 2: 
    #             # Sum over the adjustment set
    #             p_y_do_x = np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape)))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
    #         else:
    #             p_y_do_x = p_y_given_xadj
            
    #         # p_y_do_x = p_y_given_xadj * p_adj
    #         self.dbn[outcome].DO[treatment] = p_y_do_x
    def computeDoDensity(self, outcome: str):
        """
        Computes the p(outcome|do(treatment)) density

        Args:
            outcome (str): outcome variable
        """
        if self.dbn[outcome].parents is None: return
        for treatment in self.dbn[outcome].parents:
                    
            # Select the adjustment set
            adjset = self.get_adjset(treatment, outcome, self.get_lag(treatment, outcome))
                        
            # Compute the adjustment density
            p_adj = np.ones((self.nsample, 1)).squeeze()
            
            for node in adjset: p_adj = p_adj * self.dbn[self.data.features[node[0]]].CondDensity # TODO: to verify if computed like this is equal to compute the joint density directly through KDE

            # Compute the p(outcome|treatment,adjustment) density
            p_yxadj = self.dbn[outcome].CondDensity * self.dbn[treatment].CondDensity * p_adj # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_xadj = self.dbn[treatment].CondDensity * p_adj # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_y_given_xadj = p_yxadj / p_xadj
            
            # Compute the p(outcome|do(treatment)) density
            if len(p_y_given_xadj.shape) > 2: 
                # Sum over the adjustment set
                p_y_do_x = np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape)))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
            else:
                p_y_do_x = p_y_given_xadj
            
            self.dbn[outcome].DO[treatment]['adj'] = adjset
            self.dbn[outcome].DO[treatment]['p_y|do_x'] = p_y_do_x
            self.dbn[outcome].DO[treatment]['p_y|do(x)_adj'] = ...
            
            
    def evalDoDensity(self, treatment: str, outcome: str, value):
        """
        Evaluates the p(outcome|treatment = t)

        Args:
            treatment (str): treatment variable
            outcome (str): outcome variable
            value (float): treatment value

        Returns:
            tuple: p(outcome|treatment = t), E[p(outcome|treatment = t)]
        """
        indices_X = None
        column_indices = np.where(np.isclose(self.dbn[outcome].X[:, treatment], value, atol=0.25))[0]
        if indices_X is None:
            indices_X = set(column_indices)
        else:
            # The intersection is needed to take the common indices 
            indices_X = indices_X.intersection(column_indices)
        
        indices_X = np.array(sorted(indices_X))
        
        X_dens = np.zeros_like(self.dbn[treatment].MarginalDensity)
        zero_array = np.zeros_like(X_dens)
        p_y_do_X_x = copy.deepcopy(self.dbn[outcome].DO[treatment])
        p_y_do_X_x[~np.isin(np.arange(len(X_dens)), indices_X)] = zero_array[~np.isin(np.arange(len(X_dens)), indices_X)]
        p_y_do_X_x = p_y_do_X_x.reshape(-1, 1)
        
        # TODO: once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
        return p_y_do_X_x, self.dbn[outcome].expectation(p_y_do_X_x)
        
        
