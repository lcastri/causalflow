from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import numpy as np
import pandas as pd
from causalflow.causal_inference.Density import Density
from causalflow.causal_inference.Process import Process
from tigramite.causal_effects import CausalEffects
# from tigramite.causal_effects.CausalE import CausalEffects
from causalflow.causal_inference.CausalEngine import CausalEngine 

class CausalInferenceEngine():
    def __init__(self, dag: DAG, obs_data: Data):
        """
        CausalInferenceEngine contructer

        Args:
            dag (DAG): DAG to convert into a DBN
            data (Data): observational data
        """
        self.CE = CausalEngine(dag, obs_data)      
        self.outcome_var = None
    
    # TODO:  this class should work as interface between the user and the causal engine. It must: 
    # - take the desired query in input
    # - check if this type of intervention has been already performed
    #   - if so, apply the transportability formula to estimate the effect of the intervention on the desired population
    #   - still to figure out how to compute the covariate set to use in the transportability formula to re-weight the effect
    # - if that intervention is not in our 'engine' (it has not been performed yet)
    #   - compute the adjustment set from the treatment variable to the outcome variable and compute the cause-effect by back/front-door criterion
    #   - once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
        
    
    def whatHappensTo(self, outcome):
        """
        initialises the query taking in input the outcome variable

        Args:
            outcome (str): outcome variable

        Returns:
            CausalInferenceEngine: self
        """
        self.outcome_var = outcome
        return self
    
    
    @staticmethod
    def get_lag(dag: DAG, treatment: str, outcome: str):
        """
        outputs the lag-time associated to the treatment-outcome link

        Args:
            dag (DAG): DAG
            treatment (str): treatment variable
            outcome (str): outcome variable

        Returns:
            int: treatment-outcome link's lag-time 
        """
        matching_keys = [key[1] for key in dag.g[treatment].sources.keys() if key[0] == outcome]
        # if multiple, here it is returned only the minimum lag (closest to 0)
        return min(matching_keys)
    
    
    @staticmethod
    def get_adjset(dag: DAG, treatment: str, outcome: str, lag: int):
        """
        outputs the optimal adjustment set associated to the treatment-outcome intervention.
        The adjustment set is calculated through the TIGRAMITE pkg based on [1]
        
        [1] Runge, Jakob. "Necessary and sufficient graphical conditions for optimal adjustment 
            sets in causal graphical models with hidden variables." Advances in Neural Information 
            Processing Systems 34 (2021): 15762-15773.  

        Args:
            dag (DAG): DAG
            treatment (str): treatment variable
            outcome (str): outcome variable
            lag (int): lag time where the intervention is performed.

        Returns:
            int: treatment-outcome link's lag-time 
        """
        graph = CausalEffects.get_graph_from_dict(dag.get_parents(), tau_max = dag.max_lag)
        opt_adj_set = CausalEffects(graph, graph_type='stationary_dag', 
                                    X = [(dag.features.index(treatment), -abs(lag))], 
                                    Y = [(dag.features.index(outcome), 0)]).get_optimal_set()
        return opt_adj_set
    
    
    def If(self, treatment, value, lag = None):
        """
        finalises the query taking in input the treatment variable and its value

        Args:
            treatment (str): treatment variable
            value (float): treatment value
            lag (int): lag time where the intervention is performed. If None, it is retrieved from the graph
        """
        
        if lag is None: lag = self.get_lag(self.CE.dag, treatment, self.outcome_var)
        
        # TODO: check if data for this intervention already exists
        if ('int_' + str(treatment)) in self.CE.Ds:
            # TODO: search for the most similar intervention. if not present, go to else
            # TODO: apply the transportability formula to estimate the effect of the intervention on the desired population
            pass
        else:
            # Select the adjustment set
            adjset = self.get_adjset(self.CE.dag, treatment, self.outcome_var, lag)
            
            # Compute the adjustment density
            p_adj = 1
            for node in adjset: p_adj = p_adj * self.CE.dbn[node[0]].cond_density
            
            # Compute the P(outcome|treatment,adjustment) density
            p_yxadj = self.CE.dbn[self.outcome_var].cond_density * self.CE.dbn[treatment].cond_density * p_adj
            p_xadj = self.CE.dbn[treatment].cond_density * p_adj
            p_y_given_xadj = p_yxadj / p_xadj
            
            # Compute the P(outcome|do(treatment)) density
            # TODO: compute p_y_do_x for a particular value of X
            p_y_do_x = np.sum(p_y_given_xadj * p_adj)
            
            # TODO: once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
            return p_y_do_x
            
    

            
    # def If(self, given_p: Dict[str, float]):
    #     if self.parents is None: 
    #         dens = self.MarginalDensity()
    #     else:
    #         # self.given_p = given_p
    #         indices_X = None
    #         for p in given_p.keys():
    #             # if p not in self.parents:
    #             #     marginal_density = self._get_marginal_density()
    #             #     return marginal_density, self._expectation(marginal_density)
    #             column_indices = np.where(np.isclose(self.X[:, list(self.parents.keys()).index(p)], given_p[p], atol=0.25))[0]
                
    #             if indices_X is None:
    #                 indices_X = set(column_indices)
    #             else:
    #                 indices_X = indices_X.intersection(column_indices)
            
    #         indices_X = np.array(sorted(indices_X)) 

    #         zero_array = np.zeros_like(self.ParentMarginalDensity())
    #         eval_cond_density = deepcopy(self.ConditionalDensity())
    #         eval_cond_density[~np.isin(np.arange(len(self.ParentMarginalDensity())), indices_X)] = zero_array[~np.isin(np.arange(len(self.ParentMarginalDensity())), indices_X)]
    #         dens = eval_cond_density.reshape(-1, 1)
            
    #     return dens, self._expectation(dens)