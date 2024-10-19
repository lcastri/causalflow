import copy
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import numpy as np
import pandas as pd
from causalflow.causal_inference.deprecated.Density_old import Density
from causalflow.causal_inference.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.causal_inference.CausalInferenceEngine import Engine 

class CausalInferenceEngine():
    def __init__(self, dag: DAG, obs_data: Data):
        """
        CausalInferenceEngine contructer

        Args:
            dag (DAG): DAG to convert into a DBN
            data (Data): observational data
        """
        self.features = obs_data.features
        self.CE = Engine(dag, obs_data)      
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
            for node in adjset: p_adj = p_adj * self.CE.dbn[self.features.index(node[0])].dens
            
            # Compute the P(outcome|treatment,adjustment) density
            p_yxadj = self.CE.dbn[self.outcome_var].dens * self.CE.dbn[treatment].dens * p_adj
            p_xadj = self.CE.dbn[treatment].dens * p_adj
            p_y_given_xadj = p_yxadj / p_xadj
            
            # Compute the P(outcome|do(treatment)) density
            p_y_do_x = p_y_given_xadj * p_adj
            
            indices_X = None
            column_indices = np.where(np.isclose(self.CE.d.d[:, treatment], value, atol=0.25))[0]
            if indices_X is None:
                indices_X = set(column_indices)
            else:
                # The intersection is needed to take the common indices 
                indices_X = indices_X.intersection(column_indices)
            
            indices_X = np.array(sorted(indices_X))
            
            X_dens = np.zeros_like(Density.estimate(self.CE.d.d[treatment]))
            zero_array = np.zeros_like(X_dens)
            p_y_do_X_x = copy.deepcopy(p_y_do_x)
            p_y_do_X_x[~np.isin(np.arange(len(X_dens)), indices_X)] = zero_array[~np.isin(np.arange(len(X_dens)), indices_X)]
            p_y_do_X_x = p_y_do_X_x.reshape(-1, 1)

            # TODO: once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
            return p_y_do_X_x, self.CE.dbn[self.outcome_var].expectation(p_y_do_X_x)