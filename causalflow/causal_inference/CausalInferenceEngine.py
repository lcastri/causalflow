from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
import numpy as np
import pandas as pd
from causalflow.causal_inference.Density import Density
from causalflow.causal_inference.Process import Process
from tigramite.causal_effects import CausalEffects
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
    
    
    def If(self, treatment, value):
        """
        finalises the query taking in input the treatment variable and its value

        Args:
            treatment (str): treatment variable
            value (float): treatment value
        """
        # TODO: check if data for this intervention already exists
        if ('int_' + str(treatment)) in self.CE.Ds:
            # TODO: search for the most similar intervention. if not present, go to else
            # TODO: apply the transportability formula to estimate the effect of the intervention on the desired population
            pass
        else:
            # TODO: compute the adjustment set from the treatment variable to the outcome variable and compute the cause-effect by back/front-door criterion
            # TODO: once found the estimated cause-effect, use the transportability formula to estimate the effect of the intervention on the desired population 
            pass
            
    
