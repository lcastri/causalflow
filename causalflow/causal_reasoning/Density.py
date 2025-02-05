import numpy as np
import warnings
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
import causalflow.causal_reasoning.DensityUtils as DensityUtils
from typing import Dict
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
      

class Density():       
    def __init__(self, 
                 y: Process, 
                 parents: Dict[str, Process] = None,
                 max_components = 50,
                 pY = None, pJoint = None):
        """
        Class constructor.

        Args:
            y (Process): target process.
            parents (Dict[str, Process], optional): Target's parents. Defaults to None.
        """
        self.y = y
        self.parents = parents
        self.max_components = max_components

        # If precomputed densities are provided, set them directly
        self.pY = None
        self.pJoint = None
        
        # Check if any density is None and run _preprocess() if needed
        self._preprocess()
            
        # Only compute densities if they were not provided
        self.pY = self.get_pY() if pY is None else pY
        self.pJoint = self.get_pYX() if pJoint is None else pJoint
            
        
    def _preprocess(self):
        maxLag = DensityUtils.get_max_lag(self.parents)
        
        # target
        self.y.align(maxLag)
        
        if self.parents is not None:
            # parents
            for p in self.parents.values():
                p.align(maxLag)


    def get_pY(self):
        """
        Compute the prior density p(y) using GMM.

        Returns:
            dict: GMM parameters for the prior density.
        """
        CP.info("    - p(Y) density", noConsole=True)
        return DensityUtils.fit_gmm(self.max_components, 'p(Y)', self.y.aligndata)


    def get_pYX(self):
        """
        Compute the joint density p(y, parents) using GMM.

        Returns:
            dict: GMM parameters for the joint density.
        """
        CP.info("    - p(YX) density", noConsole=True)
        if self.parents:
            processes = [self.y] + list(self.parents.values())
            data = np.column_stack([p.aligndata for p in processes])
            return DensityUtils.fit_gmm(self.max_components, 'p(YX)', data)
        else:
            return self.pY


   
    def get_pY_gX(self, given_p: Dict[str, float] = None):
        """
        Predict the conditional density p(y | parents) and the expected value of y.

        Args:
            given_p (Dict[str, float], optional): A dictionary of parent variable values (e.g., {"p1": 1.5, "p2": 2.0}).

        Returns:
            float: Expected value of y.
        """
        if self.parents is None:
            conditional_params = self.pY
        else:
            # Extract parent samples and match with given parent values
            parent_values = np.array([given_p[p] for p in self.parents.keys()]).reshape(-1, 1)
            conditional_params = DensityUtils.compute_conditional(self.pJoint, parent_values)

        return conditional_params
    
    
    def predict(self, given_p: Dict[str, float] = None):
        """
        Predict the conditional density p(y | parents) and the expected value of y.

        Args:
            given_p (Dict[str, float], optional): A dictionary of parent variable values (e.g., {"p1": 1.5, "p2": 2.0}).

        Returns:
            float: Expected value of y.
        """
        cond_params = self.get_pY_gX(given_p)

        # Find the most likely value (mode)
        expected_value = DensityUtils.expectation_from_params(cond_params['means'], cond_params['weights'])

        return expected_value
