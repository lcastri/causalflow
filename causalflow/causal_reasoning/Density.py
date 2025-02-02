import numpy as np
import warnings
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
import causalflow.causal_reasoning.Utils as DensityUtils
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
        self.pY = self.compute_pY() if pY is None else pY
        self.pJoint = self.compute_pY_X() if pJoint is None else pJoint
            
        
    def _preprocess(self):
        ALL = {}
        ALL.update({self.y.varname: self.y})
        if self.parents is not None:
            ALL.update(self.parents)
        maxLag = DensityUtils.get_max_lag(ALL)
        
        # target
        self.y.align(maxLag)
        
        if self.parents is not None:
            # parents
            for p in self.parents.values():
                p.align(maxLag)


    def compute_pY(self):
        """
        Compute the prior density p(y) using GMM.

        Returns:
            dict: GMM parameters for the prior density.
        """
        logstr = f"p({self.y.varname}_t)"
        CP.info(f"    - {logstr}]", noConsole=True)
        return DensityUtils.fit_gmm(self.max_components, logstr, self.y.aligndata)


    def compute_pY_X(self):
        """
        Compute the joint density p(y, parents) using GMM.

        Returns:
            dict: GMM parameters for the joint density.
        """
        parent_str = ','.join([f'{p.varname}_t{-abs(p.lag) if p.lag != 0 else ""}' for p in self.parents.values()]) if self.parents is not None else ''
        logstr = f"p({self.y.varname}_t{',' if self.parents is not None else ''}{parent_str})"
        CP.info(f"    - {logstr}", noConsole=True)
        if self.parents:
            processes = [self.y] + list(self.parents.values())
            data = np.column_stack([p.aligndata for p in processes])
            return DensityUtils.fit_gmm(self.max_components, logstr, data)
        else:
            return self.pY


   
    def predict(self, given_p: Dict[str, float] = None):
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

        # Find the most likely value (mode)
        expected_value = DensityUtils.expectation_from_params(conditional_params['means'], conditional_params['weights'])

        return expected_value
