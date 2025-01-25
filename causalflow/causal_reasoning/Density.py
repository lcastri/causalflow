import numpy as np
import warnings
# from tqdm import tqdm
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
import causalflow.causal_reasoning.Utils as DensityUtils
from typing import Dict
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
# from scipy.stats import multivariate_normal
warnings.filterwarnings('ignore', category=ConvergenceWarning)
      

class Density():       
    def __init__(self, 
                 y: Process, 
                 parents: Dict[str, Process] = None,
                 max_components = 50):
        """
        Class constructor.

        Args:
            y (Process): target process.
            parents (Dict[str, Process], optional): Target's parents. Defaults to None.
        """
        self.y = y
        self.parents = parents
        self.max_components = max_components
        self.DO = {}

        if self.parents is not None:
            self.DO = {treatment: {ADJ: None, 
                                P_Y_GIVEN_DOX_ADJ: None, 
                                P_Y_GIVEN_DOX: None} for treatment in self.parents.keys()}
        
        # If precomputed densities are provided, set them directly
        self.PriorDensity = None
        self.JointDensity = None
        # self.ParentJointDensity = None
        
        # Check if any density is None and run _preprocess() if needed
        self._preprocess()
            
        # Only compute densities if they were not provided
        self.PriorDensity = self.compute_prior()
        self.JointDensity = self.compute_joint()
            
        
    def _preprocess(self):
        maxLag = DensityUtils.get_max_lag(self.parents)
        
        # target
        self.y.align(maxLag)
        
        if self.parents is not None:
            # parents
            for p in self.parents.values():
                p.align(maxLag)


    def compute_prior(self):
        """
        Compute the prior density p(y) using GMM.

        Returns:
            dict: GMM parameters for the prior density.
        """
        CP.info("    - Prior density", noConsole=True)
        return DensityUtils.fit_gmm(self.max_components, 'Prior', self.y.aligndata)


    def compute_joint(self):
        """
        Compute the joint density p(y, parents) using GMM.

        Returns:
            dict: GMM parameters for the joint density.
        """
        CP.info("    - Joint density", noConsole=True)
        if self.parents:
            processes = [self.y] + list(self.parents.values())
            data = np.column_stack([p.aligndata for p in processes])
            return DensityUtils.fit_gmm(self.max_components, 'Joint', data)
        else:
            return self.PriorDensity


   
    def predict(self, given_p: Dict[str, float] = None):
        """
        Predict the conditional density p(y | parents) and the expected value of y.

        Args:
            given_p (Dict[str, float], optional): A dictionary of parent variable values (e.g., {"p1": 1.5, "p2": 2.0}).

        Returns:
            float: Expected value of y.
        """
        if self.parents is None:
            conditional_params = self.PriorDensity
        else:
            # Extract parent samples and match with given parent values
            parent_values = np.array([given_p[p] for p in self.parents.keys()]).reshape(-1, 1)
            conditional_params = DensityUtils.compute_conditional(self.JointDensity, parent_values)
            
        # dens = Density.get_density(self.y.aligndata, conditional_params)
        # dens = dens / np.sum(dens)

        # Find the most likely value (mode)
        expected_value = DensityUtils.expectation_from_params(conditional_params['means'], conditional_params['weights'])

        # return dens, expected_value
        return expected_value
