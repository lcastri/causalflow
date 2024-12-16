import numpy as np
import warnings
from tqdm import tqdm
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Utils import *
from typing import Dict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import multivariate_normal
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
            batch_size (int): Batch size.
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
        self.ParentJointDensity = None
        
        # Check if any density is None and run _preprocess() if needed
        self._preprocess()
            
        # Only compute densities if they were not provided
        self.PriorDensity = self.compute_prior()
        self.JointDensity = self.compute_joint()
        self.ParentJointDensity = self.compute_parent_joint()
            
        
    @property
    def MaxLag(self):
        """
        Return max time lag between target and all its parents.

        Returns:
            int: max time lag.
        """
        if self.parents is not None: 
            return max(p.lag for p in self.parents.values())
        return 0
        
        
    def _preprocess(self):
        """Preprocess the data to have all the same length by using the maxlag."""
        # target
        self.y.align(self.MaxLag)
        
        if self.parents is not None:
            # parents
            for p in self.parents.values():
                p.align(self.MaxLag)
                
   
    def fit_gmm(self, caller, data, standardize = True):
        """
        Fit a Gaussian Mixture Model (GMM) to the data, optionally standardizing it.

        Args:
            data (ndarray): Data to fit the GMM.
            standardize (bool): Whether to standardize the data before fitting.

        Returns:
            dict: Parameters of the GMM (means, covariances, weights).
        """        
        if standardize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        # Fit GMM as usual
        components = range(1, self.max_components + 1)
        aic = []
        bic = []

        for n in tqdm(components, desc=f"[INFO]:     - {caller} density"):
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm.fit(data)
            aic.append(gmm.aic(data))
            bic.append(gmm.bic(data))

        optimal_n_components = components[np.argmin(aic)]  # Or np.argmin(bic)
        CP.debug(f"          Optimal n.components: {optimal_n_components}")

        gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=42)
        gmm.fit(data)
        
        # Extract parameters
        gmm_params = {
            "means": gmm.means_,
            "covariances": gmm.covariances_,
            "weights": gmm.weights_,
        }

        # If standardized, adjust means and covariances back to original scale
        if standardize:
            gmm_params["means"] = scaler.inverse_transform(gmm.means_)
            gmm_params["covariances"] = [
                scaler.scale_[:, None] * cov * scaler.scale_[None, :]
                for cov in gmm.covariances_
            ]

        # Return adjusted parameters
        return gmm_params

            
    @staticmethod
    def get_density(x, params):
        """
        Query the density for a given point `x` using the GMM parameters.

        Args:
            x (ndarray): The point(s) at which to evaluate the density.
            params (dict): The GMM parameters (means, covariances, weights).

        Returns:
            ndarray: The computed density at the point(s) x.
        """
        density = np.zeros(x.shape[0])
        for k in range(len(params["weights"])):
            mvn = multivariate_normal(mean=params["means"][k].flatten(), cov=params["covariances"][k].flatten())
            density += params["weights"][k] * mvn.pdf(x)
        return density


    def compute_prior(self):
        """
        Compute the prior density p(y) using GMM.

        Returns:
            dict: GMM parameters for the prior density.
        """
        CP.info("    - Prior density", noConsole=True)
        return self.fit_gmm('Prior', self.y.aligndata)


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
        else:
            data = self.y.aligndata

        return self.fit_gmm('Joint', data)


    def compute_parent_joint(self):
        """
        Compute the joint density of the parents p(parents) using GMM.

        Returns:
            dict: GMM parameters for the parents' joint density.
        """
        CP.info("    - Parent joint density", noConsole=True)
        if self.parents:
            data = np.column_stack([p.aligndata for p in self.parents.values()])
            return self.fit_gmm('Parent Joint', data)
        return None

   
    def compute_conditional(self, parent_values):
        """
        Compute the conditional density p(y | parents) = p(y, parents) / p(parents) online.

        Args:
            parent_values (ndarray): Values of the parent variables (e.g., [p1, p2, ...]) for conditioning.

        Returns:
            dict: GMM parameters for the conditional density.
        """

        # Compute conditional GMM parameters dynamically
        conditional_params = {
            "means": [],
            "covariances": [],
            "weights": self.JointDensity["weights"]
        }

        for k in range(len(self.JointDensity["weights"])):
            # Extract joint parameters for component k
            mean_joint = self.JointDensity["means"][k]
            cov_joint = self.JointDensity["covariances"][k]

            # Split into parent and target components
            dim_y = 1
            mean_parents = mean_joint[dim_y:]
            mean_target = mean_joint[:dim_y]
            cov_pp = cov_joint[dim_y:, dim_y:]  # Covariance of parents
            cov_pp = cov_pp + 1e-6 * np.eye(cov_pp.shape[0]) # To ensure invertibility

            cov_yp = cov_joint[:dim_y, dim_y:]  # Cross-covariance between y and parents
            cov_yy = cov_joint[:dim_y, :dim_y]  # Covariance of y

            # Update conditional mean and covariance
            cond_mean = mean_target + cov_yp @ np.linalg.inv(cov_pp) @ (parent_values.flatten() - mean_parents.flatten())
            cond_cov = cov_yy - cov_yp @ np.linalg.inv(cov_pp) @ cov_yp.T

            conditional_params["means"].append(cond_mean)
            conditional_params["covariances"].append(cond_cov)

        # Stack means and covariances
        conditional_params["means"] = np.array(conditional_params["means"])
        conditional_params["covariances"] = np.array(conditional_params["covariances"])

        return conditional_params
   
    
    def predict(self, given_p: Dict[str, float] = None):
        """
        Predict the conditional density p(y | parents) and the most likely value of y.

        Args:
            given_p (Dict[str, float], optional): A dictionary of parent variable values (e.g., {"p1": 1.5, "p2": 2.0}).

        Returns:
            Tuple[np.ndarray, float]: The conditional density and the most likely value of y.
        """
        if self.parents is None:
            conditional_params = self.PriorDensity
        else:
            # Extract parent samples and match with given parent values
            parent_values = np.array([given_p[p] for p in self.parents.keys()]).reshape(-1, 1)
            conditional_params = self.compute_conditional(parent_values)
            
        expected_value = expectation_from_params(conditional_params['means'], conditional_params['weights'])

        return expected_value
