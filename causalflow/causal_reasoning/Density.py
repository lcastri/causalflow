from tqdm import tqdm
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from typing import Dict
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Utils import *
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def fit_and_evaluate(data, n):
    """
    Fit a GMM with a specified number of components and compute AIC and BIC.

    Args:
        data (ndarray): Data to fit the GMM.
        n (int): Number of components.

    Returns:
        Tuple[float, float]: AIC and BIC for the fitted GMM.
    """
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(data)
    return gmm.aic(data), gmm.bic(data)

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
            
            
    def fit_gmm(self, caller, data):
        """
        Fit a Gaussian Mixture Model (GMM) to the data.

        Args:
            data (ndarray): Data to fit the GMM.
            n_components (int): Number of Gaussian components.

        Returns:
            dict: Parameters of the GMM (means, covariances, weights).
        """
        # Range of components to test
        components = range(1, self.max_components + 1)

        aic = []
        bic = []

        for n in tqdm(components, desc=f"[INFO]:     - {caller} density"):
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm.fit(data)
            aic.append(gmm.aic(data))
            bic.append(gmm.bic(data))

        # Choose the optimal number of components based on the minimum AIC or BIC
        optimal_n_components_aic = components[np.argmin(aic)]
        optimal_n_components_bic = components[np.argmin(bic)]
        CP.debug(f"    Optimal n.components (AIC): {optimal_n_components_aic} (BIC): {optimal_n_components_bic}")
        
        # Choose the optimal number of components (you can choose AIC or BIC-based on preference)
        optimal_n_components = optimal_n_components_aic  # Or use BIC: optimal_n_components_bic
        
        # Fit the GMM with the chosen optimal number of components
        gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=42)
        gmm.fit(data)
        
        # Return the parameters as a dictionary
        return {
            "means": gmm.means_,
            "covariances": gmm.covariances_,
            "weights": gmm.weights_
        }
            
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
        # Compute the density using the GMM parameters
        density = np.zeros(x.shape[0])
        for k in range(len(params["weights"])):
            mvn = multivariate_normal(mean=params["means"][k].flatten(), cov=params["covariances"][k].flatten())
            # mvn = multivariate_normal(mean=params["means"][k], cov=params["covariances"][k])
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
            
        dens = Density.get_density(self.y.aligndata, conditional_params)
        dens = dens / np.sum(dens)

        # Find the most likely value (mode)
        most_likely = mode(self.y.aligndata, dens)
        # expected_value = expectation(self.y.aligndata, dens)
        expected_value = 0

        return dens, most_likely, expected_value
