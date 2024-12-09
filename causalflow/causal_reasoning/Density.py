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
      
from causalflow.causal_reasoning.EM import EM


class Density():       
    def __init__(self, 
                 y: Process, 
                 parents: Dict[str, Process] = None,
                 max_components = 25):
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
                
                
    def fit_gmm(self, caller, data, standardize=True):
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
        best_model_params = None
        best_n = None
        best_bic = float('inf')

        for n in tqdm(components, desc=f"[INFO]:     - {caller} density"):
            em = EM(n_components=n)
            em_params = em.fit(data)
            bic = self.compute_bic(data, em_params)
            if bic < best_bic:
                best_bic = bic
                best_n = n
                best_model_params = em_params
                
        # optimal_n_components = components[np.argmin(aic)]  # Or np.argmin(bic)
        CP.debug(f"          Optimal n.components: {best_n}")
        
        # If standardized, adjust means and covariances back to original scale
        if standardize:
            best_model_params["means"] = scaler.inverse_transform(best_model_params['means'])
            best_model_params["covariances"] = [
                scaler.scale_[:, None] * cov * scaler.scale_[None, :]
                for cov in best_model_params['covariances']
            ]

        # Return parameters
        return best_model_params
    
    
    def compute_bic(self, data, gmm_params):
        """
        Compute the Bayesian Information Criterion (BIC) for a GMM.

        Args:
            data (ndarray): The dataset (N x D) where N is the number of samples and D is the dimensionality.
            gmm_params (dict): A dictionary containing the GMM parameters:
                - means (ndarray): Means of the Gaussian components (K x D).
                - covariances (list of ndarray): Covariance matrices of the Gaussian components (K x D x D).
                - weights (ndarray): Weights of the Gaussian components (K).

        Returns:
            float: The BIC score for the GMM.
        """
        means = gmm_params["means"]
        covariances = gmm_params["covariances"]
        weights = gmm_params["weights"]

        N, D = data.shape  # Number of samples (N) and dimensionality (D)
        K = len(weights)   # Number of components

        # Compute log-likelihood of the data under the GMM
        log_likelihood = 0
        for i in range(N):
            prob = 0
            for k in range(K):
                mean = means[k]
                cov = covariances[k]
                weight = weights[k]

                # Multivariate Gaussian PDF
                diff = data[i] - mean
                exp_term = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)
                norm_term = weight / np.sqrt((2 * np.pi) ** D * np.linalg.det(cov))
                prob += norm_term * exp_term

            epsilon = 1e-10  # A small value to prevent log(0)
            log_likelihood += np.log(prob + epsilon)

        # Compute the number of free parameters in the model
        num_params = K * D  # Parameters for the means
        num_params += K * D * (D + 1) // 2  # Parameters for the covariance matrices (symmetric)
        num_params += K - 1  # Parameters for the weights (K-1 independent weights)

        # Compute BIC
        bic = -2 * log_likelihood + num_params * np.log(N)
        return bic

            
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
            
        dens = Density.get_density(self.y.aligndata, conditional_params)
        dens = dens / np.sum(dens)

        # Find the most likely value (mode)
        most_likely = mode(self.y.aligndata.flatten(), dens)
        expected_value = expectation(self.y.aligndata.flatten(), dens)

        return dens, most_likely, expected_value
