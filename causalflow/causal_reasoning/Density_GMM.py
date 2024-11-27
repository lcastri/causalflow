from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Process import Process
from typing import Dict
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Utils import *
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal
 

class Density():       
    def __init__(self, 
                 y: Process, 
                 batch_size: int, 
                 parents: Dict[str, Process] = None,
                 prior_density=None, 
                 joint_density=None, 
                 parent_joint_density=None, 
                 cond_density=None, 
                 marginal_density=None):
        """
        Class constructor.

        Args:
            y (Process): target process.
            batch_size (int): Batch size.
            parents (Dict[str, Process], optional): Target's parents. Defaults to None.
            prior_density (np.array, optional): Precomputed prior density. Defaults to None.
            joint_density (np.array, optional): Precomputed joint density. Defaults to None.
            parent_joint_density (np.array, optional): Precomputed parent joint density. Defaults to None.
            cond_density (np.array, optional): Precomputed conditional density. Defaults to None.
            marginal_density (np.array, optional): Precomputed marginal density. Defaults to None.
        """
        self.y = y
        self.batch_size = batch_size
        self.parents = parents
        self.DO = {}

        if self.parents is not None:
            self.DO = {treatment: {ADJ: None, 
                                P_Y_GIVEN_DOX_ADJ: None, 
                                P_Y_GIVEN_DOX: None} for treatment in self.parents.keys()}
        
        # If precomputed densities are provided, set them directly
        self.PriorDensity = prior_density
        self.JointDensity = joint_density
        self.ParentJointDensity = parent_joint_density
        self.CondDensity = cond_density
        self.MarginalDensity = marginal_density
        
        # Check if any density is None and run _preprocess() if needed
        self._preprocess()
            
        # Only compute densities if they were not provided
        if self.PriorDensity is None: self.PriorDensity = self.computePriorDensity()
        if self.JointDensity is None: self.JointDensity = self.computeJointDensity()
        if self.ParentJointDensity is None: self.ParentJointDensity = self.computeParentJointDensity()
        if self.CondDensity is None: self.CondDensity = self.computeConditionalDensity()
        if self.MarginalDensity is None: self.MarginalDensity = self.computeMarginalDensity()
            
        
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
            
            
    @staticmethod
    def fit_gmm(data, n_components=3):
        """
        Fit a Gaussian Mixture Model (GMM) to the data.

        Args:
            data (ndarray): Data to fit the GMM.
            n_components (int): Number of Gaussian components.

        Returns:
            dict: Parameters of the GMM (means, covariances, weights).
        """
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(data)
        
        # Return the parameters as a dictionary
        return {
            "means": gmm.means_,
            "covariances": gmm.covariances_,
            "weights": gmm.weights_
        }

    def computePriorDensity(self):
        """
        Compute the prior density p(y) using GMM.

        Returns:
            dict: GMM parameters for the prior density.
        """
        CP.info("    - Prior density")
        return self.fit_gmm(self.y.aligndata)

    def computeJointDensity(self):
        """
        Compute the joint density p(y, parents) using GMM.

        Returns:
            dict: GMM parameters for the joint density.
        """
        CP.info("    - Joint density")
        if self.parents:
            processes = [self.y] + list(self.parents.values())
            data = np.column_stack([p.aligndata for p in processes])
        else:
            data = self.y.aligndata

        return self.fit_gmm(data)

    def computeParentJointDensity(self):
        """
        Compute the joint density of the parents p(parents) using GMM.

        Returns:
            dict: GMM parameters for the parents' joint density.
        """
        CP.info("    - Parent joint density")
        if self.parents:
            data = np.column_stack([p.aligndata for p in self.parents.values()])
            return self.fit_gmm(data)
        return None

    def computeConditionalDensity(self):
        """
        Compute the conditional density p(y|parents) = p(y, parents) / p(parents).

        Returns:
            dict: GMM parameters for the conditional density.
        """
        CP.info("    - Conditional density")

        if self.parents:
            # Joint GMM parameters
            joint_params = self.JointDensity
            # Parent GMM parameters (marginal over parents)
            parent_params = self.ParentJointDensity

            # Initialize containers for conditional means, covariances, and weights
            conditional_params = {
                "means": [],
                "covariances": [],
                "weights": []
            }

            for k in range(len(joint_params["weights"])):
                # Extract the parameters for the k-th component
                mean_joint = joint_params["means"][k]
                cov_joint = joint_params["covariances"][k]
                weight_joint = joint_params["weights"][k]

                # Partition the mean and covariance matrix into blocks
                mean_y = mean_joint[:len(self.y.samples)]  # Mean for the target variable y
                mean_parents = mean_joint[len(self.y.samples):]  # Mean for the parent variables
                cov_yy = cov_joint[:len(self.y.samples), :len(self.y.samples)]  # Covariance of y
                cov_pp = cov_joint[len(self.y.samples):, len(self.y.samples):]  # Covariance of parents
                cov_yp = cov_joint[:len(self.y.samples), len(self.y.samples):]  # Covariance between y and parents
                cov_py = cov_joint[len(self.y.samples):, :len(self.y.samples)]  # Covariance between parents and y

                # Compute the conditional mean and covariance
                parent_diff = parent_values - mean_parents
                cond_mean = mean_y + np.dot(cov_yp, np.linalg.inv(cov_pp)).dot(parent_diff)
                cond_cov = cov_yy - np.dot(cov_yp, np.linalg.inv(cov_pp)).dot(cov_py)

                # Store the conditional parameters for this component
                conditional_params["means"].append(cond_mean)
                conditional_params["covariances"].append(cond_cov)
                conditional_params["weights"].append(weight_joint)

            return conditional_params
        else:
            # If no parents, return the prior density
            return self.PriorDensity

    def computeMarginalDensity(self):
        """
        Compute the marginal density p(y).

        Returns:
            dict: GMM parameters for the marginal density.
        """
        CP.info("    - Marginal density")
        if not self.parents:
            return self.PriorDensity
        else:
            # Compute the marginal density over y by summing out the parent variables
            joint_params = self.JointDensity
            
            # Initialize containers for the marginal parameters
            marginal_params = {
                "means": [],
                "covariances": [],
                "weights": joint_params["weights"]  # The weights remain the same
            }

            for k in range(len(joint_params["weights"])):
                # Extract the parameters for the k-th component
                mean_joint = joint_params["means"][k]
                cov_joint = joint_params["covariances"][k]
                
                # Extract the blocks corresponding to y and the parents
                mean_y = mean_joint[:len(self.y.samples)]  # The mean for the target (y)
                cov_yy = cov_joint[:len(self.y.samples), :len(self.y.samples)]  # Covariance for y
                
                # The marginal density is obtained by summing out the parent dimensions
                # This is equivalent to removing the parent parts of the covariance matrix
                marginal_params["means"].append(mean_y)
                marginal_params["covariances"].append(cov_yy)

            return marginal_params

    def get_density(self, x, params):
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
            mvn = multivariate_normal(mean=params["means"][k], cov=params["covariances"][k])
            density += params["weights"][k] * mvn.pdf(x)
        return density
