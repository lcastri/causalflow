from typing import Dict
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from causalflow.CPrinter import CP
from scipy.stats import multivariate_normal

from causalflow.causal_reasoning.Process import Process


def expectation(y, p):
    """
    Compute the expectation E[y*p(y)/sum(p(y))].

    Args:
        y (ndarray): process samples.
        p (ndarray): probability density function. Note it must be ALREADY NORMALISED.

    Returns:
        float: expectation E[y*p(y)/sum(p(y))].
    """
    if np.sum(p) == 0:
        return np.nan
    expectation_Y_given_X = np.sum(y * p)
    return expectation_Y_given_X


def expectation_from_params(means, weights):
    """
    Compute the expectation (mean) of a Gaussian Mixture Model (GMM).
    
    Args:
        means (ndarray): The means of the Gaussian components (K x D).
        weights (ndarray): The weights of the Gaussian components (K,).
        
    Returns:
        ndarray: The expected mean (1 x D), which is the weighted sum of the component means.
    """
    # Ensure the weights sum to 1
    assert np.allclose(np.sum(weights), 1), "Weights must sum to 1."

    # Compute the weighted sum of the means
    if weights.ndim == 1:
        weights = weights[:, None]
    expectation = np.sum(weights * means, axis=0)

    
    return expectation
    
    
def mode(y, p):
    """
    Compute the mode, which is the most likely valueof y.

    Args:
        y (ndarray): process samples.
        p (ndarray): probability density function. Note it must be ALREADY NORMALISED.
        
    Returns:
        float: mode Mode(y*p(y)).
    """
    return y[np.argmax(p)]

    
def normalise(p):
    """
    Normalise the probability density function to ensure it sums to 1.

    Args:
        p (ndarray): probability density function.

    Returns:
        ndarray: normalised probability density function.
    """
    p_sum = np.sum(p)
    if p_sum > 0:
        return p / p_sum
    else:
        return np.zeros_like(p)
    
    
def format_combo(combo):
    return tuple(sorted(combo))


def get_max_lag(processes: Dict[str, Process] = None):
    """
    Return max time lag between target and all its parents.

    Returns:
        int: max time lag.
    """
    if processes is not None: 
        return max(p.lag for p in processes.values())
    return 0


def fit_gmm(max_components, caller, data, standardize = True):
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
    components = range(1, max_components + 1)
    aic = []
    bic = []

    for n in tqdm(components, desc=f"[INFO]:     - {caller}"):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(data)
        aic.append(gmm.aic(data))
        bic.append(gmm.bic(data))

    optimal_n_components = components[np.argmin(aic)]  # Or np.argmin(bic)
    # optimal_n_components = components[np.argmin(bic)]  # Switch to np.argmin(aic) if needed
    # CP.debug(f"          Optimal n.components: {optimal_n_components}")

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


def get_density(params, x):
    """
    Query the density for a given point `x` using the GMM parameters.
    
    Args:
        params (dict): The GMM parameters (means, covariances, weights).
        x (ndarray): The point(s) at which to evaluate the density.
        
    Returns:
        ndarray: The computed density at the point(s) x.
    """   
    total_weight = np.sum(params["weights"])  # Total weight sum (should be 1 in a proper GMM)
    
    # If the weights do not sum to 1, normalize them
    if total_weight != 1:
        params["weights"] = np.array(params["weights"]) / total_weight
    
    density = 0
    for k in range(len(params["weights"])):
        mvn = multivariate_normal(mean=params["means"][k].flatten(), cov=params["covariances"][k].flatten())
        density += params["weights"][k] * mvn.pdf(x)
    return density


def compute_conditional(joint_gmm_params, parent_values):
        """
        Compute the conditional density p(Y | parents) dynamically from joint GMM.

        Args:
            joint_gmm_params (dict): GMM parameters for the joint density p(Y, parents).
            parent_values (ndarray): Values of the parent variables (e.g., [x, adj]).

        Returns:
            dict: GMM parameters for the conditional density.
        """
        conditional_params = {
            "means": [],
            "covariances": [],
            "weights": joint_gmm_params["weights"],
        }

        for k in range(len(joint_gmm_params["weights"])):
            mean_joint = joint_gmm_params["means"][k]
            cov_joint = joint_gmm_params["covariances"][k]

            dim_y = 1
            mean_parents = mean_joint[dim_y:]
            mean_target = mean_joint[:dim_y]
            cov_pp = cov_joint[dim_y:, dim_y:]
            cov_pp += 1e-6 * np.eye(cov_pp.shape[0])  # Regularize for invertibility

            cov_yp = cov_joint[:dim_y, dim_y:]
            cov_yy = cov_joint[:dim_y, :dim_y]

            # Conditional mean and covariance
            cond_mean = mean_target + cov_yp @ np.linalg.inv(cov_pp) @ (parent_values.flatten() - mean_parents.flatten())
            cond_cov = cov_yy - cov_yp @ np.linalg.inv(cov_pp) @ cov_yp.T

            conditional_params["means"].append(cond_mean)
            conditional_params["covariances"].append(cond_cov)

        conditional_params["means"] = np.array(conditional_params["means"])
        conditional_params["covariances"] = np.array(conditional_params["covariances"])

        return conditional_params