import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class EM:
    def __init__(self, n_components, reg_strength=0.1, max_iter=100, tol=1e-6):
        """
        Custom implementation of Gaussian Mixture Model (GMM) using EM algorithm with regularization.

        Args:
            n_components (int): Number of Gaussian components.
            reg_strength (float): Regularization strength for the means update.
            max_iter (int): Maximum number of iterations for the EM algorithm.
            tol (float): Tolerance for convergence based on log-likelihood.
        """
        self.n_components = n_components
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None

    def initialize_means(self, data):
        """
        Initialize the means of the GMM components using K-means clustering.

        Args:
            data (ndarray): Data to fit the GMM.

        Returns:
            ndarray: Initial means of the components.
        """
        kmeans = KMeans(n_clusters=self.n_components, random_state=42)
        kmeans.fit(data)
        return kmeans.cluster_centers_

    def compute_responsibilities(self, data):
        """
        Compute the responsibilities in the E-step.

        Args:
            data (ndarray): Data to fit the GMM.

        Returns:
            ndarray: Responsibilities (N x K matrix).
        """
        n_samples, n_components = data.shape[0], self.n_components
        responsibilities = np.zeros((n_samples, n_components))

        # Compute weighted Gaussian PDFs for each component
        for k in range(n_components):
            pdf = multivariate_normal(mean=self.means[k], cov=self.covariances[k]).pdf(data)
            responsibilities[:, k] = self.weights[k] * pdf

        # Normalize to ensure responsibilities sum to 1 for each data point
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def update_means(self, data, responsibilities):
        """
        Update the means of the GMM components in the M-step.

        Args:
            data (ndarray): Data to fit the GMM.
            responsibilities (ndarray): Responsibilities (N x K matrix).

        Returns:
            ndarray: Updated means of the components.
        """
        means = np.zeros((self.n_components, data.shape[1]))
        for k in range(self.n_components):
            # Weighted average of data points for component k
            means[k] = np.sum(responsibilities[:, k][:, None] * data, axis=0) / np.sum(responsibilities[:, k])

        return means

    def update_covariances(self, data, responsibilities):
        """
        Update the covariances of the GMM components in the M-step.

        Args:
            data (ndarray): Data to fit the GMM.
            responsibilities (ndarray): Responsibilities (N x K matrix).

        Returns:
            list[ndarray]: Updated covariances of the components.
        """
        covariances = []
        for k in range(self.n_components):
            diff = data - self.means[k]
            weighted_diff = responsibilities[:, k][:, None] * diff
            cov_k = np.dot(weighted_diff.T, diff) / np.sum(responsibilities[:, k])
            covariances.append(cov_k + 1e-6 * np.eye(data.shape[1]))  # Regularization term

        return covariances

    def has_converged(self, prev_log_likelihood, log_likelihood):
        """
        Check if the EM algorithm has converged.

        Args:
            prev_log_likelihood (float): Log-likelihood from the previous iteration.
            log_likelihood (float): Current log-likelihood.

        Returns:
            bool: True if converged, False otherwise.
        """
        return np.abs(log_likelihood - prev_log_likelihood) < self.tol

    def fit(self, data):
        """
        Fit the GMM to the data using the EM algorithm.

        Args:
            data (ndarray): Data to fit the GMM.

        Returns:
            dict: GMM parameters (means, covariances, weights).
        """
        self.means = self.initialize_means(data)
        self.covariances = [np.cov(data, rowvar=False)] * self.n_components
        self.weights = np.full(self.n_components, 1 / self.n_components)

        prev_log_likelihood = -np.inf
        for _ in range(self.max_iter):
            # E-step
            responsibilities = self.compute_responsibilities(data)

            # M-step
            self.weights = responsibilities.mean(axis=0)
            self.means = self.update_means(data, responsibilities)
            self.covariances = self.update_covariances(data, responsibilities)

            # Regularize means
            empirical_mean = np.mean(data, axis=0)
            gmm_mean = np.sum(self.weights[:, None] * self.means, axis=0)
            self.means += self.reg_strength * (empirical_mean - gmm_mean)

            # Log-likelihood
            log_likelihood = np.sum(np.log(responsibilities.sum(axis=1)))
            if self.has_converged(prev_log_likelihood, log_likelihood):
                break
            prev_log_likelihood = log_likelihood

        return {
            "means": self.means,
            "covariances": self.covariances,
            "weights": self.weights,
        }
