import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

# Generate example data from different distributions
np.random.seed(0)

# Gaussian distribution
gaussian_data = np.random.exponential(scale=3, size=1000)

# Exponential distribution
exponential_data = np.random.exponential(scale=1, size=1000)

# Exponential distribution
uniform_data = np.random.uniform(low = -5, high=5, size=1000)

# Multimodal distribution
multimodal_data = np.concatenate([np.random.exponential(scale=0.5, size=250),
                                  np.random.uniform(low=3, high=5, size=250),
                                  np.random.chisquare(df = 3, size=250),
                                  np.random.beta(3, 7, size=250)])

# Combine the data from different distributions into a single array
# data = multimodal_data
data = np.concatenate([gaussian_data, exponential_data, uniform_data, multimodal_data])

# Reshape the data into a 2D array as required by scikit-learn
data = data.reshape(-1, 1)

# Choose a bandwidth for the kernel density estimation
bandwidths = np.logspace(-5, 5, 100)
grid_search = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths})
grid_search.fit(data)
best_bandwidth = grid_search.best_params_['bandwidth']

# Fit a KDE model to the combined data
kde = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
kde.fit(data)

# Fit a GMM model to the combined data
gmm = GaussianMixture(n_components=3)
gmm.fit(data)

# Fit Dirichlet Process Mixture Model
dpmm = BayesianGaussianMixture(n_components=4, max_iter=1000)
dpmm.fit(data)

# Generate points for the x-axis to visualize the density estimate
x = np.linspace(-5, 10, 1000).reshape(-1, 1)

# Evaluate the KDE model at the generated points
log_density_kde = kde.score_samples(x)
density_kde = np.exp(log_density_kde)

# Evaluate the GMM model at the generated points
log_density_gmm = gmm.score_samples(x)
density_gmm = np.exp(log_density_gmm)

# Evaluate the GMM model at the generated points
log_density_dpmm = dpmm.score_samples(x)
density_dpmm = np.exp(log_density_dpmm)

# Evaluate the SUM model at the generated points
gaussian_density = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
gaussian_density.fit(gaussian_data.reshape(-1, 1))
log_density_gaussian = gaussian_density.score_samples(x)
exponential_density = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
exponential_density.fit(exponential_data.reshape(-1, 1))
log_density_expo = exponential_density.score_samples(x)
uniform_density = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
uniform_density.fit(uniform_data.reshape(-1, 1))
log_density_uni = uniform_density.score_samples(x)
multimodal_density = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
multimodal_density.fit(multimodal_data.reshape(-1, 1))
log_density_multi = multimodal_density.score_samples(x)
density_sum = 0.25*np.exp(log_density_gaussian) + 0.25*np.exp(log_density_expo) + 0.25*np.exp(log_density_multi) + 0.25*np.exp(log_density_uni)

# Plot the combined density estimates
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.5, color='gray', label='Data')  # Plot histogram of the combined data
plt.plot(x, density_kde, color='red', label='KDE')  # Plot KDE estimate
plt.plot(x, density_gmm, color='blue', label='GMM')  # Plot GMM estimate
plt.plot(x, density_sum, color='green', label='SUM')  # Plot GMM estimate
plt.plot(x, density_dpmm, color='k', label='DPMM')  # Plot GMM estimate
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Combined Density Estimate')
plt.legend()
plt.show()
