import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

np.random.seed(seed=1)
# Generate sample data for observational and interventional cases
# observational_data = np.random.uniform(low=-10, high=10, size=1000)
observational_data = np.random.normal(loc=3, scale=1, size=1000)
interventional_data = np.random.normal(loc=7, scale=1, size=1000)
# interventional2_data = np.random.exponential(scale=4, size=1000)

# Concatenate observational and interventional data
# combined_data = np.concatenate([observational_data, interventional_data, interventional2_data])
combined_data = np.concatenate([observational_data, interventional_data])


obs = KernelDensity(bandwidth=0.5)
obs.fit(observational_data.reshape(-1, 1))

int = KernelDensity(bandwidth=0.5)
int.fit(interventional_data.reshape(-1, 1))

# int2 = KernelDensity(bandwidth=0.5)
# int2.fit(interventional2_data.reshape(-1, 1))

# Fit KDE to the combined data
kde_combined = KernelDensity(bandwidth=0.5)
kde_combined.fit(combined_data[:, np.newaxis])

# # Fit Gaussian Mixture Model to the combined data
# gmm = GaussianMixture(n_components=n_components)
# gmm.fit(combined_data[:, np.newaxis])

# Generate X values for plotting
x = np.linspace(-5, 10, 1000)

# Calculate log density estimate for combined data
log_dens_obs = obs.score_samples(x[:, np.newaxis])
log_dens_int = int.score_samples(x[:, np.newaxis])
# log_dens_int2 = int2.score_samples(x[:, np.newaxis])
log_dens_combined = kde_combined.score_samples(x[:, np.newaxis])
# gmm_dens_combined = gmm.score_samples(x[:, np.newaxis])

# Convert log density to density
dens_obs = np.exp(log_dens_obs)
dens_int = np.exp(log_dens_int)
# dens_int2 = np.exp(log_dens_int2)
dens_combined = np.exp(log_dens_combined)
# gmm_dens_combined = np.exp(gmm_dens_combined)

# Combine densities using weighted average
weight_obs = 0.5  # Adjust as needed
weight_int = 0.5  # Adjust as needed
my = weight_obs * dens_obs + weight_int * dens_int
# my = weight_obs * dens_obs + weight_int * dens_int + weight_int * dens_int2

# Plot the combined density estimated by KDE
plt.plot(x, dens_obs, label='Obs Density')
plt.plot(x, dens_int, label='Int Density')
# plt.plot(x, dens_int2, label='int2')
plt.plot(x, dens_combined, label='Combined Density (KDE)')
# plt.plot(x, gmm_dens_combined, label='GMM')
plt.plot(x, my, label='Weighted Sum')
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Combined Observational and Interventional Densities (KDE)')
plt.legend()
plt.show()
