import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set a fixed seed
np.random.seed(42)

# Define network parameters
mean_A = 0
mean_B_given_A = 1
mean_C_given_B = 2
mean_D_given_C_E = 3
mean_E = 0

sigma_A = 1
sigma_B_given_A = 1
sigma_C_given_B = 1
sigma_D_given_C_E = 1
sigma_E = 1

cov_BD = 0.5  # Correlation factor

# Analytical Sampling
def analytical_posterior_D_given_B(b, n_samples=1000):
    # Sample E first
    E_samples = np.random.normal(loc=mean_E, scale=sigma_E, size=n_samples)
    
    # Sample C given B
    C_samples = np.random.normal(loc=mean_C_given_B + b, scale=sigma_C_given_B, size=n_samples)
    
    # Compute D given C and E
    D_samples = np.random.normal(loc=mean_D_given_C_E + C_samples + E_samples, 
                                 scale=sigma_D_given_C_E, size=n_samples)
    
    return D_samples

b = 5  # Fixed observed value
n_samples = 500
analytical_samples = analytical_posterior_D_given_B(b, n_samples=n_samples)

# PyMC Model
with pm.Model() as model:
    # Priors
    A = pm.Normal("A", mu=mean_A, sigma=sigma_A)
    E = pm.Normal("E", mu=mean_E, sigma=sigma_E)

    # Conditional distributions
    B = pm.Normal("B", mu=mean_B_given_A + A, sigma=sigma_B_given_A, observed=b)
    C = pm.Normal("C", mu=mean_C_given_B + B, sigma=sigma_C_given_B)
    D = pm.Normal("D", mu=mean_D_given_C_E + C + E, sigma=sigma_D_given_C_E)

    # Use Metropolis sampling for efficiency
    step = pm.Metropolis()
    trace = pm.sample(n_samples, step=step, tune=300, return_inferencedata=True, progressbar=True, cores=1)

# Extract PyMC posterior samples
posterior_samples_pymc = trace.posterior['D'].values.flatten()

# Expected value from PyMC sampling
expected_D_pymc = np.mean(posterior_samples_pymc)
print(expected_D_pymc)
# Plot comparison
plt.figure(figsize=(12, 6))
plt.hist(analytical_samples, bins=30, alpha=0.5, label='Analytical Posterior', density=True)
plt.hist(posterior_samples_pymc, bins=30, alpha=0.5, label='PyMC Posterior', density=True)
plt.legend()
plt.title(f'Comparison of Analytical and PyMC Posterior Distributions for D | B = {b}')
plt.xlabel('D')
plt.ylabel('Density')
plt.show()



# PyMC Interventional Model
with pm.Model() as causal_model:
    # Sample E from its prior P(E)
    E = pm.Normal("E", mu=mean_E, sigma=sigma_E)

    # Intervene: Set B to a fixed value b (do(B = b))
    B = b  # Fixed intervention

    # Compute C based on the intervened B
    C = pm.Normal("C", mu=mean_C_given_B + B, sigma=sigma_C_given_B)

    # Compute D based on C and E
    D = pm.Normal("D", mu=mean_D_given_C_E + C + E, sigma=sigma_D_given_C_E)

    # Sample from the interventional distribution
    causal_trace = pm.sample(500, return_inferencedata=True, tune=300, progressbar=True, cores=1)

# Extract posterior samples for D under do(B = b)
posterior_samples_do_B_pymc = causal_trace.posterior['D'].values.flatten()

# Compute expected value of D under do(B = b) from PyMC samples
expected_D_do_B_pymc = np.mean(posterior_samples_do_B_pymc)

# Print results
print(f"PyMC E[D | do(B = b)]: {expected_D_do_B_pymc}")