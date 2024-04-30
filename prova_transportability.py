import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate synthetic data for source dataset S
np.random.seed(0)
n_samples_s = 1000
X_s = np.random.normal(0, 1, size=(n_samples_s, 1))  # Treatment variable
T_s = np.random.normal(0, 1, size=(n_samples_s, 2))  # Covariates
Y_s = 2*X_s + T_s[:, 0] - 0.5*T_s[:, 1] + np.random.normal(0, 0.5, size=n_samples_s)  # Outcome variable

# Generate synthetic data for target dataset T
n_samples_t = 1000
X_t = np.random.normal(0, 1, size=(n_samples_t, 1))  # Treatment variable
T_t = np.random.normal(0, 1, size=(n_samples_t, 2))  # Covariates
Y_t = 2*X_t + T_t[:, 0] - 0.5*T_t[:, 1] + np.random.normal(0, 0.5, size=n_samples_t)  # Outcome variable

# Apply intervention in source and target datasets
X_s_intervention = np.full_like(X_s, 5)  # Intervention: do(X) = 5 in source dataset
X_t_intervention = np.full_like(X_t, 3)  # Intervention: do(X) = 3 in target dataset

# Assess covariate balance
# Compare covariate distributions between source and target datasets
plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.hist(T_s[:, i], bins=30, alpha=0.5, color='blue', label='Source Dataset')
    plt.hist(T_t[:, i], bins=30, alpha=0.5, color='red', label='Target Dataset')
    plt.title(f'Covariate {i+1} Distribution')
    plt.xlabel(f'T{i+1}')
    plt.ylabel('Frequency')
    plt.legend()
plt.tight_layout()
plt.show()

# Estimate transportability weight
# Use Kernel Density Estimation (KDE) to estimate the transportability weight
kde_s = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde_s.fit(X_s_intervention)
log_density_t_given_s = kde_s.score_samples(X_t_intervention)  # Log density of X_t given X_s
log_density_s = kde_s.score_samples(X_s_intervention)  # Log density of X_s
transportability_weight = np.exp(log_density_t_given_s - log_density_s)  # Transportability weight

# Plot transportability weight
plt.figure(figsize=(6, 4))
plt.hist(transportability_weight, bins=30, color='green', alpha=0.5)
plt.title('Transportability Weight Distribution')
plt.xlabel('Transportability Weight')
plt.ylabel('Frequency')
plt.show()
