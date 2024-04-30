import numpy as np
from scipy.stats import gaussian_kde

def compute_causal_density(x_values, y_values, z_values):
    # Compute KDE for each value of z
    kdes = []
    for z_val in np.unique(z_values):
        # Filter data for the current value of z
        z_mask = (z_values == z_val)
        x_z = x_values[z_mask]
        y_z = y_values[z_mask]
        # Compute KDE for (x, y) pairs with z = z_val
        kde = gaussian_kde(np.vstack((x_z, y_z)))
        kdes.append((z_val, kde))
    
    # Compute probability of each value of z
    unique_z, counts_z = np.unique(z_values, return_counts=True)
    prob_z = counts_z / len(z_values)
    
    # Compute causal density by summing weighted densities
    causal_density = np.zeros_like(y_values)
    for z_val, kde in kdes:
        kde_values = kde(np.vstack((x_values, y_values)))
        causal_density += kde_values * prob_z[np.where(unique_z == z_val)[0][0]]
    
    return causal_density

# Example usage:
# Generate example data
np.random.seed(0)
x_values = np.random.normal(0, 1, 1000)
z_values = np.random.choice([0, 1], size=1000)
y_values = 2 * x_values + z_values + np.random.normal(0, 0.5, 1000)

# Compute causal density
causal_density = compute_causal_density(x_values, y_values, z_values)

# Plot causal density
import matplotlib.pyplot as plt
plt.hist(causal_density, bins=30, density=True, alpha=0.5, color='blue')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.title('Causal Density')
plt.show()