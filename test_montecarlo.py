import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

# Simulating data (same as before)
np.random.seed(42)
n_samples = 500
C = np.random.normal(loc=0, scale=1, size=n_samples)
B = np.random.normal(loc=2, scale=1.5, size=n_samples)
E = np.random.normal(loc=-1, scale=1, size=n_samples)
D = 3*C + 2*B - 1.5*E + np.random.normal(scale=1.0, size=n_samples)

df = pd.DataFrame(np.column_stack((C, B, E, D)), columns=["C", "B", "E", "D"])

# Fit standard GMM for P(D | C, B, E)
gmm_D_given_CBE = GaussianMixture(n_components=5, covariance_type='full', max_iter=1000, random_state=42)
gmm_D_given_CBE.fit(df[["C", "B", "E", "D"]].values)

# Fit GMM for P(C | B)
gmm_C_given_B = GaussianMixture(n_components=5, covariance_type='full', max_iter=1000, random_state=42)
gmm_C_given_B.fit(df[["B", "C"]].values)

# Fit GMM for P(E)
gmm_E = GaussianMixture(n_components=3, covariance_type='full', max_iter=1000, random_state=42)
gmm_E.fit(df[["E"]].values)

# Function to sample C given B
def sample_C_given_B(b, n_samples=100):
    """Sample C given B using the learned GMM P(C | B)."""
    B_samples = np.full((n_samples, 1), b)  # Fixed B
    X_test = np.hstack([B_samples, np.zeros((n_samples, 1))])  # Placeholder for C
    C_samples = gmm_C_given_B.sample(n_samples)[0][:, 1]  # Sample C values
    return C_samples

# Function to sample E
def sample_E(n_samples=100):
    """Sample E from the learned GMM P(E)."""
    E_samples = gmm_E.sample(n_samples)[0].flatten()
    return E_samples

# Function to estimate P(D | B) using Monte Carlo
def estimate_p_D_given_B(b, n_samples=100):
    """Estimate P(D | B) by marginalizing over C and E using Monte Carlo sampling."""
    C_samples = sample_C_given_B(b, n_samples)
    E_samples = sample_E(n_samples)
    
    # Compute P(D | C, B, E) for each sample
    D_values = np.linspace(D.min(), D.max(), 100)  # Possible values of D
    probs = np.zeros_like(D_values)

    for i in range(n_samples):
        c, e = C_samples[i], E_samples[i]
        X_test = np.column_stack([np.full_like(D_values, c),  # C
                                  np.full_like(D_values, b),  # B
                                  np.full_like(D_values, e),  # E
                                  D_values])  # D grid

        log_probs = gmm_D_given_CBE.score_samples(X_test)  # Log likelihoods
        probs += np.exp(log_probs)  # Convert to probabilities

    probs /= n_samples  # Average over samples
    return D_values, probs

# Example: Estimate P(D | B=2)
D_values, probs = estimate_p_D_given_B(b=2, n_samples=200)

# Plot the estimated density
plt.plot(D_values, probs, label="Estimated P(D | B=2)")
plt.xlabel("D")
plt.ylabel("Density")
plt.legend()
plt.show()
