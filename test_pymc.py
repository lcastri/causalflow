import pymc as pm
import numpy as np
import arviz as az
import pymc.math as pmm

# Generate synthetic data (following the DBN structure)
T = 100  # Number of timesteps
A = np.random.normal(0, 1, T)
E = np.random.normal(0, 1, T)
B = 0.1 * A + 1.5 * np.roll(E, 1) + np.random.normal(0, 0.1, T)
C = 0.7 * B + np.random.normal(0, 0.1, T)
D = 0.1 * C + 10.1 * np.roll(E, 1) + np.random.normal(0, 0.1, T)

with pm.Model() as bayesian_model:
    # Priors
    sigma_B = pm.Exponential("sigma_B", 1.0)
    sigma_C = pm.Exponential("sigma_C", 1.0)
    sigma_D = pm.Exponential("sigma_D", 1.0)

    # Latent variables
    A_t = pm.Normal("A", mu=0, sigma=1, shape=T)
    E_t = pm.Normal("E", mu=0, sigma=1, shape=T)

    # Lagged E
    E_t_lag = pm.Deterministic("E_lag", pmm.concatenate([[0], E_t[:-1]]))

    # System equations
    B_t = pm.Normal("B", mu=0.1 * A_t + 1.5 * E_t_lag, sigma=sigma_B, shape=T)
    C_t = pm.Normal("C", mu=0.7 * B_t, sigma=sigma_C, shape=T)
    D_t = pm.Normal("D", mu=0.1 * C_t + 10.1 * E_t_lag, sigma=sigma_D, shape=T)

    # Observation model (conditioning on observed B)
    B_obs = pm.Data("B_obs", B)
    D_obs = pm.Normal("D_obs", mu=0.6 * C_t + 10.1 * E_t_lag, sigma=sigma_D, observed=D)

    # Posterior Sampling
    trace_bayesian = pm.sample(1000, return_inferencedata=True, cores=2, target_accept=0.9)

# Extract Bayesian Posterior Prediction of D given B
D_pymc_obs = az.summary(trace_bayesian, var_names=["D"])["mean"].values


with pm.Model() as causal_model:
    # Priors
    sigma_C = pm.Exponential("sigma_C", 1.0)
    sigma_D = pm.Exponential("sigma_D", 1.0)

    # Latent variable (E)
    E_t = pm.Normal("E", mu=0, sigma=1, shape=T)
    E_t_lag = pm.Deterministic("E_lag", pmm.concatenate([[0], E_t[:-1]]))

    # Intervened variable: B is now externally set (not dependent on A or E)
    B_do = pm.Normal("B_do", mu=0.5, sigma=0.1, shape=T)  # Intervened B = 0.5

    # Compute effects
    C_t = pm.Normal("C", mu=0.7 * B_do, sigma=sigma_C, shape=T)
    D_t = pm.Normal("D", mu=0.6 * C_t + 10.1 * E_t_lag, sigma=sigma_D, shape=T)
    D_obs = pm.Normal("D_obs", mu=0.6 * C_t + 10.1 * E_t_lag, sigma=sigma_D, observed=D)

    # Posterior Sampling
    trace_causal = pm.sample(1000, return_inferencedata=True, cores=2, target_accept=0.9)

# Extract Causal Posterior Prediction of D given do(B)
D_pymc_do = az.summary(trace_causal, var_names=["D"])["mean"].values


import matplotlib.pyplot as plt

# Compute RMSE
obs_RMSE = np.sqrt(np.mean((D_pymc_obs - D) ** 2))
obs_NRMSE = obs_RMSE / np.std(D)

do_RMSE = np.sqrt(np.mean((D_pymc_do - D) ** 2))
do_NRMSE = do_RMSE / np.std(D)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(T), D, label="Ground Truth", linestyle="solid", marker="o")
plt.plot(range(T), D_pymc_obs, label=f"E[D] -- P(D | B), NRMSE={obs_NRMSE:.2f}", linestyle="dashed", marker="x", color="green")
plt.plot(range(T), D_pymc_do, label=f"E[D] -- P(D | do(B)), NRMSE={do_NRMSE:.2f}", linestyle="dashed", marker="o", color="red")

plt.xlabel("Time Step")
plt.ylabel("D Value")
plt.title("PyMC Predictions: Observational vs. Causal")
plt.legend()
plt.grid(True)
plt.show()