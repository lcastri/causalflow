import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from causalflow.basics.constants import DataType, NodeType
from causalflow.graph import DAG
from causalflow.preprocessing.data import Data
# from causalflow.causal_discovery.FPCMCI import FPCMCI
# from causalflow.selection_methods.TE import TE, TEestimator
# from tigramite.independence_tests.gpdc import GPDC
from fpcmci.CPrinter import CPLevel
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.causal_reasoning.SMCFilter import SMCFilter
import pymc as pm
import arviz as az
import pymc.math as pmm
variables = ['A', 'B', 'C', 'D', 'E']
# # Set random seed for reproducibility
# np.random.seed(42)

# # Number of time steps
# T = 1000

# # Initialize arrays for each variable
# A = np.random.normal(0, 1, T)  # Exogenous noise
# E = np.random.normal(0, 1, T)  # Exogenous noise
# B = np.zeros(T)
# C = np.zeros(T)
# D = np.zeros(T)

# #! Generate data using the system of equations
# for t in range(1, T):
#     B[t] = 0.1 * A[t] + 0.9 * E[t-1] + np.random.normal(0, 0.1)
#     C[t] = 0.7 * B[t] + np.random.normal(0, 0.1)
#     D[t] = 0.1 * C[t] + 3.5 * E[t-1] + np.random.normal(0, 0.1)

# # Store in a DataFrame
# df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E_lag': np.roll(E, shift=1)})
# df = df.iloc[1:].reset_index(drop=True)  # Remove first row since E_lag[0] is undefined

# d = Data(df, vars = variables)
# d.save_csv("results/5vars_test/data.csv")

# #! Causal Discovery
# fpcmci = FPCMCI(data = d,
#                 min_lag = 0,
#                 max_lag = 1,
#                 sel_method = TE(TEestimator.Gaussian),
#                 val_condtest = GPDC(significance = 'analytic', gp_params = None),
#                 alpha = 0.05,
#                 verbosity = CPLevel.INFO,
#                 neglect_only_autodep = False,
#                 resfolder=f"results/5vars_test")

# CM = fpcmci.run(nofilter=True)
# CM.dag(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
#        save_name=fpcmci.dag_path + '_circular')
# CM.dag(node_layout = 'dot', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
#        save_name=fpcmci.dag_path + '_dot')
# CM.ts_dag(node_size = 4, 
#           min_cross_width = 0.5, max_cross_width = 1.5, 
#           x_disp=1.5, y_disp=0.2,
#           save_name=fpcmci.ts_dag_path)


# #! Causal Inference Engine
# DAGDIR = '/home/lcastri/git/causalflow/results/5vars_test/'
# RES = 'res_modified.pkl'
# with open(DAGDIR+RES, 'rb') as f:
#     CM = DAG.load(pickle.load(f))

# DATA_TYPE = {
#     'A': DataType.Continuous,
#     'B': DataType.Continuous,
#     'C': DataType.Continuous,
#     'D': DataType.Continuous,
#     'E': DataType.Continuous,
# }
# NODE_TYPE = {
#     'A': NodeType.System,
#     'B': NodeType.System,
#     'C': NodeType.System,
#     'D': NodeType.System,
#     'E': NodeType.System,
# }
# cie = CIE(CM, 
#           data_type = DATA_TYPE, 
#           node_type = NODE_TYPE,
#           model_path = '5vars_test',
#           verbosity = CPLevel.INFO)


# cie.addObsData(Data(DAGDIR+'data.csv', vars=variables))
# cie.save(os.path.join(cie.model_path, 'cie.pkl'))


# Initialize arrays for each variable
T = 10
A = np.random.normal(0, 1, T)  # Exogenous noise
E = np.random.normal(0, 1, T)  # Exogenous noise
B = np.zeros(T)
C = np.zeros(T)
D = np.zeros(T)

#! Generate data using the system of equations
for t in range(1, T):
    B[t] = 0.1 * A[t] + 0.9 * E[t-1] + np.random.normal(0, 0.1)
    C[t] = 0.7 * B[t] + np.random.normal(0, 0.1)
    D[t] = 0.1 * C[t] + 3.5* E[t-1] + np.random.normal(0, 0.1)


cie = CIE.load('/home/lcastri/git/causalflow/results/5vars_test/5vars_test/cie.pkl')

# Initialize the SMC filter
smc = SMCFilter(cie.DBNs[('obs', 0)], num_particles=1000)

given_values = {('B', 0): B}
given_context = {}

# Run inference for 10 time steps
D_obs = smc.sequential_query("D", given_values, given_context, num_steps=T)  
obs_RMSE = np.sqrt(np.mean((D_obs - D) ** 2))
obs_NRMSE = obs_RMSE/np.std(D)
    
# Initialize SMC filter
smc_do = SMCFilter(cie.DBNs[('obs', 0)], num_particles=1000)

# Run inference for 10 time steps
D_do= smc_do.sequential_query("D", given_values, given_context, num_steps=T, 
                             intervention_var=('B', 0), adjustment_set=[('E', -1)])
do_RMSE = np.sqrt(np.mean((D_do - D) ** 2))
do_NRMSE = do_RMSE/np.std(D)

# # ========== PyMC Implementation ========== #
# with pm.Model() as model:
#     # Priors
#     sigma_B = pm.Exponential("sigma_B", 1.0)
#     sigma_C = pm.Exponential("sigma_C", 1.0)
#     sigma_D = pm.Exponential("sigma_D", 1.0)

#     # Latent variables
#     A_t = pm.Normal("A", mu=0, sigma=1, shape=T)
#     E_t = pm.Normal("E", mu=0, sigma=1, shape=T)
#     E_t_lag = pm.Deterministic("E_lag", pmm.concatenate([[0], E_t[:-1]]))

#     B_t = pm.Normal("B", mu=0.1 * A_t + 0.9 * E_t_lag, sigma=sigma_B, shape=T)
#     C_t = pm.Normal("C", mu=0.7 * B_t, sigma=sigma_C, shape=T)
#     D_t = pm.Normal("D", mu=0.1 * C_t + 1.1 * E_t_lag, sigma=sigma_D, shape=T)

#     # Observations
#     B_obs = pm.Data("B_obs", B)
#     D_obs_pymc = pm.Normal("D_obs", mu=0.6 * C_t + 0.4 * E_t, sigma=sigma_D, observed=D)

#     # Posterior Sampling
#     trace = pm.sample(1000, return_inferencedata=True, cores=2, target_accept=0.9)

# # Compute posterior expectation of D given B
# D_pymc_obs = az.summary(trace, var_names=["D"])["mean"].values
# pymc_RMSE = np.sqrt(np.mean((D_pymc_obs - D) ** 2))
# pymc_NRMSE = pymc_RMSE/np.std(D)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(range(T), D, label="Ground Truth", linestyle="solid", marker="o")
plt.plot(range(T), D_obs, label=f"E[D] -- p(D∣B = b), NRMSE={obs_NRMSE:.2f}", linestyle="dashed", marker="x", color="green")
plt.plot(range(T), D_do, label=f"E[D] -- p(D∣do(B = b)), NRMSE={do_NRMSE:.2f}", linestyle="dashed", marker="o", color="red")
# plt.plot(range(T), D_pymc_obs, label=f"E[D] - PyMC, NRMSE={pymc_NRMSE:.2f}", linestyle="dashed", marker="o", color="blue")

plt.xlabel("Time Step")
plt.ylabel("D Value")
plt.title("SMC Predictions vs. Ground Truth")
plt.legend()
plt.grid(True)
plt.show()