from matplotlib import pyplot as plt
import numpy as np
from causalflow.basics.constants import LabelType
from causalflow.preprocessing.data import Data
import pandas as pd
from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.causal_inference.CausalInferenceEngine import CausalInferenceEngine as CIE


# Parameters
alpha = 0.05
max_lag = 2
min_lag = 1


# Population A
np.random.seed(1)
T = 1000
N = 3
dA_obs = np.random.normal(0, 1, size = (T, N))
for t in range(max_lag, T):
    dA_obs[t, 1] += 0.75 * dA_obs[t-1, 0]**2
    dA_obs[t, 2] += 1.3 * dA_obs[t-1, 0] * dA_obs[t-2, 1]

dfA_obs = Data(dA_obs)
# dfA_obs.plot_timeseries()

T = 300
N = 3
dA_int = np.random.normal(0, 1, size = (T, N))
dA_int[:, 1] = np.squeeze(5 * np.ones(shape=(T,1)))
for t in range(max_lag, T):
    dA_int[t, 2] += 1.3 * dA_int[t-1, 0] * dA_int[t-2, 1]

dfA_int = Data(dA_int)
# dfA_int.plot_timeseries()


# Causal Discovery on Population A
fpcmci = FPCMCI(dfA_obs, 
                f_alpha = alpha, 
                alpha = alpha, 
                min_lag = min_lag, 
                max_lag = max_lag, 
                sel_method = TE(TEestimator.Auto), 
                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                verbosity = CPLevel.DEBUG,
                resfolder = 'results/dbn')
CM = fpcmci.run()
fpcmci.dag(label_type = LabelType.NoLabels, node_layout = 'circular')
fpcmci.timeseries_dag()

# Population B
np.random.seed(8)
T = 1000
N = 3
dB_obs = np.random.random(size = (T, N))
for t in range(max_lag, T):
    dB_obs[t, 1] += 0.75 * dB_obs[t-1, 0]**2
    dB_obs[t, 2] += 1.3 * dB_obs[t-1, 0] * dB_obs[t-2, 1]

dfB_obs = Data(dB_obs)
# dfA_obs.plot_timeseries()

cie = CIE(CM, dfA_obs)
Aint_id = cie.addIntData('X_1', dfA_int)
Bobs_id = cie.addObsData(dfB_obs)
p, e = cie.whatHappensTo('X_2').If('X_1', 5).In(Bobs_id)

# Generate an array of x values (the domain over which to plot the density)
x_values = np.linspace(min(dfB_obs.d["X_2"]), max(dfB_obs.d["X_2"]), len(p))

# Plot the density function
plt.figure(figsize=(10, 6))
plt.plot(x_values, p, label='Density Function')

# Add a vertical line for the expectation
plt.axvline(x=e, color='r', linestyle='--', label=f'Expectation: {e:.2f}')

# Add labels and a legend
plt.xlabel('X_2')
plt.ylabel('Density')
plt.title('Density Function and Expectation')
plt.legend()

# Show the plot
plt.show()