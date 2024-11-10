from matplotlib import pyplot as plt
import numpy as np
from causalflow.basics.constants import DataType
from causalflow.preprocessing.data import Data
from causalflow.causal_discovery.tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE


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
    dA_obs[t, 1] += 0.75 * dA_obs[t-1, 0]
    dA_obs[t, 2] += 1.3 * dA_obs[t-2, 1]

dfA_obs = Data(dA_obs)
# dfA_obs.plot_timeseries()

T = 300
N = 3
dA_int = np.random.normal(0, 1, size = (T, N))
dA_int[:, 1] = np.squeeze(5 * np.ones(shape = (T, 1)))
for t in range(max_lag, T):
    dA_int[t, 2] += 1.3 * dA_int[t-2, 1]

dfA_int = Data(dA_int)
# dfA_int.plot_timeseries()


# Causal Discovery on Population A
NODE_COLOURS = {
    "X_0": "orange",
    "X_1": "lightgray",
    "X_2": "red",
}  
fpcmci = FPCMCI(dfA_obs, 
                f_alpha = alpha, 
                alpha = alpha, 
                min_lag = min_lag, 
                max_lag = max_lag, 
                sel_method = TE(TEestimator.Gaussian), 
                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                verbosity = CPLevel.DEBUG,
                resfolder = 'results/dbn')
CM = fpcmci.run()
CM.dag(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=fpcmci.dag_path, node_color=NODE_COLOURS)
CM.ts_dag(node_size = 4, 
          min_cross_width = 0.5, max_cross_width = 1.5, 
          x_disp=1.5, y_disp=0.2,
          save_name=fpcmci.ts_dag_path, node_color=NODE_COLOURS)

# Population B
np.random.seed(8)
T = 1000
N = 3
dB_obs = np.random.random(size = (T, N))
for t in range(max_lag, T):
    dB_obs[t, 1] += 0.75 * dB_obs[t-1, 0]
    dB_obs[t, 2] += 1.3 * dB_obs[t-2, 1]

dfB_obs = Data(dB_obs)
# dfB_obs.plot_timeseries()
DATA_TYPE = {
    "X_0": DataType.Continuous,
    "X_1": DataType.Continuous,
    "X_2": DataType.Continuous,
}  
cie = CIE(CM, nsample=100, data_type=DATA_TYPE, atol=0.1)
Aobs_id = cie.addObsData(dfA_obs)
Aint_id = cie.addIntData('X_1', dfA_int)
# cie.save('/home/lcastri/git/causalflow/results/dbn/cie.pkl')

res = cie.whatIf('X_1', 
                 dfB_obs.d.values[int(len(dfB_obs.d.values)/2):int(len(dfB_obs.d.values)/2)+50, 1], 
                 dfB_obs.d.values[:int(len(dfB_obs.d.values)/2), :])

result = np.concatenate((dfB_obs.d.values[:int(len(dfB_obs.d.values) / 2), :], res), axis=0)
# Get the number of columns
num_columns = result.shape[1]

# Set up the subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 3), sharex=True)

# Plot each column in a different subplot
for i in range(num_columns):
    axes[i].plot(result[:, i])
    axes[i].plot(dfB_obs.d.values[:int(len(dfB_obs.d.values)/2 + 50), i])
    axes[i].set_ylabel(dfA_int.features[i])
    axes[i].grid(True)

# Show the plot
plt.xlabel('Index')
plt.tight_layout()
plt.show()