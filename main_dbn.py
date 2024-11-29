from matplotlib import pyplot as plt
import numpy as np
from causalflow.basics.constants import DataType, NodeType
from causalflow.graph import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_discovery.tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.basics.metrics import fully_connected_dag

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
# CM.dag(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
#        save_name=fpcmci.dag_path, node_color=NODE_COLOURS)
# CM.ts_dag(node_size = 4, 
#           min_cross_width = 0.5, max_cross_width = 1.5, 
#           x_disp=1.5, y_disp=0.2,
#           save_name=fpcmci.ts_dag_path, node_color=NODE_COLOURS)

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
NODE_TYPE = {
    "X_0": NodeType.System,
    "X_1": NodeType.System,
    "X_2": NodeType.System,
}
cie = CIE(CM, data_type=DATA_TYPE, node_type=NODE_TYPE, model_path='testDBN', verbosity=CPLevel.DEBUG)
Aobs_id = cie.addObsData(dfA_obs)

fullg = fully_connected_dag(list(CM.features), min_lag, max_lag)
fulldag = DAG(list(CM.features), min_lag, max_lag, scm = fullg)
cie2 = CIE(fulldag, data_type=DATA_TYPE, node_type=NODE_TYPE, model_path='testDBNfull', verbosity=CPLevel.DEBUG)
Aobs_id = cie2.addObsData(dfA_obs)

res = cie.whatIf('X_1', 
                 dfB_obs.d.values[int(len(dfB_obs.d.values)/2):int(len(dfB_obs.d.values)/2)+50, 1], 
                 dfB_obs.d.values[:int(len(dfB_obs.d.values)/2), :],
                 {'X_0':dfB_obs.d.values[int(len(dfB_obs.d.values)/2):int(len(dfB_obs.d.values)/2)+50, 0]})
# res2 = cie2.whatIf('X_1', 
#                  dfB_obs.d.values[int(len(dfB_obs.d.values)/2):int(len(dfB_obs.d.values)/2)+50, 1], 
#                  dfB_obs.d.values[:int(len(dfB_obs.d.values)/2), :],
#                  {'X_0':dfB_obs.d.values[int(len(dfB_obs.d.values)/2):int(len(dfB_obs.d.values)/2)+50, 0]})

result = np.concatenate((dfB_obs.d.values[:int(len(dfB_obs.d.values) / 2), :], res), axis=0)
# result2 = np.concatenate((dfB_obs.d.values[:int(len(dfB_obs.d.values) / 2), :], res2), axis=0)
# Get the number of columns
num_columns = result.shape[1]

# Set up the subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 3), sharex=True)

# Plot each column in a different subplot
for i in range(num_columns):
    axes[i].plot(result[:, i], linestyle = '--', color = "tab:blue")
    # axes[i].plot(result2[:, i], linestyle = '--', color = "tab:green")
    axes[i].plot(dfB_obs.d.values[:int(len(dfB_obs.d.values)/2 + 50), i], linestyle = '-', color = "tab:orange")
    axes[i].set_ylabel(dfA_int.features[i])
    axes[i].grid(True)

# Show the plot
plt.xlabel('Index')
plt.tight_layout()
plt.show()