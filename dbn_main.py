from matplotlib import pyplot as plt
import numpy as np
from causalflow.preprocessing.data import Data
import pandas as pd

from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator

from causalflow.dbn.DynamicBayesianNetwork import DynamicBayesianNetwork as DBN

alpha = 0.05
max_lag = 2
min_lag = 1

np.random.seed(1)
T = 2000
N = 5
d_obs = np.random.normal(0, 1, size = (T, N))
for t in range(max_lag, T):
    d_obs[t, 1] += 0.75 * d_obs[t-1, 0]**2
    d_obs[t, 2] += 1.3 * d_obs[t-1, 0] * d_obs[t-2, 1]
    d_obs[t, 3] += 0.7 * d_obs[t-1, 3] * d_obs[t-2, 4]
    

df_obs = Data(d_obs)
df_obs.plot_timeseries()

fpcmci = FPCMCI(df_obs, 
                f_alpha = alpha, 
                alpha = alpha, 
                min_lag = min_lag, 
                max_lag = max_lag, 
                sel_method = TE(TEestimator.Auto), 
                val_condtest = GPDC(significance = 'analytic', gp_params = None),
                verbosity = CPLevel.DEBUG,
                resfolder = 'results/dbn')
CM = fpcmci.run()


T = 300
N = 5
d_int = np.random.normal(0, 1, size = (T, N))
d_int[:, 1] = np.squeeze(5 * np.ones(shape=(T,1)))
for t in range(max_lag, T):
    d_int[t, 2] += 1.3 * d_int[t-1, 0] * d_int[t-2, 1]
    d_int[t, 3] += 0.7 * d_int[t-1, 3] * d_int[t-2, 4]

df_int = Data(d_int)
df_int.plot_timeseries()

DyBN = DBN(CM, df_obs)
DyBN.addInterventionData("X_1", df_int)




# df_test = df_test[:, [df_train.orig_features.index(f) for f in sel_var]]
# DyBN.addInterventionData("X_1", df_test[:, df_train.features.index("X_1")])


# Prediction of the next T_test time step of the sys given X_0 (common driver)
# df_test = df_test[:, [df_train.orig_features.index(f) for f in CM.features]]
# estimated_data = DyBN.predictEffect("X_0", df_test[:, df_train.features.index("X_0")])

# fig, axs = plt.subplots(4)

# axs[0].plot(range(train_len, T), df_test[:, 1], label="Observed $X_1$", color='green')
# axs[0].plot(range(train_len, T), estimated_data[:, 1], label="Estimated $\hat X_1$", color='red')
# axs[0].set_title("$X_1$ observed vs estimated")
# axs[0].legend()

# axs[1].plot(range(train_len, T), df_test[:, 2], label="Observed $X_2$", color='green')
# axs[1].plot(range(train_len, T), estimated_data[:, 2], label="Estimated $\hat X_2$", color='red')
# axs[1].set_title("$X_2$ observed vs estimated")
# axs[1].legend()

# axs[2].plot(range(train_len, T), df_test[:, 3], label="Observed $X_3$", color='green')
# axs[2].plot(range(train_len, T), estimated_data[:, 3], label="Estimated $\hat X_3$", color='red')
# axs[2].set_title("$X_3$ observed vs estimated")
# axs[2].legend()

# axs[3].plot(range(train_len, T), df_test[:, 4], label="Observed $X_4$", color='green')
# axs[3].plot(range(train_len, T), estimated_data[:, 4], label="Estimated $\hat X_4$", color='red')
# axs[3].set_title("$X_4$ observed vs estimated")
# axs[3].legend()

# plt.show()

# fig, axs = plt.subplots(2)

# n = len(df_test[:, 0])
# rmse = np.sqrt(np.sum((df_test[:, 0] - estimated_data[:, 0]) ** 2) / n)
# axs[0].plot(range(train_len, T), abs(df_test[:, 0] - estimated_data[:, 0]), label="Abs err", color='blue')
# axs[0].set_title(f"$X_0$ Abs Error | Root Mean Squared Error: {rmse:.2f}")
# axs[0].legend()

# n = len(df_test[:, 2])
# rmse = np.sqrt(np.sum((df_test[:, 2] - estimated_data[:, 2]) ** 2) / n)
# axs[1].plot(range(train_len, T), abs(df_test[:, 2] - estimated_data[:, 2]), label="Abs err", color='blue')
# axs[1].set_title(f"$X_2$ Abs Error | Root Mean Squared Error: {rmse:.2f}")
# axs[1].legend()
# plt.show()