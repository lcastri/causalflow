import numpy as np

from tigramite import data_processing as pp
# from tigramite.lpcmci import LPCMCI
# from causalflow.causal_discovery.tigramite.lpcmci import LPCMCI
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
from causalflow.causal_discovery.tigramite.independence_tests.parcorr import ParCorr as myParCorr
from tigramite.independence_tests.parcorr import ParCorr
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.CAnDOIT_lpcmci import CAnDOIT
# from causalflow.causal_discovery.CAnDOIT_pcmciplus import CAnDOIT
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator


tau_max = 2
pc_alpha = 0.05
np.random.seed(19)
nsample_obs = 500
nsample_int = 150
nfeature = 4

d = np.random.random(size = (nsample_obs, nfeature))
for t in range(tau_max, nsample_obs):
  d[t, 0] += 0.9 * d[t-1, 0] + 0.6 * d[t, 1]
  d[t, 2] += 0.9 * d[t-1, 2] + 0.4 * d[t-1, 1]
  d[t, 3] += 0.9 * d[t-1, 3] -0.5 * d[t-2, 2]

# Remove the unobserved component time series
# data_obs = d
data_obs = d[:, [0, 2, 3]]

# Number of observed variables
N = data_obs.shape[1]

var_names = ['X_0', 'X_2', 'X_3']
# var_names = ['X_0', 'X_1', 'X_2', 'X_3']
# Initialize dataframe object, specify variable names
dataframe = pp.DataFrame(data_obs, var_names=var_names)

# Create a LPCMCI object, passing the dataframe and (conditional)
# independence test objects.
parcorr = ParCorr(significance='analytic')
# lpcmci = LPCMCI(dataframe=dataframe, 
#                 cond_ind_test=parcorr,
#                 verbosity=2)
lpcmci = LPCMCI(Data(data_obs, vars = var_names),
                min_lag = 0,
                max_lag = tau_max,
                sys_context=[],
                val_condtest = parcorr,
                verbosity = CPLevel.DEBUG,
                alpha = pc_alpha,
                resfolder = 'results/toy/lpcmci',
                )

# Run LPCMCI
results = lpcmci.run()
# results = lpcmci.run(link_assumptions=JCI_assumptions(N, tau_max))

# Plot the learned time series DPAG
# tp.plot_time_series_graph(graph=results['graph'],
#                           val_matrix=results['val_matrix'])
# plt.show()



######################################################## INTERVENTION ########################################################
# I want to do an intervention on X_1. So, I create a context variable CX_1 which models the intervention
# CX_1 does not have any parent and it is connected ONLY to X_1 @ time t-1
int_data = dict()
    
# X_0
d_int0 = np.random.random(size = (nsample_int, nfeature))
d_int0[0:tau_max, :] = d[len(d)-tau_max:,:]
d_int0[:, 0] = 3 * np.ones(shape = (nsample_int,)) 
# d_int0[:, 2] = 3 * np.ones(shape = (nsample_int,)) 
for t in range(tau_max, nsample_int):
    # d_int0[t, 0] += 0.9 * d_int0[t-1, 0] + 0.6 * d_int0[t, 1]
    d_int0[t, 2] += 0.9 * d_int0[t-1, 2] + 0.4 * d_int0[t-1, 1]
    d_int0[t, 3] += 0.9 * d_int0[t-1, 3] -0.5 * d_int0[t-2, 2]
        
data_int = d_int0
data_int = d_int0[:, [0, 2, 3]]
df_int = Data(data_int, vars = var_names)
int_data['X_0'] =  df_int
# int_data['X_2'] =  df_int
# int_data = {}



# candoit_lpcmci = CAnDOIT(Data(data_obs, vars = ['X_0', 'X_1', 'X_2', 'X_3']), 
candoit_lpcmci = CAnDOIT(Data(data_obs, vars = ['X_0', 'X_2', 'X_3']), 
                         int_data,
                         f_alpha = 0.5, 
                         alpha = pc_alpha, 
                         min_lag = 0, 
                         max_lag = tau_max, 
                         sel_method = TE(TEestimator.Gaussian), 
                         val_condtest = myParCorr(significance='analytic'),
                         verbosity = CPLevel.DEBUG,
                         neglect_only_autodep = True,
                         plot_data = True,
                         exclude_context = False,
                         resfolder = 'results/toy/candoit')
    
candoit_lpcmci_cm = candoit_lpcmci.run(nofilter=True)
candoit_lpcmci_cm.ts_dag(min_width=3, max_width=5, x_disp=0.5)
