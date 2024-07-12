import numpy as np

from matplotlib import pyplot as plt    
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.CAnDOIT_lpcmci import CAnDOIT
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn
# from tigramite.independence_tests.cmisymb import CMIsymb


# Set a seed for reproducibility
# seed = 19

# Choose the time series length
# T = 500

# # Specify the model (note that here, unlike in the typed equations, variables
# # are indexed starting from 0)
# def lin(x): return x

# links = {0: [((0, -1), 0.9, lin), ((1, 0), 0.6, lin)],
#          1: [],
#          2: [((2, -1), 0.9, lin), ((1, -1), 0.4, lin)],
#          3: [((3, -1), 0.9, lin), ((2, -2), -0.5, lin)]                                    
#         }


min_lag = 1
max_lag = 2
np.random.seed(19)
nsample = 500
nfeature = 4

d = np.random.random(size = (nsample, nfeature))
for t in range(max_lag, nsample):
  d[t, 0] += 0.9 * d[t-1, 0] + 0.6 * d[t, 1]
  d[t, 2] += 0.9 * d[t-1, 2] + 0.4 * d[t-1, 1]
  d[t, 3] += 0.9 * d[t-1, 3] -0.5 * d[t-2, 2]




# Specify dynamical noise term distributions, here unit variance Gaussians
# random_state = np.random.RandomState(seed)
# noises = noises = [random_state.randn for j in links.keys()]

# Generate data according to the full structural causal process
# data_full, nonstationarity_indicator = toys.structural_causal_process(links=links, T=T, noises=noises, seed=seed)
# assert not nonstationarity_indicator

# Remove the unobserved component time series
data_obs = d[:, [0, 2, 3]]

# Number of observed variables
N = data_obs.shape[1]

# Initialize dataframe object, specify variable names
var_names = [r'$X^{%d}$' % j for j in range(N)]
dataframe = pp.DataFrame(data_obs, var_names=var_names)

# Create a LPCMCI object, passing the dataframe and (conditional)
# independence test objects.
parcorr = ParCorr(significance='analytic')
lpcmci = LPCMCI(dataframe=dataframe, 
                cond_ind_test=parcorr,
                verbosity=1)

# Define the analysis parameters.
tau_max = 2
pc_alpha = 0.01

# Run LPCMCI
results = lpcmci.run_lpcmci(tau_max=tau_max,
                            pc_alpha=pc_alpha)

# Plot the learned time series DPAG
tp.plot_time_series_graph(graph=results['graph'],
                          val_matrix=results['val_matrix'])
plt.show()



######################################################## INTERVENTION ########################################################
# I want to do an intervention on X_1. So, I create a context variable CX_1 which models the intervention
# CX_1 does not have any parent and it is connected ONLY to X_1 @ time t-1
int_data = dict()
    
# X_0
d_int0 = np.random.random(size = (100, nfeature))
d_int0[:, 2] = 3 * np.ones(shape = (100,)) 
for t in range(max_lag, 100):
    d_int0[t, 0] += 0.9 * d_int0[t-1, 0] + 0.6 * d_int0[t, 1]
    # d_int0[t, 2] += 0.9 * d_int0[t-1, 2] + 0.4 * d_int0[t-1, 1]
    d_int0[t, 3] += 0.9 * d_int0[t-1, 3] -0.5 * d_int0[t-2, 2]
        
data_int = d_int0[:, [0, 2, 3]]
df_int = Data(data_int, vars = ['X_0', 'X_2', 'X_3'])
int_data['X_2'] =  df_int
# int_data = {}



 

candoit_lpcmci = CAnDOIT(Data(data_obs, vars = ['X_0', 'X_2', 'X_3']), 
                         int_data,
                         f_alpha = 0.5, 
                         alpha = pc_alpha, 
                         min_lag = 0, 
                         max_lag = tau_max, 
                         sel_method = TE(TEestimator.Gaussian), 
                         val_condtest = parcorr,
                         verbosity = CPLevel.DEBUG,
                         neglect_only_autodep = True,
                         plot_data = False,
                         exclude_context = True)
    
candoit_lpcmci_cm = candoit_lpcmci.run(nofilter=True)
candoit_lpcmci_cm.ts_dag(min_width=3, max_width=5, x_disp=0.5)
