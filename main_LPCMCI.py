from tigramite.lpcmci import LPCMCI as T_LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import data_processing as pp
from matplotlib import pyplot as plt
import numpy as np
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.baseline.LPCMCI import LPCMCI
from causalflow.preprocessing.data import Data
# Set a seed for reproducibility
seed = 19

# Choose the time series length
T = 500

# Specify the model (note that here, unlike in the typed equations, variables
# are indexed starting from 0)
def lin(x): return x

links = {0: [((0, -1), 0.9, lin), ((1, 0), 0.6, lin)],
         1: [],
         2: [((2, -1), 0.9, lin), ((1, -1), 0.4, lin)],
         3: [((3, -1), 0.9, lin), ((2, -2), -0.5, lin)]                                    
        }

# Specify dynamical noise term distributions, here unit variance Gaussians
random_state = np.random.RandomState(seed)
noises = noises = [random_state.randn for j in links.keys()]

# Generate data according to the full structural causal process
data_full, nonstationarity_indicator = toys.structural_causal_process(
    links=links, T=T, noises=noises, seed=seed)
assert not nonstationarity_indicator

# Remove the unobserved component time series
data_obs = data_full[:, [0, 2, 3]]

# Number of observed variables
N = data_obs.shape[1]

# Initialize dataframe object, specify variable names
var_names = [r'$X_{%d}$' % j for j in range(N)]
dataframe = pp.DataFrame(data_obs, var_names=var_names)










# Create a LPCMCI object, passing the dataframe and (conditional)
# independence test objects.
parcorr = ParCorr(significance='analytic')
lpcmci = T_LPCMCI(dataframe=dataframe, 
                cond_ind_test=parcorr,
                verbosity=1)

# Define the analysis parameters.
tau_max = 5
pc_alpha = 0.01

# Run LPCMCI
results = lpcmci.run_lpcmci(tau_max=tau_max,
                            pc_alpha=pc_alpha)

var_names = [r'X_{%d}' % j for j in range(N)]
df = Data(data_obs, vars = var_names)
ahah = LPCMCI(df,
                min_lag = 0, 
                max_lag = tau_max, 
                val_condtest = parcorr,
                verbosity = CPLevel.INFO,
                alpha = pc_alpha, 
                neglect_only_autodep = False)
ahah_res = ahah.run()
                    


# Plot the learned time series DPAG
# tp.plot_time_series_graph(graph=results['graph'],
#                           val_matrix=results['val_matrix'])
# plt.show()
ahah.CM.g['X_{1}'].sources[('X_{2}', 2)]["type"] = 'o->'
# ahah.timeseries_dag()


# Plot the learned time series DPAG
tp.plot_graph(graph=results['graph'],
              val_matrix=results['val_matrix'])
# plt.show()
ahah.dag(node_layout="circular")