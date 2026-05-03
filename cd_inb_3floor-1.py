# Imports
import os
from time import time
import numpy as np
import pandas as pd
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import DataType, LabelType

from causalflow.causal_discovery.baseline.JPCMCIplus import JPCMCIplus
from causalflow.causal_discovery.tigramite.independence_tests.regressionCI import RegressionCI
from causalflow.preprocessing.data import Data
from utils import *

def detrend(signal, window_size):
    detrended_signal = np.copy(signal)
    # Loop through the signal and subtract the window mean
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i+window_size]
        window_median = np.median(window)
        detrended_signal[i:i+window_size] -= window_median
        
    return detrended_signal


INDIR = '/home/lcastri/git/causal-sim2real/HRISim_docker/src/HRISim/postprocessing/hrisim_postprocess/csv_pp-1/'
BAGNAME= ['DISCOVERY_POSTER', 'DISCOVERY_BUFFET', 'DISCOVERY_OFF']
USE_SUBSAMPLED = True

var_names =  ["S", "V", "L", "W", "D", "O"]
# var_names =  ["S", "V", "L", "W", "D", "O", "DIST"]
node_classification = {
    0: "space_context",
    1: "system",
    2: "system",
    3: "time_context",
    4: "system",
    5: "space_context",
}

NODE_COLOR = {}
for node, classification in node_classification.items():
    NODE_COLOR[var_names[node]] = 'orange' if classification == "system" else 'lightgray'

DATA_DICT = {}
DATA_TYPE = {}
dfs = []
for bagname in BAGNAME:
    print(f"### Loading : {bagname}")
    filename = os.path.join(INDIR, f"{bagname}", f"{bagname}.csv")
    DF = pd.read_csv(filename)
            
    # Check for NaN values
    if DF.isnull().values.any():
        print(f"Warning: NaN values found in {filename}. Skipping this file.")
            
    idx = len(DATA_DICT)
    DATA_DICT[idx] = Data(DF[var_names], varnames = var_names)

# for idx, data in DATA_DICT.items():
#     # Detrend the data
#     # data.d['TOD'] = detrend(data.d['TOD'].values, window_size=100)
#     data.plot_timeseries()


DATA_TYPE = {
    'S': DataType.Discrete,
    'V': DataType.Continuous,
    'L': DataType.Continuous,
    'W': DataType.Discrete,
    'D': DataType.Continuous,
    'O': DataType.Discrete,
}

MIN_LAG = 0
MAX_LAG = 1

jpcmciplus = JPCMCIplus(data = DATA_DICT,
                        min_lag = 0,
                        max_lag = 1,
                        val_condtest = RegressionCI(), 
                        node_classification = node_classification,
                        data_type = DATA_TYPE,
                        alpha = 0.05,
                        verbosity=CPLevel.INFO,
                        resfolder="results/INB_3floor")

link_assumptions = {}
for i in range(len(var_names)):
    link_assumptions[i] = {}
    for j in range(len(var_names)):
        for lag in range(MIN_LAG, MAX_LAG + 1):
            if not (i == j and lag == 0):
                if lag == 0:
                    link_assumptions[i][(j, 0)] = 'o?o'
                else:
                    link_assumptions[i][(j, -lag)] = '-?>'
link_assumptions[var_names.index('S')][(var_names.index('D'), 0)] = '-->'
link_assumptions[var_names.index('D')][(var_names.index('S'), 0)] = '<--'
link_assumptions[var_names.index('V')][(var_names.index('D'), 0)] = '-->'
link_assumptions[var_names.index('D')][(var_names.index('V'), 0)] = '<--'
del link_assumptions[var_names.index('V')][(var_names.index('L'), 0)]
del link_assumptions[var_names.index('L')][(var_names.index('V'), 0)]
# del link_assumptions[var_names.index('D')][(var_names.index('V'), 0)]


# Run J-PCMCI+
start_time = time()
CM = jpcmciplus.run(link_assumptions=link_assumptions)
end_time = time()
print(f"J-PCMCI+ completed in {end_time - start_time:.2f} seconds.")

# CM = CM.filter_alpha(0.00001)
# CM.plot_graph(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
#        save_name=jpcmciplus.dag_path + '_circular', node_color=NODE_COLOR)
CM.plot_graph(node_layout = 'dot', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_dot', node_color=NODE_COLOR, label_type = LabelType.OnlyLagged)
node_layout = {'$O$': np.array([0.25, 0.95]), 
               '$W$': np.array([0.68333333, 0.95]), 
               '$S$': np.array([1., 0.95]), 
               '$L$': np.array([0.05, 0.475]), 
               '$D$': np.array([0.84166667, 0.475]), 
               '$V$': np.array([0.45, 0.475])}
CM.plot_graph(node_layout = node_layout, node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_my', node_color=NODE_COLOR, label_type = LabelType.OnlyLagged)
CM.plot_ts_graph(node_size = 4, 
          min_cross_width = 0.5, max_cross_width = 1.5, 
          x_disp=1.5, y_disp=0.2,
          save_name=jpcmciplus.ts_dag_path, node_color=NODE_COLOR)
jpcmciplus.save()