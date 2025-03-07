# Imports
import os
import numpy as np
import pandas as pd
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import DataType

from causalflow.causal_discovery.baseline.JPCMCIplus import JPCMCIplus
from tigramite.independence_tests.regressionCI import RegressionCI
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


INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['test-obs-21022025']
USE_SUBSAMPLED = True

var_names =  ["EC", "R_V", "OBS"]
node_classification = {
    0: "system",
    1: "system",
    2: "space_context",
}

NODE_COLOR = {}
for node, classification in node_classification.items():
    # if classification == "system":
    NODE_COLOR[var_names[node]] = 'orange' if classification == "system" else 'lightgray'
    # elif classification == "space_context":
    #     NODE_COLOR[list(NODES)[node].value] = 'lightgray'

DATA_DICT = {}
DATA_TYPE = {}
for bagname in BAGNAME:
    for wp in WP:
        dfs = []
        if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
        print(f"### Loading : {bagname}-{wp.value}")
        for tod in TOD:
            print(f"- {tod.value}")
            if USE_SUBSAMPLED:
                filename = os.path.join(INDIR, "HH/my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            else:
                filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            WPDF = pd.read_csv(filename)
            
            # Check for NaN values
            if WPDF.isnull().values.any():
                print(f"Warning: NaN values found in {filename}. Skipping this file.")
            
            dfs.append(WPDF[var_names])
        concatenated_df = pd.concat(dfs, ignore_index=True)
        idx = len(DATA_DICT)
        # concatenated_df["TOD"] = detrend(concatenated_df["TOD"], 500)

        DATA_DICT[idx] = Data(concatenated_df, varnames = var_names)
        DATA_DICT[idx].plot_timeseries()

DATA_TYPE = {
    # NODES.WP.value: DataType.Discrete,
    # NODES.TOD.value: DataType.Discrete,
    # NODES.PD.value: DataType.Continuous,
    'EC': DataType.Continuous,
    'R_V': DataType.Continuous,
    'OBS': DataType.Discrete,
    # NODES.EC.value: DataType.Continuous,
    # NODES.RV.value: DataType.Continuous,
    # NODES.OBS.value: DataType.Discrete,
}

jpcmciplus = JPCMCIplus(data = DATA_DICT,
                        min_lag = 0,
                        max_lag = 1,
                        val_condtest = RegressionCI(), 
                        node_classification = node_classification,
                        data_type = DATA_TYPE,
                        alpha = 0.05,
                        verbosity=CPLevel.INFO,
                        resfolder=f"results/{'__'.join(BAGNAME)}_new")

# Run J-PCMCI+
CM = jpcmciplus.run()

# CM = CM.filter_alpha(0.00001)
CM.dag(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_circular', node_color=NODE_COLOR)
CM.dag(node_layout = 'dot', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_dot', node_color=NODE_COLOR)
CM.ts_dag(node_size = 4, 
          min_cross_width = 0.5, max_cross_width = 1.5, 
          x_disp=1.5, y_disp=0.2,
          save_name=jpcmciplus.ts_dag_path, node_color=NODE_COLOR)
jpcmciplus.save()