# Imports
import os
import numpy as np
import pandas as pd
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import DataType

from causalflow.causal_discovery.baseline.JPCMCIplus import JPCMCIplus
from tigramite.independence_tests.regressionCI import RegressionCI
from causalflow.preprocessing.data import Data
from utils_2 import *

def detrend(signal, window_size):
    detrended_signal = np.copy(signal)
    # Loop through the signal and subtract the window mean
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i+window_size]
        window_median = np.median(window)
        detrended_signal[i:i+window_size] -= window_median
        
    return detrended_signal


INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024']
USE_SUBSAMPLED = True

node_classification = {
    list(NODES).index(NODES.TOD): "space_context",
    list(NODES).index(NODES.RV): "system",
    list(NODES).index(NODES.RB): "system",
    list(NODES).index(NODES.CS): "space_context",
    list(NODES).index(NODES.PD): "system",
    list(NODES).index(NODES.ELT): "system",
    # list(NODES).index(NODES.OBS): "system",
    list(NODES).index(NODES.OBS): "space_context",
    list(NODES).index(NODES.WP): "space_context",
}

NODE_COLOR = {}
for node, classification in node_classification.items():
    if classification == "system":
        NODE_COLOR[list(NODES)[node].value] = 'orange'
    elif classification == "space_context":
        NODE_COLOR[list(NODES)[node].value] = 'lightgray'

var_names = [n.value for n in NODES]
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
                filename = os.path.join(INDIR, "TOD/my_noised", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            else:
                filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            WPDF = pd.read_csv(filename)
            
            # Check for NaN values
            if WPDF.isnull().values.any():
                print(f"Warning: NaN values found in {filename}. Skipping this file.")
            
            dfs.append(WPDF[var_names])
        concatenated_df = pd.concat(dfs, ignore_index=True)
        idx = len(DATA_DICT)
        concatenated_df["TOD"] = detrend(concatenated_df["TOD"], 500)

        DATA_DICT[idx] = Data(concatenated_df, vars = var_names)
        # if wp == WP.DOOR_ENTRANCE:
        #     dd = copy.deepcopy(DATA_DICT[idx])
        #     dd.shrink(['TOD', 'T', 'R_V'])
        #     dd.plot_timeseries()
        # if wp == WP.DOOR_ENTRANCE: DATA_DICT[idx].plot_timeseries()
            # idx = len(DATA_DICT)
            # DATA_DICT[idx] = Data(WPDF[var_names], vars = var_names)
            # DATA_DICT[idx].plot_timeseries()

DATA_TYPE = {
    NODES.TOD.value: DataType.Continuous,
    NODES.RV.value: DataType.Continuous,
    NODES.RB.value: DataType.Continuous,
    NODES.CS.value: DataType.Discrete,
    NODES.PD.value: DataType.Continuous,
    NODES.ELT.value: DataType.Continuous,
    NODES.OBS.value: DataType.Discrete,
    NODES.WP.value: DataType.Discrete,
}
jpcmciplus = JPCMCIplus(data = DATA_DICT,
                        min_lag = 0,
                        max_lag = 1,
                        val_condtest = RegressionCI(), 
                        node_classification = node_classification,
                        data_type = DATA_TYPE,
                        alpha = 0.01,
                        verbosity=CPLevel.INFO,
                        resfolder=f"results/{'__'.join(BAGNAME)}_new")

# Run J-PCMCI+
CM = jpcmciplus.run()

CM = CM.filter_alpha(0.00001)
CM.dag(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_circular', node_color=NODE_COLOR)
CM.dag(node_layout = 'dot', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_dot', node_color=NODE_COLOR)
CM.ts_dag(node_size = 4, 
          min_cross_width = 0.5, max_cross_width = 1.5, 
          x_disp=1.5, y_disp=0.2,
          save_name=jpcmciplus.ts_dag_path, node_color=NODE_COLOR)
jpcmciplus.save()