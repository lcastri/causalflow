# Imports
import os
from time import time
import numpy as np
import pandas as pd
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import DataType

from causalflow.causal_discovery.baseline.JPCMCIplus import JPCMCIplus
from causalflow.causal_discovery.tigramite.independence_tests.regressionCI import RegressionCI
from causalflow.preprocessing.data import Data
from utils import *


class TOD(Enum):
    STARTING = "STARTING"
    POSTER = "POSTER"
    BUFFET = "BUFFET"
    OFF = "OFF"

TODS = {t.value: i for i, t in enumerate(TOD)}

class WP(Enum):
    ROOM1 = "r1"
    ROOM2 = "r2"
    CORRIDOR1 = "c1"
    CORRIDOR2 = "c2"
    CORRIDOR3 = "c3"
    CORRIDOR4 = "c4"
    CORRIDOR5 = "c5"
    CORRIDOR6 = "c6"
    CORRIDOR7 = "c7"
    CORRIDOR8 = "c8"
    CORRIDOR9 = "c9"
    CORRIDOR10 = "c10"
    CORRIDOR11 = "c11"
    CORRIDOR12 = "c12"
    CORRIDOR13 = "c13"
    CORRIDOR14 = "c14"
    CORRIDOR15 = "c15"
    CORRIDOR16 = "c16"
    CORRIDOR17 = "c17"
    CORRIDOR18 = "c18"
    CORRIDOR19 = "c19"
    CORRIDOR20 = "c20"
    CORRIDOR21 = "c21"
    CORRIDOR22 = "c22"
    CORRIDOR23 = "c23"
    CORRIDOR24 = "c24"
    CORRIDOR25 = "c25"
    CORRIDOR26 = "c26"

WPS = {wp.value: i for i, wp in enumerate(WP)}
ID_TO_WP = {v: k for k, v in WPS.items()}  


INDIR = '/home/lcastri/git/causal-sim2real/HRISim_docker/src/HRISim/postprocessing/hrisim_postprocess/csv_pp-2/'

var_names =  ["S", "W", "V", "L", "D", "O"]
# var_names =  ["S", "W", "V", "L", "D", "O", "DIST"]
node_classification = {
    0: "time_context",
    1: "space_context",
    2: "system",
    3: "system",
    4: "system",
    5: "space_context",
}

NODE_COLOR = {}
for node, classification in node_classification.items():
    NODE_COLOR[var_names[node]] = 'orange' if classification == "system" else 'lightgray'

DATA_DICT = {}
DATA_TYPE = {}
for wp in WP:
    dfs = []
    for tod in TOD:
        if tod == TOD.STARTING: continue
        bagname = f"DISCOVERY_{tod.value}_{wp.value}"
        print(f"### Loading : {bagname}")
        filename = os.path.join(INDIR, f"DISCOVERY_{tod.value}", f"{bagname}.csv")
        DF = pd.read_csv(filename)
            
        # Check for NaN values
        if DF.isnull().values.any():
            print(f"Warning: NaN values found in {filename}. Skipping this file.")
            
        dfs.append(DF[var_names])
        
    idx = len(DATA_DICT)
    DATA_DICT[idx] = Data(pd.concat(dfs, axis=0), varnames = var_names)

# for idx, data in DATA_DICT.items():
#     # Detrend the data
#     # data.d['TOD'] = detrend(data.d['TOD'].values, window_size=100)
#     data.plot_timeseries()


DATA_TYPE = {
    'S': DataType.Discrete,
    'W': DataType.Discrete,
    'V': DataType.Continuous,
    'L': DataType.Continuous,
    'D': DataType.Continuous,
    'O': DataType.Discrete,
}

jpcmciplus = JPCMCIplus(data = DATA_DICT,
                        min_lag = 0,
                        max_lag = 1,
                        val_condtest = RegressionCI(), 
                        node_classification = node_classification,
                        data_type = DATA_TYPE,
                        alpha = 0.05,
                        verbosity=CPLevel.INFO,
                        resfolder="results/INB_3floor")

# Run J-PCMCI+
start_time = time()
CM = jpcmciplus.run()
end_time = time()
print(f"J-PCMCI+ completed in {end_time - start_time:.2f} seconds.")

# CM = CM.filter_alpha(0.00001)
CM.plot_graph(node_layout = 'circular', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_circular', node_color=NODE_COLOR)
CM.plot_graph(node_layout = 'dot', node_size = 4, min_cross_width = 0.5, max_cross_width = 1.5,
       save_name=jpcmciplus.dag_path + '_dot', node_color=NODE_COLOR)
CM.plot_ts_graph(node_size = 4, 
          min_cross_width = 0.5, max_cross_width = 1.5, 
          x_disp=1.5, y_disp=0.2,
          save_name=jpcmciplus.ts_dag_path, node_color=NODE_COLOR)
jpcmciplus.save()