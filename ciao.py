# Imports
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *

INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME = ['BL100_21102024'] 
USE_SUBSAMPLED = True


var_names = [n.value for n in NODES]
DATA_DICT = {}
for bagname in BAGNAME:
    dfs = []
    for tod in [TOD.MORNING]:
        print(f"Loading : {bagname}-{tod.value}-{WP.DELIVERY_POINT.value}")
        if USE_SUBSAMPLED:
            filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{WP.DELIVERY_POINT.value}.csv")
        else:
            filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{WP.DELIVERY_POINT.value}.csv")
        dfs.append(pd.read_csv(filename))
            
    concatenated_df = pd.concat(dfs, ignore_index=True)
    OBS = Data(concatenated_df[var_names].values, vars = var_names)
    dfs = []
    for tod in [TOD.LUNCH, TOD.AFTERNOON]:
            print(f"Loading : {bagname}-{tod.value}-{WP.DELIVERY_POINT.value}")
            if USE_SUBSAMPLED:
                filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{WP.DELIVERY_POINT.value}.csv")
            else:
                filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{WP.DELIVERY_POINT.value}.csv")
            dfs.append(pd.read_csv(filename))
            
    concatenated_df = pd.concat(dfs, ignore_index=True)
    DF = Data(concatenated_df[var_names].values, vars = var_names)

# DF.plot_timeseries()

with open('test.npy', 'rb') as f:
    res = np.load(f)
    
# result = np.concatenate((OBS.d.values, res), axis=0)

# Get the number of columns
num_columns = res.shape[1]
# Set up the subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 3), sharex=True)

# Plot each column in a different subplot
for i in range(num_columns):
    axes[i].plot(res[:, i], label = "predicted")
    axes[i].plot(DF.d.values[:, i], label = "ground-truth")
    axes[i].set_ylabel(DF.features[i])
    axes[i].grid(True)

# Show the plot
plt.xlabel('Index')
plt.tight_layout()
plt.legend()
plt.show()