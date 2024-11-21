# Imports
import copy
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *
import time

DAGDIR = '/home/lcastri/git/causalflow/results/RAL/causal discovery/res.pkl'
CIEDIR = '/home/lcastri/git/causalflow/CIE_optimised/cie.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024']
# BAGNAME= ['BL100_21102024', 'BL75_29102024', 'BL50_22102024', 'BL25_28102024']

with open(CIEDIR, 'rb') as f:
    cie = CIE.load(pickle.load(f))
with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))
    
treatment_len = 25
dfs = []
for bagname in BAGNAME:
    for wp in [WP.TABLE3]:
        for tod in [TOD.LUNCH]:
            if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            dfs.append(pd.read_csv(filename))
            
concat_df = pd.concat(dfs, ignore_index=True)
concat_df.drop(concat_df[concat_df['B_S'] == 1].index, inplace=True)
DATA_DICT_TRAIN = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[:len(concat_df) - treatment_len], vars = CM.features + ["pf_elapsed_time"])
DATA_DICT_TEST = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[len(concat_df) - treatment_len:], vars = CM.features + ["pf_elapsed_time"])
T = np.concatenate((DATA_DICT_TRAIN.d["pf_elapsed_time"].values[- CM.max_lag:], DATA_DICT_TEST.d["pf_elapsed_time"].values[0:]), axis=0)
DATA_DICT_TRAIN.shrink(CM.features)
DATA_DICT_TEST.shrink(CM.features)
prior_knowledge = {f: DATA_DICT_TEST.d[f].values[:] for f in ['TOD', 'B_S', 'WP']}

res = cie.whatIf(NODES.RV.value, 
                 DATA_DICT_TEST.d.values[:, DATA_DICT_TEST.features.index(NODES.RV.value)], 
                 DATA_DICT_TRAIN.d.values,
                 prior_knowledge
                 )

FEATURES = [f for f in CM.features if cie.node_type[f] is not NodeType.Context]
N = len(FEATURES)
# Set up the subplots
fig, axes = plt.subplots(N, 1, figsize=(8, N * 3), sharex=True)

# Plot each column in a different subplot
for f in FEATURES:
    i = DATA_DICT_TRAIN.features.index(f)
    observation = np.concatenate((DATA_DICT_TRAIN.d.values[-CM.max_lag:], DATA_DICT_TEST.d.values[0,:].reshape(1,-1), np.nan*np.ones_like(res)), axis=0)
    prediction = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:]), res), axis=0)
    ground_truth = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:]), DATA_DICT_TEST.d.values), axis=0)
    axes[FEATURES.index(f)].plot(observation[:, i], linestyle = '-', color = "black", label = "observation")
    axes[FEATURES.index(f)].plot(ground_truth[:, i], linestyle = '-', color = "tab:orange", label = "ground-truth")
    axes[FEATURES.index(f)].plot(prediction[:, i], linestyle = '--', color = "tab:blue", label = "prediction")
    axes[FEATURES.index(f)].set_ylabel(DATA_DICT_TRAIN.features[i])
    axes[FEATURES.index(f)].grid(True)
    RMSE = np.sqrt(np.mean((res[:, i] - DATA_DICT_TEST.d.values[:, i]) ** 2))
    NRMSE = RMSE/np.std(DATA_DICT_TEST.d.values[:, i]) if np.std(DATA_DICT_TEST.d.values[:, i]) != 0 else 0
    axes[FEATURES.index(f)].set_title(f"NRMSE: {NRMSE:.4f}")
    axes[FEATURES.index(f)].legend(loc='best')

# Show the plot
plt.xticks(ticks = list(range(len(T))),
           labels = [time.strftime("%H:%M:%S", time.gmtime(8*3600+t)) for t in list(T)],
           rotation=45
    )
plt.tight_layout()
plt.show()