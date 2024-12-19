# Imports
import os
import joblib
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

DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024/res.pkl'
CIEDIR = '/home/lcastri/git/causalflow/CIE_100_HH_noBAC/cie.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024']
# BAGNAME= ['BL100_21102024', 'BL75_29102024', 'BL50_22102024', 'BL25_28102024']

cie = CIE.load(CIEDIR)
with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))
    
treatment_len = 120
dfs = []
wp = WP.TABLE2
tod = 10
for bagname in BAGNAME:
    files = [f for f in os.listdir(os.path.join(INDIR, "TOD/HH", f"{bagname}"))]
    files_split = [f.split('_') for f in files]
    wp_files = [f for f in files_split if len(f) == 3 and f[1] == f"{tod}h" and f[2].split('.')[0] == wp.value]
    wp_files = sorted(wp_files, key=lambda x: int(x[1].replace('h', '')))
    wp_files = ['_'.join(wp_f) for wp_f in wp_files]
    for file in wp_files:
        print(f"Loading : {file}")
        filename = os.path.join(INDIR, "TOD/HH", f"{bagname}", file)
        dfs.append(pd.read_csv(filename))
            
concat_df = pd.concat(dfs, ignore_index=True)
DATA_DICT_TRAIN = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[:len(concat_df) - treatment_len], vars = CM.features + ["pf_elapsed_time"])
DATA_DICT_TEST = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[len(concat_df) - treatment_len:], vars = CM.features + ["pf_elapsed_time"])
T = np.concatenate((DATA_DICT_TRAIN.d["pf_elapsed_time"].values[- CM.max_lag:], DATA_DICT_TEST.d["pf_elapsed_time"].values[0:]), axis=0)
DATA_DICT_TRAIN.shrink(CM.features)
DATA_DICT_TEST.shrink(CM.features)
prior_knowledge = {f: DATA_DICT_TEST.d[f].values[:] for f in ['TOD', 'B_S', 'WP']}

res_f, res_s = cie.whatIf(NODES.RV.value, 
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
    observation = np.concatenate((DATA_DICT_TRAIN.d.values[-CM.max_lag:, i].reshape(-1, 1), DATA_DICT_TEST.d.values[0,i].reshape(-1, 1), np.nan*np.ones_like(res_f[:, i]).reshape(-1, 1)), axis=0).reshape(-1)
    prediction_f = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:, i]), res_f[:, i]), axis=0)
    ground_truth = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:, i]), DATA_DICT_TEST.d.values[:, i]), axis=0)

    if f in [NODES.RB.value, NODES.BAC.value]: 
        observation = np.floor(observation)
        prediction_f = np.floor(prediction_f)
        ground_truth = np.floor(ground_truth)
    elif f in [NODES.PD.value]:
        obs_1 = np.mean(DATA_DICT_TRAIN.d.values[-CM.max_lag:, i].reshape(-1, 1)) * np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:, i].reshape(-1, 1))
        obs_2 = np.mean(DATA_DICT_TEST.d.values[0,i].reshape(-1, 1)) * np.ones_like(DATA_DICT_TEST.d.values[0,i].reshape(-1, 1))
        observation = np.concatenate((obs_1, obs_2, np.nan*np.ones_like(res_f[:, i]).reshape(-1, 1)), axis=0)
        prediction_f = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:, i]), np.mean(res_f[:, i]) * np.ones_like(res_f[:, i])), axis=0)
        ground_truth = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:, i]), np.mean(DATA_DICT_TEST.d.values[:, i]) * np.ones_like(DATA_DICT_TEST.d.values[:, i])), axis=0)
         
    axes[FEATURES.index(f)].plot(observation, linestyle = '-', color = "black", label = "observation")
    axes[FEATURES.index(f)].plot(ground_truth, linestyle = '-', color = "tab:orange", label = "ground-truth")
    axes[FEATURES.index(f)].plot(prediction_f, linestyle = '--', color = "tab:blue", label = "prediction")
    axes[FEATURES.index(f)].set_ylabel(DATA_DICT_TRAIN.features[i])
    axes[FEATURES.index(f)].grid(True)
    title = {}

    RMSE = np.sqrt(np.mean((res_f[:, i] - DATA_DICT_TEST.d.values[:, i]) ** 2))
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