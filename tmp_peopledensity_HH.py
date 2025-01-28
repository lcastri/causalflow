# Imports
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

DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024_wBAC/res.pkl'
CIEDIR = '/home/lcastri/git/causalflow/CIE_100_HH_v3/cie.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['noncausal_27122024']

cie = CIE.load(CIEDIR)
with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))
    
dfs = []
PD_means = []
starting = None
starting_len = 400
wp = WP.WA_2_L
for bagname in BAGNAME:
    for tod in TOD:
        files = [f for f in os.listdir(os.path.join(INDIR, "HH/shrunk", f"{bagname}", f"{tod.value}", "static"))]
        files_split = [f.split('_') for f in files]
        wp_files = [f for f in files_split if len(f) == 3 and f[2].split('.')[0] == wp.value][0]
        wp_file = '_'.join(wp_files)
        print(f"Loading : {wp_file}")
        filename = os.path.join(INDIR, "HH/shrunk", f"{bagname}", f"{tod.value}", "static", wp_file)
        if tod == TOD.H1:
            starting = pd.read_csv(filename)
        df = pd.read_csv(filename)
        dfs.append(df)
        PD_means.append(df['PD'].mean() * np.ones(df.shape[0]))


concat_PD = np.concatenate(PD_means, axis=0)       
concat_df = pd.concat(dfs, ignore_index=True)
DATA_DICT_TRAIN = Data(starting[CM.features + ["pf_elapsed_time"]].values[:len(starting) - starting_len], vars = CM.features + ["pf_elapsed_time"])
DATA_DICT_TEST = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[len(starting) - starting_len:], vars = CM.features + ["pf_elapsed_time"])
DATA_PD_TRAIN = Data(concat_PD[:len(starting) - starting_len], vars = ["PD"])
DATA_PD_TEST = Data(concat_PD[len(starting) - starting_len:], vars = ["PD"])
T = np.concatenate((DATA_DICT_TRAIN.d["pf_elapsed_time"].values[- CM.max_lag:], DATA_DICT_TEST.d["pf_elapsed_time"].values), axis=0)
DATA_DICT_TRAIN.shrink(CM.features)
DATA_DICT_TEST.shrink(CM.features)
prior_knowledge = {f: DATA_DICT_TEST.d[f].values[:] for f in ['TOD', 'B_S', 'WP']}
res_f, res_s = cie.whatIf(NODES.RV.value, 
                 DATA_DICT_TEST.d.values[:, DATA_DICT_TEST.features.index(NODES.RV.value)], 
                 DATA_DICT_TRAIN.d.values,
                 prior_knowledge
                 )

FEATURES = ['TOD', 'PD']
N = len(FEATURES)
# Set up the subplots
fig, axes = plt.subplots(N, 1, figsize=(8, 1 * 3), sharex=True)

# Plot each column in a different subplot
DATA_DICT_TRAIN.d['PD'] = DATA_PD_TRAIN.d['PD']
DATA_DICT_TEST.d['PD'] = DATA_PD_TEST.d['PD']
for f in FEATURES:
    i = DATA_DICT_TRAIN.features.index(f)
    ground_truth = DATA_DICT_TEST.d.values[::10]
    prediction_f = res_f[::10]
    axes[FEATURES.index(f)].plot(ground_truth[:, i], linestyle = '-', color = "tab:orange", label = "ground-truth")
    axes[FEATURES.index(f)].plot(prediction_f[:, i], linestyle = '--', color = "tab:blue", label = "prediction")
    axes[FEATURES.index(f)].set_ylabel(DATA_DICT_TRAIN.features[i])
    axes[FEATURES.index(f)].grid(True)
    title = {}
    RMSE = np.sqrt(np.mean((res_f[:, i] - DATA_DICT_TEST.d.values[:, i]) ** 2))
    NRMSE = RMSE/np.std(DATA_DICT_TEST.d.values[:, i]) if np.std(DATA_DICT_TEST.d.values[:, i]) != 0 else 0
    axes[FEATURES.index(f)].set_title(f"NRMSE: {NRMSE:.4f}")
    axes[FEATURES.index(f)].legend(loc='best')

num_ticks = 25
tick_indices = np.linspace(0, len(ground_truth) - 1, num_ticks, dtype=int)
safe_indices = [idx for idx in tick_indices if idx * 10 < len(T)]
tick_labels = [time.strftime("%H:%M:%S", time.gmtime(8 * 3600 + T[idx * 10])) for idx in safe_indices]
plt.xticks(ticks=safe_indices, labels=tick_labels, rotation=45)

plt.tight_layout()
plt.show()
