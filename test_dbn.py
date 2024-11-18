# Imports
import copy
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *
import time

DAGDIR = '/home/lcastri/git/causalflow/results/RAL/causal discovery/res.pkl'
CIEDIR = '/home/lcastri/git/causalflow/results/RAL/causal reasoning/cie_context.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024']
# BAGNAME= ['BL100_21102024', 'BL75_29102024', 'BL50_22102024', 'BL25_28102024']


with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))
    _tmp = copy.deepcopy(CM)
    for t in CM.g:
        for s in CM.g[t].sources:
            if s[0] in [NODES.TOD.value, NODES.BS.value, NODES.PD.value, NODES.WP.value]:
                _tmp.del_source(t, s[0], s[1])
    CM = _tmp
    for n in [NODES.TOD.value, NODES.BS.value, NODES.PD.value, NODES.WP.value]:
        CM.g.pop(n)

DATA_TYPE = {
    NODES.RV.value: DataType.Continuous,
    NODES.RB.value: DataType.Continuous,
    NODES.BAC.value: DataType.Continuous,
}
NODE_TYPE = {
    NODES.RV.value: NodeType.System,
    NODES.RB.value: NodeType.System,
    NODES.BAC.value: NodeType.System,
}
cie = CIE(CM, 
          data_type = DATA_TYPE, 
          node_type = NODE_TYPE, 
          nsample = 500, 
          use_gpu = False, 
          model_path = 'testDBN')

treatment_len = 25
dfs = []
for bagname in BAGNAME:
    for wp in [WP.CORR_CANTEEN_1]:
        for tod in TOD:
            if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            dfs.append(pd.read_csv(filename))
            
concatenated_df = pd.concat(dfs, ignore_index=True)
concatenated_df.drop(concatenated_df[concatenated_df['B_S'] == 1].index, inplace=True)
DATA_DICT_TRAIN = Data(concatenated_df[CM.features].values[:len(concatenated_df) - treatment_len], vars = CM.features)
DATA_DICT_TEST = Data(concatenated_df[CM.features].values[len(concatenated_df) - treatment_len:], vars = CM.features)
cie.addObsData(DATA_DICT_TRAIN)
cie.save(os.path.join(cie.model_path, 'cie.pkl'))
 
res = cie.whatIf(NODES.RV.value, 
                 DATA_DICT_TEST.d.values[:, 0], 
                 DATA_DICT_TRAIN.d.values
                 )

# Set up the subplots
fig, axes = plt.subplots(DATA_DICT_TEST.N, 1, figsize=(8, DATA_DICT_TEST.N * 3), sharex=True)

# Plot each column in a different subplot
for i in range(DATA_DICT_TEST.N):
    prediction = np.concatenate((DATA_DICT_TRAIN.d.values[-10:], res), axis=0)
    ground_truth = np.concatenate((DATA_DICT_TRAIN.d.values[-10:], DATA_DICT_TEST.d.values), axis=0)
    axes[i].plot(prediction[:, i], linestyle = '--', color = "tab:blue")
    axes[i].plot(ground_truth[:, i], linestyle = '-', color = "tab:orange")
    axes[i].set_ylabel(DATA_DICT_TRAIN.features[i])
    axes[i].grid(True)
    NRMSE = np.sqrt(np.mean((prediction[:, i] - ground_truth[:, i]) ** 2))/np.std(ground_truth[:, i])
    axes[i].set_title(f"NRMSE: {NRMSE:.4f}")

# Show the plot
plt.tight_layout()
plt.show()