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
          batch_size = 12500,
          nsample= 100,
          model_path = 'testDBN_2')

treatment_len = 25
dfs = []
for bagname in BAGNAME:
    for wp in [WP.CORR_CANTEEN_1]:
        for tod in TOD:
            if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            dfs.append(pd.read_csv(filename))
            
concat_df = pd.concat(dfs, ignore_index=True)
concat_df.drop(concat_df[concat_df['B_S'] == 1].index, inplace=True)
DATA_DICT_TRAIN = Data(concat_df[CM.features+ ["pf_elapsed_time"]].values[:len(concat_df) - treatment_len], vars = CM.features + ["pf_elapsed_time"])
DATA_DICT_TEST = Data(concat_df[CM.features+ ["pf_elapsed_time"]].values[len(concat_df) - treatment_len:], vars = CM.features + ["pf_elapsed_time"])
T = np.concatenate((DATA_DICT_TRAIN.d["pf_elapsed_time"].values[- CM.max_lag:], DATA_DICT_TEST.d["pf_elapsed_time"].values[0:]), axis=0)
# T = concat_df["pf_elapsed_time"].values[len(concat_df) - (treatment_len + CM.max_lag):]
DATA_DICT_TRAIN.shrink(CM.features)
DATA_DICT_TEST.shrink(CM.features)

np.save('T.npy', T)
np.save('DATA_DICT_TRAIN.npy', DATA_DICT_TRAIN.d.values)
np.save('DATA_DICT_TEST.npy', DATA_DICT_TEST.d.values)

cie.addObsData(DATA_DICT_TRAIN)
cie.save(os.path.join(cie.model_path, 'cie.pkl'))
 
res = cie.whatIf(NODES.RV.value, 
                 DATA_DICT_TEST.d.values[:, 0], 
                 DATA_DICT_TRAIN.d.values
                 )

np.save('res.npy', res)

# Set up the subplots
fig, axes = plt.subplots(DATA_DICT_TEST.N, 1, figsize=(8, DATA_DICT_TEST.N * 3), sharex=True)

# Plot each column in a different subplot
for i in range(DATA_DICT_TEST.N):
    observation = np.concatenate((DATA_DICT_TRAIN.d.values[-CM.max_lag:], DATA_DICT_TEST.d.values[0,:].reshape(1,-1), np.nan*np.ones_like(res)), axis=0)
    prediction = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:]), res), axis=0)
    ground_truth = np.concatenate((np.nan*np.ones_like(DATA_DICT_TRAIN.d.values[-CM.max_lag:]), DATA_DICT_TEST.d.values), axis=0)
    axes[i].plot(observation[:, i], linestyle = '-', color = "black", label = "observation")
    axes[i].plot(ground_truth[:, i], linestyle = '-', color = "tab:orange", label = "ground-truth")
    axes[i].plot(prediction[:, i], linestyle = '--', color = "tab:blue", label = "prediction")
    axes[i].set_ylabel(DATA_DICT_TRAIN.features[i])
    axes[i].grid(True)
    NRMSE = np.sqrt(np.mean((res[:, i] - DATA_DICT_TEST.d.values[:, i]) ** 2))/np.std(DATA_DICT_TEST.d.values[:, i])
    axes[i].set_title(f"NRMSE: {NRMSE:.4f}")
    axes[i].legend(loc='best')

# Show the plot
plt.xticks(ticks = list(range(len(T))),
           labels = [time.strftime("%H:%M:%S", time.gmtime(8*3600+t)) for t in list(T)],
           rotation=45  # Rotate labels by 45 degrees
    )
plt.tight_layout()
plt.show()