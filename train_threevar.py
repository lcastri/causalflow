# Imports
import copy
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *
import time

DAGDIR = '/home/lcastri/git/causalflow/results/RAL/causal discovery/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024']


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
          model_path = 'CIE_only1bag',
          verbosity = CPLevel.DEBUG)

treatment_len = 25
dfs = []
for bagname in BAGNAME:
    for wp in WP:
        for tod in TOD:
            if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            dfs.append(pd.read_csv(filename))
            
    concat_df = pd.concat(dfs, ignore_index=True)
    concat_df.drop(concat_df[concat_df['B_S'] == 1].index, inplace=True)
    DATA_DICT_TRAIN = Data(concat_df[CM.features+ ["pf_elapsed_time"]].values[:len(concat_df) - treatment_len], varnames = CM.features + ["pf_elapsed_time"])
    DATA_DICT_TEST = Data(concat_df[CM.features+ ["pf_elapsed_time"]].values[len(concat_df) - treatment_len:], varnames = CM.features + ["pf_elapsed_time"])
    T = np.concatenate((DATA_DICT_TRAIN.d["pf_elapsed_time"].values[- CM.max_lag:], DATA_DICT_TEST.d["pf_elapsed_time"].values[0:]), axis=0)
    DATA_DICT_TRAIN.shrink(CM.features)
    DATA_DICT_TEST.shrink(CM.features)

    cie.addObsData(DATA_DICT_TRAIN)
    cie.save(os.path.join(cie.model_path, 'cie.pkl'))