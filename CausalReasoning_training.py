import os
import pickle

import pandas as pd
from causalflow.CPrinter import CPLevel
from causalflow.basics.constants import *
from causalflow.graph.DAG import DAG
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *
import time

DAGDIR = '/home/lcastri/git/darko-peopleflow/HRISim_docker/HRISim/prediction/hrisim_prediction_manager/DAGs/DARKO_v1/res.pkl'
INDIR = '/home/lcastri/git/darko-peopleflow/utilities_ws/src/bag_postprocess/csv'
BAGNAME= ['06-02-2025-DARKO']

USE_SUBSAMPLED = True

with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))

DATA_TYPE = {
    NODES.TOD.value: DataType.Discrete,
    NODES.RV.value: DataType.Continuous,
    NODES.RB.value: DataType.Discrete,
    NODES.BS.value: DataType.Discrete,
    NODES.PD.value: DataType.Continuous,
    NODES.BAC.value: DataType.Discrete,
    NODES.WP.value: DataType.Discrete,
}
NODE_TYPE = {
    NODES.TOD.value: NodeType.Context,
    NODES.RV.value: NodeType.System,
    NODES.RB.value: NodeType.System,
    NODES.BS.value: NodeType.Context,
    NODES.PD.value: NodeType.System,
    NODES.BAC.value: NodeType.System,
    NODES.WP.value: NodeType.Context,
}
cie = CIE(CM, 
          data_type = DATA_TYPE, 
          node_type = NODE_TYPE,
          model_path = 'CIE_DARKO_v2',
          verbosity = CPLevel.INFO)

start_time = time.time()

var_names = [n.value for n in NODES]
DATA_DICT = {}
dfs = []
for bagname in BAGNAME:
    for wp in WP:
        if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
        for tod in TOD:
            if tod in [TOD.OFF, TOD.T0, TOD.T19]: continue
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            if USE_SUBSAMPLED:
                filename = os.path.join(INDIR, "shrunk", f"{bagname}", tod.value, 'static', f"{bagname}_{tod.value}_{wp.value}.csv")
            else:
                filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            dfs.append(pd.read_csv(filename))
        concatenated_df = pd.concat(dfs, ignore_index=True)
        dfs = []
        idx = len(DATA_DICT)
        DATA_DICT[idx] = Data(concatenated_df[var_names].values, vars = var_names)
        del concatenated_df
        cie.addObsData(DATA_DICT[idx])
    
cie.save(os.path.join(cie.model_path, 'cie.pkl'))

end_time = time.time()
elapsed_time = end_time - start_time
days = elapsed_time // (24 * 3600)
remaining_time = elapsed_time % (24 * 3600)
formatted_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
print(f"Training time: {days} days, {formatted_time}")