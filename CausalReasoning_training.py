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

DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024']
# BAGNAME= ['BL100_21102024', 'BL75_29102024', 'BL50_22102024', 'BL25_28102024']
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
          model_path = 'CIE',
          verbosity = CPLevel.INFO)

start_time = time.time()

var_names = [n.value for n in NODES]
DATA_DICT = {}
dfs = []
for bagname in BAGNAME:
    for wp in WP:
        if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
        for tod in TOD:
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            if USE_SUBSAMPLED:
                if wp is WP.DOOR_ENTRANCE_CANTEEN:
                    tmp_wp = "door_entrance-canteen"
                else:
                    tmp_wp = wp.value.replace("-", "_")
                filename = os.path.join(INDIR, "TOD/my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{tmp_wp}.csv")
            else:
                filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{tmp_wp}.csv")
            dfs.append(pd.read_csv(filename))
            
        concatenated_df = pd.concat(dfs, ignore_index=True)
        dfs = []
        idx = len(DATA_DICT)
        DATA_DICT[idx] = Data(concatenated_df[var_names].values, varnames = var_names)
        del concatenated_df
        cie.addObsData(DATA_DICT[idx])
    
cie.save(os.path.join(cie.model_path, 'cie.pkl'))

end_time = time.time()
elapsed_time = end_time - start_time
days = elapsed_time // (24 * 3600)
remaining_time = elapsed_time % (24 * 3600)
formatted_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
print(f"Training time: {days} days, {formatted_time}")