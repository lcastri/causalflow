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

DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024_new/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['noncausal-03012025']


with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))

DATA_TYPE = {
    NODES.TOD.value: DataType.Discrete,
    NODES.RV.value: DataType.Continuous,
    NODES.RB.value: DataType.Discrete,
    NODES.CS.value: DataType.Discrete,
    NODES.PD.value: DataType.Continuous,
    NODES.ELT.value: DataType.Discrete,
    NODES.OBS.value: DataType.Discrete,
    NODES.WP.value: DataType.Discrete,
}
NODE_TYPE = {
    NODES.TOD.value: NodeType.Context,
    NODES.RV.value: NodeType.System,
    NODES.RB.value: NodeType.System,
    NODES.CS.value: NodeType.Context,
    NODES.PD.value: NodeType.System,
    NODES.ELT.value: NodeType.System,
    NODES.OBS.value: NodeType.Context,
    NODES.WP.value: NodeType.Context,
}
cie = CIE(CM, 
          data_type = DATA_TYPE, 
          node_type = NODE_TYPE,
          model_path = 'CIE_100_HH_v4',
          verbosity = CPLevel.DEBUG)

start_time = time.time()



var_names = [n.value for n in NODES]
DATA_DICT = {}
dfs = []

for bagname in BAGNAME:
    for wp in WP:
        if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
        for tod in TOD:
            files = [f for f in os.listdir(os.path.join(INDIR, "HH/my_nonoise", f"{bagname}", f"{tod.value}"))]
            files_split = [f.split('_') for f in files]
            
            wp_file = [f for f in files_split if len(f) == 3 and f[2].split('.')[0] == wp.value][0]
            wp_file = '_'.join(wp_file)
            print(f"Loading : {wp_file}")
            filename = os.path.join(INDIR, "HH/my_nonoise", f"{bagname}", f"{tod.value}", f"{wp_file}")
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