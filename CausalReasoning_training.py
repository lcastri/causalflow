# Imports
import os
import pickle

import pandas as pd
from causalflow.basics.constants import DataType
from causalflow.graph.DAG import DAG
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *
import time

DAGDIR = '/home/lcastri/git/causalflow/results/FINAL/BL100_21102024/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['BL100_21102024', 'BL75_29102024', 'BL50_22102024', 'BL25_28102024']
USE_SUBSAMPLED = True


with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))

DATA_TYPE = {
    NODES.TOD.value: DataType.Discrete,
    NODES.RV.value: DataType.Continuous,
    NODES.RB.value: DataType.Continuous,
    NODES.BS.value: DataType.Discrete,
    NODES.PD.value: DataType.Continuous,
    NODES.BAC.value: DataType.Continuous,
    NODES.WP.value: DataType.Discrete,
}
cie = CIE(CM, data_type = DATA_TYPE, nsample = 60, use_gpu = False)

start_time = time.time()

var_names = [n.value for n in NODES]
DATA_DICT = {}
for bagname in BAGNAME:
    dfs = []
    for wp in WP:
        for tod in TOD:
            if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
            print(f"Loading : {bagname}-{tod.value}-{wp.value}")
            if USE_SUBSAMPLED:
                filename = os.path.join(INDIR, "my_nonoise", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            else:
                filename = os.path.join(INDIR, "original", f"{bagname}", tod.value, f"{bagname}_{tod.value}_{wp.value}.csv")
            dfs.append(pd.read_csv(filename))
            
    concatenated_df = pd.concat(dfs, ignore_index=True)
    idx = len(DATA_DICT)
    DATA_DICT[idx] = Data(concatenated_df[var_names].values, vars = var_names)
    cie.addObsData(DATA_DICT[idx])
    
cie.save(f"/home/lcastri/git/causalflow/results/FINAL/BL100_21102024/cie.pkl")

end_time = time.time()

# Calculate and print the elapsed time in HH:mm:ss format
elapsed_time = end_time - start_time
# Calculate days, hours, minutes, and seconds
days = elapsed_time // (24 * 3600)
remaining_time = elapsed_time % (24 * 3600)
formatted_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
print(f"Training time: {days} days, {formatted_time}")