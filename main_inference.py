import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.causal_reasoning.SMCFilter import SMCFilter
from causalflow.graph import DAG
from causalflow.preprocessing.data import Data
from utils import *

# DATA
DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['noncausal-03012025']
with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))

starting_t = 100
treatment_len = 100

for bagname in BAGNAME:
    for wp in WP:
        dfs = []
        if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
        for tod in TOD:
            files = [f for f in os.listdir(os.path.join(INDIR, "HH/my_nonoise", f"{bagname}", f"{tod.value}"))]
            files_split = [f.split('_') for f in files]
            wp_files = [f for f in files_split if len(f) == 3 and f[2].split('.')[0] == wp.value][0]
            wp_file = '_'.join(wp_files)
            print(f"Loading : {wp_file}")
            filename = os.path.join(INDIR, "HH/my_nonoise", f"{bagname}", f"{tod.value}", wp_file)

            df = pd.read_csv(filename)
            dfs.append(df)
        concat_df = pd.concat(dfs, ignore_index=True)
        break
    
DATA_DICT_TRAIN = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[:starting_t], varnames = CM.features + ["pf_elapsed_time"])
DATA_DICT_TEST = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[starting_t:starting_t+treatment_len], varnames = CM.features + ["pf_elapsed_time"])
T = DATA_DICT_TEST.d["pf_elapsed_time"].values
DATA_DICT_TRAIN.shrink(CM.features)
DATA_DICT_TEST.shrink(CM.features)
DATA_DICT = Data(np.concatenate((DATA_DICT_TRAIN.d.values, DATA_DICT_TEST.d.values), axis=0), varnames = CM.features)

# Define initial evidence (at t = -1)
given_values = {
    ('R_V', -1): DATA_DICT_TEST.d['R_V'].values.tolist(),  
    ('ELT', -1): np.concatenate((DATA_DICT_TRAIN.d['ELT'].values[-1:], DATA_DICT_TEST.d['ELT'].values[:-1])).tolist(),  
}
given_context = {
    'C_S': 0,  
    'TOD': 0,  
    'WP': 1
}

cie = CIE.load('CIE_100_HH_v6/cie.pkl')

smc = SMCFilter(cie.DBNs[('obs', 0)], num_particles=500)
ELT_obs = smc.sequential_query("ELT", given_values, given_context, num_steps=treatment_len)
obs_RMSE = np.sqrt(np.mean((np.array(ELT_obs) - DATA_DICT_TEST.d['ELT'].values) ** 2))
obs_NRMSE = obs_RMSE/np.std(DATA_DICT_TEST.d['ELT'].values)   
    
smc_do = SMCFilter(cie.DBNs[('obs', 0)], num_particles=500)
ELT_do = smc_do.sequential_query("ELT", given_values, given_context, num_steps=treatment_len, 
                                                    intervention_var=('R_V', -1), adjustment_set=[('OBS', -1)])
do_RMSE = np.sqrt(np.mean((np.array(ELT_do) - DATA_DICT_TEST.d['ELT'].values) ** 2))
do_NRMSE = do_RMSE/np.std(DATA_DICT_TEST.d['ELT'].values)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(T, DATA_DICT_TEST.d['ELT'].values.tolist(), label="Ground Truth", linestyle="solid", marker="o")
plt.plot(T, ELT_obs, label=f"E[ELT] -- p(ELTt∣RV_t-1 = rv, ELT_t-1 = elt), NRMSE={obs_NRMSE:.2f}", linestyle="dashed", marker="x", color="green")
plt.plot(T, ELT_do, label=f"E[ELT] -- p(D∣do(ELTt∣RV_t-1 = rv), ELT_t-1 = elt), NRMSE={do_NRMSE:.2f}", linestyle="dashed", marker="o", color="red")
plt.legend()
plt.show()