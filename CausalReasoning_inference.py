# Imports
import os
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.preprocessing.data import Data
from utils import *

INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME = 'BL100_21102024'
tod = TOD.LUNCH.value
wp = WP.TABLE2.value
with open('CIE_allbags_again/cie.pkl', 'rb') as f:
    cie = CIE.load(pickle.load(f))
lag = cie.DAG['complete'].max_lag
treatment_len = 10

var_names = [n.value for n in NODES]
DATA_DICT = {}
print(f"Loading : {BAGNAME}-{tod}-{wp}")
filename = os.path.join(INDIR, "my_nonoise", f"{BAGNAME}", tod, f"{BAGNAME}_{tod}_{wp}.csv")
df = pd.read_csv(filename)
OBS = Data(df[var_names].values[:int(len(df.values)/2)], vars = var_names)
DF = Data(df[var_names].values[int(len(df.values)/2):int(len(df.values)/2)+treatment_len], vars = var_names)
T = np.array(df["pf_elapsed_time"].values[int(len(df.values)/2)-2:int(len(df.values)/2)+treatment_len])

prior_knowledge = {f: DF.d.values[:treatment_len, DF.features.index(f)] for f in ['TOD', 'B_S']}
res = cie.whatIf('WP', 
                 DF.d.values[:treatment_len, DF.features.index('WP')], 
                 OBS.d.values,
                 prior_knowledge)

# result = np.concatenate((OBS.d.values, res), axis=0)

# Get the number of columns
num_columns = res.shape[1]
# Set up the subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 3), sharex=True)

# Plot each column in a different subplot
for i in range(num_columns):
    axes[i].plot(np.concatenate([np.nan*np.ones_like(OBS.d.values[-lag:, i]), DF.d.values[:treatment_len, i]], axis=0), 
                 label = "ground-truth",
                 color = "red",
                 linestyle = "-")
    axes[i].plot(np.concatenate([np.nan*np.ones_like(OBS.d.values[-lag:, i]), res[:, i]], axis=0), 
                 label = "predicted",
                 color = "blue",
                 linestyle = "--")
    axes[i].plot(np.concatenate([OBS.d.values[-1-lag:, i], np.nan*np.ones_like(res[:, i])], axis=0), 
                 label = "observed", 
                 color = "black",
                 linestyle = "-")
    axes[i].set_ylabel(DF.features[i])

# Show the plot
plt.xlabel('Time')
plt.xticks(ticks = list(range(len(np.concatenate([OBS.d.values[-1-lag:, i], np.nan*np.ones_like(res[:, i])], axis=0)))),
           labels = [time.strftime("%H:%M:%S", time.gmtime(8*3600+t)) for t in list(T)])
plt.tight_layout()
plt.legend()
plt.show()