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

INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['noncausal-03012025']
cie = CIE.load('CIE_100_HH_v4_round1/cie.pkl')

start_time = time.time()

var_names = [n.value for n in NODES]
DATA_DICT = {}
dfs = []

for obs_id in cie.DBNs.keys():
        print(f"Computing density for obs_id: {obs_id}")
        cie.DBNs[obs_id].compute_single_do_density(cie.DAG['complete'], 
                                                   cie.Ds[obs_id]["complete"], 
                                                   'ELT', ('R_V', -1), 
                                                   conditions = [('C_S', -1), ('ELT', -1)],
                                                   max_adj_size = 1)
        # cie.DBNs[obs_id].compute_single_do_density(cie.DAG['complete'], 
        #                                            cie.Ds[obs_id]["complete"], 
        #                                            'PD', ('TOD', 0), 
        #                                            conditions = [('PD', -1)],
        #                                            max_adj_size = 1)
        cie.save(os.path.join(cie.model_path, 'cie.pkl'))

end_time = time.time()
elapsed_time = end_time - start_time
days = elapsed_time // (24 * 3600)
remaining_time = elapsed_time % (24 * 3600)
formatted_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
print(f"Training time: {days} days, {formatted_time}")