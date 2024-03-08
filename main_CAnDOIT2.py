import copy
from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.CAnDOIT import CAnDOIT
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.basics.constants import LabelType
import numpy as np

from time import time
from datetime import timedelta

if __name__ == '__main__':   
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample_obs = 1000
    nsample_int = 300
    nfeature = 5
    d = np.random.random(size = (nsample_obs, nfeature))
    for t in range(max_lag, nsample_obs):
        d[t, 1] += 0.5 * d[t-1, 0]
        d[t, 2] += 0.5 * d[t-2, 0] * 0.75 * d[t-1, 3] 
        d[t, 3] += 0.7 * d[t-1, 3] * d[t-2, 4]
        # d[t, 4] += 0.3 * d[t-1, 1]

    df = Data(d, vars = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    
    df.shrink(['X_1', 'X_2', 'X_3', 'X_4'])
    fpcmci = FPCMCI(copy.deepcopy(df),
                    f_alpha = f_alpha, 
                    alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True)
    
    new_start = time()
    cm = fpcmci.run()
    elapsed_fpcmci = time() - new_start
    print(str(timedelta(seconds = elapsed_fpcmci)))
    fpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    fpcmci.timeseries_dag()  
       
    
    ######################################################## INTERVENTION ########################################################
    # I want to do an intervention on X_0. So, I create a context variable CX_0 which models the intervention
    # CX_0 does not have any parent and it is connected ONLY to X_0 @ time t-1
    int_data = dict()
    
    # X_1
    d_int1 = np.random.random(size = (nsample_int, nfeature))
    d_int1[:, 1] = 10 * np.ones(shape = (nsample_int,)) 
    for t in range(max_lag, nsample_int):
        d_int1[t, 2] += 0.5 * d_int1[t-2, 0] * 0.75 * d_int1[t-2, 3] 
        d_int1[t, 3] += 0.7 * d_int1[t-1, 3] * d_int1[t-2, 4]
        # d_int1[t, 4] += 0.3 * d_int1[t-1, 1]
        
    df_int = Data(d_int1, vars = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    df_int.shrink(['X_1', 'X_2', 'X_3', 'X_4'])

    int_data['X_1'] =  df_int
        
    
    candoit = CAnDOIT(copy.deepcopy(df), 
                      int_data,
                      f_alpha = f_alpha, 
                      alpha = pcmci_alpha, 
                      min_lag = min_lag, 
                      max_lag = max_lag, 
                      sel_method = TE(TEestimator.Gaussian), 
                      val_condtest = GPDC(significance = 'analytic', gp_params = None),
                      verbosity = CPLevel.DEBUG,
                      neglect_only_autodep = True,
                      plot_data = False,
                      exclude_context = True)
    
    new_start = time()
    cm = candoit.run()
    elapsed_candoit = time() - new_start
    print(str(timedelta(seconds = elapsed_candoit)))
    candoit.dag(label_type = LabelType.Lag, node_layout = 'dot')
    candoit.timeseries_dag()
