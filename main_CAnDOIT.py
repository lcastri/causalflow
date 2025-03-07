from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.CAnDOIT import CAnDOIT
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.basics.constants import LabelType
import numpy as np

from time import time
from datetime import timedelta

if __name__ == '__main__':   
    f_alpha = 0.1
    pcmci_alpha = 0.05
    max_lag = 2

    np.random.seed(1)
    nsample = 750
    nfeature = 5
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 1] += 0.5 * d[t-1, 0]**2
        d[t, 2] += 0.3 * d[t-1, 0] * 0.75 * d[t-2, 1] 
        d[t, 3] += 0.7 * d[t-1, 3] * d[t-2, 4]
       
    df = Data(d, varnames = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
       
    
    ######################################################## INTERVENTION ########################################################
    # I want to do an intervention on X_1. So, I create a context variable CX_1 which models the intervention
    # CX_1 does not have any parent and it is connected ONLY to X_1 @ time t-1
    int_data = dict()
    
    # X_0
    d_int0 = np.random.random(size = (int(nsample/2), nfeature))
    d_int0[:, 0] = 2 * np.ones(shape = (int(nsample/2),)) 
    for t in range(max_lag, int(nsample/2)):
        d_int0[t, 1] += 0.5 * d_int0[t-1, 0]**2
        d_int0[t, 2] += 0.3 * d_int0[t-1, 0] * 0.75 * d_int0[t-2, 1] 
        d_int0[t, 3] += 0.7 * d_int0[t-1, 3] * d_int0[t-2, 4]
        
    df_int = Data(d_int0, varnames = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    int_data['X_0'] =  df_int
    
    # # X_1
    # d_int1 = np.random.random(size = (int(nsample/2), nfeature))
    # d_int1[:, 1] = 5 * np.ones(shape = (int(nsample/2),)) 
    # for t in range(max_lag, int(nsample/2)):
    #     d_int1[t, 2] += 0.3 * d_int1[t-1, 0] * 0.75 * d_int1[t-2, 1] 
    #     d_int1[t, 3] += 0.7 * d_int1[t-1, 3] * d_int1[t-2, 4]
        
    # df_int = Data(d_int1, vars = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    # int_data['X_1'] =  df_int

    # # X_4
    # d_int2 = np.random.random(size = (int(nsample/2), nfeature))
    # d_int2[:, 4] = 0.5 * np.ones(shape = (int(nsample/2),)) 
    # for t in range(max_lag, int(nsample/2)):
    #     d_int2[t, 1] += 0.5 * d_int2[t-1, 0]**2
    #     d_int2[t, 2] += 0.3 * d_int2[t-1, 0] * 0.75 * d_int2[t-2, 1] 
    #     d_int2[t, 3] += 0.7 * d_int2[t-1, 3] * d_int2[t-2, 4]
        
    # df_int = Data(d_int2, vars = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    # int_data['X_4'] =  df_int
        
    
    candoit = CAnDOIT(df, 
                        int_data,
                        f_alpha = f_alpha, 
                        alpha = pcmci_alpha, 
                        max_lag = max_lag, 
                        sel_method = TE(TEestimator.Gaussian), 
                        val_condtest = GPDC(significance = 'analytic', gp_params = None),
                        verbosity = CPLevel.DEBUG,
                        neglect_only_autodep = True,
                        plot_data = False,
                        exclude_context = True,
                        resfolder='results/candoitsasas')
    
    new_start = time()
    cm = candoit.run(nofilter=True)
    elapsed_candoit = time() - new_start
    print(str(timedelta(seconds = elapsed_candoit)))
    cm.dag(save_name = candoit.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = candoit.ts_dag_path,)