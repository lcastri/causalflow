from tigramite.independence_tests.gpdc import GPDC
from connectingdots.CPrinter import CPLevel
from connectingdots.causal_discovery.FPCMCI import FPCMCI
from connectingdots.causal_discovery.baseline.DYNOTEARS import DYNOTEARS
from connectingdots.causal_discovery.baseline.VarLiNGAM import VarLiNGAM
from connectingdots.causal_discovery.baseline.PCMCI import PCMCI
from connectingdots.causal_discovery.baseline.oCSE import oCSE
from connectingdots.preprocessing.data import Data
from connectingdots.selection_methods.TE import TE, TEestimator
from connectingdots.basics.constants import LabelType
import numpy as np


if __name__ == '__main__':   
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 1
    max_lag = 2

    np.random.seed(1)
    nsample = 1000
    nfeature = 5
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 1] += 0.5 * d[t-1, 0]**2
        d[t, 2] += 0.3 * d[t-1, 0] * 0.75 * d[t-2, 1] - 2.5
        d[t, 3] += 0.7 * d[t-1, 3] * d[t-2, 4]
       
    df = Data(d, vars = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    
    
    # fpcmci = FPCMCI(df,
    #                 f_alpha = f_alpha, 
    #                 alpha = pcmci_alpha, 
    #                 min_lag = min_lag, 
    #                 max_lag = max_lag, 
    #                 sel_method = TE(TEestimator.Gaussian), 
    #                 val_condtest = GPDC(significance = 'analytic', gp_params = None),
    #                 verbosity = CPLevel.DEBUG,
    #                 neglect_only_autodep = True)
    
    # cm = fpcmci.run()
    # fpcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    # fpcmci.timeseries_dag()
    
    
    
    # pcmci = PCMCI(df,
    #               min_lag = min_lag, 
    #               max_lag = max_lag, 
    #               val_condtest = GPDC(significance = 'analytic', gp_params = None),
    #               verbosity = CPLevel.DEBUG,
    #               alpha = pcmci_alpha, 
    #               neglect_only_autodep = True)
    
    # cm = pcmci.run()
    # pcmci.dag(label_type = LabelType.Lag, node_layout = 'dot')
    # pcmci.timeseries_dag()
    
    
    
    # dynotears = DYNOTEARS(df,
    #                       max_lag = max_lag,
    #                       verbosity = CPLevel.DEBUG,
    #                       alpha = pcmci_alpha,
    #                       neglect_only_autodep = True)
    # cm = dynotears.run()
    # dynotears.dag(label_type = LabelType.Lag, node_layout = 'dot')
    # dynotears.timeseries_dag()
    
    # varlingam = VarLiNGAM(df,
    #                       max_lag = max_lag,
    #                       verbosity = CPLevel.DEBUG,
    #                       alpha = pcmci_alpha,
    #                       neglect_only_autodep = True)
    # cm = varlingam.run()
    # varlingam.dag(label_type = LabelType.Lag, node_layout = 'dot')
    # varlingam.timeseries_dag()
    
    ocse = oCSE(df,
                     max_lag = max_lag,
                     verbosity = CPLevel.DEBUG,
                     alpha = pcmci_alpha,
                     neglect_only_autodep = True)
    cm = ocse.run()
    ocse.dag(label_type = LabelType.Lag, node_layout = 'dot')
    ocse.timeseries_dag()