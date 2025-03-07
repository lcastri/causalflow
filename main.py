from tigramite.independence_tests.gpdc import GPDC
from causalflow.CPrinter import CPLevel
from causalflow.causal_discovery.FPCMCI import FPCMCI
from causalflow.causal_discovery.baseline.DYNOTEARS import DYNOTEARS
from causalflow.causal_discovery.baseline.VarLiNGAM import VarLiNGAM
from causalflow.causal_discovery.baseline.PCMCI import PCMCI
from causalflow.causal_discovery.baseline.PCMCIplus import PCMCIplus
from causalflow.causal_discovery.baseline.TCDF import TCDF
from causalflow.causal_discovery.baseline.tsFCI import tsFCI
from causalflow.preprocessing.data import Data
from causalflow.selection_methods.TE import TE, TEestimator
from causalflow.basics.constants import LabelType
import numpy as np


if __name__ == '__main__':   
    f_alpha = 0.1
    pcmci_alpha = 0.05
    min_lag = 0
    max_lag = 2

    np.random.seed(1)
    nsample = 500
    nfeature = 5
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 1] += 0.5 * d[t-1, 0]**2
        d[t, 2] += 0.3 * d[t-1, 0] * 0.75 * d[t-2, 1] - 2.5
        d[t, 3] += 0.7 * d[t-1, 3] * d[t-2, 4]
       
    df = Data(d, varnames = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4'])
    
    
    fpcmci = FPCMCI(df,
                    f_alpha = f_alpha, 
                    alpha = pcmci_alpha, 
                    min_lag = min_lag, 
                    max_lag = max_lag, 
                    sel_method = TE(TEestimator.Gaussian), 
                    val_condtest = GPDC(significance = 'analytic', gp_params = None),
                    verbosity = CPLevel.DEBUG,
                    neglect_only_autodep = True,
                    resfolder = "results/fpcmci")
    
    cm = fpcmci.run()
    cm.dag(save_name = fpcmci.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = fpcmci.ts_dag_path)    
    
    
    dynotears = DYNOTEARS(df,
                          min_lag = min_lag,
                          max_lag = max_lag,
                          verbosity = CPLevel.DEBUG,
                          alpha = pcmci_alpha,
                          neglect_only_autodep = True,
                          resfolder = "results/dynotears")
    cm = dynotears.run()
    cm.dag(save_name = dynotears.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = dynotears.ts_dag_path) 
    
       
    pcmci = PCMCI(df,
                  min_lag = min_lag, 
                  max_lag = max_lag, 
                  val_condtest = GPDC(significance = 'analytic', gp_params = None),
                  verbosity = CPLevel.DEBUG,
                  alpha = pcmci_alpha, 
                  neglect_only_autodep = True,
                  resfolder = "results/pcmci")
    
    cm = pcmci.run()
    cm.dag(save_name = pcmci.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = pcmci.ts_dag_path) 
    
    pcmciplus = PCMCIplus(df,
                  min_lag = 0, 
                  max_lag = max_lag, 
                  val_condtest = GPDC(significance = 'analytic', gp_params = None),
                  verbosity = CPLevel.DEBUG,
                  alpha = pcmci_alpha, 
                  neglect_only_autodep = True,
                  resfolder = "results/pcmciplus")
    
    cm = pcmciplus.run()
    cm.dag(save_name = pcmciplus.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = pcmciplus.ts_dag_path) 
    

    tcdf = TCDF(df,
                min_lag = min_lag,
                max_lag = max_lag,
                verbosity = CPLevel.DEBUG,
                neglect_only_autodep = True,
                resfolder = "results/tcdf")
    cm = tcdf.run(cuda=True)
    cm.dag(save_name = tcdf.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = tcdf.ts_dag_path) 
    
    
    tsfci = tsFCI(df,
                min_lag = min_lag,
                max_lag = max_lag,
                verbosity = CPLevel.DEBUG,
                alpha = pcmci_alpha,
                neglect_only_autodep = True,
                resfolder = "results/tsfci")
    cm = tsfci.run()
    cm.dag(save_name = tsfci.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = tsfci.ts_dag_path) 
    
    
    varlingam = VarLiNGAM(df,
                          min_lag = min_lag,
                          max_lag = max_lag,
                          verbosity = CPLevel.DEBUG,
                          alpha = pcmci_alpha,
                          neglect_only_autodep = True,
                          resfolder = "results/varlingam")
    cm = varlingam.run()
    cm.dag(save_name = varlingam.dag_path, label_type = LabelType.Lag, node_layout = 'dot')
    cm.ts_dag(save_name = varlingam.ts_dag_path) 