# Imports
import os
import pickle
import time
import numpy as np
import pandas as pd
from causalflow.basics.constants import DataType
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.graph import DAG
from causalflow.preprocessing.data import Data
from utils import *
from pgmpy.models import BayesianNetwork
import matplotlib.pyplot as plt

def get_DBN(link_assumptions, tau_max) -> BayesianNetwork:
    DBN = BayesianNetwork()
    DBN.add_nodes_from([f"{t}__{-abs(l)}" for t in link_assumptions.keys() for l in range(0, tau_max + 1)])

    # Edges
    edges = []
    for t in link_assumptions.keys():
        for source in link_assumptions[t]:
            if len(source) == 0: continue
            elif len(source) == 2: s, l = source
            elif len(source) == 3: s, l, _ = source
            else: raise ValueError("Source not well defined")
            edges.append((f"{s}__{-abs(l)}", f"{t}__0"))
            # Add edges across time slices from -1 to -tau_max
            for lag in range(1, tau_max + 1):
                if l - lag >= -tau_max:
                    edges.append((f"{s}__{-abs(l - lag)}", f"{t}__{-abs(lag)}"))
    DBN.add_edges_from(edges)
    return DBN

# DATA
DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['noncausal-03012025']
# cie = CIE.load('CIE_100_HH_v4_int/cie.pkl')
cie = CIE.load('CIE_100_HH_v4/cie.pkl')
cie_context = CIE.load('CIE_100_HH_v4_context/cie.pkl')
cie_bayes = CIE.load('CIE_100_HH_v4_bayes/cie.pkl')
with open(DAGDIR, 'rb') as f:
    CM = DAG.load(pickle.load(f))
    
DATA_TYPE = {
    NODES.TOD.value: DataType.Discrete,
    NODES.RV.value: DataType.Continuous,
    NODES.RB.value: DataType.Continuous,
    NODES.CS.value: DataType.Discrete,
    NODES.PD.value: DataType.Continuous,
    NODES.ELT.value: DataType.Continuous,
    NODES.OBS.value: DataType.Discrete,
    NODES.WP.value: DataType.Discrete,
}

var_names = [n.value for n in NODES]
PD_means = []
starting_t = 500
treatment_len = 20

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
            PD_means.append(df['PD'].mean() * np.ones(df.shape[0]))
        concat_df = pd.concat(dfs, ignore_index=True)
        concat_PD = np.concatenate(PD_means, axis=0)       
        break
    
DATA_DICT_TRAIN = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[:starting_t], vars = CM.features + ["pf_elapsed_time"])
DATA_DICT_TEST = Data(concat_df[CM.features + ["pf_elapsed_time"]].values[starting_t:starting_t+treatment_len], vars = CM.features + ["pf_elapsed_time"])
DATA_PD_TRAIN = Data(concat_PD[:starting_t], vars = ["PD"])
DATA_PD_TEST = Data(concat_PD[starting_t:starting_t+treatment_len], vars = ["PD"])
T = np.concatenate((DATA_DICT_TRAIN.d["pf_elapsed_time"].values[- CM.max_lag:], DATA_DICT_TEST.d["pf_elapsed_time"].values), axis=0)
DATA_DICT_TRAIN.shrink(CM.features)
DATA_DICT_TEST.shrink(CM.features)
DATA_DICT = Data(np.concatenate((DATA_DICT_TRAIN.d.values, DATA_DICT_TEST.d.values), axis=0), vars = CM.features)

#! Causal Inference: this must be used for the original CIE -- CIE_100_HH_v4
resELT = cie.whatIfDo(outcome = NODES.ELT.value, 
                   treatment = NODES.RV.value, lag = -1, treatment_values = 0.5*np.ones(shape = (treatment_len, 1)),
                   conditions = {(NODES.CS.value, -1): np.zeros(shape = (treatment_len, 1)).astype(int),
                                 (NODES.ELT.value, -1): DATA_DICT_TEST.d['ELT'].to_numpy().reshape(-1, 1)},
                   adjustment = [(NODES.OBS.value, -1)])
resELT_context = cie_context.whatIfDo_context(outcome = NODES.ELT.value, 
                                      treatment = NODES.RV.value, lag = -1, treatment_values = 0.5*np.ones(shape = (treatment_len, 1)),
                                      conditions = {(NODES.CS.value, -1): np.zeros(shape = (treatment_len, 1)).astype(int),
                                            (NODES.ELT.value, -1): DATA_DICT_TEST.d['ELT'].to_numpy().reshape(-1, 1),
                                            (NODES.WP.value, 0): DATA_DICT_TEST.d['WP'].to_numpy().reshape(-1, 1)},
                                      adjustment = [(NODES.OBS.value, -1)])
resELT_bayes = cie_bayes.whatIfDo_bayes(outcome = NODES.ELT.value, 
                                        treatment = NODES.RV.value, lag = -1, treatment_values = 0.5*np.ones(shape = (treatment_len, 1)),
                                        conditions = {(NODES.CS.value, -1): np.zeros(shape = (treatment_len, 1)).astype(int),
                                                        (NODES.ELT.value, -1): DATA_DICT_TEST.d['ELT'].to_numpy().reshape(-1, 1)})
resPD = cie.whatIfDo(outcome = NODES.PD.value, 
                     treatment = NODES.TOD.value, lag = 0, treatment_values = 0*np.ones(shape = (treatment_len, 1)),
                     conditions = {(NODES.PD.value, -1): DATA_DICT_TEST.d['PD'].to_numpy().reshape(-1, 1),
                                   (NODES.WP.value, 0): DATA_DICT_TEST.d['WP'].to_numpy().reshape(-1, 1)},
                     adjustment = None)
resPD_bayes = cie_bayes.whatIfDo_bayes(outcome = NODES.PD.value, 
                     treatment = NODES.TOD.value, lag = 0, treatment_values = 0*np.ones(shape = (treatment_len, 1)),
                     conditions = {(NODES.PD.value, -1): DATA_DICT_TEST.d['PD'].to_numpy().reshape(-1, 1),
                                   (NODES.WP.value, 0): DATA_DICT_TEST.d['WP'].to_numpy().reshape(-1, 1)})
# resPD_noDO = cie.whatIf(treatment = NODES.TOD.value, values = DATA_DICT_TEST.d['TOD'].values,
#                    data = DATA_DICT_TRAIN.d.values,
#                    prior_knowledge = {f: DATA_DICT_TEST.d[f].values[:] for f in ['TOD', 'C_S', 'OBS', 'WP']})[:, CM.features.index('PD')]


# #! Causal Inference: this must be used for the int CIE -- CIE_100_HH_v4_int
# resELT = cie.whatIfDo(outcome = NODES.ELT.value, 
#                    treatment = NODES.RV.value, lag = -1, treatment_values = 0.5*np.ones(shape = (treatment_len, 1)),
#                    conditions = {(NODES.CS.value, -1): np.zeros(shape = (treatment_len, 1)).astype(int),
#                                  (NODES.ELT.value, -1): DATA_DICT[0].d['ELT'][t:t+treatment_len].to_numpy().astype(int).reshape(-1, 1)},
#                    adjustment = [(NODES.OBS.value, -1)])
# resPD = cie.whatIfDo(outcome = NODES.PD.value, 
#                     treatment = NODES.TOD.value, lag = 0, treatment_values = 0*np.ones(shape = (treatment_len, 1)),
#                     conditions = {(NODES.PD.value, -1): DATA_DICT[0].d['PD'][t:t+treatment_len].to_numpy().astype(int).reshape(-1, 1)},
#                     adjustment = None)


# #! Causal Inference: this must be used for the round1 CIE -- CIE_100_HH_v4_round1
# resELT = cie.whatIfDo(outcome = NODES.ELT.value, 
#                    treatment = NODES.RV.value, lag = -1, treatment_values = 0.5*np.ones(shape = (treatment_len, 1)),
#                    conditions = {(NODES.CS.value, -1): np.zeros(shape = (treatment_len, 1)).astype(int),
#                                  (NODES.ELT.value, -1): np.round(DATA_DICT[0].d['ELT'][t:t+treatment_len].to_numpy().reshape(-1, 1), 1)},
#                    adjustment = [(NODES.OBS.value, -1)])
# resPD = cie.whatIfDo(outcome = NODES.PD.value, 
#                     treatment = NODES.TOD.value, lag = 0, treatment_values = 0*np.ones(shape = (treatment_len, 1)),
#                     conditions = {(NODES.PD.value, -1): np.round(DATA_DICT[0].d['PD'][t:t+treatment_len].to_numpy().reshape(-1, 1), 1)},
#                     adjustment = None)

gtELT = DATA_DICT_TEST.d['ELT'].to_numpy().reshape(-1, 1)

# # Bayesian Inference
# bn = get_DBN(cie.DAG["complete"].get_Adj(), cie.DAG["complete"].max_lag)

# data = {}
# for node in bn.nodes():
#     name = node.split("__")[0]
#     lag = int(node.split("__")[1])
#     if DATA_TYPE[name] == DataType.Continuous:
#         data[node] = np.round(DATA_DICT.d[name][cie.DAG["complete"].max_lag - abs(lag):DATA_DICT_TRAIN.T - abs(lag)].to_numpy(), 1)
#         # data[node] = np.round(DATA_DICT[0].d[name][cie.DAG["complete"].max_lag - abs(lag):DATA_DICT[0].T - abs(lag)].to_numpy(), 1)
#     else:
#         data[node] = DATA_DICT.d[name][cie.DAG["complete"].max_lag - abs(lag):DATA_DICT_TRAIN.T - abs(lag)].to_numpy(dtype=int)
# dataframe = pd.DataFrame(data)

# # Learn CPDs
# bn.fit(dataframe, estimator=MaximumLikelihoodEstimator)

# # Validate the model
# if not bn.check_model():
#     raise ValueError("Bayesian Network is invalid. Check structure and CPDs.")
# # save_bn_to_json(bn, "learned_bn.json")
# inference = VariableElimination(bn)

# bayesian_result = []
# for i in range(treatment_len):
#     rv_value = 0.5  # Treatment value
    
#     # Querying the conditional probability for ELT' given RV-1, ELT-1, OBS0, RB-1, WP-1
#     query_variable = f"{NODES.ELT.value}__{0}"
#     evidence = {
#         f"{NODES.RV.value}__{-1}": round(rv_value, 1),
#         f"{NODES.CS.value}__{-1}": int(DATA_DICT_TEST.d[NODES.CS.value][i].item()),
#         f"{NODES.ELT.value}__{-1}": round(DATA_DICT_TEST.d[NODES.ELT.value][i].item(), 1),
#         # f"{NODES.ELT.value}__{-1}": round(DATA_DICT_TEST.d[NODES.ELT.value][i].item() if i == 0 else bayesian_result[-1], 1),
#         # f"{NODES.OBS.value}__{0}": DATA_DICT[0].d['OBS'][i].item(),
#         # f"{NODES.RB.value}__{-1}": int(DATA_DICT[0].d['R_B'][i].item()),
#         # f"{NODES.WP.value}__{-1}": DATA_DICT[0].d['WP'][i].item(),
#     }
    
#     # result = inference.query(variables=[query_variable])
#     result = inference.query(variables=[query_variable], evidence=evidence)
    
#     # Compute the expected value of the distribution for ELT'
#     expected_value = sum(state * prob for state, prob in zip(result.state_names[query_variable], result.values))
#     bayesian_result.append(expected_value)

# bayesian_result = np.array(bayesian_result).reshape(-1, 1)

plt.figure()
plt.plot(gtELT, label='ELT - Ground Truth', color='k')
plt.plot(resELT, label=r"E[ELT] - $p(\mathrm{ELT}_{t} | \mathrm{Do}(R_{V_{t-1}}), \mathrm{C_{S_{t-1}}}, \mathrm{ELT}_{t-1})$", color='blue', linestyle='--')
# plt.plot(resELT_context, label=r"E[ELT] - $p(\mathrm{ELT}_{t} | \mathrm{Do}(R_{V_{t-1}}), \mathrm{C_{S_{t-1}}}, \mathrm{ELT}_{t-1})$", color='green', linestyle=':')
plt.plot(resELT_bayes, label=r"E[ELT] - $p(\mathrm{ELT}_{t} | \mathrm{R_{V_{t-1}}}, \mathrm{C_{S_{t-1}}}, \mathrm{ELT}_{t-1})$", color='red', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Values')
RMSE_do = np.sqrt(np.mean((resELT - gtELT) ** 2))
NRMSE_do = RMSE_do/np.std(gtELT) if np.std(gtELT) != 0 else 0
RMSE_bayes = np.sqrt(np.mean((resELT_bayes - gtELT) ** 2))
NRMSE_bayes = RMSE_bayes/np.std(gtELT) if np.std(gtELT) != 0 else 0
plt.title(f'Comparison of GT - DO (NRMSE: {NRMSE_do:.4f}) - Bayes (NRMSE: {NRMSE_bayes:.4f})')
plt.legend()
num_ticks = 25
tick_indices = np.linspace(0, len(gtELT) - 1, num_ticks, dtype=int)
safe_indices = [idx for idx in tick_indices if idx < len(T)]
tick_labels = [time.strftime("%H:%M:%S", time.gmtime(8 * 3600 + T[idx])) for idx in safe_indices]
plt.xticks(ticks=safe_indices, labels=tick_labels, rotation=45)

plt.figure()
plt.plot(DATA_PD_TEST.d.values, label='PD - Ground Truth', color='k')
plt.plot(resPD, label=r"E[PD] - $p(\mathrm{PD}_{t} | \mathrm{Do}(TOD_{t}), \mathrm{PD_{t-1}})$", color='blue', linestyle='--')
# plt.plot(resPD_noDO, label=r"E[PD] - $p(\mathrm{PD}_{t} | TOD_{t}, \mathrm{PD_{t-1}})$", color='tab:blue', linestyle=':')
plt.plot(resPD_bayes, label=r"E[PD] - $p(\mathrm{PD}_{t} | \mathrm{TOD_{t}}, \mathrm{PD_{t-1}})$", color='red', linestyle=':',)
plt.xlabel('Time')
plt.ylabel('Values')
RMSE_do = np.sqrt(np.mean((resPD - DATA_PD_TEST.d.values) ** 2))
NRMSE_do = RMSE_do/np.std(DATA_PD_TEST.d.values) if np.std(DATA_PD_TEST.d.values) > 0.001 else RMSE_do
RMSE_bayes = np.sqrt(np.mean((resPD_bayes - DATA_PD_TEST.d.values) ** 2))
NRMSE_bayes = RMSE_bayes/np.std(DATA_PD_TEST.d.values) if np.std(DATA_PD_TEST.d.values) > 0.001 else RMSE_bayes
plt.title(f'Comparison of GT - DO (NRMSE: {NRMSE_do:.4f}) - Bayes (NRMSE: {NRMSE_bayes:.4f})')
plt.legend()
num_ticks = 25
tick_indices = np.linspace(0, len(DATA_PD_TEST.d.values) - 1, num_ticks, dtype=int)
safe_indices = [idx for idx in tick_indices if idx < len(T)]
tick_labels = [time.strftime("%H:%M:%S", time.gmtime(8 * 3600 + T[idx])) for idx in safe_indices]
plt.xticks(ticks=safe_indices, labels=tick_labels, rotation=45)
plt.show()