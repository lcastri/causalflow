# Imports
import os
import numpy as np
import pandas as pd
from causalflow.basics.constants import DataType
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.graph import DAG
from causalflow.preprocessing.data import Data
from utils import *
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization
import matplotlib.pyplot as plt
import json

def get_DBN(link_assumptions, tau_max) -> BayesianNetwork:
        """
        Create a DAG represented by a Baysian Network.

        Args:
            link_assumptions (dict): DAG link assumptions.
            tau_max (int): max time lag.

        Raises:
            ValueError: source not well defined.

        Returns:
            BayesianNetwork: DAG represented by a Baysian Network.
        """
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


def save_bn_to_json(bn, filename):
    # Extract structure (edges)
    structure = list(bn.edges())
    
    # Extract CPDs
    cpd_dict = {}
    for cpd in bn.get_cpds():
        cpd_dict[str(cpd.variable)] = {
            "values": cpd.values.tolist(),  # Convert numpy array to list for JSON compatibility
            "evidence": cpd.variables[1:],  # Parents of the variable
            "cardinality": cpd.cardinality.tolist()  # Cardinality of the variable and its parents
        }
    
    # Combine structure and CPDs into a dictionary
    bn_data = {"structure": structure, "cpds": cpd_dict}
    
    # Save to JSON
    with open(filename, "w") as f:
        json.dump(bn_data, f)
    print(f"Bayesian Network saved to {filename}")
    
    
# DATA
DAGDIR = '/home/lcastri/git/causalflow/results/BL100_21102024_new/res.pkl'
INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
BAGNAME= ['noncausal-03012025']
cie = CIE.load('CIE_100_HH_v4/cie.pkl')
    
    
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
        DATA_DICT[idx] = Data(concatenated_df[var_names].values, varnames = var_names)
        del concatenated_df
        break

treatment_len = 120
t = 100

# Bayesian Inference
bn = get_DBN(cie.DAG["complete"].get_Adj(), cie.DAG["complete"].max_lag)

data = {}
for node in bn.nodes():
    name = node.split("__")[0]
    lag = int(node.split("__")[1])
    if DATA_TYPE[name] == DataType.Continuous:
        if name in [NODES.RB.value, NODES.ELT.value]:
            data[node] = DATA_DICT[0].d[name][cie.DAG["complete"].max_lag - abs(lag):DATA_DICT[0].T - abs(lag)].to_numpy(dtype=int)
        else:
            data[node] = np.round(DATA_DICT[0].d[name][cie.DAG["complete"].max_lag - abs(lag):DATA_DICT[0].T - abs(lag)].to_numpy(), 1)
    else:
        data[node] = DATA_DICT[0].d[name][cie.DAG["complete"].max_lag - abs(lag):DATA_DICT[0].T - abs(lag)].to_numpy(dtype=int)
dataframe = pd.DataFrame(data)

for node in bn.nodes():
    print(f"\nVariable: {node}, Cardinality: {dataframe[node].nunique()}")
    parents = list(bn.predecessors(node))
    if parents:
        print(f"Parents of {node}: {parents}")
        parent_card = [dataframe[parent].nunique() for parent in parents]
        print(f"Parent Cardinalities: {parent_card}")
        
# Learn CPDs
bn.fit(dataframe, estimator=MaximumLikelihoodEstimator)

# Validate the model
if not bn.check_model():
    raise ValueError("Bayesian Network is invalid. Check structure and CPDs.")
# save_bn_to_json(bn, "learned_bn.json")
inference = VariableElimination(bn)

bayesian_result = []
for i in range(treatment_len):
    rv_value = 0.5  # Treatment value
    
    # Querying the conditional probability for ELT'
    query_variable = f"{NODES.ELT.value}__{0}"
    evidence = {
        f"{NODES.RV.value}__{-1}": rv_value,
        f"{NODES.ELT.value}__{-1}": int(DATA_DICT[0].d['ELT'][t + i].item()),
        f"{NODES.OBS.value}__{0}": DATA_DICT[0].d['OBS'][t + i].item(),
        f"{NODES.RB.value}__{-1}": int(DATA_DICT[0].d['R_B'][t + i].item()),
        f"{NODES.WP.value}__{-1}": DATA_DICT[0].d['WP'][t + i].item(),
    }
    
    # result = inference.query(variables=[query_variable])
    result = inference.query(variables=[query_variable], evidence=evidence)
    
    # Compute the expected value of the distribution for ELT'
    # factor = result[query_variable]
    expected_value = sum(state * prob for state, prob in zip(result.state_names[query_variable], result.values))
    bayesian_result.append(expected_value)

bayesian_result = np.array(bayesian_result).reshape(-1, 1)