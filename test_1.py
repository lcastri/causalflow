from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE    
import numpy as np
cie = CIE.load('CIE_100_HH_v5/cie.pkl')
value_ranges = cie.DBNs[('obs', 0)].data.d['ELT'].unique()
bn = BayesianNetwork.load("pgmpy_network.bif")
inference = VariableElimination(bn)
phi_query = inference.query(variables=['ELT_t'], evidence={'R_V_t_1': 2, 'C_S_t_1': 0, 'ELT_t_1': 45})
expected_value = np.sum(value_ranges*phi_query.values)
print(f"Expected Value of ELT_t: {expected_value}")

# expected_value = sum(i * result[i] for i in range(result.shape[0]))
# print(f"Expected Value of ELT_t: {expected_value}")

# import pyAgrum as gum

# bn = gum.loadBN("network.bif")

# ie = gum.LazyPropagation(bn)
# ie.setEvidence({'R_V_t_1': 2, 'C_S_t_1': 0, 'ELT_t_1': 45})
# # ie.setEvidence({'R_V_t_1': 0, 'C_S_t_1': 0, 'ELT_t_1': 88})
# ie.setTargets({'ELT_t'})
# ie.makeInference()

# result = ie.posterior("ELT_t")  # Get the posterior distribution of D

# expected_value = sum(i * result[i] for i in range(result.shape[0]))
# print(f"Expected Value of ELT_t: {expected_value}")
