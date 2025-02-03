from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.causal_reasoning.SMCFilter import SMCFilter

cie = CIE.load('CIE_100_HH_v6/cie.pkl')

# Initialize the SMC filter
smc = SMCFilter(cie.DBNs[('obs', 0)], num_particles=1000)

# # Initialize particles for all relevant nodes
# for node in cie.DBNs[('obs', 0)].dbn.keys():
#     smc.initialize_particles(node)

# Run inference over 10 time steps
for t in range(10):
    given_values = {
        ('R_V', -1): 0.5,  
        ('C_S', -1): 0,  
        ('ELT', -1): 88,  
        ('TOD', -1): 0,  
        ('WP', -1): 1
    }
    smc.step_ancestor_query("ELT", given_values)

# Answer queries
posterior_gmm = smc.estimate_posterior("ELT_t")
expected_value = smc.compute_expectation("ELT_t")
map_estimate = smc.compute_map_estimate("ELT_t")

print("Posterior Distribution (GMM):", posterior_gmm)
print("Expected Value of ELT_t:", expected_value)
print("MAP Estimate of ELT_t:", map_estimate)
