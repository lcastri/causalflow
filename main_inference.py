from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE
from causalflow.causal_reasoning.SMCFilter import SMCFilter

cie = CIE.load('CIE_100_HH_v6/cie.pkl')

# Initialize the SMC filter
smc = SMCFilter(cie.DBNs[('obs', 0)], num_particles=500)

# Define initial evidence (at t = -1)
given_values = {
    ('R_V', -1): 0.5,  
    ('ELT', -1): 89,  
}
given_context = {
    'C_S': 0,  
    'TOD': 0,  
    'WP': 1
}

# Run inference for 10 time steps
expected_values_over_time = smc.sequential_query("ELT", given_values, given_context, num_steps=10)

# Print expectations
for t, exp_value in enumerate(expected_values_over_time):
    print(f"Step {t}: Expected Value of ELT = {exp_value}")
    
    
# Initialize SMC filter
smc_do = SMCFilter(cie.DBNs[('obs', 0)], num_particles=100)

# Define initial evidence (at t = -1)
given_values = {
    ('R_V', -1): 0.5,  
    ('ELT', -1): 89,  
}
given_context = {
    'C_S': 0,  
    'TOD': 0,  
    'WP': 1
}

# Run inference for 10 time steps
expected_values_over_time = smc_do.sequential_query("ELT", given_values, given_context, num_steps=10, 
                                                    intervention_var=('R_V', -1), adjustment_set=[('OBS', -1)])

# Print expectations
for t, exp_value in enumerate(expected_values_over_time):
    print(f"Step {t}: Expected Value of ELT = {exp_value}")
