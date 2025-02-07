from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyAgrum
import pyAgrum.skbn as skbn
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gimg
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.causal as pyc
from cairosvg import svg2png
import pyAgrum.causal.notebook as cslnb

#! Generate data using the system of equations
# Set random seed for reproducibility
np.random.seed(42)

# Number of time steps
T = 1000

# Initialize arrays for each variable
A = np.random.normal(0, 1, T)  # Exogenous noise
E = np.random.normal(0, 1, T)  # Exogenous noise
B = np.zeros(T)
C = np.zeros(T)
D = np.zeros(T)

for t in range(1, T):
    B[t] = 0.1 * A[t] + 0.9 * E[t-1] + np.random.normal(0, 0.1)
    C[t] = 0.7 * B[t] + np.random.normal(0, 0.1)
    D[t] = 0.1 * C[t] + 3.5 * E[t-1] + np.random.normal(0, 0.1)

# Store in a DataFrame
df = pd.DataFrame({'At': A, 'Bt': B, 'Ct': C, 'Dt': D, 'Et': E})

# Add lagged variable
df['A0'] = np.concatenate([df['At'].values[1:], [0]])
df['B0'] = np.concatenate([df['Bt'].values[1:], [0]])
df['C0'] = np.concatenate([df['Ct'].values[1:], [0]])
df['D0'] = np.concatenate([df['Dt'].values[1:], [0]])
df['E0'] = np.concatenate([df['Et'].values[1:], [0]])
df = df.iloc[1:].reset_index(drop=True)  # Remove first row
df = df.iloc[:-1].reset_index(drop=True)  # Remove last row

# df.to_csv("results/5vars_test_pyAgrum/data.csv")
discretizer=skbn.BNDiscretizer(defaultDiscretizationMethod='quantile', discretizationThreshold=5, defaultNumberOfBins=10)
# discretizer=skbn.BNDiscretizer(defaultDiscretizationMethod='quantile', discretizationThreshold=10, defaultNumberOfBins='elbowMethod')
auditDict=discretizer.audit(df)

print()
print("** audit **")
for var in auditDict:
    print(f"- {var} : ")
    for k,v in auditDict[var].items():
        print(f"    + {k} : {v}")

template = discretizer.discretizedBN(df)
template.addArc("A0", "B0")
template.addArc("B0", "C0")
template.addArc("C0", "D0")
template.addArc("At", "Bt")
template.addArc("Bt", "Ct")
template.addArc("Ct", "Dt")
template.addArc("E0", "Bt")
template.addArc("E0", "Dt")
learner = pyAgrum.BNLearner(df, discretizer.discretizedBN(df))
learner.useEM(1e-6)
bn = learner.learnParameters(template)

time_slices_bn = gdyn.getTimeSlices(bn)
svg2png(bytestring=time_slices_bn,write_to='results/pyAgrum/dbn.png')

gimg.export(bn,"results/pyAgrum/bn.pdf")
gimg.exportInference(bn, "results/pyAgrum/inference.pdf")

cm = pyc.CausalModel(bn)

# ------------------------------
# Helper Function: Find Bin Index
# ------------------------------
def find_bin(value, edges):
    """
    Given a continuous value and an array of bin edges,
    return the index of the bin that contains the value.
    """
    idx = np.digitize(value, edges, right=False) - 1
    return int(max(0, min(idx, len(edges) - 2)))

# ------------------------------
# Precompute Bin Edges and Midpoints for Bt and Dt
# ------------------------------
num_bins = 10

# For Bt (used as evidence)
quantiles = np.linspace(0, 100, num_bins + 1)
edges_Bt = np.percentile(df['Bt'].values, quantiles)

# For Dt (used for expected value computation)
edges_Dt = np.percentile(df['Dt'].values, quantiles)
midpoints_Dt = [(edges_Dt[i] + edges_Dt[i+1]) / 2.0 for i in range(num_bins)]

# ------------------------------
# Evaluate Predictions on a Test Set using BN inference and CausalModel (do-intervention)
# ------------------------------
n_test = 100
predicted_Dt_bn = []      # Predictions using BN conditioning (setEvidence)
predicted_Dt_causal = []  # Predictions using CausalModel (do-intervention)
ground_truth_Dt = []

for i in range(n_test):
    # Use the continuous Bt value from the test instance
    b_val = df.iloc[i]['Bt']
    bin_idx = find_bin(b_val, edges_Bt)
    
    # --- BN prediction: Conditioning on Bt = bin_idx ---
    ie = pyAgrum.LazyPropagation(bn)
    ie.setEvidence({"Bt": bin_idx})
    ie.addTarget('Dt')
    ie.makeInference()
    bn_posterior = ie.posterior("Dt")
    bn_posterior_values = bn_posterior.toarray()
    pred_bn = sum(bn_posterior_values[j] * midpoints_Dt[j] for j in range(num_bins))
    predicted_Dt_bn.append(pred_bn)

    # --- CausalModel prediction: Intervention do(Bt = bin_idx) ---
    # Convert the BN to a CausalModel.
    # Note: In pyAgrum, the CausalModel is built directly from the BN.
    formula, adj, exp = pyc.causalImpact(cm, "Dt", "Bt", values={"Bt":bin_idx})
    posterior_causal = adj.toarray()
    pred_causal = sum(posterior_causal[j] * midpoints_Dt[j] for j in range(num_bins))
    predicted_Dt_causal.append(pred_causal)
    
    # Ground-truth Dt value
    ground_truth_Dt.append(df.iloc[i]['Dt'])

predicted_Dt_bn = np.array(predicted_Dt_bn)
predicted_Dt_causal = np.array(predicted_Dt_causal)
ground_truth_Dt = np.array(ground_truth_Dt)

# ------------------------------
# Compare Predictions to Ground Truth
# ------------------------------
import matplotlib.pyplot as plt
bn_RMSE = np.sqrt(np.mean((predicted_Dt_bn - ground_truth_Dt) ** 2))
bn_NRMSE = bn_RMSE/np.std(ground_truth_Dt)
cm_RMSE = np.sqrt(np.mean((predicted_Dt_causal - ground_truth_Dt) ** 2))
cm_NRMSE = cm_RMSE/np.std(ground_truth_Dt)

# ------------------------------
# Plot Predictions and Residuals
# ------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot predicted vs. ground truth
ax1.plot(ground_truth_Dt, label=r'Ground Truth: $D_{t}$', marker='o', linestyle='-')
ax1.plot(predicted_Dt_bn, label=r'BN: $E[D_{t}|B_{t}]$' + f" (NRMSE: {bn_NRMSE:.2f})", marker='s', linestyle='--')
ax1.plot(predicted_Dt_causal, label=r'CM: $E[D_{t}|do(B_{t})]$' + f" (NRMSE: {cm_NRMSE:.2f})", marker='^', linestyle=':')
ax1.set_ylabel('$D_{t}$')
ax1.set_title('Predictions: BN vs. CausalModel')
ax1.legend()

# Plot absolute errors for each method
abs_error_bn = np.abs(predicted_Dt_bn - ground_truth_Dt)
abs_error_cm = np.abs(predicted_Dt_causal - ground_truth_Dt)
ax2.plot(abs_error_bn, label='BN Absolute Error', marker='s', linestyle='--')
ax2.plot(abs_error_cm, label='CM Absolute Error', marker='^', linestyle=':')
ax2.set_xlabel('Test Instance')
ax2.set_ylabel('Absolute Error')
ax2.set_title('Absolute Prediction Errors')
ax2.legend()
plt.show()