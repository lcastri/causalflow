
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import *
import numpy as np
import pandas as pd
import pyAgrum
import pyAgrum.skbn as skbn
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.causal as pyc
import pyAgrum.causal.notebook as cslnb
from pyAgrum.lib.discretizer import Discretizer
import re
import pickle
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.stats import iqr, gaussian_kde, entropy


def format_evidence_for_latex(evidence):
    formatted_evidence = []
    
    for var in evidence.keys():
        # Replace the part after the underscore with curly braces, if applicable
        var = re.sub(r'_(\w+)', r'_{\1}', var)  # Add {} around the part after the underscore
        
        # Replace '0' with '_{t-1}'
        var = var.replace('t', '_t')
        var = var.replace('0', '_{t-1}')

        # Append formatted variable to the list
        formatted_evidence.append(f"${var}$")  # Adding $ for LaTeX format
        
    return ', '.join(formatted_evidence)

def plot_distributions(var, distributions, var_midpoints, xlabel, ylabel):
    # Number of distributions to plot
    n = len(distributions)
    
    # Create subplots (one for each distribution)
    fig, axes = plt.subplots(1, n, figsize=(12, 4))

    # If there's only one plot, axes is not a list, so we handle it separately
    if n == 1:
        axes = [axes]
    var_midpoints = [round(float(m),2) for m in var_midpoints]
    for i, (data, title) in enumerate(distributions):
        # Plot the distribution in the corresponding subplot
        axes[i].bar(range(len(data)), data)
        axes[i].set_xticks(range(len(data)), var_midpoints)
        var_str = format_evidence_for_latex({var: None})
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)

    plt.tight_layout()
    plt.show()
    
def find_bin(value, edges):
    """
    Given a continuous value and an array of bin edges,
    return the index of the bin that contains the value.
    """
    idx = np.digitize(value, edges, right=False) - 1
    return int(max(0, min(idx, len(edges) - 2)))


def get_info(D, auditDict, var):
    if 'param' in auditDict[var] and isinstance(auditDict[var]['param'], list):
        edges = auditDict[var]['param']
        midpoints = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)]
        return auditDict[var]['param'], edges, midpoints
    quantiles = np.linspace(0, 100, auditDict[var]['param'] + 1 if 'param' in auditDict[var] else auditDict[var]['nbBins'] + 1)
    edges = np.percentile(D[var].values, quantiles)
    midpoints = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)]
    return quantiles, edges, midpoints


def optimal_bins_elbow_method(time_series, min_bins=2, max_bins=15, plot=True):
    """
    Determines the optimal number of bins using the elbow method with KMeans clustering.

    Parameters:
    - time_series: numpy array or list of numerical values (1D time series)
    - min_bins: minimum number of bins to consider
    - max_bins: maximum number of bins to consider
    - plot: whether to plot the elbow curve

    Returns:
    - optimal_bins: The optimal number of bins for discretization
    """
    time_series = np.array(time_series).reshape(-1, 1)  # Ensure data is 2D for KMeans

    inertia_values = []
    bin_range = range(min_bins, max_bins + 1)

    for k in bin_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(time_series)
        inertia_values.append(kmeans.inertia_)

    # Use KneeLocator to find the elbow point
    kneedle = KneeLocator(bin_range, inertia_values, curve="convex", direction="decreasing")
    optimal_bins = kneedle.elbow

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(bin_range, inertia_values, marker="o", linestyle="-", label="Inertia")
        plt.axvline(optimal_bins, linestyle="--", color="red", label=f"Optimal Bins: {optimal_bins}")
        plt.xlabel("Number of Bins (Clusters)")
        plt.ylabel("Inertia (WCSS)")
        plt.title("Elbow Method for Optimal Bin Selection")
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_bins

def loaf_D(indir, bagname):
    for wp in WP:
        dfs = []
        if wp == WP.PARKING or wp == WP.CHARGING_STATION: continue
        for tod in TOD:
            if tod is TOD.OFF: continue
            files = [f for f in os.listdir(os.path.join(indir, "HH/my_nonoise", f"{bagname}", f"{tod.value}"))]
            files_split = [f.split('_') for f in files]
            wp_files = [f for f in files_split if len(f) == 3 and f[2].split('.')[0] == wp.value][0]
            wp_file = '_'.join(wp_files)
            print(f"Loading : {wp_file}")
            filename = os.path.join(indir, "HH/my_nonoise", f"{bagname}", f"{tod.value}", wp_file)

            df = pd.read_csv(filename)
            dfs.append(df)
        concat_df = pd.concat(dfs, ignore_index=True)
        break
 
    D = concat_df.drop('pf_elapsed_time', axis=1)
    D = D.drop('T', axis=1)
    D = D.drop('R_X', axis=1)
    D = D.drop('R_Y', axis=1)
    D = D.drop('G_X', axis=1)
    D = D.drop('G_Y', axis=1)
    D = D.drop('NP', axis=1)
    D = D.drop('R_B', axis=1)
    D = D.drop('C_S', axis=1)
    D = D.drop('TOD', axis=1)
    D = D.drop('WP', axis=1)
    D = D.drop('PD', axis=1)
    original_names = list(D.columns)
    D.columns = [f'{v}t' for v in original_names]

    # Add lagged variable
    for v in original_names:
        D[f'{v}0'] = np.concatenate([D[f'{v}t'].values[1:], [0]])
    D = D.iloc[1:].reset_index(drop=True)  # Remove first row
    D = D.iloc[:-1].reset_index(drop=True)  # Remove last row
    return D


INDIR = '/home/lcastri/git/PeopleFlow/utilities_ws/src/RA-L/hrisim_postprocess/csv'
D = loaf_D(INDIR, 'noncausal-28022025')

# DISCRETIZATION
discretizer = Discretizer(defaultDiscretizationMethod='quantile')
RV_bins = optimal_bins_elbow_method(D['R_Vt'].values, plot=False)
EC_bins = optimal_bins_elbow_method(D['ECt'].values, plot=False)

# Set the same binning for all datasets
discretizer.setDiscretizationParameters('R_V0', 'quantile', RV_bins)
discretizer.setDiscretizationParameters('R_Vt', 'quantile', RV_bins)
discretizer.setDiscretizationParameters('EC0', 'quantile', EC_bins)
discretizer.setDiscretizationParameters('ECt', 'quantile', EC_bins)
# discretizer.setDiscretizationParameters('R_V0', 'expert', [0, 0.088, 0.499, 0.509])
# discretizer.setDiscretizationParameters('R_Vt', 'expert', [0, 0.088, 0.499, 0.509])
# discretizer.setDiscretizationParameters('EC0', 'quantile', 3)
# discretizer.setDiscretizationParameters('ECt', 'quantile', 3)

# Define structure
template = discretizer.discretizedBN(D)  # Use one dataset to define structure

template.addArc("WP0", "PD0")
template.addArc("TOD0", "PD0")
template.addArc("WPt", "PDt")
template.addArc("TODt", "PDt")
template.addArc("PD0", "PDt")

template.addArc("OBS0", "EC0")
template.addArc("OBS0", "R_V0")
template.addArc("R_V0", "EC0")
template.addArc("OBSt", "R_Vt")
template.addArc("OBSt", "ECt")
template.addArc("R_Vt", "ECt")
template.addArc("R_V0", "R_Vt")

# Learn parameters for each dataset
discretized_D = discretizer.discretizedBN(D)  # Apply the same discretization
auditDict=discretizer.audit(D)

learner = pyAgrum.BNLearner(D, template)
learner.useSmoothingPrior()
bn = learner.learnParameters(template)

cm = pyc.CausalModel(bn)

target_var = 'ECt'
target_var_str = format_evidence_for_latex({target_var: None})
intervention_var = "R_Vt"

static_duration = 5
dynamic_duration = 4
charging_time = 2
LOAD_FACTOR = 5
ROBOT_MAX_VEL = 0.5
K_nl_s = 100 / (static_duration * 3600)
K_nl_d = (100 / (dynamic_duration * 3600) - K_nl_s)/(ROBOT_MAX_VEL)
K_l_s = K_nl_s * LOAD_FACTOR
K_l_d = K_nl_d * LOAD_FACTOR

quantiles_R_V, edges_R_V, midpoints_R_V = get_info(test_03, auditDict, intervention_var)
quantiles_EC, edges_EC, midpoints_EC = get_info(test_03, auditDict, target_var)
STEP = 5
time = 5

plot_fontsize = 20

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'  # Set background to white globally
plt.rcParams['axes.facecolor'] = 'white'    # Set axis background to white
STEP = 5
time = 5
gt_noload = []
gt_load = []
gts = []
bayes_preds = []
causal_preds = []

RV_range = midpoints_R_V
for bn, cm in zip(bns, cms):
    tmp_gt_noload = []
    tmp_gt_load = []
    tmp_gts = []
    tmp_bayes_preds = []
    tmp_causal_preds = []
    for RV_val in RV_range:
        RV_bin_idx = find_bin(RV_val, edges_R_V)
        RV_val = midpoints_R_V[RV_bin_idx]
        GTLOAD = time * (K_l_s + K_l_d * RV_val) 
        GTNOLOAD = time * (K_nl_s + K_nl_d * RV_val)
        
        # --- BN prediction ---
        ie = pyAgrum.VariableElimination(bn)
        bn_prior = ie.posterior(target_var)
        evidence = {intervention_var: RV_bin_idx}
        evidence_str = format_evidence_for_latex(evidence)
        ie.setEvidence(evidence)
        ie.makeInference()
        bn_posterior = ie.posterior(target_var)
        bn_posterior_values = bn_posterior.toarray()
        pred_bn = sum(bn_posterior_values[j] * midpoints_EC[j] for j in range(len(bn_posterior_values)))
        bayes_pred = (time/STEP)*pred_bn
                
        # --- CausalModel prediction ---
        formula, adj, exp = pyc.causalImpact(cm, on=target_var, doing=intervention_var, values=evidence)
        posterior_causal = adj.toarray()
        pred_causal = sum(posterior_causal[j] * midpoints_EC[j] for j in range(len(posterior_causal)))
        causal_pred = (time/STEP)*pred_causal
                
        tmp_gt_noload.append(GTNOLOAD)
        tmp_gt_load.append(GTLOAD)
        tmp_gts.append((GTNOLOAD+GTLOAD)/2)
        tmp_bayes_preds.append(bayes_pred)
        tmp_causal_preds.append(causal_pred)
    gt_noload.append(tmp_gt_noload)
    gt_load.append(tmp_gt_load)
    gts.append(tmp_gts)
    bayes_preds.append(tmp_bayes_preds)
    causal_preds.append(tmp_causal_preds)

# Define bar width and offsets
x_indexes = np.arange(len(RV_range))

# Compute absolute errors
for i in range(len(bns)):

    cm_errors = np.abs(np.array(causal_preds[i]) - np.array(gts[i]))
    bn_errors = np.abs(np.array(bayes_preds[i]) - np.array(gts[i]))

    # Compute Mean Absolute Error (MAE)
    cm_MAE = np.mean(cm_errors)
    bn_MAE = np.mean(bn_errors)

    # Plot bars for each category with offset
    plt.figure(figsize=(12, 6))
    bar_width = 0.375
    plt.bar(x_indexes, bn_errors, width=bar_width, label=r"$\mid (\text{GT})~\Delta R_{B}(t) - (\text{BN})~\Delta R_{B}(t) \mid$", color='tab:orange', alpha=1)
    plt.bar(x_indexes + bar_width, cm_errors, width=bar_width, label=r"$\mid (\text{GT})~\Delta R_{B}(t) - (\text{CM})~\Delta R_{B}(t) \mid$", color='tab:green', alpha=1)
    
    # Labeling
    plt.xlabel(r"$R_V(t)$ Bins", fontdict={'fontsize': plot_fontsize})
    plt.ylabel(r"$\mid (\text{GT})~\Delta R_{B}(t) - (\text{Estimated})~\Delta R_{B}(t) \mid$", fontdict={'fontsize': plot_fontsize})
    plt.xticks(x_indexes+bar_width/2, [f"({edges_R_V[i-1]:.3f}, {edges_R_V[i]:.3f})" for i in range(1, len(edges_R_V))], fontdict={'fontsize': plot_fontsize})  # Format x-axis labels
    plt.yticks(fontsize=plot_fontsize)  # Format x-axis labels
    plt.title(r"Comparison of $\Delta R_B(t)$ " + f"Estimation Errors per Velocity Band\nBN MAE: {bn_MAE:.2f}% -- CM MAE: {cm_MAE:.2f}%", fontdict={'fontsize': plot_fontsize})
    plt.legend(fontsize=plot_fontsize, loc=(0.2, 0.75))
    plt.tight_layout()

    plt.show()
    
myGT = np.array(gts)[1:,:]
myBN = np.array(bayes_preds)[1:,:]
myCM = np.array(causal_preds)[1:,:]
    
print("GT:")
print(myGT)
print("BN:")
print(myBN)
print("CM:")
print(myCM)

print("BN Error:")
print(np.abs(myBN-myGT))
print("CM Error:")
print(np.abs(myCM-myGT))

print(f"BN Mean Abs Error: {np.abs(np.mean(myBN-myGT, axis=0))}")
print(f"CM Mean Abs Error: {np.abs(np.mean(myCM-myGT, axis=0))}")

print(f"BN RMSE: {np.sqrt(np.mean((myBN - myGT)**2, axis=0))}")
print(f"CM RMSE: {np.sqrt(np.mean((myCM - myGT)**2, axis=0))}")


# print("BN-CM Error:")
# print(np.abs(myBN-myGT)-np.abs(myCM-myGT))
    
bn_errors = np.abs(np.mean(myBN-myGT, axis=0))
cm_errors = np.abs(np.mean(myCM-myGT, axis=0))
# bn_errors = np.sqrt(np.mean((myBN - myGT)**2, axis=0))
# cm_errors = np.sqrt(np.mean((myCM - myGT)**2, axis=0))
# cm_errors = np.abs(np.mean(np.array(causal_preds)[[2,4], :], axis=0) - np.array(gts[0]))
# bn_errors = np.abs(np.mean(np.array(bayes_preds)[[2,4], :], axis=0) - np.array(gts[0]))

# Compute Mean Absolute Error (MAE)
cm_MAE = np.mean(cm_errors)
bn_MAE = np.mean(bn_errors)
# Plot bars for each category with offset
plt.figure(figsize=(12, 6))
bar_width = 0.375
plt.bar(x_indexes, bn_errors, width=bar_width, label=r"$\mid (\text{GT})~\Delta R_{B}(t) - (\text{BN})~\Delta R_{B}(t) \mid$", color='tab:orange', alpha=1)
plt.bar(x_indexes + bar_width, cm_errors, width=bar_width, label=r"$\mid (\text{GT})~\Delta R_{B}(t) - (\text{CM})~\Delta R_{B}(t) \mid$", color='tab:green', alpha=1)

# Labeling
plt.xlabel(r"$R_V(t)$ Bins", fontdict={'fontsize': plot_fontsize})
plt.ylabel(r"Mean $\mid (\text{GT})~\Delta R_{B}(t) - (\text{Estimated})~\Delta R_{B}(t) \mid$", fontdict={'fontsize': plot_fontsize})
plt.xticks(x_indexes+bar_width/2, [f"({edges_R_V[i-1]:.3f}, {edges_R_V[i]:.3f})" for i in range(1, len(edges_R_V))], fontdict={'fontsize': plot_fontsize})  # Format x-axis labels
plt.yticks(fontsize=plot_fontsize)  # Format x-axis labels
plt.title(r"Comparison of $\Delta R_B(t)$ " + f"Estimation Errors per Velocity Band\nBN MAE: {bn_MAE:.2f}% -- CM MAE: {cm_MAE:.2f}%", fontdict={'fontsize': plot_fontsize})
plt.legend(fontsize=plot_fontsize)
plt.tight_layout()
plt.show()


# %%
RV_val = 0.5
RV_bin_idx = find_bin(RV_val, edges_R_V)
RV_val = midpoints_R_V[RV_bin_idx]
GTLOAD = time * (K_l_s + K_l_d * RV_val) 
GTNOLOAD = time * (K_nl_s + K_nl_d * RV_val)

# --- BN prediction ---
ie = pyAgrum.VariableElimination(bn)
bn_prior = ie.posterior(target_var)
evidence = {intervention_var: RV_bin_idx}
evidence_str = format_evidence_for_latex(evidence)
ie.setEvidence(evidence)
ie.makeInference()
bn_posterior = ie.posterior(target_var)
bn_posterior_values = bn_posterior.toarray()
pred_bn = sum(bn_posterior_values[j] * midpoints_EC[j] for j in range(len(bn_posterior_values)))
bayes_pred = (time/STEP)*pred_bn
        
# --- CausalModel prediction ---
formula, adj, exp = pyc.causalImpact(cm, on=target_var, doing=intervention_var, values=evidence)
posterior_causal = adj.toarray()
pred_causal = sum(posterior_causal[j] * midpoints_EC[j] for j in range(len(posterior_causal)))
causal_pred = (time/STEP)*pred_causal
        
print(f"GT {(GTNOLOAD+GTLOAD)/2:.3f}")
print(f"BN {bayes_pred:.3f} -- AE {np.abs(bayes_pred - (GTNOLOAD+GTLOAD)/2):.3f}")
print(f"CM {causal_pred:.3f} -- AE {np.abs(causal_pred - (GTNOLOAD+GTLOAD)/2):.3f}")

# %%
# OBS PRIOR DISTRIBUTION
D = test_05
R_V_selection = D
counts, bins = np.histogram(R_V_selection['OBSt'].values, bins=2)
plt.hist([0, 1], bins, weights=counts/sum(counts), width=0.4, alpha=1, label='OBS')
plt.xticks([0.2, 0.7], ['0', '1'])
plt.xlim([-0.1, 1])
plt.xlabel('OBS')
plt.ylabel(f'p(OBS)')
plt.title(f"Prior Distribution of OBS")
plt.show()

# RV 
RV_val = 0.25
RV_bin_idx = find_bin(RV_val, edges_R_V)

## OBS GIVEN RV
R_V_selection = D[abs(D['R_Vt'] - RV_val) < 0.05]
counts, bins = np.histogram(R_V_selection['OBSt'].values, bins=2)
plt.hist([0, 1], bins, weights=counts/sum(counts), width=0.4, alpha=1, label='OBS')
plt.xticks([0.2, 0.7], ['0', '1'])
plt.xlim([-0.1, 1])
plt.xlabel('OBS')
plt.ylabel(f'p(OBS|RV = {RV_val})')
plt.title(f"Distribution of OBS given RV")
plt.show()

## BC GIVEN RV OBS = 0
OBS_val = 0
ie = pyAgrum.VariableElimination(bn)
bn_prior = ie.posterior(target_var)
evidence = {"R_Vt": RV_bin_idx, "OBSt": OBS_val}
evidence_str = format_evidence_for_latex(evidence)
ie.setEvidence(evidence)
ie.makeInference()
bn_posterior = ie.posterior(target_var)
bn_posterior_values = bn_posterior.toarray()
plot_distributions("ECt", [(bn_posterior_values, "")], midpoints_EC, xlabel="BC", ylabel=f"p(BC|OBS = {OBS_val}, RV = {RV_val})")

## BC GIVEN RV OBS = 1
OBS_val = 1
ie = pyAgrum.VariableElimination(bn)
bn_prior = ie.posterior(target_var)
evidence = {"R_Vt": RV_bin_idx, "OBSt": OBS_val}
evidence_str = format_evidence_for_latex(evidence)
ie.setEvidence(evidence)
ie.makeInference()
bn_posterior = ie.posterior(target_var)
bn_posterior_values = bn_posterior.toarray()
plot_distributions("ECt", [(bn_posterior_values, "")], midpoints_EC, xlabel="BC", ylabel=f"p(BC|OBS = {OBS_val}, RV = {RV_val})")


