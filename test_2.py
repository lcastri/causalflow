import copy
import itertools
from causalflow.causal_reasoning.CausalInferenceEngine import CausalInferenceEngine as CIE    
import causalflow.causal_reasoning.Utils as DensityUtils  
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from causalflow.graph import DAG


def get_node(node):
    name, lag = node
    return f"{name}_t{'_'+str(abs(lag)) if abs(lag) > 0 else ''}"
            
def generate_histogram_from_gmm(node, value_range, parents_values=None):
    """
    Generate histogram bin probabilities from a GMM.
    Args:
        node (tuple): The node (variable, lag).
        value_range (tuple): The (min, max) range of the variable.
    Returns:
        np.array: The probability of each bin.
    """
    gmm = cie.DBNs[('obs', 0)].dbn_data[node].pJoint
    
    bin_edges = sorted(value_range)
    bin_probs = np.zeros(len(value_range))
    
    # If the node has parents, iterate over all possible combinations of parent values
    if parents_values:
        conditioned_gmm_params = DensityUtils.compute_conditional(gmm, np.array([p for p in parents_values.values()]).reshape(-1, 1))

        # Compute the probability of each bin using the conditioned GMM
        for i in range(1, len(value_range)):
            bin_start = bin_edges[i-1]
            bin_end = bin_edges[i]
            bin_probs[i] += integrate_gmm(conditioned_gmm_params, bin_start, bin_end)
            
        # Normalize the probabilities
        bin_probs /= bin_probs.sum()  # Normalize to ensure it sums to 1
        return bin_probs
    else:
        for i in range(1, len(value_range)):
            bin_start = bin_edges[i-1]
            bin_end = bin_edges[i]
            bin_probs[i] += integrate_gmm(gmm, bin_start, bin_end)
        bin_probs /= bin_probs.sum()  # Normalize to ensure it sums to 1
        return bin_probs

    
    
def integrate_gmm(gmm_params, start, end, num_points=1000):
    """
    Integrate the GMM over a range [start, end].

    Args:
        gmm_params (dict): Dictionary containing GMM parameters: {'means': [], 'weights': [], 'covariances': []}.
        start (float): Start of the range.
        end (float): End of the range.
        num_points (int): Number of points for numerical integration.

    Returns:
        float: The integrated probability.
    """
    x = np.linspace(start, end, num_points)
    pdf_values = DensityUtils.get_density(gmm_params, x)
    return np.trapz(pdf_values, x)


def populate_cpt_for_node(node, value_range, parents_dict=None):
    """
    Populate the CPT for a given node, accounting for its parents and their values.
    Args:
        node (tuple): The node (variable, lag).
        value_range (tuple): The (min, max) range of the variable.
        parents_dict (dict, optional): A dictionary of parent nodes and their values.
    """
    if parents_dict is None:
        # If no parents, simply fill the CPT with the marginal probabilities
        bin_probs = generate_histogram_from_gmm(node, value_range)
        cpd = TabularCPD(variable=get_node(node), variable_card=len(value_range), values=bin_probs.reshape(-1, 1))
    else:
        # If the node has parents, iterate over all combinations of parent values
        parent_combinations = list(itertools.product(*parents_dict.values()))  # Cartesian product of parent values
        
        with tqdm(total=len(parent_combinations), desc=f'Populating CPT for node {get_node(node)}') as pbar:
            cpd_values = []
            for parent_values in parent_combinations:
                # Map parent values to their corresponding node
                parent_dict = dict(zip(parents_dict.keys(), parent_values))
                
                # Generate the conditional bin probabilities based on parent values
                conditional_bin_probs = generate_histogram_from_gmm(node, value_range, parents_values=parent_dict)
                    
                cpd_values.append(conditional_bin_probs)
                pbar.update(1)

            # Reshape to match the expected format
            cpd_values = np.array(cpd_values).T
            cpd = TabularCPD(variable=get_node(node), variable_card=len(value_range), values=cpd_values,
                             evidence=list(parents_dict.keys()), evidence_card=[len(v) for v in parents_dict.values()])
            
    return cpd

# Example usage                     
cie = CIE.load('CIE_100_HH_v5/cie.pkl')
nodes = [(f, -abs(l)) for f in cie.DBNs[('obs', 0)].dag.features if f != 'WP' for l in range(cie.DBNs[('obs', 0)].dag.max_lag + 1)]
value_ranges = {var: cie.DBNs[('obs', 0)].data.d[var[0]].unique() for var in nodes}

dag = copy.deepcopy(cie.DBNs[('obs', 0)].dag)
for t in cie.DBNs[('obs', 0)].dag.g:
    for s in cie.DBNs[('obs', 0)].dag.g[t].sources:
        if s[0] == 'WP':
            dag.del_source(t, s[0], s[1])
del dag.g['WP']
bn_pgmpy = DAG.get_DBN(cie.DBNs[('obs', 0)].dag.get_Adj(), cie.DBNs[('obs', 0)].dag.max_lag)
            
# Add CPDs
for node in nodes:
    if node[0] == 'WP': continue
    
    # Generate histogram and bin probabilities from the GMM
    parent_value = None
    if cie.DBNs[('obs', 0)].dbn_data[node].parents is not None:
        parent_value = {get_node((parent, -abs(cie.DBNs[('obs', 0)].dbn_data[node].parents[parent].lag))): value_ranges[(parent, -abs(cie.DBNs[('obs', 0)].dbn_data[node].parents[parent].lag))] for parent in cie.DBNs[('obs', 0)].dbn_data[node].parents}
    
    cpd = populate_cpt_for_node(node, value_ranges[node], parents_dict=parent_value)
    bn_pgmpy.add_cpds(cpd)
    
bn_pgmpy.save("pgmpy_network.bif")