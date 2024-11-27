import ast
import copy
import gc
import itertools
import json
import numpy as np
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Utils import *
from causalflow.basics.constants import SampleMode
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density_GMM import Density
from causalflow.causal_reasoning.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.basics.constants import *
from typing import Dict
from scipy.stats import entropy
import h5py

class DynamicBayesianNetwork():
    def __init__(self, 
                 filename, id, 
                 dag: DAG, 
                 data: Data, 
                 nsample, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType], 
                 batch_size: int,
                 dbn = None,
                 recycle = None):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            nsample (SampleMode/int/dict): if SampleMode - Mode to discovery the number of samples.
                                           if int - Common number of samples for all the variables.
                                           if dict - Number of samples for each variable.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}.
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}.
            batch_size (int): Batch Size.
            dbn (DynamicBayesianNetwork): DBN to load. Default None.
            recycle ((list[str], DynamicBayesianNetwork)): tuple containing the list of features to recycle and the 
                                                           DBN from which extract their densities. Default None.
        """
        if not isinstance(nsample, SampleMode) and not isinstance(nsample, int) and not isinstance(nsample, dict):
            raise ValueError("nsample field must be either SampleMode or int or dict!")
        
        self.data_type = data_type
        self.node_type = node_type
        
        if dbn is None:
            for node in dag.g:
                if self.node_type[node] == NodeType.Context: continue
                self.dbn = {node: None}
                anchestors = dag.get_node_anchestors(node)
                context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                contexts = self._extract_contexts(data, context_anchestors)
                for context in contexts:
                    # Retrieve data for each node
                    d = self.get_context_specific_data(data, context, node, dag.g[node].sourcelist)
                
                    # Retrieve the number of samples for each node
                    if isinstance(nsample, SampleMode):
                        nsamples = DynamicBayesianNetwork.estimate_optimal_samples(d, nsample)
                    elif isinstance(nsample, int):
                        nsamples = {f: nsample for f in d.columns}
                    elif isinstance(nsample, dict):
                        nsamples = nsample          
                    
                    Y = Process(d[node].to_numpy(), node, 0, nsamples[node], self.data_type[node], self.node_type[node])
                    X = self._extract_parents(node, d, dag, nsamples, self.data_type, self.node_type)

                    parents_str = f" - parents {', '.join(list(X.keys()))}" if X else ""
                    CP.info(f"\n    ### Target variable: {node}{parents_str}")
                    CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
                    for k, v in nsamples.items(): CP.debug(f"    {k} samples : {v}")
                    if self.dbn[node] is None: self.dbn[node] = {}
                    self.dbn[node][context] = Density(Y, batch_size, X)
                    self.save(filename, id, node, nsamples, context)
                
        else:
            self.dbn = dbn          
              
        del dag, data, recycle
        gc.collect()
        
        
    def save(self, filename: str, id, node, samples, context = None):
        """Save the DBN object to an HDF5 file."""
        with h5py.File(filename, 'a') as f:
            group_name = f"DBNs/{id}/{node}/{context}" if context is not None else f"DBNs/{id}/{node}"
                
            if group_name in f: del f[group_name]
            node_group = f.require_group(group_name)
            node_group.attrs["nsamples"] = json.dumps(samples)
            density = self.dbn[node][context] if context is not None else self.dbn[node]
                        
            node_group.create_dataset("PriorDensity", data=density.PriorDensity)
            node_group.create_dataset("JointDensity", data=density.JointDensity)
            if density.parents is not None:
                node_group.create_dataset("ParentJointDensity", data=density.ParentJointDensity)
            else:
                node_group.create_dataset("ParentJointDensity", data=[])
            node_group.create_dataset("CondDensity", data=density.CondDensity)
            node_group.create_dataset("MarginalDensity", data=density.MarginalDensity)
            
    @property
    def isThereContext(self):
        return any(n is NodeType.Context for n in self.node_type.values())

    @staticmethod
    def estimate_optimal_samples(data, mode=SampleMode.Entropy):
        """
        Allocate samples based on the entropy or variance of each variable.
        
        Parameters:
            data (DataFrame): Data obj.
            mode (Sample mode): 'entropy': Allocate samples based on the entropy of each variable.
                                'variance': Allocate samples based on the variance of each variable.
                                'full': Allocate all samples to each variable.
        
        Returns:
            optimal_samples : dict
                Sample sizes for each variable based on the selected mode.
        """
        optimal_samples = {}
        tolerance = 0.3
        
        for node in data.columns:
            series = data[node].dropna()
            if mode is SampleMode.Entropy:
                _, counts = np.unique(series, return_counts=True)
                prob_dist = counts / len(series)
                series_entropy = entropy(prob_dist) 

                # Estimate sample size based on entropy
                sample_size = int(np.ceil(series_entropy / tolerance ** 2))  
            
            elif mode is SampleMode.Variance:
                variance = np.var(series)  # Calculate the variance of the series
                std_dev = np.sqrt(variance)  # Standard deviation

                # Estimate sample size based on variance (rule of thumb)
                sample_size = int(np.ceil(std_dev ** 2 / tolerance ** 2))  # Adjust based on the desired tolerance
            
            elif mode is SampleMode.Full:
                return {feature: data.shape[0] for feature in data.columns}
                
            sample_size = min(max(sample_size, 10), len(data))
            optimal_samples[node] = sample_size
        
        return optimal_samples
        
        
    def _extract_parents(self, node, data, dag: DAG, nsamples, data_type, node_type):
        """
        Extract the parents of a specified node.

        Args:
            node (str): Node belonging to the dag.

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)].
        """
        parents = {s[0]: Process(data[s[0]].to_numpy(), s[0], s[1], nsamples[s[0]], data_type[s[0]], node_type[s[0]]) 
                   for s in dag.g[node].sources if self.node_type[s[0]] is not NodeType.Context}
        if not parents: return None
        return parents
    
    
    def get_context_specific_data(self, data, context, node, parents):
        context_dict = dict(context)

        # Filter the dataframe based on the context dictionary
        filtered_data = data.d
        for key, value in context_dict.items():
            filtered_data = filtered_data[filtered_data[key] == value]

        system_parents = list(set([node] + [p for p in parents]))
        system_parents = [p for p in system_parents if self.node_type[p] is not NodeType.Context]
        # Check if the filtered data is non-empty (i.e., if the context combination exists)
        return filtered_data[system_parents]
        
        
    def _extract_contexts(self, data, contexts):
        res = {}
        for n, ntype in self.node_type.items():
            if n in contexts and ntype is NodeType.Context:
                res[n] = np.array(np.unique(data.d[n]), dtype=int if self.data_type[n] is DataType.Discrete else float)
        
        # Generate the combinations
        tmp = list(itertools.product(*[[(k, float(v) if self.data_type[k] is DataType.Continuous else int(v)) for v in values] for k, values in res.items()]))
        return [format_combo(c) for c in tmp]
    
    
    def get_lag(self, treatment: str, outcome: str, dag: DAG):
        """
        Output the lag-time associated to the treatment -> outcome link.

        Args:
            treatment (str): treatment variable.
            outcome (str): outcome variable.

        Returns:
            int: treatment -> outcome link's lag-time.
        """
        matching_keys = [key[1] for key in dag.g[outcome].sources.keys() if key[0] == treatment]
        # if multiple, here it is returned only the minimum lag (closest to 0)
        return min(matching_keys)

    
    def get_adjset(self, treatment: str, outcome: str, dag: DAG):
        """
        Output the optimal adjustment set associated to the treatment-outcome intervention.
        
        The adjustment set is calculated through the TIGRAMITE pkg based on [1]
        
        [1] Runge, Jakob. "Necessary and sufficient graphical conditions for optimal adjustment 
            sets in causal graphical models with hidden variables." Advances in Neural Information 
            Processing Systems 34 (2021): 15762-15773.  

        Args:
            treatment (str): treatment variable.
            outcome (str): outcome variable.

        Returns:
            tuple: optimal adjustment set for the treatment -> outcome link.
        """
        lag = self.get_lag(treatment, outcome, dag)
        
        graph = CausalEffects.get_graph_from_dict(dag.get_Adj(indexed=True), tau_max = dag.max_lag)
        opt_adj_set = CausalEffects(graph, graph_type='stationary_dag', 
                                    X = [(dag.features.index(treatment), -abs(lag))], 
                                    Y = [(dag.features.index(outcome), 0)]).get_optimal_set()
        del graph
        return opt_adj_set
    

    def compute_do_density(self, outcome: str, features, dag: DAG):
        """
        Compute the p(outcome|do(treatment)) density.

        Args:
            outcome (str): outcome variable.
        """
        CP.info(f"\n### DO Densities - Outcome {outcome}")
        if self.dbn[outcome].parents is None: return
        for treatment in self.dbn[outcome].parents:
            CP.info(f"- Treatment {treatment}")
                    
            # Select the adjustment set
            adjset = self.get_adjset(treatment, outcome, dag)
                        
            # Compute the adjustment density
            p_adj = np.ones((self.nsample, 1)).squeeze()
            
            for node in adjset: p_adj = p_adj * self.dbn[features[node[0]]].CondDensity # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_adj = normalise(p_adj)
            
            # Compute the p(outcome|treatment,adjustment) density
            p_yxadj = normalise(self.dbn[outcome].CondDensity * self.dbn[treatment].CondDensity * p_adj) # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_xadj = normalise(self.dbn[treatment].CondDensity * p_adj) # TODO: to verify if computed like this is equal to compute the joint density directly through KDE
            p_y_given_xadj = normalise(p_yxadj / p_xadj)
            
            # Compute the p(outcome|do(treatment)) and p(outcome|do(treatment),adjustment)*p(adjustment) densities
            if len(p_y_given_xadj.shape) > 2: 
                # Sum over the adjustment set
                p_y_do_x_adj = normalise(p_y_given_xadj * p_adj)
                p_y_do_x = normalise(np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape))))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
            else:
                p_y_do_x_adj = p_y_given_xadj
                p_y_do_x = p_y_given_xadj
            
            self.dbn[outcome].DO[treatment][ADJ] = np.array(adjset, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ] = np.array(p_y_do_x_adj, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX] = np.array(p_y_do_x, dtype=np.float16)
            
            del adjset, p_y_do_x_adj, p_y_do_x, p_adj, p_yxadj, p_xadj, p_y_given_xadj
            gc.collect()
            
    @staticmethod
    def load(filename: str, dag, data, batch_size, data_type, node_type, id, context = None):
        """Load the DBN object from an HDF5 file."""
        with h5py.File(filename, 'r') as f:          
            
            # Load the DBN group
            group_name = f"DBNs/{id}/{context}" if context is not None else f"DBNs/{id}"
            dbn_group = f.get(group_name)
            if dbn_group is None: raise KeyError(f"{group_name} group not found in the file.")
            nsamples = json.loads(dbn_group.attrs["nsamples"])
            
            dbn = {}
            for node in dbn_group:
                node_group = dbn_group.get(node)
                
                Y = Process(data.d[node].to_numpy(), node, 0, 
                            nsamples[node], 
                            data_type[node], 
                            node_type[node])
                if not node_group.attrs["linked"]:
                    prior = node_group["PriorDensity"][:]
                    joint = node_group["JointDensity"][:]
                    parent_joint_density = node_group["ParentJointDensity"][:]
                    if parent_joint_density.size == 0: parent_joint_density = None
                    parentjoint = node_group["ParentJointDensity"][:]
                    conditional = node_group["CondDensity"][:]
                    marginal = node_group["MarginalDensity"][:]
                else:
                    # Resolve linked node from the other DBN
                    linked_info = json.loads(node_group.attrs["linked_to"])
                    linked_id = ast.literal_eval(linked_info["id"])
                    linked_context = ast.literal_eval(linked_info["context"])
                    
                    # Only load the linked node's densities, not the entire DBN
                    prior, joint, parentjoint, conditional, marginal = DynamicBayesianNetwork.load_density_from_dbn_file(
                        filename, linked_id, linked_context, node
                    )
                    
                dbn[node] = Density(
                        y = Y,
                        batch_size=batch_size,
                        parents=DynamicBayesianNetwork._extract_parents(node, data, dag, nsamples, data_type, node_type), # Adjust as needed if parents are available in the HDF5 file
                        
                        # Load precomputed densities directly
                        prior_density=prior,
                        joint_density=joint,
                        parent_joint_density=parentjoint,
                        cond_density=conditional,
                        marginal_density=marginal
                    )

        # Rebuild the DBN object, assuming dag and data need to be recreated
        return DynamicBayesianNetwork(
            dag=dag,  
            data=data,
            nsample=nsamples,
            data_type=data_type,
            node_type=node_type,
            batch_size=batch_size,
            dbn=dbn
        )
        
    @staticmethod
    def load_density_from_dbn_file(filename, linked_id, linked_context, node):
        """Helper function to load a specific node's density from the DBN file, handling recursive links."""
        with h5py.File(filename, 'r') as f:
            # Define the group name based on the linked ID and context
            group_name = f"DBNs/{linked_id}/{linked_context}"
            dbn_group = f.get(group_name)
            if dbn_group is None:
                raise KeyError(f"{group_name} group not found in the file.")

            # Get the specific node's group
            node_group = dbn_group.get(node)
            if node_group is None:
                raise KeyError(f"Node {node} not found in the file.")

            # Check if the node is linked to another DBN
            if node_group.attrs.get("linked", False):
                # Parse the linked ID and context
                linked_info = json.loads(node_group.attrs["linked_to"])
                new_linked_id = linked_info["id"]
                new_linked_context = linked_info["context"]

                # Recurse with the new linked ID, context, and node
                return DynamicBayesianNetwork.load_density_from_dbn_file(
                    filename, new_linked_id, new_linked_context, node
                )
            else:
                # Load the densities for the specific node
                prior_density = node_group["PriorDensity"][:]
                joint_density = node_group["JointDensity"][:]
                parent_joint_density = node_group["ParentJointDensity"][:]
                cond_density = node_group["CondDensity"][:]
                marginal_density = node_group["MarginalDensity"][:]

        # Return the densities as a tuple or a custom object if necessary
        return (prior_density, joint_density, parent_joint_density, cond_density, marginal_density)
