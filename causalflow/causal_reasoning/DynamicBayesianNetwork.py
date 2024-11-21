import copy
import numpy as np
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Density_utils import normalise
from causalflow.basics.constants import SampleMode
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density_CPU import Density
from causalflow.causal_reasoning.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.basics.constants import *
from typing import Dict
from sklearn.neighbors import KernelDensity

class DynamicBayesianNetwork():
    def __init__(self, 
                 dag: DAG, 
                 data: Data, 
                 nsample, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType], 
                 batch_size: int,
                 recycle = None):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            nsample (SampleMode/int): if SampleMode - Mode to discovery the number of samples.
                                      if int- Number of samples used for density estimation.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}.
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}.
            batch_size (int): Batch Size.
            recycle ((list[str], DynamicBayesianNetwork)): tuple containing the list of features to recycle and the 
                                                           DBN from which extract their densities. Default None.
        """
        if not isinstance(nsample, SampleMode) and not isinstance(nsample, int):
            raise ValueError("nsample field must be either SampleMode or int!")
        
        if recycle is not None: Fs, DBN = recycle
        
        self.data_type = data_type
        self.node_type = node_type
        if isinstance(nsample, SampleMode):
            self.nsamples = DynamicBayesianNetwork.estimate_optimal_samples(data, nsample)
        else:
            self.nsamples = {f: nsample for f in data.features}
        for k, v in self.nsamples.items(): CP.info(f"    {k} samples : {v}")
        
        self.dbn = {node: None for node in dag.g}
        for node in self.dbn:
            Y = Process(data.d[node].to_numpy(), node, 0, self.nsamples[node], self.data_type[node], self.node_type[node])
            parents = self._extract_parents(node, data, dag)
            if recycle is not None and node in Fs and all(p in Fs for p in parents.keys()):
                CP.info(f"\n    ### Recycling {node} densities")
                self.dbn[node] = copy.deepcopy(DBN.dbn[node])
            else:
                if parents is None:
                    CP.info(f"\n    ### Target variable: {node}")
                else:
                    CP.info(f"\n    ### Target variable: {node} - parents {', '.join(list(parents.keys()))}")
                self.dbn[node] = Density(Y, batch_size, parents)
            
        # for node in self.dbn: self.computeDoDensity(node, data.features, dag)        
        del dag, data, recycle
        
    @staticmethod
    def estimate_entropy(data, bandwidth=0.5):
        """
        Estimate the entropy of each variable using Kernel Density Estimation (KDE).
        
        Args:
            data (ndarray): The data matrix (samples x variables).
            bandwidth (float): The bandwidth parameter for KDE.
        
        Returns:
            entropy (ndarray): Estimated entropy for each variable.
        """
        
        _, n_features = data.shape
        entropy = np.zeros(n_features)
        
        # Fit KDE for each feature (variable)
        for i in range(n_features):
            # Use KDE to estimate the density for the i-th variable
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(data[:, i].reshape(-1, 1))  # Fit KDE to the i-th variable
            # Compute the log likelihood for the density at each point
            log_density = kde.score_samples(data[:, i].reshape(-1, 1))
            
            # Entropy is estimated as the negative average of the log likelihoods
            entropy[i] = -np.mean(log_density)
        
        return entropy

    @staticmethod
    def estimate_optimal_samples(data, mode = SampleMode.Entropy):
        """
        Allocate samples based on the entropy of each variable.
        
        Args:
            data (ndarray): The input data matrix (samples x variables).
            max_samples (int): Maximum number of samples to allocate.
        
        Returns:
            optimal_samples (ndarray): Sample sizes for each variable based on entropy.
        """
        if mode is SampleMode.Entropy:
            entropy = DynamicBayesianNetwork.estimate_entropy(data.d.values)
            proportions = entropy / np.sum(entropy)  # Normalize entropy values            
        
        elif mode is SampleMode.Variance:
            variances = np.var(data.d.values, axis=0)
            total_variance = np.sum(variances)
            proportions = variances / total_variance
            
        # Scale the sample size based on the variance proportion
        optimal_samples = (proportions * data.T).astype(int)
            
        # Ensure at least 10 samples per variable (or some minimum threshold)
        optimal_samples = np.maximum(optimal_samples, 10)
        return {f: s for f, s in zip(*[data.features, optimal_samples])}    
        
        
    def _extract_parents(self, node, data, dag: DAG):
        """
        Extract the parents of a specified node.

        Args:
            node (str): Node belonging to the dag.

        Returns:
            dict: parents express as dict[parent name (str), parent process (Process)].
        """
        parents = {s[0]: Process(data.d[s[0]].to_numpy(), s[0], s[1], self.nsamples[s[0]], self.data_type[s[0]], self.node_type[s[0]]) 
                   for s in dag.g[node].sources}
        if not parents: return None
        return parents
    
    
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
    

    def computeDoDensity(self, outcome: str, features, dag: DAG):
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
        
        