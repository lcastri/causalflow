import copy
import gc
import itertools
import numpy as np
import pandas as pd
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Utils import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.DODensity import DODensity, DOType
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
from typing import Dict
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os


class DynamicBayesianNetwork():
    def __init__(self, 
                 dag: DAG, 
                 data: Data, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType],
                 recycle = None,
                 max_components = 50):
        """
        Class constructor.

        Args:
            dag (DAG): DAG from which deriving the DBN.
            data (Data): Data associated with the DAG.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}.
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}.
        """
        self.data_type = data_type
        self.node_type = node_type
        self.max_components = max_components
        self.dbn = {node: None for node in dag.g}
        self.DO = {node: None for node in dag.g}
        self.data = {node: None for node in dag.g}
        
        self.compute_density(dag, data, recycle)
        # self.compute_do_density(dag, data, recycle)
        
        del dag, data, recycle
        gc.collect()
        
        
    def gmm_density(self, x, means, covariances, weights):
        """
        Computes the Gaussian Mixture Model density.
        
        Args:
            x (np.ndarray): Input points where the density is evaluated.
            means (np.ndarray): Means of the Gaussian components.
            covariances (np.ndarray): Covariances of the Gaussian components.
            weights (np.ndarray): Weights of the Gaussian components.
        
        Returns:
            np.ndarray: Evaluated density at each point in x.
        """
        density = np.zeros_like(x)
        for mean, cov, weight in zip(means, covariances, weights):
            density += weight * multivariate_normal.pdf(x, mean=mean, cov=cov)
            
        return density
        
        
    def plot_density(self, node, context):
        # Extract ground-truth data
        ground_truth_data = self.data[node][context]['full'].d[node]

        # Extract estimated density (assumes Density object has a method to generate density values)
        estimated_density = self.dbn[node][context]['full'].PriorDensity

        # Generate X (independent variable range)
        x_range = np.linspace(ground_truth_data.min(), ground_truth_data.max(), 100)  # Adjust as needed

        # Evaluate densities
        ground_truth_density = np.histogram(ground_truth_data, bins=30, density=True)  # Histogram approximation
        estimated_density_values = self.gmm_density(x_range, 
                                                    estimated_density['means'], 
                                                    estimated_density['covariances'], 
                                                    estimated_density['weights'])
    
        # Plot ground-truth density
        plt.plot(ground_truth_density[1][:-1], ground_truth_density[0], label='Ground Truth', linestyle='-', color='orange')

        # Plot estimated density
        plt.plot(x_range, estimated_density_values, label='Estimated', linestyle='--', color='blue')

        # Add labels, legend, and title
        plt.ylabel("Density")
        plt.title(f"Density Comparison for Node: {node} in Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
        plt.legend()
        plt.grid(True)

        # Define the output folder and filename
        output_folder = "/home/lcastri/Desktop/Densities"
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
        filename = f"{node}__{','.join([f'{c[0]}={c[1]}' for c in context])}.png"
        filepath = os.path.join(output_folder, filename)

        # Save the plot as a PNG image
        plt.savefig(filepath, format='png')
        plt.close()  # Close the plot to free memory

        
        
    def _get_Y_X(self, data, node, dag):
        Y = Process(data[node].to_numpy(), node, 0, self.data_type[node], self.node_type[node])
        X = {s[0]: Process(data[s[0]].to_numpy(), s[0], s[1], self.data_type[s[0]], self.node_type[s[0]])
            for s in dag.g[node].sources if self.node_type[s[0]] is not NodeType.Context}
        return Y, X
    
    
    def _get_Y_X_ADJ(self, data, outcome, treatment, cond = None, adj = None):
        Y = Process(data[outcome].to_numpy(), outcome, 0, self.data_type[outcome], self.node_type[outcome])
        X = Process(data[treatment[0]].to_numpy(), treatment[0], abs(treatment[1]), self.data_type[treatment[0]], self.node_type[treatment[0]])
        COND = {s: Process(data[s[0]].to_numpy(), s[0], abs(s[1]), self.data_type[s[0]], self.node_type[s[0]])
               for s in cond} if cond is not None else None
        ADJ = {s: Process(data[s[0]].to_numpy(), s[0], abs(s[1]), self.data_type[s[0]], self.node_type[s[0]])
               for s in adj} if adj is not None else None
        return Y, X, COND, ADJ
    
        
    def combine_segment_densities(self, segment_densities, segment_sizes):
        """
        Combine densities from multiple segments into a single density.

        Args:
            segment_densities (list): List of GMM parameters (means, covariances, weights) for each segment.
            segment_sizes (list): List of sizes (number of points) for each segment.

        Returns:
            dict: Combined GMM parameters.
        """
        total_points = sum(segment_sizes)
        combined_means = []
        combined_covariances = []
        combined_weights = []

        for segment_density, size in zip(segment_densities, segment_sizes):
            weight_factor = size / total_points
            for k in range(len(segment_density["weights"])):
                combined_means.append(segment_density["means"][k])
                combined_covariances.append(segment_density["covariances"][k])
                combined_weights.append(segment_density["weights"][k] * weight_factor)

        return {
            "means": np.array(combined_means),
            "covariances": np.array(combined_covariances),
            "weights": np.array(combined_weights)
        }
        
        
    def _extract_contexts(self, data, contexts):
        res = {}
        for n, ntype in self.node_type.items():
            if n in contexts and ntype is NodeType.Context:
                res[n] = np.array(np.unique(data.d[n]), dtype=int if self.data_type[n] is DataType.Discrete else float)
        
        # Generate the combinations
        tmp = list(itertools.product(*[[(k, float(v) if self.data_type[k] is DataType.Continuous else int(v)) for v in values] for k, values in res.items()]))
        return [format_combo(c) for c in tmp]
    
       
    def split_by_context(self, data, target_context_conditions):
        """
        Split the data into separate datasets based on continuous segments 
        that match all the specified context conditions.

        Args:
            data (pd.DataFrame): The input dataset.
            target_context_conditions (dict): Dictionary of context conditions 
                                            (e.g., {'context1': 'a', 'context2': 1}).

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each representing a continuous segment 
                                matching all the context conditions.
        """
        def isDuplicate():
            for existing_segment in segments:
                if len(current_segment_df) == len(existing_segment):  # Check lengths first
                    if (current_segment_df.columns == existing_segment.columns).all():
                        if all((current_segment_df[col].values == existing_segment[col].values).all()
                            for col in current_segment_df.columns):
                            return True
            return False

        # Ensure that all rows satisfy the context conditions
        filtered_data = data.copy()
        for context_column, target_context in target_context_conditions.items():
            filtered_data = filtered_data[filtered_data[context_column] == target_context]

        segments = []
        current_segment = []

        for i in range(len(filtered_data)):
            if i == 0 or filtered_data.index[i] == filtered_data.index[i - 1] + 1:
                current_segment.append(filtered_data.iloc[i])
            else:
                if current_segment:
                    current_segment_df = pd.DataFrame(current_segment)
                    if not isDuplicate():
                        segments.append(current_segment_df)
                    current_segment = [filtered_data.iloc[i]]

        # Add the last segment if it exists
        if current_segment:
            current_segment_df = pd.DataFrame(current_segment)
            if not isDuplicate():
                segments.append(current_segment_df)

        return segments


    def get_context_specific_segments(self, data, context, node, parents):
        """
        Extract continuous segments of data that match the specified context combination.

        Args:
            data (DataFrame): The dataset.
            context (dict): A dictionary specifying the desired context conditions.
            node (str): The target node for which segments are extracted.
            parents (list): List of parent nodes.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each representing a continuous segment
                                matching the specified context combination.
        """
        context_dict = dict(context)

        # Filter the dataframe to keep only relevant columns
        system_parents = list(set([node] + [p for p in parents]))
        system_parents = [p for p in system_parents if self.node_type[p] is not NodeType.Context]
        filtered_data = data.d[system_parents + list(context_dict.keys())]

        # Get segments that match the intersection of all context conditions
        segments = self.split_by_context(filtered_data, context_dict)
        return [segment[system_parents] for segment in segments]
                    
       
    def compute_density(self, dag: DAG, data: Data, recycle = None):
        for node in dag.g:
            if recycle is not None and node in recycle:
                self.dbn[node] = recycle[node]['dbn']
                self.data[node] = recycle[node]['data']
            else:
                if self.node_type[node] == NodeType.Context:
                    CP.info(f"\n    ### Target context variable: {node}")
                    Y, _ = self._get_Y_X(data.d, node, dag)
                    self.dbn[node] = Density(Y, None, max_components=self.max_components)
                else:
                    anchestors = dag.get_anchestors(node)
                    context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                    contexts = self._extract_contexts(data, context_anchestors)
                    self.dbn[node] = {context: None for context in contexts}
                    self.data[node] = {context: None for context in contexts}
                    
                    for context in contexts:
                        # Retrieve data for each node
                        segments = self.get_context_specific_segments(data, context, node, dag.g[node].sourcelist) 
                        if not segments: 
                            CP.info(f"\n    ### Target variable: {node}")
                            if context: CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
                            CP.info(f"    ### No context-specific segments found")
                            continue
                        self.dbn[node][context] = {idx: None for idx, _ in enumerate(segments)}
                        self.data[node][context] = {idx: None for idx, _ in enumerate(segments)}
                        
                        # Full DBN using all the segments concatenated
                        full_data = pd.concat([segment for segment in segments])
                        Y, X = self._get_Y_X(full_data, node, dag)
                        parents_str = f" - parents {', '.join(list(X.keys()))}" if X else ""
                        CP.info(f"\n    ### Target variable: {node}{parents_str}")
                        if context: CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
                        CP.info(f"    ### Full - {len(full_data)} samples")
                        self.dbn[node][context]['full'] = Density(Y, X if X else None, max_components=self.max_components)
                        self.data[node][context]['full'] = Data(full_data)
                        
                        
    def compute_do_density(self, dag: DAG, data: Data, recycle = None):
        """
        Compute the p(outcome|do(treatment)) density for all treatment-outcome combinations.
        Adjust for variables that block backdoor paths.
        
        Args:
            dag (DAG): Directed acyclic graph representing the causal relationships.
        """
        CP.info("\n## DO Densities Computation")
        
        all_nodes = [(t, -l) for t in dag.get_Adj().keys() for l in range(0, dag.max_lag + 1)]
        self.DO = {outcome[0]: {treatment: {} for treatment in all_nodes if treatment != outcome} for outcome in all_nodes if outcome[1] == 0}
        for outcome in all_nodes:
            
            if outcome[1] != 0: continue
            
            if recycle is not None and outcome[0] in recycle:
                self.DO[outcome[0]] = recycle[outcome[0]]['do']
                
            else:
                for treatment in all_nodes:
                    
                    # No self-loops
                    if treatment == outcome: continue
                    
                    # No treatment before outcome
                    if treatment[1] > outcome[1]: continue
                    
                    CP.info(f"\n### p({outcome}|do({treatment}))")
                    
                    # Treatment not affecting outcome
                    if (treatment[0], abs(treatment[1])) not in dag.get_anchestors(outcome[0], include_lag=True):
                        CP.info(f"- {treatment} not an anchestor of {outcome}")
                        CP.info(f"- p({outcome}|do({treatment})) = p({outcome})")
                        Y, X, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment)
                        self.DO[outcome[0]][treatment][frozenset()] = DODensity(Y, X, adjustments = None, 
                                                                                doType = DOType.pY, 
                                                                                max_components=self.max_components)
                        continue
                                
                    open_backdoor_paths = dag.get_open_backdoors_paths(treatment, outcome)
                    self.DO[outcome[0]][treatment]['backdoor_paths'] = open_backdoor_paths
                    
                    # Treatment affects outcome (no open Backdoor Paths)
                    if not open_backdoor_paths: 
                        CP.info(f"- No adjustment needed for {treatment} -> {outcome}")
                        CP.info(f"- p({outcome}|do({treatment})) = p({outcome}|{treatment})")
                        Y, X, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment)
                        self.DO[outcome[0]][treatment][frozenset()] = DODensity(Y, X, adjustments = None, 
                                                                                doType = DOType.pY_given_X, 
                                                                                max_components=self.max_components)
                        continue
                    
                    # Treatment affects outcome (Open Backdoor Paths)
                    else:
                        pY = None
                        pY_X = None
                        
                        adjustment_sets = dag.find_all_d_separators(treatment, outcome, open_backdoor_paths)    
                        CP.info(f"- Adjustment needed for {treatment} -> {outcome}")
                        CP.info(f"- Backdoor paths: {open_backdoor_paths}")
                        
                        
                        self.DO[outcome[0]][treatment]['backdoor_paths'] = open_backdoor_paths
                        for i, adj_set in enumerate(adjustment_sets):
                            
                            adj_set_key = frozenset(adj_set)
                            CP.info(f"    #### Adjustment set: {adj_set}")
                            CP.info(f"    - p({outcome}|do({treatment})) = sum[p({outcome}|{treatment},{','.join([str(s) for s in adj_set])})*p({','.join([str(s) for s in adj_set])})]")
                            
                            
                            Y, X, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment, adj = adj_set)
                            self.DO[outcome[0]][treatment][adj_set_key] = DODensity(Y, X, adjustments = ADJ, 
                                                                                    doType = DOType.pY_given_X_Adj, 
                                                                                    max_components=self.max_components,
                                                                                    pY=pY, pY_X=pY_X)
                            if i == 0:
                                pY = self.DO[outcome[0]][treatment][adj_set_key].pY
                                pY_X = self.DO[outcome[0]][treatment][adj_set_key].pY_X
                    gc.collect()
                    
                    
    def compute_single_do_density(self, dag: DAG, data: Data, outcome: str, treatment: tuple, conditions: list = None):
        """
        Compute the p(outcome|do(treatment)) density for all treatment-outcome combinations.
        Adjust for variables that block backdoor paths.
        
        Args:
            dag (DAG): Directed acyclic graph representing the causal relationships.
        """
        CP.info("\n## DO Densities Computation")
        outcome = (outcome, 0)
        if conditions is not None: 
            conditions = [conditions] if not isinstance(conditions, list) else conditions
            conditions_str = ','.join([str(c) for c in conditions])
                              
        all_nodes = [(t, -l) for t in dag.get_Adj().keys() for l in range(0, dag.max_lag + 1)]
        self.DO = {outcome[0]: {treatment: {} for treatment in all_nodes if treatment != outcome} for outcome in all_nodes if outcome[1] == 0}
            
        #! No self-loops
        if treatment == outcome:
            CP.info(f"- {treatment} is the same as {outcome}")
            return
                    
        #! No treatment before outcome
        if treatment[1] > outcome[1]: 
            CP.info(f"- {treatment} is after {outcome}")
            return
                    
        CP.info(f"\n### p({outcome}|do({treatment}))")
                    
        # Treatment not affecting outcome
        if (treatment[0], abs(treatment[1])) not in dag.get_anchestors(outcome[0], include_lag=True):
            CP.info(f"- {treatment} not an anchestor of {outcome}")
            CP.info(f"- p({outcome}|do({treatment})) = p({outcome})")
            Y, X, COND, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment)
            self.DO[outcome[0]][treatment][frozenset()] = DODensity(Y, X, 
                                                                    adjustments = None,
                                                                    conditions = None, # FIXME: this should be different 
                                                                    doType = DOType.pY, 
                                                                    max_components=self.max_components)
            return
                                
        open_backdoor_paths = dag.get_open_backdoors_paths(treatment, outcome, conditions)
        self.DO[outcome[0]][treatment]['backdoor_paths'] = open_backdoor_paths
                    
        #! Treatment affects outcome (no open Backdoor Paths)
        if not open_backdoor_paths:
            if conditions is not None:
                CP.info(f"- No adjustment needed for {treatment} -> {outcome} conditioning on {conditions_str}")
                CP.info(f"- p({outcome}|do({treatment},{conditions_str})) = p({outcome}|{treatment},{conditions_str})")
            else:
                CP.info(f"- No adjustment needed for {treatment} -> {outcome}")
                CP.info(f"- p({outcome}|do({treatment})) = p({outcome}|{treatment})")
                
            Y, X, COND, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment)
            self.DO[outcome[0]][treatment][frozenset()] = DODensity(Y, X, 
                                                                    adjustments = None, 
                                                                    conditions = None, # FIXME: this should be different
                                                                    doType = DOType.pY_given_X, 
                                                                    max_components=self.max_components)
            return
                    
        #! Treatment affects outcome (Open Backdoor Paths)
        else:
            pY = None
            pY_X = None
            
            adjustment_sets = dag.find_all_d_separators(treatment, outcome, open_backdoor_paths, conditions)    
            if conditions is not None:
                CP.info(f"- Adjustment needed for {treatment} -> {outcome} conditioning on {conditions_str}")
            else:
                CP.info(f"- Adjustment needed for {treatment} -> {outcome}")
            CP.info(f"- Backdoor paths: {open_backdoor_paths}")
            if conditions is not None: CP.info(f"- Conditioning set: {conditions}")
            
            self.DO[outcome[0]][treatment]['backdoor_paths'] = open_backdoor_paths
            for i, adj_set in enumerate(adjustment_sets):
                
                adj_set_key = frozenset(adj_set)
                CP.info(f"    #### Adjustment set: {adj_set}")
                if conditions is not None:
                    CP.info(f"    - p({outcome}|do({treatment}),{conditions_str}) = sum[p({outcome}|{treatment},{conditions_str},{','.join([str(s) for s in adj_set])})*p({','.join([str(s) for s in adj_set])})]")
                else:
                    CP.info(f"    - p({outcome}|do({treatment})) = sum[p({outcome}|{treatment},{','.join([str(s) for s in adj_set])})*p({','.join([str(s) for s in adj_set])})]")
                
                
                Y, X, COND, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment, cond = conditions, adj = adj_set)
                self.DO[outcome[0]][treatment][adj_set_key] = DODensity(Y, X, 
                                                                        adjustments = ADJ,
                                                                        conditions = COND, 
                                                                        doType = DOType.pY_given_X_Adj if conditions is None else DOType.pY_given_X_Cond_Adj, 
                                                                        max_components=self.max_components,
                                                                        pY=pY, pY_X=pY_X)
                if i == 0:
                    pY = self.DO[outcome[0]][treatment][adj_set_key].pY
                    pY_X = self.DO[outcome[0]][treatment][adj_set_key].pY_X



    # def compute_do_density(self, dag: DAG, data: Data, recycle=None):
    #     """
    #     Compute the p(outcome | do(treatment)) density for all treatment-outcome combinations, 
    #     optionally conditioning on context variables.

    #     Args:
    #         dag (DAG): Directed acyclic graph representing the causal relationships.
    #         data (Data): Observational data.
    #         recycle (dict, optional): Reuse previously computed densities.
    #     """
    #     CP.info("\n## DO Densities Computation")

    #     all_nodes = [(t, -l) for t in dag.get_Adj().keys() for l in range(0, dag.max_lag + 1)]
    #     self.DO = {outcome[0]: {treatment: {} for treatment in all_nodes if treatment != outcome} for outcome in all_nodes if outcome[1] == 0}

    #     for outcome in all_nodes:
    #         if self.node_type[outcome[0]] == NodeType.Context: continue
    #         if outcome[1] != 0: 
    #             continue
            
    #         # Handle context-based segmentation
    #         anchestors = dag.get_anchestors(outcome[0])
    #         context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
    #         contexts = self._extract_contexts(data, context_anchestors)

    #         for context in contexts:
    #             # Filter data by the current context
    #             segments = self.get_context_specific_segments(data, context, outcome[0], dag.g[outcome[0]].sourcelist)
    #             if not segments:
    #                 CP.info(f"\n    ### No data for context: {context}")
    #                 continue

    #             # Full data for context
    #             full_data = pd.concat([segment for segment in segments])

    #             for treatment in all_nodes:
    #                 # Standard checks
    #                 if treatment == outcome: continue
    #                 if treatment[1] > outcome[1]: continue

    #                 CP.info(f"\n\t### p({outcome} | do({treatment}))")
    #                 CP.info(f"\t### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")      
                                  
    #                 # Standard Do-Density logic as before, using full_data for this context
    #                 if (treatment[0], abs(treatment[1])) not in dag.get_anchestors(outcome[0], include_lag=True):
    #                     Y, X, ADJ = self._get_Y_X_ADJ(full_data, outcome[0], treatment)
    #                     self.DO[outcome[0]][treatment][frozenset()] = DODensity(Y, X, adjustments=None, 
    #                                                                             doType=DOType.pY, 
    #                                                                             max_components=self.max_components)
    #                     continue
                                    
    #                 open_backdoor_paths = dag.get_open_backdoors_paths(treatment, outcome)
    #                 if not open_backdoor_paths:
    #                     Y, X, ADJ = self._get_Y_X_ADJ(full_data, outcome[0], treatment)
    #                     self.DO[outcome[0]][treatment][frozenset()] = DODensity(Y, X, adjustments=None, 
    #                                                                             doType=DOType.pY_given_X, 
    #                                                                             max_components=self.max_components)
    #                 else:
    #                     adjustment_sets = dag.find_all_d_separators(treatment, outcome, open_backdoor_paths)
    #                     for adj_set in adjustment_sets:
    #                         adj_set_key = frozenset(adj_set)
    #                         Y, X, ADJ = self._get_Y_X_ADJ(full_data, outcome[0], treatment, adj_set)
    #                         self.DO[outcome[0]][treatment][adj_set_key] = DODensity(Y, X, adjustments=ADJ, 
    #                                                                                 doType=DOType.pY_given_X_Adj, 
    #                                                                                 max_components=self.max_components)
