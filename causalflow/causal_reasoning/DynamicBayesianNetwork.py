import copy
import gc
import itertools
import numpy as np
import pandas as pd
from causalflow.CPrinter import CP
from causalflow.causal_reasoning.DensityUtils import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.DODensity import DODensity, DOType
import causalflow.causal_reasoning.DensityUtils as DensityUtils
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
        self.dag = dag
        self.data = data
        self.data_type = data_type
        self.node_type = node_type
        self.max_components = max_components
        
        self.dbn = {node: None for node in dag.g}
        
        self.compute_density(dag, data, recycle)
        
        
        
    def _get_Y_X(self, data, node, dag):
        Y = Process(data[node].to_numpy(), node, 0, self.data_type[node], self.node_type[node])
        X = {s[0]: Process(data[s[0]].to_numpy(), s[0], s[1], self.data_type[s[0]], self.node_type[s[0]])
            for s in dag.g[node].sources if self.node_type[s[0]] is not NodeType.Context}
        return Y, X
       
        
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
                self.dbn[node] = recycle[node]
            else:
                if self.node_type[node] == NodeType.Context:
                    CP.info(f"\n    ### Target context variable: {node}")
                    Y, _ = self._get_Y_X(data.d, node, dag)
                    self.dbn[node] = {() : None}
                    self.dbn[node][()] = Density(Y, None, max_components=self.max_components)
                else:
                    anchestors = dag.get_anchestors(node)
                    context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                    contexts = self._extract_contexts(data, context_anchestors)
                    self.dbn[node] = {context: None for context in contexts}
                    
                    for context in contexts:
                        # Retrieve data for each node
                        segments = self.get_context_specific_segments(data, context, node, dag.g[node].sourcelist) 
                        if not segments: 
                            CP.info(f"\n    ### Target variable: {node}")
                            if context: CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
                            CP.info(f"    ### No context-specific segments found")
                            continue
                        
                        self.dbn[node][context] = None
                        
                        # Full DBN using all the segments concatenated
                        full_data = pd.concat([segment for segment in segments])
                        Y, X = self._get_Y_X(full_data, node, dag)
                        parents_str = f" - parents {', '.join(list(X.keys()))}" if X else ""
                        CP.info(f"\n    ### Target variable: {node}{parents_str}")
                        if context: CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
                        self.dbn[node][context] = Density(Y, X if X else None, max_components=self.max_components)
                        
                        
                        
                        
# # Nodes
# "WP_t"
# "WP_t_1"
# "TOD_t"
# "TOD_t_1"
# "PD_t"
# "PD_t_1"
# "RV_t"
# "RV_t_1"
# "RB_t"
# "RB_t_1"
# "ELT_t"
# "ELT_t_1"
# "OBS_t"
# "OBS_t_1"
# "CS_t"
# "CS_t_1"

# # Edges
# ("WP_t", "PD_t")
# ("WP_t_1", "PD_t_1")
# ("TOD_t", "PD_t")
# ("TOD_t_1", "PD_t_1")
# ("PD_t_1", "PD_t")
# ("ELT_t_1", "ELT_t")
# ("RB_t", "ELT_t")
# ("WP_t", "ELT_t")
# ("OBS_t", "ELT_t")
# ("RB_t_1", "ELT_t_1")
# ("OBS_t_1", "ELT_t_1")
# ("WP_t_1", "ELT_t_1")
# ("CS_t", "RB_t")
# ("RB_t_1", "RB_t")
# ("RV_t_1", "RB_t")
# ("CS_t_1", "RB_t_1")
# ("RV_t_1", "RV_t")
# ("CS_t", "RV_t")
# ("OBS_t", "RV_t")
# ("CS_t_1", "RV_t_1")
# ("OBS_t_1", "RV_t_1")