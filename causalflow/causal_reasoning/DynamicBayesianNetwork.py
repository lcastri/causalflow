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
from causalflow.causal_reasoning.Process import Process
from tigramite.causal_effects import CausalEffects
from causalflow.basics.constants import *
from typing import Dict

class DynamicBayesianNetwork():
    def __init__(self, 
                 dag: DAG, 
                 data: Data, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType]):
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
        
        self.dbn = {node: None for node in dag.g}
        self.data = {node: None for node in dag.g}
        for node in dag.g:
            if self.node_type[node] == NodeType.Context: continue
            anchestors = dag.get_node_anchestors(node)
            context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
            contexts = self._extract_contexts(data, context_anchestors)
            self.dbn[node] = {context: None for context in contexts}
            self.data[node] = {context: None for context in contexts}
            
            for context in contexts:
                # Retrieve data for each node
                segments = self.get_context_specific_segments(data, context, node, dag.g[node].sourcelist) 
                self.dbn[node][context] = {idx: None for idx, _ in enumerate(segments)}
                self.data[node][context] = {idx: None for idx, _ in enumerate(segments)}
                
                for idx, segment in enumerate(segments):
                    # Target Y process
                    Y = Process(segment[node].to_numpy(), node, 0, self.data_type[node], self.node_type[node])
                                                    
                    # Parent(s) X process
                    X = {s[0]: Process(segment[s[0]].to_numpy(), s[0], s[1], self.data_type[s[0]], self.node_type[s[0]])
                        for s in dag.g[node].sources if self.node_type[s[0]] is not NodeType.Context}
                    
                    # Density params estimation
                    parents_str = f" - parents {', '.join(list(X.keys()))}" if X else ""
                    CP.info(f"\n    ### Target variable: {node}{parents_str}")
                    CP.info(f"    ### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])} -- ### Segment: {idx + 1}/{len(segments)}")

                    self.dbn[node][context][idx] = Density(Y, X if X else None)
                    self.data[node][context][idx] = Data(segment)
                self.dbn[node][context]['combined'] = copy.deepcopy(self.dbn[node][context][0])
                self.dbn[node][context]['combined'].PriorDensity = self.combine_segment_densities([self.dbn[node][context][idx].PriorDensity for idx in range(len(segments))], [len(segment) for segment in segments])
                self.dbn[node][context]['combined'].JointDensity = self.combine_segment_densities([self.dbn[node][context][idx].JointDensity for idx in range(len(segments))], [len(segment) for segment in segments])
                self.dbn[node][context]['combined'].ParentJointDensity = self.combine_segment_densities([self.dbn[node][context][idx].ParentJointDensity for idx in range(len(segments))], [len(segment) for segment in segments])
                
        del dag, data
        gc.collect()
        
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
    
    
    # def split_by_context(self, data, context_column, target_context):
    #     """
    #     Split the data into separate datasets based on continuous segments of the specified context.

    #     Args:
    #         data (pd.DataFrame): The input dataset containing a context column.
    #         context_column (str): The name of the context column.
    #         target_context (any): The target context value to extract.

    #     Returns:
    #         List[pd.DataFrame]: A list of DataFrames, each representing a continuous segment of the target context.
    #     """
        
    #     def isDuplicate():
    #         for existing_segment in segments:
    #             if len(current_segment_df) == len(existing_segment):  # Check lengths first
    #                 if (current_segment_df.columns == existing_segment.columns).all():
    #                     if all((current_segment_df[col].values == existing_segment[col].values).all()
    #                         for col in current_segment_df.columns):
    #                         return True
    #         return False
        
    #     segments = []
    #     current_segment = []

    #     for i in range(len(data)):
    #         if data[context_column].iloc[i] == target_context:
    #             current_segment.append(data.iloc[i])
    #         else:
    #             if current_segment:
    #                 current_segment_df = pd.DataFrame(current_segment)
    #                 if not isDuplicate():
    #                     segments.append(current_segment_df)
    #                 current_segment = []

    #     # Add the last segment if it exists
    #     if current_segment:
    #         current_segment_df = pd.DataFrame(current_segment)
    #         if not isDuplicate():
    #             segments.append(pd.DataFrame(current_segment))
        
    #     return segments

    
    
    # def get_context_specific_segments(self, data, context, node, parents):
    #     context_dict = dict(context)

    #     # Filter the dataframe based on the context dictionary
    #     filtered_data = data.d[list(set([node] + [p for p in parents]))]
    #     for key, value in context_dict.items():
    #         segments = self.split_by_context(filtered_data, key, value)

    #     system_parents = list(set([node] + [p for p in parents]))
    #     system_parents = [p for p in system_parents if self.node_type[p] is not NodeType.Context]
    #     # Check if the filtered data is non-empty (i.e., if the context combination exists)
    #     return filtered_data[system_parents]
    
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
        lag = min([key[1] for key in dag.g[outcome].sources.keys() if key[0] == treatment])
        
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
                
            # Compute the p(adjustment) density            
            for node in adjset: p_adj = p_adj * self.dbn[features[node[0]]].CondDensity
            
            # Compute the p(outcome|treatment,adjustment) density
            p_yxadj = self.dbn[outcome].CondDensity * self.dbn[treatment].CondDensity * p_adj
            p_xadj = self.dbn[treatment].CondDensity * p_adj
            p_y_given_xadj = p_yxadj / p_xadj
            
            # Compute the p(outcome|do(treatment)) and p(outcome|do(treatment),adjustment)*p(adjustment) densities
            if len(p_y_given_xadj.shape) > 2: 
                # Sum over the adjustment set
                p_y_do_x_adj = p_y_given_xadj * p_adj
                p_y_do_x = normalise(np.sum(p_y_given_xadj * p_adj, axis=tuple(range(2, len(p_y_given_xadj.shape))))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
            else:
                p_y_do_x_adj = p_y_given_xadj
                p_y_do_x = p_y_given_xadj
            
            self.dbn[outcome].DO[treatment][ADJ] = np.array(adjset, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ] = np.array(p_y_do_x_adj, dtype=np.float16)
            self.dbn[outcome].DO[treatment][P_Y_GIVEN_DOX] = np.array(p_y_do_x, dtype=np.float16)
            
            del adjset, p_y_do_x_adj, p_y_do_x, p_adj, p_yxadj, p_xadj, p_y_given_xadj
            gc.collect()