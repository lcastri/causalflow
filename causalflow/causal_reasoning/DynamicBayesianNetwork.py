from causalflow.CPrinter import CP
from causalflow.causal_reasoning.Utils import *
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.DODensity import DODensity, DOType
from causalflow.causal_reasoning.Process import Process
from causalflow.basics.constants import *
from typing import Dict
import pyAgrum as gum

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
        
        nodes = [(f, -abs(l)) for f in self.dag.features for l in range(self.dag.max_lag + 1)]
        value_ranges = {var: self.data.d[var[0]].unique() for var in nodes}

        self.DBN = {node: None for node in nodes}

        self.pyAgrum_dbn = gum.BayesNet('MyBN')
        bn_pgmpy = DAG.get_DBN(self.dag.get_Adj(), self.dag.max_lag)

        # Create the DBN structure
        for node in bn_pgmpy.nodes:
            self.pyAgrum_dbn.add(gum.LabelizedVariable(str(node), str(node), len(value_ranges[node])))

        # Add edges based on the DAG
        for edge in list(bn_pgmpy.edges):
            self.pyAgrum_dbn.addArc(str(edge[0]), str(edge[1]))
            
        self.compute_density(recycle)

        # Populate CPTs with discretized GMM probabilities
        for node in nodes:
            if self.DBN[node] is None: continue
            
            # Generate histogram and bin probabilities from the GMM
            bin_probs = self.generate_histogram_from_gmm(node, value_ranges[node])
            
            # Populate the CPT
            cpt = self.pyAgrum_dbn.cpt(str(node))
            cpt.fillWith(bin_probs)
            
            
    def generate_histogram_from_gmm(self, node, value_range):
        """
        Generate histogram bin probabilities from a GMM.

        Args:
            node (tuple): The node (variable, lag).
            value_range (tuple): The (min, max) range of the variable.

        Returns:
            np.array: The probability of each bin.
        """
        gmm = self.DBN[node].pJoint
        min_val, max_val = min(value_range), max(value_range)
        bin_edges = sorted(value_range)
        bin_probs = np.zeros(len(value_range))

        # Compute the probability of each bin using the GMM
        for i in range(len(value_range)):
            # Integrate the GMM over the bin range
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_probs[i] = self.integrate_gmm(gmm, bin_start, bin_end)

        # Normalize the probabilities
        bin_probs /= bin_probs.sum()
        return bin_probs
    
    
    def integrate_gmm(self, gmm, start, end, num_points=1000):
        """
        Integrate the GMM over a range [start, end].

        Args:
            gmm (GaussianMixture): The GMM to integrate.
            start (float): Start of the range.
            end (float): End of the range.
            num_points (int): Number of points for numerical integration.

        Returns:
            float: The integrated probability.
        """
        x = np.linspace(start, end, num_points)
        pdf_values = np.exp(gmm.score_samples(x.reshape(-1, 1)))
        return np.trapz(pdf_values, x)
        

        
        
    def _get_Y_X(self, data, node, dag):
        f, lag = node
        Y = Process(data[f].to_numpy(), f, abs(lag), self.data_type[f], self.node_type[f])
        X = {}
        for s in dag.g[f].sources:
            if s[1] + abs(lag) > dag.max_lag: continue
            if s[0] == 'WP': continue
            X[s[0]] = Process(data[s[0]].to_numpy(), s[0], s[1] + abs(lag), self.data_type[s[0]], self.node_type[s[0]])
        # X = {s[0]: Process(data[s[0]].to_numpy(), s[0], s[1], self.data_type[s[0]], self.node_type[s[0]])
        #     for s in dag.g[node].sources}
        return Y, X
    
    
    def _get_Y_X_ADJ(self, data, outcome, treatment, cond = None, adj = None):
        Y = Process(data[outcome].to_numpy(), outcome, 0, self.data_type[outcome], self.node_type[outcome])
        X = Process(data[treatment[0]].to_numpy(), treatment[0], abs(treatment[1]), self.data_type[treatment[0]], self.node_type[treatment[0]])
        COND = {s: Process(data[s[0]].to_numpy(), s[0], abs(s[1]), self.data_type[s[0]], self.node_type[s[0]])
               for s in cond} if cond is not None else None
        ADJ = {s: Process(data[s[0]].to_numpy(), s[0], abs(s[1]), self.data_type[s[0]], self.node_type[s[0]])
               for s in adj} if adj is not None else None
        return Y, X, COND, ADJ
                    
       
    def compute_density(self, recycle = None):
        for node in self.DBN.keys():
            if node[0] == 'WP': continue
            # Y and Y's parents retrieved from data                                             
            Y, X = self._get_Y_X(self.data.d, node, self.dag)
            parents_str = []
            for x in X.keys():
                parents_str.append(f"{X[x].varname}_t{-abs(X[x].lag) if X[x].lag != 0 else ''}")
            parents_str = f" {', '.join(parents_str)}" if len(parents_str) > 0 else ''
            CP.info(f"\n    ### Variable: {node[0]}_t{-abs(node[1]) if node[1] != 0 else ''}{f' -- Parent(s): {parents_str}' if X else ''}")
            self.DBN[node] = Density(Y, X if X else None, max_components=self.max_components)
                        
                                            
    def compute_single_do_density(self, dag: DAG, data: Data, outcome: str, treatment: tuple, conditions: list = None, max_adj_size = 2):
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
                              
        #! No self-loops
        if treatment == outcome:
            CP.info(f"- {treatment} is the same as {outcome}")
            return
                    
        #! No treatment before outcome
        if treatment[1] > outcome[1]: 
            CP.info(f"- {treatment} is after {outcome}")
            return
                    
        CP.info(f"\n### p({outcome}|do({treatment}))")
                    
        #! Treatment not affecting outcome
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
            self.DO[outcome[0]][treatment][frozenset()] = self.dbn[outcome[0]]
                
            return
                    
        #! Treatment affects outcome (Open Backdoor Paths)
        else:
            pY = None
            pY_X = None
            
            adjustment_sets = dag.find_all_d_separators(treatment, outcome, open_backdoor_paths, conditions, max_adj_size = max_adj_size)
            adjustment_sets = [{('OBS', -1)}] # FIXME: remove me this is a hack
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
                    
                    
    def compute_single_bayes_density(self, dag: DAG, data: Data, outcome: str, treatment: tuple, conditions: list = None):
        if not hasattr(self, 'Bayes'):
            all_nodes = [(t, -l) for t in dag.get_Adj().keys() for l in range(0, dag.max_lag + 1)]
            self.Bayes = {outcome[0]: {treatment: {} for treatment in all_nodes if treatment != outcome} for outcome in all_nodes if outcome[1] == 0}
        
        CP.info("\n## Bayes Densities Computation")
        outcome = (outcome, 0)
        if conditions is not None: 
            conditions = [conditions] if not isinstance(conditions, list) else conditions
            conditions_str = ','.join([str(c) for c in conditions])
                              
        #! No self-loops
        if treatment == outcome:
            CP.info(f"- {treatment} is the same as {outcome}")
            return
                    
        #! No treatment before outcome
        if treatment[1] > outcome[1]: 
            CP.info(f"- {treatment} is after {outcome}")
            return
                    
        CP.info(f"\n### p({outcome}|do({treatment}))")
                    
        #! Treatment not affecting outcome
        if (treatment[0], abs(treatment[1])) not in dag.get_anchestors(outcome[0], include_lag=True):
            CP.info(f"- {treatment} not an anchestor of {outcome}")
            CP.info(f"- p({outcome}|do({treatment})) = p({outcome})")
            Y, X, COND, ADJ = self._get_Y_X_ADJ(data.d, outcome[0], treatment)
            self.Bayes[outcome[0]][treatment] = DODensity(Y, X, 
                                                                    adjustments = None,
                                                                    conditions = None, # FIXME: this should be different 
                                                                    doType = DOType.pY, 
                                                                    max_components=self.max_components)
            return
                                
        open_backdoor_paths = dag.get_open_backdoors_paths(treatment, outcome, conditions)                    
        #! Treatment affects outcome (no open Backdoor Paths)
        if not open_backdoor_paths:
            self.Bayes[outcome[0]][treatment] = self.dbn[outcome[0]]
            return
                    
        #! Treatment affects outcome (Open Backdoor Paths)
        else:
            if conditions is not None:
                CP.info(f"    - p({outcome}|{treatment},{conditions_str})")
            else:
                CP.info(f"    - p({outcome}|{treatment})")
            if conditions is not None: CP.info(f"    - Conditioning set: {conditions}")
                
            Y, X, COND, _ = self._get_Y_X_ADJ(data.d, outcome[0], treatment, cond = conditions)
            self.Bayes[outcome[0]][treatment] = DODensity(Y, X, 
                                                          adjustments = None,
                                                          conditions = COND, 
                                                          doType = DOType.pY_given_X if conditions is None else DOType.pY_given_X_Cond, 
                                                          max_components=self.max_components)
    
    # def compute_single_do_density(self, dag: DAG, data: Data, outcome: str, treatment: tuple, conditions: list = None, max_adj_size = 2):
    #     """
    #     Compute the p(outcome|do(treatment)) density for all treatment-outcome combinations.
    #     Adjust for variables that block backdoor paths.
        
    #     Args:
    #         dag (DAG): Directed acyclic graph representing the causal relationships.
    #     """
    #     CP.info("\n## DO Densities Computation")
    #     outcome = (outcome, 0)
    #     if conditions is not None: 
    #         conditions = [conditions] if not isinstance(conditions, list) else conditions
    #         conditions_str = ','.join([str(c) for c in conditions])
                              
    #     #! Case 1. No self-loops
    #     if treatment == outcome:
    #         CP.info(f"- {treatment} is the same as {outcome}")
    #         return
                    
    #     #! Case 2. No treatment before outcome
    #     if treatment[1] > outcome[1]: 
    #         CP.info(f"- {treatment} is after {outcome}")
    #         return
                    
    #     CP.info(f"\n### p({outcome}|do({treatment}))")
                    
    #     #! Case 3. Treatment not affecting outcome
    #     if (treatment[0], abs(treatment[1])) not in dag.get_anchestors(outcome[0], include_lag=True):
    #         CP.info(f"- {treatment} not an anchestor of {outcome}")
    #         CP.info(f"- p({outcome}|do({treatment})) = p({outcome})")
    #         self.DO[outcome[0]][treatment][frozenset()] = self.dbn[outcome[0]]
    #         return
                                
    #     open_backdoor_paths = dag.get_open_backdoors_paths(treatment, outcome, conditions)
    #     self.DO[outcome[0]][treatment]['backdoor_paths'] = open_backdoor_paths
                    
    #     #! Case 4. Treatment affects outcome (no open Backdoor Paths)
    #     if not open_backdoor_paths:
    #         CP.info(f"- No adjustment needed for {treatment} -> {outcome} conditioning on {conditions_str}")
    #         CP.info(f"- p({outcome}|do({treatment}),{conditions_str}) = p({outcome}|{treatment},{conditions_str})")
    #         self.DO[outcome[0]][treatment][frozenset()] = self.dbn[outcome[0]]
    #         return
                    
    #     #! Case 5. Treatment affects outcome (Open Backdoor Paths)
    #     else:
    #         adjustment_sets = dag.find_all_d_separators(treatment, outcome, open_backdoor_paths, conditions, max_adj_size = max_adj_size)
    #         adjustment_sets = [{('OBS', -1)}] #! FIXME: remove me this is a hack
    #         if conditions is not None:
    #             CP.info(f"- Adjustment needed for {treatment} -> {outcome} conditioning on {conditions_str}")
    #         else:
    #             CP.info(f"- Adjustment needed for {treatment} -> {outcome}")
    #         CP.info(f"- Backdoor paths: {open_backdoor_paths}")
    #         if conditions is not None: CP.info(f"- Conditioning set: {conditions}")
            
    #         self.DO[outcome[0]][treatment]['backdoor_paths'] = open_backdoor_paths
    #         for adj_set in adjustment_sets:
    #             adj_set_key = frozenset(adj_set)
    #             CP.info(f"    #### Adjustment set: {adj_set}")
    #             if conditions is not None:
    #                 CP.info(f"    - p({outcome}|do({treatment}),{conditions_str}) = sum[p({outcome}|{treatment},{conditions_str},{','.join([str(s) for s in adj_set])})*p({','.join([str(s) for s in adj_set])})]")
    #             else:
    #                 CP.info(f"    - p({outcome}|do({treatment})) = sum[p({outcome}|{treatment},{','.join([str(s) for s in adj_set])})*p({','.join([str(s) for s in adj_set])})]")
                
                
    #             anchestors = dag.get_anchestors(outcome[0])
    #             context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
    #             contexts = self._extract_contexts(data, context_anchestors)
    #             self.DO[outcome[0]][treatment][adj_set_key] = {c: None for c in contexts}
    #             self.DO[outcome[0]][treatment][adj_set_key][(('C_S', 0), ('WP', 1))] = None
    #             self.DO[outcome[0]][treatment][adj_set_key][(('C_S', 1), ('WP', 1))] = None
                
    #             parents = [treatment[0]] + [cond[0] for cond in conditions] + [adj[0] for adj in adj_set]
    #             system_parents = [a for a in parents if self.node_type[a] == NodeType.System]

    #             system_conditions = {cond for cond in conditions if self.node_type[cond[0]] == NodeType.System}
    #             system_adj = {adj for adj in adj_set if self.node_type[adj[0]] == NodeType.System}
    #             for context in contexts:
                    
    #                 ## FIRST TERM
    #                 CP.info(f"    #### Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")
    #                 # Retrieve data for each node
    #                 segments = self.get_context_specific_segments(data, context, outcome[0], system_parents) 
    #                 if not segments: 
    #                     CP.info(f"    ### No context-specific segments found")
    #                     continue
                    
    #                 # Full DBN using all the segments concatenated
    #                 full_data = pd.concat([segment for segment in segments])
    #                 Y, X, COND, ADJ = self._get_Y_X_ADJ(full_data, outcome[0], treatment, cond=system_conditions, adj=system_adj)
    #                 ALL = {treatment[0]: X}
    #                 ALL.update({c[0]: COND[c] for c in COND})
    #                 ALL.update({c[0]: ADJ[c] for c in ADJ})
    #                 self.DO[outcome[0]][treatment][adj_set_key][context] = Density(Y, ALL, 
    #                                                                                max_components=self.max_components, 
    #                                                                                pY=self.dbn[outcome[0]][context]['full'].PriorDensity)
    #                 # self.DO[outcome[0]][treatment][adj_set_key][context] = Density(Y, ALL, 
    #                 #                                                                max_components=self.max_components, 
    #                 #                                                                pY=self.dbn[outcome[0]][context].PriorDensity)
    #                 # ## SECOND TERM (Adjustment)
    #                 # pJoint = self.DO[outcome[0]][treatment][adj_set_key][context].pJoint
    #                 # pAdj = self.dbn['OBS'].PriorDensity #! FIXME: remove me this is a hack
    #                 #                                     #! it should be computed by using gmm
                                      
                    

            
    #                 # p_adj = DensityUtils.get_density(pAdj, np.array([c[1] for c in context if c[0] == 'OBS'][0]).reshape(-1, 1)) #! FIXME: remove me this is a hack
                            
    #                 # # Accumulate weighted means and weights
    #                 # for k in range(len(self.DO[outcome[0]][treatment][adj_set_key][context].pJoint["weights"])):
    #                 #     self.DO[outcome[0]][treatment][adj_set_key][context].pJoint["weights"][k] *=  p_adj

    #                 # # Construct conditional_params from the accumulated means and weights
    #                 # total_weight = sum(self.DO[outcome[0]][treatment][adj_set_key][context].pJoint["weights"])
    #                 # if total_weight == 0:
    #                 #     raise ValueError("Total weight is zero. Check adjustment sets or inputs.")
    #                 # self.DO[outcome[0]][treatment][adj_set_key][context].pJoint["weights"] /= total_weight


    #             # pJoint_OBS0 = self.DO[outcome[0]][treatment][adj_set_key][(('C_S', 0), ('OBS', 0), ('WP', 1))].pJoint
    #             # pJoint_OBS1 = self.DO[outcome[0]][treatment][adj_set_key][(('C_S', 0), ('OBS', 1), ('WP', 1))].pJoint

    #             # # Extract GMM parameters
    #             # means_OBS0, covs_OBS0, weights_OBS0 = np.array(pJoint_OBS0["means"]), np.array(pJoint_OBS0["covariances"]), np.array(pJoint_OBS0["weights"])
    #             # means_OBS1, covs_OBS1, weights_OBS1 = np.array(pJoint_OBS1["means"]), np.array(pJoint_OBS1["covariances"]), np.array(pJoint_OBS1["weights"])

    #             # # Compute p(OBS)
    #             # pOBS_0 = DensityUtils.get_density(self.dbn['OBS'].PriorDensity, np.array([0]).reshape(-1, 1))  
    #             # pOBS_1 = DensityUtils.get_density(self.dbn['OBS'].PriorDensity, np.array([1]).reshape(-1, 1))  

    #             # # Weight the components
    #             # weighted_weights_OBS0 = weights_OBS0 * pOBS_0
    #             # weighted_weights_OBS1 = weights_OBS1 * pOBS_1

    #             # # Concatenate GMM components
    #             # total_means = np.vstack([means_OBS0, means_OBS1])
    #             # total_covs = np.vstack([covs_OBS0, covs_OBS1])
    #             # total_weights = np.concatenate([weighted_weights_OBS0, weighted_weights_OBS1])

    #             # # Normalize the weights
    #             # total_weights /= np.sum(total_weights)

    #             # # Store the combined GMM in `total`
    #             # total = {"means": total_means, "covariances": total_covs, "weights": total_weights}
    #             # self.DO[outcome[0]][treatment][adj_set_key][(('C_S', 0), ('WP', 1))] = Density(Y, ALL, 
    #             #                                                                                max_components=self.max_components, 
    #             #                                                                                pY=self.dbn[outcome[0]][context].PriorDensity,
    #             #                                                                                pJoint=total)
                    



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
