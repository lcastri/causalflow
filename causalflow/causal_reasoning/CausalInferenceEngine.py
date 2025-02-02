import itertools
import os
import pickle
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from causalflow.CPrinter import CP
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.DODensity import DODensity
import causalflow.causal_reasoning.Utils as DensityUtils
from causalflow.causal_reasoning.DynamicBayesianNetwork_2 import DynamicBayesianNetwork
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.CPrinter import CPLevel, CP
import copy
import networkx as nx
import causalflow.causal_reasoning.Utils as DensityUtils


class CausalInferenceEngine():
    def __init__(self, 
                 dag: DAG, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType],
                 max_components = 50,
                 model_path: str = '', 
                 verbosity = CPLevel.INFO):
        """
        CausalEngine constructor.

        Args:
            dag (DAG): observational dataset extracted from a causal discovery method.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}
            node_type (dict[str:NodeType]): node type for each node (system|context). E.g., {"X_2": NodeType.Context}
            verbosity (CPLevel, optional): Verbosity level. Defaults to DEBUG.
        """
        os.makedirs(model_path, exist_ok=True)
        CP.set_verbosity(verbosity)
        CP.set_logpath(os.path.join(model_path, 'log.txt'))
        CP.info("\n##")
        CP.info("## Causal Inference Engine")
        CP.info(f"## - Data Type:")
        for k, v in data_type.items(): CP.info(f"##    {k} : {v.name}")
        CP.info(f"## - Node Type:")
        for k, v in node_type.items(): CP.info(f"##    {k} : {v.name}")
        CP.info("##")
        
        self.data_type = data_type
        self.node_type = node_type
        self.max_components = max_components
        self.DAG = {'complete': dag, 'system': self.remove_context(dag)}
        self.model_path = model_path
        
        self.obs_id = -1
        
        self.DBNs = {}
        
    def save(self, respath):
        """
        Save a CausalInferenceEngine object from a pickle file.

        Args:
            respath (str): pickle save path.
        """
        pkl = dict()
        pkl['DAG'] = self.DAG
        pkl['DBNs'] = self.DBNs
        pkl['data_type'] = self.data_type 
        pkl['node_type'] = self.node_type 
        pkl['max_components'] = self.max_components
        pkl['model_path'] = self.model_path
        pkl['verbosity'] = CP.verbosity
        pkl['obs_id'] = self.obs_id

        with open(respath, 'wb') as resfile:
            pickle.dump(pkl, resfile)    
    
    @classmethod
    def load(cls, pklpath):
        """
        Load a CausalInferenceEngine object from a pickle file.

        Args:
            pklpath (str): pickle filepath.

        Returns:
            CausalInferenceEngine: loaded CausalInferenceEngine object.
        """
        with open(pklpath, 'rb') as f:
            pkl = pickle.load(f)
            # cie = cls(pkl['DAG']['complete'], pkl['data_type'], pkl['node_type'], 50, pkl['model_path'], pkl['verbosity'])
            cie = cls(pkl['DAG']['complete'], pkl['data_type'], pkl['node_type'], pkl['max_components'], pkl['model_path'], pkl['verbosity'])
            cie.obs_id = pkl['obs_id']
            cie.DBNs = pkl['DBNs']
            return cie
           
        
    def remove_context(self, dag: DAG):
        tmp = copy.deepcopy(dag)
        for t in dag.g:
            if self.node_type[t] is NodeType.Context: continue
            for s in dag.g[t].sources:
                if self.node_type[s[0]] is NodeType.Context:
                    tmp.del_source(t, s[0], s[1])
        for t in dag.g:
            if self.node_type[t] is NodeType.Context: 
                tmp.g.pop(t)
        return tmp 
    
    
    @staticmethod
    def remove_intVarParents(dag: DAG, target):
        if target not in dag.g: return dag
        tmp = copy.deepcopy(dag)
        for s in dag.g[target].sources:
            tmp.del_source(target, s[0], s[1])
        return tmp


    def addObsData(self, data: Data):
        """
        Add new observational dataset.

        Args:
            data (Data): new observational dataset.
        """
        self.obs_id += 1
        id = ('obs', self.obs_id)
        CP.info(f"\n## Building DBN for DAG ID {str(id)}")
        
        recycle = {}
        for existing_id, dbn in self.DBNs.items():
            for node in dbn.data.features:
                if node in recycle: continue
                if np.array_equal(dbn.data.d[node].values, data.d[node].values):
                    CP.info(f"Recycling node: {node} from DBN ID {str(existing_id)}")
                    recycle[node] = self.DBNs[existing_id].dbn[node],
                    continue
    
        self.DBNs[id] = DynamicBayesianNetwork(self.DAG['complete'], 
                                               data, 
                                               self.data_type, self.node_type, 
                                               recycle if recycle else None, 
                                               max_components = self.max_components)
        return id
    
    
    
    def Query(self, outcome: str, treatment: Dict[tuple, np.array], evidence : Dict[tuple, np.array] = None, wp: int = 0):
        import pyAgrum as gum
        bn_pgmpy = DAG.get_DBN(self.DBNs[('obs', wp)].dag.get_Adj(), self.DBNs[('obs', wp)].dag.max_lag)

        nodes = [(f, -abs(l)) for f in self.DAG['complete'].features for l in range(self.DAG['complete'].max_lag + 1)]
        outcome = (outcome, 0)
        
        treatment_f = list(treatment.keys())[0][0]
        treatment_values = treatment[list(treatment.keys())[0]]
        treatment_lag = list(treatment.keys())[0][1]
         
        # Get all possible values for non-evidence variables
        value_ranges = {var: self.DBNs[('obs', wp)].data.d[var[0]].unique() for var in nodes}
        
        # Step 1: Create Bayesian Network
        bn = gum.BayesNet('MyBN')

        # Add Nodes
        for n in nodes:
            bn.add(gum.LabelizedVariable(str(n), str(n), len(value_ranges[n])))

        # Step 2: Add Edges based on the Graph
        for edge in list(bn_pgmpy.edges):
            bn.addArc(str(edge[0]), str(edge[1]))
        

        # Step 3: Compute P(ELT' | RV, CS, ELT)
        ie = gum.LazyPropagation(bn)
        all_evidence = {}
        all_evidence.update({str((treatment_f, treatment_lag)): treatment_values[0]})
        for e in evidence:
            all_evidence.update({str(e): evidence[e][0]})
        ie.setEvidence(all_evidence)  # Example: setting observed values
        ie.makeInference()

        # Query P(ELT' | RV, CS, ELT)
        print(ie.posterior("ELT'"))

    
    
    # def Query(self, outcome: str, treatment: Dict[tuple, np.array], evidence : Dict[tuple, np.array] = None, wp: int = 0):
    #     bn = DAG.get_DBN(self.DBNs[('obs', wp)].dag.get_Adj(), self.DBNs[('obs', wp)].dag.max_lag)

    #     nodes = [(f, -abs(l)) for f in self.DAG['complete'].features if f != 'WP' for l in range(self.DAG['complete'].max_lag + 1)]
    #     outcome = (outcome, 0)
    #     dconnected_nodes = [n for n in nodes if bn.is_dconnected(n, outcome, list(treatment.keys()) + list(evidence.keys()))]
        
    #     treatment_f = list(treatment.keys())[0][0]
    #     treatment_values = treatment[list(treatment.keys())[0]]
    #     treatment_lag = list(treatment.keys())[0][1]
    #     evidence_fs = list(evidence.keys())
    #     nonevidence_fs = [n for n in nodes if n != (treatment_f, treatment_lag) and n not in evidence_fs and n != outcome]
        
    #     intT = len(treatment_values)
    #     maxLag = max(abs(treatment_lag), max([abs(c[1]) for c in evidence.keys()]))        
    #     res = {(treatment_f, treatment_lag): np.full((intT + maxLag, 1), np.nan),
    #            outcome: np.full((intT + maxLag, 1), np.nan)}
    #     for c in evidence.keys(): res[c] = np.full((intT + maxLag, 1), np.nan)
            
    #     res[(treatment_f, treatment_lag)][maxLag-abs(treatment_lag): len(res[c])-abs(treatment_lag)] = treatment_values
    #     for c in evidence.keys():
    #         res[c][maxLag-abs(c[1]): len(res[c])-abs(c[1])] = evidence[c]   
                 
    #     # Get all possible values for non-evidence variables
    #     value_ranges = {var: self.DBNs[('obs', wp)].data.d[var[0]].unique() for var in nonevidence_fs}
        
    #     def _get_density(node, p):
    #         f, lag = node
    #         if node in nonevidence_fs:
    #             # Marginalize the GMM parameters over all values
    #             return np.sum([DensityUtils.get_density(p, tmp_value.reshape(-1, 1)) for tmp_value in value_ranges[node]])
    #         # f observed => P(f=f)
    #         elif node in evidence_fs or node == (treatment_f, treatment_lag):
    #             value = res[(f, -abs(lag))][t-abs(lag)]
    #             # value = [evidence[e] for e in evidence if e[0] == f][0][t-abs(lag)]
    #             return DensityUtils.get_density(p, np.array(value).reshape(-1, 1))
    #         # f outcome
    #         elif node == outcome:
    #             return p
        
    #     for t in range(maxLag, intT + maxLag):
            
    #         if self.DBNs[('obs', wp)].DBN[outcome].parents is None:
    #             cond_params = self.DBNs[('obs', wp)].DBN[outcome].pY
    #         else:
    #             # p(outcome, treatment, evidence)
    #             cond_params = {'means': [], 'covariances': [], 'weights': []}
    #             accumulate_weight = 1.0

    #             # Chain rule for computing p(outcome, treatment, evidence)
    #             for node in nodes:
    #                 f, lag = node
    #                 if self.DBNs[('obs', wp)].DBN[node].parents is not None:
    #                     parents = [(name, -abs(process.lag)) for name, process in self.DBNs[('obs', wp)].DBN[node].parents.items() if name != 'WP']
    #                     p = self.DBNs[('obs', wp)].DBN[node].pJoint
    #                     parent_values = {}
    #                     for parent in parents:
    #                         parent_name, parent_lag = parent
    #                         if parent in evidence_fs or parent == (treatment_f, treatment_lag):
    #                             parent_values[parent] = res[parent][t-abs(parent_lag)]
    #                         else:
    #                             parent_values[parent] = value_ranges[parent]
                                
    #                     parent_values_combos = list(itertools.product(*parent_values.values()))
    #                     dens_accum = 0
    #                     for parent_values_combo in parent_values_combos:
    #                         pconditional = DensityUtils.compute_conditional(p, np.array(parent_values_combo).reshape(-1, 1))
    #                         dens = _get_density(node, pconditional)
    #                         if isinstance(dens, dict): 
    #                             cond_params = dens
    #                         elif isinstance(dens, float): 
    #                             dens_accum += dens
    #                     if isinstance(dens, dict): 
    #                         cond_params = dens_accum
    #                     elif isinstance(dens, float): 
    #                         accumulate_weight *= dens_accum

    #                 else:
    #                     # Use prior P(f) if no parents
    #                     p = self.DBNs[('obs', wp)].DBN[node].pY
    #                     dens = _get_density(node, p)
    #                     if isinstance(dens, dict): 
    #                         cond_params = dens
    #                     elif isinstance(dens, float): 
    #                         accumulate_weight *= dens
                            
    #         # Accumulate weighted means and weights
    #         weighted_means = []
    #         weighted_weights = []
    #         for k in range(len(cond_params["weights"])):
    #             weighted_means.append(cond_params["means"][k])
    #             weighted_weights.append(cond_params["weights"][k] * accumulate_weight)

    #         # Construct conditional_params from the accumulated means and weights
    #         total_weight = sum(weighted_weights)
    #         if total_weight == 0:
    #             raise ValueError("Total weight is zero. Check adjustment sets or inputs.")

    #         cond_params = {
    #             "means": np.array(weighted_means),
    #             "weights": np.array(weighted_weights) / total_weight,
    #         }
            
    #         expected_value = DensityUtils.expectation_from_params(cond_params['means'], cond_params['weights'])
    #         res[(outcome, 0)][t] = expected_value
    #     return res[(outcome, 0)][maxLag:intT+maxLag]

       
    def whatIf(self, 
               treatment: str, 
               values: np.array, 
               data: np.array, 
               prior_knowledge: Dict[str, np.array] = None):
        """
        Predict system behaviours in response to a certain time-series intervention.

        Args:
            treatment (str): treatment variable.
            values (np.array): treatment values.
            data (np.array): data where to start to predict from.
            prior_knowledge (dict[str: np.array], optional): future prior knowledge about variables different from the treatment one. Defaults to None.

        Returns:
            np.array: prediction.
        """
        if prior_knowledge is not None and len(values) != len(list(prior_knowledge.values())[0]):
            raise ValueError("prior_knowledge items must have the same length of the treatment values")
        intT = len(values)
        res = np.full((intT + self.DAG['complete'].max_lag, len(self.DAG['complete'].features)), np.nan)
        res[:self.DAG['complete'].max_lag, :] = data[-self.DAG['complete'].max_lag:, :]
        
        if prior_knowledge is not None:
            for f, f_data in prior_knowledge.items():
                res[self.DAG['complete'].max_lag:, self.DAG['complete'].features.index(f)] = f_data
        
        # Build an interventional DAG where the treatment variable has no parents
        dag = self.remove_intVarParents(self.DAG['complete'], treatment)
        g = self._DAG2NX(dag)
        calculation_order = list(nx.topological_sort(g))
        for t in range(self.DAG['complete'].max_lag, intT + self.DAG['complete'].max_lag):
            self.Q[TREATMENT] = treatment
            self.Q[VALUE] = values[t-self.DAG['complete'].max_lag]
            
            for f, lag in calculation_order:
                if np.isnan(res[t - abs(lag), f]):
                    if self.DAG['complete'].features[f] == self.Q[TREATMENT]:
                        res[t, f] = self.Q[VALUE]
                    else:
                        var = self.DAG['system'].features.index(self.DAG['complete'].features[f])
                        system_p = {}
                        context_p = {}
                        pID = None

                        for s in self.DAG['system'].g[self.DAG['system'].features[var]].sources:
                            system_p[s[0]] = res[t - abs(s[1]), self.DAG['complete'].features.index(s[0])]
                        anchestors = self.DAG['complete'].get_anchestors(self.DAG['complete'].features[f])
                        context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                        for a in context_anchestors:
                            context_p[a] = int(res[t, self.DAG['complete'].features.index(a)])

                        # pID, occ = self._findSource_intersection(system_p)
                        pID, pContext, pSegment, occ = self._findSource_intersection(self.DAG['complete'].features[f], system_p, context_p)

                        res[t, f] = self.DBNs[pID].dbn[self.DAG['system'].features[var]][pContext]['full'].predict(system_p) if pID is not None else np.nan
                        
        return res[self.DAG['complete'].max_lag:, :]
    
    
    def whatIfDo(self, outcome: str, 
                       treatment: str, lag: int, treatment_values: np.array, 
                       conditions: Dict[tuple, np.array] = None,
                       adjustment: list = None):
        
        # Check if treatment and conditions values have the same length
        if conditions is not None and len(conditions[list(conditions.keys())[0]]) != len(treatment_values):
            raise ValueError("conditions items must have the same length of the treatment values")
        
        intT = len(treatment_values)
        maxLag = max(abs(lag), max([abs(c[1]) for c in conditions.keys()]))
        
        # Initialize result
        res = {(treatment, lag): np.full((intT + maxLag, 1), np.nan),
               (outcome, 0): np.full((intT + maxLag, 1), np.nan)}
        for c in conditions.keys():
            res[c] = np.full((intT + maxLag, 1), np.nan)
            
        # Fill result with prior knowledge
        res[(treatment, lag)][maxLag-abs(lag): len(res[c])-abs(lag)] = treatment_values
        for c in conditions.keys():
            res[c][maxLag-abs(c[1]): len(res[c])-abs(c[1])] = conditions[c]
        
        # Predict outcome
        for t in range(maxLag, intT + maxLag):            
            sID = ('obs', 0) 
            adj_set = frozenset(adjustment) if adjustment is not None else frozenset()
            # if True:
            if isinstance(self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set], DODensity):
                res[(outcome, 0)][t] = self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set].predict(res[(treatment, lag)][t-abs(lag)], 
                                                                                                    {c: res[c][t-abs(c[1])] for c in conditions.keys()})               
            else:
                system_p = {}
                context_p = {}

                for s in self.DAG['system'].g[outcome].sources:
                    system_p[s[0]] = res[(s[0], -abs(s[1]))][t -abs(s[1])]
                anchestors = self.DAG['complete'].get_anchestors(outcome)
                context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                for a in context_anchestors:
                    context_p[a] = int(res[(a, 0)][t])
                pID, pContext, pSegment, occ = self._findSource_intersection(outcome, system_p, context_p)
                res[(outcome, 0)][t] = self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set][pContext]['full'].predict(system_p)
            # res[(outcome, 0)][t] = np.ceil(res[(outcome, 0)][t])
            if conditions is not None and any(c[0] == outcome for c in conditions.keys()):
                res[(outcome, [c[1] for c in conditions if c[0] == outcome][0])][t] = res[(outcome, 0)][t]
        return res[(outcome, 0)][maxLag:intT+maxLag]
    
    def whatIfDo_bayes(self, outcome: str, 
                       treatment: str, lag: int, treatment_values: np.array, 
                       conditions: Dict[tuple, np.array] = None):
        
        # Check if treatment and conditions values have the same length
        if conditions is not None and len(conditions[list(conditions.keys())[0]]) != len(treatment_values):
            raise ValueError("conditions items must have the same length of the treatment values")
        
        intT = len(treatment_values)
        maxLag = max(abs(lag), max([abs(c[1]) for c in conditions.keys()]))
        
        # Initialize result
        res = {(treatment, lag): np.full((intT + maxLag, 1), np.nan),
               (outcome, 0): np.full((intT + maxLag, 1), np.nan)}
        for c in conditions.keys():
            res[c] = np.full((intT + maxLag, 1), np.nan)
            
        # Fill result with prior knowledge
        res[(treatment, lag)][maxLag-abs(lag): len(res[c])-abs(lag)] = treatment_values
        for c in conditions.keys():
            res[c][maxLag-abs(c[1]): len(res[c])-abs(c[1])] = conditions[c]
        
        # Predict outcome
        for t in range(maxLag, intT + maxLag):            
            sID = ('obs', 0) 
            # if True:
            # res[(outcome, 0)][t] = self.DBNs[sID].Bayes[outcome][(treatment, lag)].predict(res[(treatment, lag)][t-abs(lag)], 
            #                                                                             {c: res[c][t-abs(c[1])] for c in conditions.keys()})  
            if isinstance(self.DBNs[sID].DO[outcome][(treatment, lag)], DODensity):
                res[(outcome, 0)][t] = self.DBNs[sID].DO[outcome][(treatment, lag)].predict(res[(treatment, lag)][t-abs(lag)], 
                                                                                                    {c: res[c][t-abs(c[1])] for c in conditions.keys()})               
            else:
                system_p = {}
                context_p = {}

                for s in self.DAG['system'].g[outcome].sources:
                    system_p[s[0]] = res[(s[0], -abs(s[1]))][t -abs(s[1])]
                anchestors = self.DAG['complete'].get_anchestors(outcome)
                context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                for a in context_anchestors:
                    context_p[a] = int(res[(a, 0)][t])
                pID, pContext, pSegment, occ = self._findSource_intersection(outcome, system_p, context_p)
                res[(outcome, 0)][t] = self.DBNs[sID].Bayes[outcome][(treatment, lag)][pContext]['full'].predict(system_p)             

            
            # res[(outcome, 0)][t] = np.ceil(res[(outcome, 0)][t])
            if conditions is not None and any(c[0] == outcome for c in conditions.keys()):
                res[(outcome, [c[1] for c in conditions if c[0] == outcome][0])][t] = res[(outcome, 0)][t]
        return res[(outcome, 0)][maxLag:intT+maxLag]
    
    
    def whatIfDo_context(self, outcome: str, 
                       treatment: str, lag: int, treatment_values: np.array, 
                       conditions: Dict[tuple, np.array] = None,
                       adjustment: list = None):
        
        # Check if treatment and conditions values have the same length
        if conditions is not None and len(conditions[list(conditions.keys())[0]]) != len(treatment_values):
            raise ValueError("conditions items must have the same length of the treatment values")
        
        intT = len(treatment_values)
        maxLag = max(abs(lag), max([abs(c[1]) for c in conditions.keys()]))
        
        # Initialize result
        res = {(treatment, lag): np.full((intT + maxLag, 1), np.nan),
               (outcome, 0): np.full((intT + maxLag, 1), np.nan)}
        for c in conditions.keys():
            res[c] = np.full((intT + maxLag, 1), np.nan)
            
        # Fill result with prior knowledge
        res[(treatment, lag)][maxLag-abs(lag): len(res[c])-abs(lag)] = treatment_values
        for c in conditions.keys():
            res[c][maxLag-abs(c[1]): len(res[c])-abs(c[1])] = conditions[c]
        
        # Predict outcome
        for t in range(maxLag, intT + maxLag):            
            sID = ('obs', 0) 
            adj_set = frozenset(adjustment) if adjustment is not None else frozenset()
            if True:
            # if isinstance(self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set], DODensity):
            #     res[(outcome, 0)][t] = self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set].predict(res[(treatment, lag)][t-abs(lag)], 
            #                                                                                         {c: res[c][t-abs(c[1])] for c in conditions.keys()})
                
                pJoint_OBS0 = self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set][(('C_S', 0), ('OBS', 0), ('WP', 1))].pJoint
                pJoint_OBS1 = self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set][(('C_S', 0), ('OBS', 1), ('WP', 1))].pJoint
                
                parent_values = np.concatenate((res[(treatment, lag)][t-abs(lag)], res[('ELT', -1)][t-1]))
                pJoint_OBS0 = DensityUtils.compute_conditional(pJoint_OBS0, parent_values)
                pJoint_OBS1 = DensityUtils.compute_conditional(pJoint_OBS1, parent_values)


                # Extract GMM parameters
                means_OBS0, covs_OBS0, weights_OBS0 = np.array(pJoint_OBS0["means"]), np.array(pJoint_OBS0["covariances"]), np.array(pJoint_OBS0["weights"])
                means_OBS1, covs_OBS1, weights_OBS1 = np.array(pJoint_OBS1["means"]), np.array(pJoint_OBS1["covariances"]), np.array(pJoint_OBS1["weights"])

                # Compute p(OBS)
                pOBS_0 = DensityUtils.get_density(self.DBNs[sID].dbn['OBS'].PriorDensity, np.array([0]).reshape(-1, 1))  
                pOBS_1 = DensityUtils.get_density(self.DBNs[sID].dbn['OBS'].PriorDensity, np.array([1]).reshape(-1, 1))  

                # Weight the components
                weighted_weights_OBS0 = weights_OBS0 * pOBS_0
                weighted_weights_OBS1 = weights_OBS1 * pOBS_1

                # Concatenate GMM components
                total_means = np.vstack([means_OBS0, means_OBS1])
                total_covs = np.vstack([covs_OBS0, covs_OBS1])
                total_weights = np.concatenate([weighted_weights_OBS0, weighted_weights_OBS1])

                # Normalize the weights
                total_weights /= np.sum(total_weights)
                
                p_Y_given_doX = {'means': total_means,
                                 'covariances': total_covs,
                                 'weights': total_weights}
                res[(outcome, 0)][t] = DensityUtils.expectation_from_params(p_Y_given_doX['means'], p_Y_given_doX['weights'])

                
            else:
                system_p = {}
                context_p = {}

                for s in self.DAG['system'].g[outcome].sources:
                    system_p[s[0]] = res[(s[0], -abs(s[1]))][t -abs(s[1])]
                anchestors = self.DAG['complete'].get_anchestors(outcome)
                context_anchestors = [a for a in anchestors if self.node_type[a] == NodeType.Context]
                for a in context_anchestors:
                    context_p[a] = int(res[(a, 0)][t])
                pID, pContext, pSegment, occ = self._findSource_intersection(outcome, system_p, context_p)
                res[(outcome, 0)][t] = self.DBNs[sID].DO[outcome][(treatment, lag)][adj_set][pContext]['full'].predict(system_p)
            # res[(outcome, 0)][t] = np.ceil(res[(outcome, 0)][t])
            if conditions is not None and any(c[0] == outcome for c in conditions.keys()):
                res[(outcome, [c[1] for c in conditions if c[0] == outcome][0])][t] = res[(outcome, 0)][t]
        return res[(outcome, 0)][maxLag:intT+maxLag]
    # def whatIfDo(self, outcome: str, 
    #                    treatment: str, lag: int, treatment_values: np.array, 
    #                    conditions: Dict[tuple, np.array] = None,
    #                    adjustment: list = None):
        
    #     # Check if treatment and conditions values have the same length
    #     if conditions is not None and len(conditions[list(conditions.keys())[0]]) != len(treatment_values):
    #         raise ValueError("conditions items must have the same length of the treatment values")
        
    #     intT = len(treatment_values)
    #     maxLag = max(abs(lag), max([abs(c[1]) for c in conditions.keys()]))
        
    #     # Initialize result
    #     res = {(treatment, lag): np.full((intT + maxLag, 1), np.nan),
    #            (outcome, 0): np.full((intT + maxLag, 1), np.nan)}
    #     for c in conditions.keys():
    #         res[c] = np.full((intT + maxLag, 1), np.nan)
            
    #     # Fill result with prior knowledge
    #     res[(treatment, lag)][:intT] = treatment_values
    #     for c in conditions.keys():
    #         res[c][:intT] = conditions[c]
        
    #     # Predict outcome
    #     for t in range(maxLag, intT + maxLag):            
    #         # pID, pContext, pSegment, occ = self._findDoSource(outcome, system_p, context_p)
    #         sID = ('obs', 0) 
    #         res[(outcome, 0)][t] = self.DBNs[sID].DO[outcome][(treatment, lag)][frozenset(adjustment)].predict(res[(treatment, lag)][t-abs(lag)], 
    #                                                                                                            {c: res[c][t-abs(c[1])] for c in conditions.keys()})
    #         res[(outcome, 0)][t] = np.round(res[(outcome, 0)][t], 1)
    #     return res[(outcome, 0)][maxLag:intT+maxLag]
    
    
    def _DAG2NX(self, dag: DAG) -> nx.DiGraph:
        G = nx.DiGraph()

        # 1. Nodes definition
        for i in range(len(dag.features)):
            for j in range(dag.max_lag, -1, -1):
                G.add_node((i, -j))

        # 2. Edges definition
        edges = list()
        for t in dag.g:
            for s in dag.g[t].sources:
                s_index = dag.features.index(s[0])
                t_index = dag.features.index(t)
                
                if s[1] == 0:
                    for j in range(dag.max_lag, -1, -1):
                        s_node = (s_index, -j)
                        t_node = (t_index, -j)
                        edges.append((s_node, t_node))
                        
                else:
                    s_lag = -s[1]
                    t_lag = 0
                    while s_lag >= -dag.max_lag:
                        s_node = (s_index, s_lag)
                        t_node = (t_index, t_lag)
                        edges.append((s_node, t_node))
                        s_lag -= 1
                        t_lag -= 1
                    
        G.add_edges_from(edges)
                
        return G
      
    
    def _findDoSource(self, target, parents, context=None):
        """
        Find the source population with the maximum number of occurrences 
        of a given intersection of parent values.

        Args:
            parents_values (dict): Dictionary where keys are parent variable names 
                                and values are the desired values for those parents.

        Returns:
            tuple: number of occurrences, source dataset
        """
        max_occurrences = 0
        pID = None
        pContext = None
        pSegment = None
        
        context = DensityUtils.format_combo(tuple(context.items()))
        
        def _find_occurrences(d, parents, atol):
            mask = np.ones(len(d.d), dtype=bool)
            for parent, value in parents.items():
                # mask &= np.isclose(d.d[parent], value, atol = atol)
                mask &= np.isclose(d.d[parent], value, atol = atol)
                # mask &= np.isclose(d.d[parent], value, atol = atol, rtol=atol*10)
                    
            # Count the number of occurrences where all parents match
            occurrences = np.sum(mask)
            return occurrences
        
        def compute_adaptive_atol(parent_values, data, scale_factor=0.05):
            """
            Compute an adaptive atol based on the variability of the parent values.

            Args:
                parent_values (dict): Dictionary of parent variable values.
                data (pd.DataFrame): Data containing the parent variables.
                scale_factor (float): Factor to scale the variability.

            Returns:
                float: Adaptive atol.
            """
            variances = []
            for parent in parent_values.keys():
                parent_data = data[parent].values
                variances.append(np.std(parent_data))  # Or use np.ptp(parent_data), np.median_absolute_deviation, etc.

            return scale_factor * np.mean(variances)  # Average variability scaled by the factor      
                        
        # return pID, pContext, pSegment, max_occurrences
        for id in self.Ds:
            if context not in self.Ds[id]['specific'][target]: continue
            d = self.Ds[id]['specific'][target][context]['full']
            adaptive_atol = compute_adaptive_atol(parents, d.d)
            occurrences = _find_occurrences(d, parents, adaptive_atol)
                
            # Update the best source if this dataset has more occurrences
            if occurrences > max_occurrences:
                max_occurrences = occurrences
                pID = id
                pContext = context
                pSegment = 'full'
                        
        if pID is None:
            max_samples = 0
            for id in self.Ds:
                if context not in self.Ds[id]['specific'][target]: continue
                d = self.Ds[id]['specific'][target][context]['full']
                num_samples = d.T
                if num_samples > max_samples:
                    max_samples = num_samples
                    pID = id
                    pContext = context
                    pSegment = 'full'
                        
        return pID, pContext, pSegment, max_occurrences
    
    
    def _findSource_intersection(self, target, parents, context=None):
        """
        Find the source population with the maximum number of occurrences 
        of a given intersection of parent values.

        Args:
            parents_values (dict): Dictionary where keys are parent variable names 
                                and values are the desired values for those parents.

        Returns:
            tuple: number of occurrences, source dataset
        """
        max_occurrences = 0
        pID = None
        pContext = None
        pSegment = None
        
        context = DensityUtils.format_combo(tuple(context.items()))
        
        def _find_occurrences(d, parents, atol):
            mask = np.ones(len(d.d), dtype=bool)
            for parent, value in parents.items():
                # mask &= np.isclose(d.d[parent], value, atol = atol)
                mask &= np.isclose(d.d[parent], value, atol = atol)
                # mask &= np.isclose(d.d[parent], value, atol = atol, rtol=atol*10)
                    
            # Count the number of occurrences where all parents match
            occurrences = np.sum(mask)
            return occurrences
        
        def compute_adaptive_atol(parent_values, data, scale_factor=0.05):
            """
            Compute an adaptive atol based on the variability of the parent values.

            Args:
                parent_values (dict): Dictionary of parent variable values.
                data (pd.DataFrame): Data containing the parent variables.
                scale_factor (float): Factor to scale the variability.

            Returns:
                float: Adaptive atol.
            """
            variances = []
            for parent in parent_values.keys():
                parent_data = data[parent].values
                variances.append(np.std(parent_data))  # Or use np.ptp(parent_data), np.median_absolute_deviation, etc.

            return scale_factor * np.mean(variances)  # Average variability scaled by the factor      
                        
        # return pID, pContext, pSegment, max_occurrences
        for id in self.Ds:
            if context not in self.Ds[id]['specific'][target]: continue
            d = self.Ds[id]['specific'][target][context]['full']
            adaptive_atol = compute_adaptive_atol(parents, d.d)
            occurrences = _find_occurrences(d, parents, adaptive_atol)
                
            # Update the best source if this dataset has more occurrences
            if occurrences > max_occurrences:
                max_occurrences = occurrences
                pID = id
                pContext = context
                pSegment = 'full'
                        
        if pID is None:
            max_samples = 0
            for id in self.Ds:
                if context not in self.Ds[id]['specific'][target]: continue
                d = self.Ds[id]['specific'][target][context]['full']
                num_samples = d.T
                if num_samples > max_samples:
                    max_samples = num_samples
                    pID = id
                    pContext = context
                    pSegment = 'full'
                        
        return pID, pContext, pSegment, max_occurrences
        
            
    def plot_pE(self, y, parent, density, expectation = None, show = False, path = None):
        plt.figure(figsize=(10, 6))
        plt.plot(y.aligndata, density, label='Density')
        if expectation is not None: plt.axvline(expectation, color='r', linestyle='--', label=f'Expectation = {expectation:.2f}')
        plt.xlabel(f'${y.varname}$')
        pa = []
        for p in parent:
            if p == self.Q[TREATMENT]: 
                pa.append(f'do(${self.Q[TREATMENT]}$ = {self.Q[VALUE]:.2f})')
            else:
                pa.append(f'${p}$')
        pa = ','.join(pa)
        plt.ylabel(f'p(${y.varname}$|{pa})')
        plt.legend()
        if show: 
            plt.show()
        else:
            plt.savefig(os.path.join(path, f'p({self.Q[OUTCOME]}|do({self.Q[TREATMENT]} = {str(self.Q[VALUE])})).png'))