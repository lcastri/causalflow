import itertools
import os
import pickle
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from causalflow.CPrinter import CP
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Density_utils import *
from causalflow.causal_reasoning.DynamicBayesianNetwork import DynamicBayesianNetwork
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.CPrinter import CPLevel, CP
import copy
import networkx as nx
import gc
import h5py


class CausalInferenceEngine():
    def __init__(self, dag: DAG, 
                 data_type: Dict[str, DataType], 
                 node_type: Dict[str, NodeType], 
                 nsample = 100, 
                 use_gpu: bool = False, 
                 model_path: str = '', 
                 verbosity = CPLevel.INFO):
        """
        CausalEngine constructor.

        Args:
            dag (DAG): observational dataset extracted from a causal discovery method.
            data_type (dict[str:DataType]): data type for each node (continuous|discrete). E.g., {"X_2": DataType.Continuous}
            nsample (int, optional): Number of samples used for density estimation. Defaults to 100.
            use_gpu (bool): If True, use GPU for density estimation; otherwise, use CPU.
            verbosity (CPLevel, optional): Verbosity level. Defaults to DEBUG.
        """
        CP.set_verbosity(verbosity)
        CP.info("\n##")
        CP.info("## Causal Inference Engine")
        CP.info("##")
        
        self.nsample = nsample
        self.Q = {}
        self.data_type = data_type
        self.node_type = node_type
        self.DAG = {'complete': dag, 'system': self.remove_context(dag)}
        self.use_gpu = use_gpu
        self.model_path = model_path
        
        self.contexts = []
        self.tols = {}
        self.obs_id = -1
        self.int_id = -1
        
        os.makedirs(model_path, exist_ok=True)
        filename = os.path.join(model_path, 'CIE_DB.h5')
        self.filename = filename
        # Check if the file exists
        if not os.path.exists(filename):
            CP.debug(f"File {filename} does not exist. Creating a new one.")
            with h5py.File(filename, 'w') as f:
                # Initialize the structure of the file if needed
                f.create_group('DAGs')
                f.create_group('DBNs')
                f.create_group('Ds')  
        else:
            CP.debug(f"Model exists, proceeding to load data.")
            
                      
    @property
    def isThereContext(self):
        return any(n is NodeType.Context for n in self.node_type.values())
    
    def get_resolutions(self):
        tol = {}
        for f in self.DAG['system'].features:
            Ds = self.load_obsDs()
            if Ds:
                for id in Ds:
                    if id not in tol: tol[id] = {}
                    if isinstance(Ds[id], dict):
                        for context, d in Ds[id].items():
                            if context not in tol[id]: tol[id][context] = {}
                            tol[id][context][f] = (np.max(np.array(d.d[f])) - np.min(np.array(d.d[f])))/self.nsample
                    else:
                        d = Ds[id]
                        tol[id][f] = (np.max(np.array(d.d[f])) - np.min(np.array(d.d[f])))/self.nsample
        return tol
    
    # def save_DAG(self, id, dag):
    #     with h5py.File(self.filename, 'a') as f:
    #         group_name = f"DAGs/{id}"
    #         # Convert the DAG object to bytes using pickle
    #         dag_data = pickle.dumps(dag)
    #         if group_name in f:
    #             del f[group_name]
    #         f.create_dataset(group_name, data=np.void(dag_data))  # Use np.void to store as binary

    # def save_DBN(self, dbn, id, context = None):
    #     with h5py.File(self.filename, 'a') as f:
    #         group_name = f"DBNs/{id}/{context}" if context is not None else f"DBNs/{id}"
    #         dbn_data = pickle.dumps(dbn)  # Serialize dbn object
    #         if group_name in f:
    #             del f[group_name]
    #         # f.create_dataset(group_name, data=np.void(dbn_data))
    #         f.create_dataset(group_name, data=np.void(dbn_data), compression="gzip", compression_opts=9)

            
    # def save_D(self, d, id, context = None):
    #     with h5py.File(self.filename, 'a') as f:
    #         group_name = f"Ds/{id}/{context}" if context is not None else f"Ds/{id}"
    #         d_data = pickle.dumps(d)
    #         if group_name in f:
    #             del f[group_name]
    #         f.create_dataset(group_name, data=np.void(d_data))
            
    # def load_DAG(self, id):
    #     with h5py.File(self.filename, 'r') as f:
    #         group_name = f"DAGs/{id}"
    #         if group_name in f:
    #             # Deserialize the stored data back into a DAG object
    #             return pickle.loads(f[group_name][()].tobytes())
    #         else:
    #             raise None
            
    # def load_DBN(self, id, context = None):
    #     with h5py.File(self.filename, 'r') as f:
    #         group_name = f"DBNs/{id}/{context}" if context is not None else f"DBNs/{id}"
    #         if group_name in f:
    #             return pickle.loads(f[group_name][()].tobytes())
    #         else:
    #             raise None
            
    # def load_D(self, id, context = None):
    #     with h5py.File(self.filename, 'r') as f:
    #         group_name = f"Ds/{id}/{context}" if context is not None else f"Ds/{id}"
    #         if group_name in f:
    #             return pickle.loads(f[group_name][()].tobytes())
    #         else:
    #             return 
    
    def save_DAG(self, id, dag):
        pickle_path = os.path.join(self.model_path, f"DAG_{id}.pkl")
        with open(pickle_path, 'wb') as file:
            pickle.dump(dag, file)
        with h5py.File(self.filename, 'a') as f:
            group_name = f"DAGs/{id}"
            if group_name in f:
                del f[group_name]
            f.create_dataset(group_name, data=pickle_path)

    def save_DBN(self, dbn, id, context=None):
        pickle_name = f"DBN_{id}_{context}.pkl" if context is not None else f"DBN_{id}.pkl"
        pickle_path = os.path.join(self.model_path, pickle_name)
        with open(pickle_path, 'wb') as file:
            pickle.dump(dbn, file)
        with h5py.File(self.filename, 'a') as f:
            group_name = f"DBNs/{id}/{context}" if context is not None else f"DBNs/{id}"
            if group_name in f:
                del f[group_name]
            f.create_dataset(group_name, data=pickle_path)

    def save_D(self, d, id, context=None):
        pickle_name = f"D{id}_{context}.pkl" if context is not None else f"D_{id}.pkl"
        pickle_path = os.path.join(self.model_path, pickle_name)
        with open(pickle_path, 'wb') as file:
            pickle.dump(d, file)
        with h5py.File(self.filename, 'a') as f:
            group_name = f"Ds/{id}/{context}" if context is not None else f"Ds/{id}"
            if group_name in f:
                del f[group_name]
            f.create_dataset(group_name, data=pickle_path)

    def load_DAG(self, id):
        with h5py.File(self.filename, 'r') as f:
            group_name = f"DAGs/{id}"
            if group_name in f:
                pickle_path = f[group_name][()].decode()  # Read the path
                with open(pickle_path, 'rb') as file:
                    return pickle.load(file)
            else:
                raise FileNotFoundError(f"No DAG found for id {id}")

    def load_DBN(self, id, context=None):
        with h5py.File(self.filename, 'r') as f:
            group_name = f"DBNs/{id}/{context}" if context is not None else f"DBNs/{id}"
            if group_name in f:
                pickle_path = f[group_name][()].decode()  # Read the path
                with open(pickle_path, 'rb') as file:
                    return pickle.load(file)
            else:
                raise FileNotFoundError(f"No DBN found for id {id} and context {context}")

    def load_D(self, id, context=None):
        with h5py.File(self.filename, 'r') as f:
            group_name = f"Ds/{id}/{context}" if context is not None else f"Ds/{id}"
            if group_name in f:
                pickle_path = f[group_name][()].decode()  # Read the path
                with open(pickle_path, 'rb') as file:
                    return pickle.load(file)
            else:
                raise FileNotFoundError(f"No D found for id {id} and context {context}")
            
    def load_obsDs(self, selected_context = None):
        Ds = {}
        for id in range(self.obs_id + 1):
            d_id = ('obs', id)
            if selected_context is None and self.contexts:
                for context in self.contexts:
                    d = self.load_D(d_id, context)
                    if d is not None:
                        if d_id not in Ds: Ds[d_id] = {}
                        Ds[d_id][context] = d
            elif selected_context is None and not self.contexts:
                d = self.load_D(d_id)
                if d is not None:
                    if d_id not in Ds: Ds[d_id] = {}
                    Ds[d_id] = d
            else:
                d = self.load_D(d_id, selected_context)
                if d is not None:
                    if d_id not in Ds: Ds[d_id] = {}
                    Ds[d_id][selected_context] = d
        return Ds


    def getContextData(self, data, context):
        context_dict = dict(context)

        # Filter the dataframe based on the context dictionary
        filtered_data = data.d
        for key, value in context_dict.items():
            filtered_data = filtered_data[filtered_data[key] == value]

        # Check if the filtered data is non-empty (i.e., if the context combination exists)
        return filtered_data    
    
    
    def _extract_contexts(self, data):
        contexts = {}
        for n, ntype in self.node_type.items():
            if ntype is NodeType.Context:
                contexts[n] = np.array(np.unique(data.d[n]), dtype=int if self.data_type[n] is DataType.Discrete else float)
        
        # Generate the combinations
        tmp = list(itertools.product(*[[(k, float(v) if self.data_type[k] is DataType.Continuous else int(v)) for v in values] for k, values in contexts.items()]))
        combinations = []
        for c in tmp:
            combinations.append(CausalInferenceEngine.get_combo(c))
        return combinations
    
    
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
        
    @staticmethod
    def get_combo(combo):
        return tuple(sorted(combo))
               

    def addObsData(self, data: Data):
        """
        Add new observational dataset.

        Args:
            data (Data): new observational dataset.
        """
        id = ('obs', self.obs_id + 1)
        if not self.isThereContext:
            CP.info(f"\n## Building DBN for DAG ID {str(id)}")
            _tmp_dbn = DynamicBayesianNetwork(self.DAG['system'], data, self.nsample, self.data_type, self.node_type, use_gpu=self.use_gpu)
            self.save_DBN(_tmp_dbn, id)
            self.save_D(data, id)
        else:
            self.contexts = self._extract_contexts(data)
            for context in self.contexts:
                d = self.getContextData(data, context)
                if not d.empty:
                    # if len([context]) == 1:
                    #     CP.info(f"\n## Building DBN for DAG ID {id} -- Context: {', '.join([f'{c[0]}={c[1]}'for c in [context]])}")
                    # else:
                    CP.info(f"\n## Building DBN for DAG ID {id} -- Context: {', '.join([f'{c[0]}={c[1]}' for c in context])}")

                    _tmp_d = Data(d, vars=d.columns)
                    _tmp_d.shrink([c for c in d.columns if c not in list(dict(context).keys())])
                    _tmp_dbn = DynamicBayesianNetwork(self.DAG['system'], _tmp_d, self.nsample, self.data_type, self.node_type, use_gpu=self.use_gpu)
                    self.save_DBN(_tmp_dbn, id, context)
                    self.save_D(_tmp_d, id, context)
                    gc.collect()
        self.obs_id += 1
        
        self.tols = self.get_resolutions()
        return id
        
        
    def addIntData(self, target: str, data: Data):
        """
        Add new interventional dataset.

        Args:
            target (str): Intervention treatment variable.
            data (Data): Interventional data.
        """
        dag = CausalInferenceEngine.remove_intVarParents(self.DAG, target)
            
        id = ('int', str(target), self.nextInt)
        self.DAGs[id] = dag
        self.Ds[id] = data
        CP.info(f"\n## Building DBN for DAG ID {str(id)}")
        self.DBNs[id] = DynamicBayesianNetwork(dag, data, self.nsample, self.data_type, self.node_type, use_gpu=self.use_gpu)
        return id
    
    
    def save(self, respath):
        """
        Save a CausalInferenceEngine object from a pickle file.

        Args:
            respath (str): pickle save path.
        """
        pkl = dict()
        pkl['DAG'] = self.DAG
        pkl['data_type'] = self.data_type 
        pkl['node_type'] = self.node_type 
        pkl['nsample'] = self.nsample
        pkl['use_gpu'] = self.use_gpu
        pkl['model_path'] = self.model_path
        pkl['verbosity'] = CP.verbosity
        pkl['contexts'] = self.contexts
        pkl['obs_id'] = self.obs_id
        pkl['int_id'] = self.int_id

        with open(respath, 'wb') as resfile:
            pickle.dump(pkl, resfile)    
    
    @classmethod
    def load(cls, pkl):
        """
        Load a CausalInferenceEngine object from a pickle file.

        Args:
            pkl (pickle): pickle file.

        Returns:
            CausalInferenceEngine: loaded CausalInferenceEngine object.
        """
        cie = cls(pkl['DAG']['complete'], pkl['data_type'], pkl['node_type'], pkl['nsample'], pkl['use_gpu'], pkl['model_path'], pkl['verbosity'])
        cie.contexts = pkl['contexts']
        cie.obs_id = pkl['obs_id']
        cie.int_id = pkl['int_id']
        cie.tols = cie.get_resolutions()
        return cie
        
        
    def whatHappens(self, outcome: str, treatment: str, value, targetP: tuple):
        """
        Calculate p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))].

        Args:
            outcome (str): outcome variable.
            treatment (str): treatment variable.
            value (float): treatment value.
            targetP (tuple): target population ID (e.g., ("obs", 3)).

        Returns:
            tuple: (outcome samples, p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))]).
        """
        self.Q[OUTCOME] = outcome
        self.Q[TREATMENT] = treatment
        self.Q[VALUE] = value
        
        CP.info("\n## Query")
        
        # searches the population with greatest number of occurrences treatment == treatment's value
        otherDs = copy.deepcopy(self.Ds) # A: all populations
        otherDs.pop(targetP, None) # A: all populations - target population
        intDs = {key: value for key, value in self.Ds.items() if key[0] == 'int' and key[1] == self.Q[TREATMENT]}  # B: all interventional populations with treatment variable == treatment
        for key in intDs.keys(): # A: A - B
            otherDs.pop(key, None)
            
        intOcc, intSource = self._findSourceQ(intDs)
        otherOcc, otherSource = self._findSourceQ(otherDs)
        
        sourceP = intSource if intOcc != 0 else otherSource
        
        # Source population's p(output|do(treatment), adjustment)
        pS_y_do_x_adj = self.DBNs[sourceP][outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ]
        
        p_y_do_x = self.transport(pS_y_do_x_adj, targetP, self.Q[TREATMENT], self.Q[OUTCOME])
        
        y, p_y_do_X_x, E_p_y_do_X_x = self.evalDoDensity(p_y_do_x, sourceP)
        CP.info(f"## What happens to {self.Q[OUTCOME]} if {self.Q[TREATMENT]} = {str(self.Q[VALUE])} in population {str(targetP)} ? {self.Q[OUTCOME]} = {E_p_y_do_X_x}")
            
        return y, p_y_do_X_x, E_p_y_do_X_x
    
    
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
                        given_p = {}
                        pID = None
                        sources = []

                        for s in self.DAG['system'].g[self.DAG['system'].features[var]].sources:
                            given_p[s[0]] = res[t - abs(s[1]), self.DAG['complete'].features.index(s[0])]

                        # Initialize variables for tracking the best source
                        sources = []
                        
                        # For non-interventional population, find the source using the intersection of all parent values # FIXME: this must be inside the ELSE
                        # context = CausalInferenceEngine.get_combo((('B_S', int(res[t, self.DAG['complete'].features.index('B_S')])), 
                        #                                            ('WP', int(res[t, self.DAG['complete'].features.index('WP')])), 
                        #                                            ('TOD', int(res[t, self.DAG['complete'].features.index('TOD')]))))
                        context = None
                        tmp_pID, tmp_occ = self._findSource_intersection(given_p, context) # FIXME: this must be inside the ELSE
                        sources.append((tmp_pID, tmp_occ)) # FIXME: this must be inside the ELSE

                        # Choose the source with the maximum occurrences from the intersection-based matches
                        # if len(dag.g[self.DAG.features[f]].sources): # FIXME: uncomment me
                        #     sourceP = intSource if intOcc != 0 else max(sources, key=lambda x: x[1])[0] # FIXME: uncomment me
                        pID = max(sources, key=lambda x: x[1])[0] # FIXME: remove me
                        if pID is None:
                            e = np.nan
                        else:
                            dbn = self.load_DBN(pID, context)
                            tol = self.tols[pID][context] if context is not None else self.tols[pID]
                            d, e = dbn.dbn[self.DAG['system'].features[var]].predict(given_p, tol)
                        # self.plot_pE(self.DBNs[sourceP][self.DAG.features[f]].y, list(given_p.keys()), d, e, True)
                        res[t, f] = e
        return res[self.DAG['complete'].max_lag:, :]
    
    
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
    
        
    def _findSourceQ(self, Ds):
        """
        finds source population with maximum number of occurrences treatment = value

        Args:
            Ds (dict): dataset dictionary {id (str): d (Data)}

        Returns:
            tuple: number of occurrences, source dataset
        """
        occurrences = 0
        sourceP = None
        for id, d in Ds.items():
            indexes = np.where(np.isclose(d.d[self.Q[TREATMENT]], self.Q[VALUE], atol = self.resolutions[TREATMENT]))[0]
            if len(indexes) > occurrences: 
                occurrences = len(indexes)
                sourceP = id
                
        return occurrences, sourceP
    
      
    def _findSource(self, Ds, target, value):
        """
        finds source population with maximum number of occurrences treatment = value

        Args:
            Ds (dict): dataset dictionary {id (str): d (Data)}

        Returns:
            tuple: number of occurrences, source dataset
        """
        occurrences = 0
        for id, d in Ds.items():
            indexes = np.where(np.isclose(d.d[target], value, atol = self.resolutions[target]))[0]
            if len(indexes) > occurrences: 
                occurrences = len(indexes)
                sourceP = id
                
        return occurrences, sourceP
    
    
    # def _findSource_intersection(self, parents_values, context):
    #     """
    #     Find the source population with the maximum number of occurrences 
    #     of a given intersection of parent values.

    #     Args:
    #         parents_values (dict): Dictionary where keys are parent variable names 
    #                                and values are the desired values for those parents.

    #     Returns:
    #         tuple: number of occurrences, source dataset
    #     """
    #     max_occurrences = 0
    #     pID = None
    #     Ds = self.load_obsDs(context)
                    
    #     for id in Ds:
    #         for context, d in Ds[id].items():
    #             # Create a mask for each parent that matches the desired value with tolerance
    #             mask = np.ones(len(d.d), dtype=bool)
    #             for parent, value in parents_values.items():
    #                 mask &= np.isclose(d.d[parent], value, atol=self.resolutions[parent])
                
    #             # Count the number of occurrences where all parents match
    #             occurrences = np.sum(mask)
                
    #             # Update the best source if this dataset has more occurrences
    #             if occurrences > max_occurrences:
    #                 max_occurrences = occurrences
    #                 pID = id

    #     return pID, max_occurrences
    
    def _findSource_intersection(self, parents_values, context=None, resolution_multiplier=1):
        """
        Find the source population with the maximum number of occurrences 
        of a given intersection of parent values.

        Args:
            parents_values (dict): Dictionary where keys are parent variable names 
                                and values are the desired values for those parents.
            resolution_multiplier (float): A multiplier for adjusting the resolution 
                                        (default is 1 for the original resolution).

        Returns:
            tuple: number of occurrences, source dataset
        """
        max_occurrences = 0
        pID = None
        Ds = self.load_obsDs(context)
        
        # Update the resolution based on the multiplier
        # current_resolutions = {parent: self.resolutions[parent] * resolution_multiplier for parent in parents_values}
        
        for id in Ds:
            if context is not None:
                for context, d in Ds[id].items():
                    # Create a mask for each parent that matches the desired value with the adjusted resolution
                    mask = np.ones(len(d.d), dtype=bool)
                    current_resolutions = {parent: self.tols[id][context][parent] * resolution_multiplier for parent in parents_values}
                    for parent, value in parents_values.items():
                        mask &= np.isclose(d.d[parent], value, atol=current_resolutions[parent])
                    
                    # Count the number of occurrences where all parents match
                    occurrences = np.sum(mask)
                    
                    # Update the best source if this dataset has more occurrences
                    if occurrences > max_occurrences:
                        max_occurrences = occurrences
                        pID = id
            else:
                d = Ds[id]
                mask = np.ones(len(d.d), dtype=bool)
                current_resolutions = {parent: self.tols[id][parent] * resolution_multiplier for parent in parents_values}
                for parent, value in parents_values.items():
                    mask &= np.isclose(d.d[parent], value, atol=current_resolutions[parent])
                    
                # Count the number of occurrences where all parents match
                occurrences = np.sum(mask)
                    
                # Update the best source if this dataset has more occurrences
                if occurrences > max_occurrences:
                    max_occurrences = occurrences
                    pID = id

        # If no matching source is found, call the function recursively with a higher resolution multiplier
        if pID is None and resolution_multiplier < 10:  # Arbitrary limit to prevent infinite recursion
            return self._findSource_intersection(parents_values, context, resolution_multiplier * 2)

        return pID, max_occurrences
        
    # TODO: to change self.DBNs[targetP].data -- self.DBNs[targetP] does not have data attribute
    def transport(self, pS_y_do_x_adj, targetP: tuple, treatment: str, outcome: str):
        """
        Computes the target population's p_y_do(x) from the source population by using the transportability formula [1].
        
        [1] Bareinboim, Elias, and Judea Pearl. "Causal inference and the data-fusion problem." 
            Proceedings of the National Academy of Sciences 113.27 (2016): 7345-7352.

        Args:
            pS_y_do_x_adj (tuple): p(output|do(treatment), adjustment) of the source population
            targetP (tuple): target population ID
            treatment (str): treatment variable
            outcome (str): outcome variable

        Returns:
            nd.array: Target population's p_y_do(x)
        """
        adjset = self.DBNs[targetP].get_adjset(treatment, outcome) # TODO: to test
        
        # # Source population's p(output|do(treatment), adjustment)
        # pS_y_do_x_adj = self.DBNs[sourceP][outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ]
        
        # Compute the adjustment density for the target population
        pT_adj = np.ones((self.nsample, 1)).squeeze()
            
        for node in adjset: pT_adj = pT_adj * self.DBNs[targetP][self.DBNs[targetP].data.features[node[0]]].CondDensity
        pT_adj = normalise(pT_adj)
        
        # Compute the p(outcome|do(treatment)) density
        if len(pS_y_do_x_adj.shape) > 2: 
            # Sum over the adjustment set
            p_y_do_x = normalise(np.sum(pS_y_do_x_adj * pT_adj, axis = tuple(range(2, len(pS_y_do_x_adj.shape)))))
        else:
            p_y_do_x = pS_y_do_x_adj
        
        return p_y_do_x
    
    
    def evalDoDensity(self, p_y_do_x, sourceP: tuple):
        """
        Evaluates the p(outcome|do(treatment = t))

        Args:
            p_y_do_x: p(outcome|do(treatment)) density
            sourceP (tuple): source population ID

        Returns:
            tuple: outcome samples, p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))], Mode(outcome*p(outcome|do(treatment = t))
        """
        indices_X = np.where(np.isclose(self.DBNs[sourceP][self.Q[OUTCOME]].parents[self.Q[TREATMENT]].samples, 
                                        self.Q[VALUE], 
                                        atol = self.resolutions[TREATMENT]))[0]
        indices_X = np.array(sorted(indices_X))               
        
        # I am taking all the outcome's densities associated to the treatment == value
        # Normalise the density to ensure it sums to 1
        p_y_do_X_x = normalise(np.sum(p_y_do_x[:, indices_X], axis = 1))
        E_p_y_do_X_x = expectation(self.DBNs[sourceP][self.Q[OUTCOME]].y.samples, p_y_do_X_x)
        M_p_y_do_X_x = mode(self.DBNs[sourceP][self.Q[OUTCOME]].y.samples, p_y_do_X_x)
        # self.plot_pE(self.DBNs[sourceP][self.Q[OUTCOME]].y.samples, p_y_do_X_x, E_p_y_do_X_x, show = True)
        return self.DBNs[sourceP][self.Q[OUTCOME]].y.samples, p_y_do_X_x, E_p_y_do_X_x, M_p_y_do_X_x
    
    
    def plot_pEQ(self, ysamples, density, expectation = None, show = False, path = None):
        plt.figure(figsize=(10, 6))
        plt.plot(ysamples, density, label='Density')
        if expectation is not None: plt.axvline(expectation, color='r', linestyle='--', label=f'Expectation = {expectation:.2f}')
        plt.xlabel(f'${self.Q[OUTCOME]}$')
        plt.ylabel(f'p(${self.Q[OUTCOME]}$|do(${self.Q[TREATMENT]}$ = {str(self.Q[VALUE])}))')
        plt.legend()
        if show: 
            plt.show()
        else:
            plt.savefig(os.path.join(path, f'p({self.Q[OUTCOME]}|do({self.Q[TREATMENT]} = {str(self.Q[VALUE])})).png'))
            
            
    def plot_pE(self, y, parent, density, expectation = None, show = False, path = None):
        plt.figure(figsize=(10, 6))
        plt.plot(y.samples, density, label='Density')
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