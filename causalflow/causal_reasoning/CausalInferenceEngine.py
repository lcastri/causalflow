import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from causalflow.CPrinter import CP
from causalflow.basics.constants import *
from causalflow.causal_reasoning.Density import Density
from causalflow.causal_reasoning.DynamicBayesianNetwork import DynamicBayesianNetwork
from causalflow.graph.DAG import DAG
from causalflow.preprocessing.data import Data
from causalflow.CPrinter import CPLevel, CP
import copy


class CausalInferenceEngine():
    def __init__(self, dag: DAG, nsample = 100, atol = 0.25, verbosity = CPLevel.DEBUG):
        """
        CausalEngine constructor.

        Args:
            dag (DAG): observational dataset extracted from a causal discovery method.
            nsample (int, optional): Number of samples used for density estimation. Defaults to 100.
            atol (float, optional): absolute tolerance used to check if a specific intervention has been already observed. Defaults to 0.25.
            verbosity (CPLevel, optional): verbosity level. Defaults to DEBUG.
        """
        CP.set_verbosity(verbosity)
        CP.info("\n##")
        CP.info("## Causal Inference Engine")
        CP.info("##")
        
        self.nsample = nsample
        self.atol = atol
        self.Q = {}
        self.DAG = dag
        
        self.DAGs = {}
        self.Ds = {}
        self.DBNs = {}
        
        
    @property        
    def nextObs(self):
        """
        Return next observation ID.

        Returns:
            int: next observation ID.
        """
        arg = [key for key in self.DAGs.keys() if key[0] == 'obs']
        if arg:
            return max(arg, key=lambda x: x[1])[1] + 1
        else:
            return 0
    
    @property
    def nextInt(self):
        """
        Return next intervention ID.

        Returns:
            int: next intervention ID.
        """
        arg = [key for key in self.DAGs.keys() if key[0] == 'int']
        if arg:
            return max(arg, key=lambda x: x[1])[1] + 1
        else:
            return 0
       
            
    def addObsData(self, data: Data):
        """
        Add new observational dataset.

        Args:
            data (Data): new observational dataset.
        """
        id = ('obs', self.nextObs)
        self.DAGs[id] = self.DAG
        self.Ds[id] = data
        CP.info(f"\n## Building DBN for DAG ID {str(id)}")
        self.DBNs[id] = DynamicBayesianNetwork(self.DAG, data, self.nsample)
        return id
        
        
    def addIntData(self, target: str, data: Data):
        """
        Add new interventional dataset.

        Args:
            target (str): Intervention treatment variable.
            data (Data): Interventional data.
        """
        dag = copy.deepcopy(self.DAG)
        for s in self.DAG.g[target].sources:
            dag.del_source(target, s[0], s[1])
            
        id = ('int', str(target), self.nextInt)
        self.DAGs[id] = dag
        self.Ds[id] = data
        CP.info(f"\n## Building DBN for DAG ID {str(id)}")
        self.DBNs[id] = DynamicBayesianNetwork(dag, data, self.nsample)
        return id
    
    
    def save(self, respath):
        """
        Save a CausalInferenceEngine object from a pickle file.

        Args:
            respath (str): pickle save path.
        """
        pkl = dict()
        pkl['DAG'] = self.DAG
        pkl['nsample'] = self.nsample
        pkl['atol'] = self.atol
        pkl['verbosity'] = CP.verbosity
        pkl['DAGs'] = self.DAGs
        pkl['Ds'] = self.Ds
        pkl['DBNs'] = self.DBNs
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
        cie = cls(pkl['DAG'], pkl['nsample'], pkl['atol'], pkl['verbosity'])
        cie.DAGs = pkl['DAGs']
        cie.Ds = pkl['Ds']
        cie.DBNs = pkl['DBNs']
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
        otherDs = copy.deepcopy(self.Ds)
        otherDs.pop(targetP, None)
        intDs = {key: value for key, value in self.Ds.items() if key[0] == 'int' and key[1] == self.Q[TREATMENT]}
        # Remove the items in intDs from self.Ds
        for key in intDs.keys():
            otherDs.pop(key, None)
            
        intOcc, intSource = self._findSource(intDs)
        otherOcc, otherSource = self._findSource(otherDs)
        
        sourceP = intSource if intOcc != 0 else otherSource
        
        # Source population's p(output|do(treatment), adjustment)
        pS_y_do_x_adj = self.DBNs[sourceP].dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ]
        
        p_y_do_x = self.transport(pS_y_do_x_adj, targetP, self.Q[TREATMENT], self.Q[OUTCOME])
        
        y, p_y_do_X_x, E_p_y_do_X_x = self.evalDoDensity(p_y_do_x, sourceP)
        CP.info(f"## What happens to {self.Q[OUTCOME]} if {self.Q[TREATMENT]} = {str(self.Q[VALUE])} in population {str(targetP)} ? {self.Q[OUTCOME]} = {E_p_y_do_X_x}")
            
        return y, p_y_do_X_x, E_p_y_do_X_x
    
    
    def whatHappensNOTRANSPORT(self, outcome: str, treatment: str, value):
        """
        Calculate p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))].

        Args:
            outcome (str): outcome variable.
            treatment (str): treatment variable.
            value (float): treatment value.

        Returns:
            tuple: (outcome samples, p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))]).
        """
        self.Q[OUTCOME] = outcome
        self.Q[TREATMENT] = treatment
        self.Q[VALUE] = value
        
        CP.info("\n## Query")
        
        # searches the population with greatest number of occurrences treatment == treatment's value
        otherDs = copy.deepcopy(self.Ds)
        intDs = {key: value for key, value in self.Ds.items() if key[0] == 'int' and key[1] == self.Q[TREATMENT]}
        # Remove the items in intDs from self.Ds
        for key in intDs.keys():
            otherDs.pop(key, None)
            
        intOcc, intSource = self._findSource(intDs)
        otherOcc, otherSource = self._findSource(otherDs)
        
        sourceP = intSource if intOcc != 0 else otherSource
        
        # Source population's p(output|do(treatment), adjustment)
        pS_y_do_x_adj = self.DBNs[sourceP].dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ]
        
        y, p_y_do_X_x, E_p_y_do_X_x = self.evalDoDensity(pS_y_do_x_adj, sourceP)
        CP.info(f"## What happens to {self.Q[OUTCOME]} if {self.Q[TREATMENT]} = {str(self.Q[VALUE])} in population {str(targetP)} ? {self.Q[OUTCOME]} = {E_p_y_do_X_x}")
            
        return y, p_y_do_X_x, E_p_y_do_X_x
    
    
    def predict(self, outcome: str, treatment: str, value, data: Data):
        """
        Calculate p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))].

        Args:
            outcome (str): outcome variable.
            treatment (str): treatment variable.
            value (float/ndarray): treatment value.
            data (Data): target population ID (e.g., ("obs", 3)).

        Returns:
            tuple: (outcome samples, p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))]).
        """
        pass
        # # TODO: to do the whatHappens method for timeseries interventions
        # self.Q[OUTCOME] = outcome
        # self.Q[TREATMENT] = treatment
        # self.Q[VALUE] = value
              
        # # searches the population with greatest number of occurrences treatment == treatment's value
        # otherDs = copy.deepcopy(self.Ds)
        # otherDs.pop(targetP, None)
        # intDs = {key: value for key, value in self.Ds.items() if key[0] == 'int' and key[1] == self.Q[TREATMENT]}
        # # Remove the items in intDs from self.Ds
        # for key in intDs.keys():
        #     otherDs.pop(key, None)
            
        # intOcc, intSource = self._findSource(intDs)
        # otherOcc, otherSource = self._findSource(otherDs)
        
        # sourceP = intSource if intOcc != 0 else otherSource
        
        # p_y_do_x = self.transport(sourceP, targetP, self.Q[TREATMENT], self.Q[OUTCOME])
        
        # y, p_y_do_X_x, E_p_y_do_X_x = self.evalDoDensity(p_y_do_x, sourceP)
            
        # return y, p_y_do_X_x, E_p_y_do_X_x
    
    
    def _findSource(self, Ds):
        """
        finds source population with maximum number of occurrences treatment = value

        Args:
            Ds (dict): dataset dictionary {id (str): d (Data)}

        Returns:
            tuple: number of occurrences, source dataset
        """
        occurrences = 0
        for id, d in Ds.items():
            # TODO: not sure if here I need to take the d or the DBN samples
            indexes = np.where(np.isclose(d.d[self.Q[TREATMENT]], self.Q[VALUE], atol = self.atol))[0]
            if len(indexes) > occurrences: 
                occurrences = len(indexes)
                sourceP = id
                
        return occurrences, sourceP     
        
        
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
        # pS_y_do_x_adj = self.DBNs[sourceP].dbn[outcome].DO[treatment][P_Y_GIVEN_DOX_ADJ]
        
        # Compute the adjustment density for the target population
        pT_adj = np.ones((self.nsample, 1)).squeeze()
            
        for node in adjset: pT_adj = pT_adj * self.DBNs[targetP].dbn[self.DBNs[targetP].data.features[node[0]]].CondDensity
        pT_adj = Density.normalise(pT_adj)
        
        # Compute the p(outcome|do(treatment)) density
        if len(pS_y_do_x_adj.shape) > 2: 
            # Sum over the adjustment set
            p_y_do_x = Density.normalise(np.sum(pS_y_do_x_adj * pT_adj, axis = tuple(range(2, len(pS_y_do_x_adj.shape))))) #* np.sum(p_adj, axis=tuple(range(0, len(p_adj.shape))))
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
            tuple: outcome samples, p(outcome|do(treatment = t)), E[p(outcome|do(treatment = t))]
        """
        indices_X = np.where(np.isclose(self.DBNs[sourceP].dbn[self.Q[OUTCOME]].parents[self.Q[TREATMENT]].samples, self.Q[VALUE], atol = self.atol))[0]
        indices_X = np.array(sorted(indices_X))               
        
        # I am taking all the outcome's densities associated to the treatment == value
        # Normalise the density to ensure it sums to 1
        p_y_do_X_x = Density.normalise(np.sum(p_y_do_x[:, indices_X], axis = 1))
        E_p_y_do_X_x = Density.expectation(self.DBNs[sourceP].dbn[self.Q[OUTCOME]].y.samples, p_y_do_X_x)
        # self.plot_pE(self.DBNs[sourceP].dbn[self.Q[OUTCOME]].y.samples, p_y_do_X_x, E_p_y_do_X_x, show = True)
        return self.DBNs[sourceP].dbn[self.Q[OUTCOME]].y.samples, p_y_do_X_x, E_p_y_do_X_x
    
    
    def plot_pE(self, ysamples, density, expectation = None, show = False, path = None):
        plt.figure(figsize=(10, 6))
        plt.plot(ysamples, density, label='Density')
        if expectation is not None: plt.axvline(expectation, color='r', linestyle='--', label=f'Expectation = {str(round(expectation, 2))}')
        plt.xlabel(f'${self.Q[OUTCOME]}$')
        plt.ylabel(f'p(${self.Q[OUTCOME]}$|do(${self.Q[TREATMENT]}$ = {str(self.Q[VALUE])}))')
        plt.legend()
        if show: 
            plt.show()
        else:
            plt.savefig(os.path.join(path, f'p({self.Q[OUTCOME]}|do({self.Q[TREATMENT]} = {str(self.Q[VALUE])})).png'))