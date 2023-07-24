import copy
from enum import Enum
import math
import random
import numpy as np
from fpcmci.graph.DAG import DAG
from fpcmci.preprocessing.data import Data
import networkx as nx


class NoiseType(Enum):
    Uniform = 0
    Gaussian = 1
    
    
class PriorityOp(Enum):
    M = '*'
    D = '/'
    

class RandomSystem:
    def __init__(self, nvars, nsamples, max_terms, coeff_range: tuple, 
                 min_lag, max_lag, max_exp = None, noise_config: tuple = None, 
                 operators = ['+', '-', '*'], 
                 functions = ['','sin', 'cos', 'exp', 'abs', 'pow'],
                 n_hidden_confounders = 0,
                 n_confounded = 0):
        """
        RandomSystem constructor

        Args:
            nvars (int): Number of variable
            nsamples (int): Number of samples
            max_terms (int): Max number of parents per variable
            coeff_range (tuple): Coefficient range. E.g. (-1, 1)
            min_lag (int): Min lagged dependency
            max_lag (int): Max lagged dependency
            max_exp (int): Max permitted exponent used by the 'pow' function. Used only if 'pow' is in the list of functions. Defaults to None.
            noise_config (tuple, optional): Noise configuration, e.g. (NoiseType.Uniform, -0.1, 0.1). Defaults to None.
            operators (list, optional): list of possible operators between variables. Defaults to ['+', '-', '*'].
            functions (list, optional): list of possible functions. Defaults to ['','sin', 'cos', 'exp', 'abs', 'pow'].
            n_hidden_confounders (int, optional): Number of hidden confounders. Defaults to 0.

        Raises:
            ValueError: max_exp cannot be None if functions list contains pow
        """
        
        if 'pow' in functions and max_exp is None:
            raise ValueError('max_exp cannot be None if functions list contains pow')
        
        # random.seed(random.randint(1, 2000))
        
        self.T = nsamples
        self.max_terms = max_terms
        self.coeff_range = coeff_range
        self.exponents = list(range(0, max_exp))
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.n_hidden_confounders = n_hidden_confounders
        self.n_confounded = n_confounded
        
        self.obsVar = ['X_' + str(i) for i in range(nvars)]
        self.hiddenVar = ['H_' + str(i) for i in range(n_hidden_confounders)]
        self.operators = operators
        self.functions = functions
        self.equations = {var: list() for var in self.obsVar + self.hiddenVar}
        self.confounders = {h: list() for h in self.hiddenVar}
        self.confintvar = dict()
        
        self.noise_config = noise_config
        self.noise = None
        if noise_config is not None:
            if noise_config[0] is NoiseType.Uniform:
                self.noise = np.random.uniform(noise_config[1], noise_config[2], (self.T, self.N))
            elif noise_config[1] is NoiseType.Gaussian:
                self.noise = np.random.normal(noise_config[1], noise_config[2], (self.T, self.N))
    
    
    @property            
    def variables(self):
        """
        Complete set of variables (observed and hidden)

        Returns:
            list: complete set of variables
        """
        return self.obsVar + self.hiddenVar 
    
    
    @property            
    def Nobs(self):
        """
        Number of observable variables

        Returns:
            int: number of observable variables
        """
        return len(self.obsVar) 
    
         
    @property            
    def N(self):
        """
        Total number of variables (observed and hidden)

        Returns:
            int: total number of variables
        """
        return len(self.obsVar) + len(self.hiddenVar)
    
    
    @property
    def obsEquations(self):
        """
        Equations corresponding to the observed variables

        Returns:
            dict: equations corresponding to the observed variables
        """
        tmp = copy.deepcopy(self.equations)
        for h in self.hiddenVar: del tmp[h]
        return tmp
    
                
    def __build_equation(self, var_choice: list):
        """
        Generates random equations

        Args:
            var_choice (list): list of possible parents for the target variable

        Returns:
            list: equation (list of tuple)
        """
        equation = []
        for _ in range(random.randint(1, self.max_terms)):
            coefficient = random.uniform(self.coeff_range[0], self.coeff_range[1])
            variable = random.choice(var_choice)
            var_choice.remove(variable)
            operator = random.choice(self.operators)
            lag = random.randint(self.min_lag, self.max_lag)
            function = random.choice(self.functions)
            if function == 'pow':
                exponent = random.choice(self.exponents)
                term = (operator, coefficient, function, variable, lag, exponent)
            else:
                term = (operator, coefficient, function, variable, lag)
            equation.append(term)
        return equation


    def gen_equations(self):
        """
        Generates random equations using the operator and function lists provided in the constructor 
        """
        for var in self.obsVar:
            var_choice = copy.deepcopy(self.obsVar)
            self.equations[var] = self.__build_equation(var_choice)
            
        for hid in self.hiddenVar:
            var_choice = copy.deepcopy(self.obsVar + self.hiddenVar)
            self.equations[hid] = self.__build_equation(var_choice)
            
        self.__add_conf_links()
        
        
    def __add_conf_links(self):
        """
        Adds confounder links to a predefined causal model
        """
        self.expected_spurious_links = list()
        firstvar_choice = copy.deepcopy(self.obsVar)
        for hid in self.hiddenVar:
            firstConf = True
            var_choice = copy.deepcopy(self.obsVar)
            n_confounded = random.randint(2, self.Nobs) if self.n_confounded == 0 else self.n_confounded 
            
            var_t1 = None
            var_t = list()
            for _ in range(n_confounded):
                coefficient = random.uniform(self.coeff_range[0], self.coeff_range[1])
                if firstConf:
                    variable = random.choice(firstvar_choice)
                    firstvar_choice.remove(variable)
                    var_choice.remove(variable)
                    
                    var_t1 = variable
                else:
                    variable = random.choice(var_choice)
                    var_choice.remove(variable)
                    var_t.append(variable)
                    
                    if (var_t1, -1) in self.get_SCM()[variable]:
                        self.equations[variable] = list(filter(lambda item: item[3] != var_t1 and item[3] != -1, self.equations[variable]))
                    
                operator = random.choice(self.operators)
                if firstConf:
                    lag = self.min_lag
                    self.confintvar[hid] = variable
                    firstConf = False
                else:
                    lag = random.randint(self.min_lag + 1, self.max_lag)
                
                function = random.choice(self.functions)
                if function == 'pow':
                    exponent = random.choice(self.exponents)
                    term = (operator, coefficient, function, hid, lag, exponent)
                else:
                    term = (operator, coefficient, function, hid, lag)
                self.equations[variable].append(term)
            
                self.confounders[hid].append((variable, lag))
            for v in var_t:
                if not (var_t1, -1) in self.get_SCM()[v]:
                    self.expected_spurious_links.append((var_t1, v))


    def print_equations(self):
        """
        Prints the generated equations
        """
        for target, eq in self.equations.items():
            equation_str = target + '(t) = '
            for i, term in enumerate(eq):
                if len(term) == 6:
                    operator, coefficient, function, variable, lag, exponent = term
                    if i != 0: 
                        term_str = f"{operator} {coefficient} * {function}({variable}, {exponent})(t-{lag}) "
                    else:
                        term_str = f"{coefficient} * {function}({variable}, {exponent})(t-{lag}) "
                else:
                    operator, coefficient, function, variable, lag = term
                    if function != '':
                        if i != 0: 
                            term_str = f"{operator} {coefficient} * {function}({variable})(t-{lag}) "
                        else:
                            term_str = f"{coefficient} * {function}({variable})(t-{lag}) "
                    else:
                        if i != 0: 
                            term_str = f"{operator} {coefficient} * {variable}(t-{lag}) "
                        else:
                            term_str = f"{coefficient} * {variable}(t-{lag}) "
                        
                equation_str += term_str
            print(equation_str)
            
            
    def __evaluate_term(self, term, t, data):
        """
        Evaluates single term componing an equation

        Args:
            term (tuple): term to evaluate
            t (int): time step
            data (numpy array): time-series

        Returns:
            tuple: operator and value of the term
        """
        operator, coefficient, function, variable, *args = term
        if function == '':
            lag = args[0]
            term_value = coefficient * (data[t - lag, self.variables.index(variable)])
        elif function == 'pow':
            exponent, lag = args
            term_value = coefficient * data[t - lag, self.variables.index(variable)] ** exponent
        elif function == 'abs':
            lag = args[0]
            term_value = coefficient * abs(data[t - lag, self.variables.index(variable)])
        else:
            lag = args[0]
            term_value = coefficient * getattr(math, function)(data[t - lag, self.variables.index(variable)])
        return operator, term_value
    
    
    def __handle_priority_operator(self, eq):
        """
        Evaluates all the terms with operato * ans /

        Args:
            eq (list): equation (list of term)

        Returns:
            list: equation with all * and / evaluated
        """
        op = '*'
        while (op in eq):
            op_i = eq.index(op)
            op1_i = op_i - 1
            op2_i = op_i + 1
            eq[op1_i] = eq[op1_i] * eq[op2_i]
                
            indices_set = set([op_i, op2_i])
            eq = [item for i, item in enumerate(eq) if i not in indices_set]
            
        op = '/'
        while (op in eq):
            op_i = eq.index(op)
            op1_i = op_i - 1
            op2_i = op_i + 1
            eq[op1_i] = eq[op1_i] / eq[op2_i]
                
            indices_set = set([op_i, op2_i])
            eq = [item for i, item in enumerate(eq) if i not in indices_set]
            
        return eq
     

    def __evaluate_equation(self, equation, t, data):
        """
        Evaluates equation

        Args:
            equation (list): equation (list of term)
            t (int): time step

        Returns:
            float: equation value
        """
        eq = list()
        for i, term in enumerate(equation):
            operator, term = self.__evaluate_term(term, t, data)
            if i == 0:
                eq.append(term)
            else:
                eq.append(operator)
                eq.append(term)
                
        # Handle * and / before + and -
        eq = self.__handle_priority_operator(eq)
        
        equation_value = eq.pop(0)
        for i in range(0, len(eq), 2):
            op = eq[i]
            term = eq[i+1]
            if op == '+': equation_value = equation_value + term
            elif op == '-': equation_value = equation_value - term
        return equation_value


    def gen_obs_ts(self):
        """
        Generates time-series data

        Returns:
            Data: generated data
        """
        np_data = np.zeros((self.T, self.N))
        for t in range(self.max_lag, self.T):
            for target, eq in self.equations.items():
                np_data[t, self.variables.index(target)] = self.__evaluate_equation(eq, t, np_data)
                if self.noise is not None: np_data[t, self.variables.index(target)] += self.noise[t, self.variables.index(target)]
                    
        data = Data(np_data, self.variables)
        data.shrink(self.obsVar)
        return data
    
    
    def gen_interv_ts(self, interventions):
        """
        Generates time-series corresponding to intervention(s)

        Args:
            interventions (dict): dictionary {INT_VAR : {INT_LEN: int_len, INT_VAL: int_val}}

        Returns:
            Data: interventional time-series data
        """
                
        int_data = dict()
        for int_var in interventions:
            T = int(interventions[int_var]["T"])
            if self.noise_config is not None:
                if self.noise_config[0] is NoiseType.Uniform:
                    int_noise = np.random.uniform(self.noise_config[1], self.noise_config[2], (T, self.N))
                elif self.noise_config[1] is NoiseType.Gaussian:
                    int_noise = np.random.normal(self.noise_config[1], self.noise_config[2], (T, self.N))
            np_data = np.zeros((T, self.N))
            for t in range(self.max_lag, T):
                for target, eq in self.equations.items():
                    if target != int_var:
                        np_data[t, self.variables.index(target)] = self.__evaluate_equation(eq, t, np_data)
                        if self.noise_config is not None: np_data[t, self.variables.index(target)] += int_noise[t, self.variables.index(target)]
                    else:
                        np_data[t, self.variables.index(target)] = interventions[int_var]["VAL"]
                        
            int_data[int_var] = Data(np_data, self.variables)
            int_data[int_var].shrink(self.obsVar)
        return int_data
    
    
    def get_SCM(self, withHidden = False):
        """
        Outputs the Structural Causal Model

        Returns:
            dict: scm
        """
        eqs = self.equations if withHidden else self.obsEquations
        scm = {target : list() for target in eqs.keys()}
        for target, eq in eqs.items():
            for term in eq:
                if len(term) == 6:
                    _, _, _, variable, _, lag = term
                else:
                    _, _, _, variable, lag = term
                if variable not in scm.keys(): continue # NOTE: this is needed to avoid adding hidden vars
                scm[target].append((variable, -abs(lag)))
        return scm
    
    
    def print_SCM(self):
        """
        Prints the Structural Causal Model
        """
        scm = self.get_SCM()
        for t in scm: print(t + ' : ' + str(scm[t]))    
          
        
    def intervene(self, int_var, int_len, int_value):
        """
        Generates intervention on a single variable

        Args:
            int_var (str): variable name
            int_len (int): intervention length
            int_value (float): intervention value

        Returns:
            Data: interventional time-series data
        """
        return self.gen_interv_ts({int_var: {"T": int_len, "VAL": int_value}})
    
    
    def ts_dag(self, withHidden = False, save_name = None):
        gt = self.get_SCM(withHidden)
        var = self.variables if withHidden else self.obsVar
        g = DAG(var, self.min_lag, self.max_lag, False, gt)
        
        node_color = 'orange'
        edge_color = 'grey'
        if withHidden:
            
            # Nodes color definition
            node_color = dict()
            tmpG = nx.grid_2d_graph(self.max_lag + 1, len(g.g.keys()))
            for n in tmpG.nodes():
                if var[abs(n[1] - (self.N - 1))] in self.hiddenVar:
                    node_color[n] = 'peachpuff'
                else:
                    node_color[n] = 'orange'
                    
            # Edges color definition
            edge_color = dict()
            for t in g.g:
                for s in g.g[t].sources:
                    s_index = len(g.g.keys())-1 - list(g.g.keys()).index(s[0])
                    t_index = len(g.g.keys())-1 - list(g.g.keys()).index(t)
                    
                    s_lag = self.max_lag - s[1]
                    t_lag = self.max_lag
                    while s_lag >= 0:
                        s_node = (s_lag, s_index)
                        t_node = (t_lag, t_index)
                        if s[0] in self.hiddenVar:
                            edge_color[(s_node, t_node)] = 'gainsboro'
                        else:
                            edge_color[(s_node, t_node)] = 'gray'
                            
                        s_lag -= s[1]
                        t_lag -= s[1]
                        
        g.ts_dag(self.max_lag, save_name = save_name, node_color = node_color, edge_color = edge_color)
        
        
    def get_TP(self, cm):
        """
        True positive number:
        edge present in the causal model 
        and present in the groundtruth

        Args:
            cm (dict): estimated SCM

        Returns:
            int: true positive
        """
        gt = self.get_SCM()
        counter = 0
        for node in cm.keys():
            for edge in cm[node]:
                if edge in gt[node]: counter += 1
        return counter


    def get_TN(self, cm):
        """
        True negative number:
        edge absent in the groundtruth 
        and absent in the causal model
        
        Args:
            cm (dict): estimated SCM

        Returns:
            int: true negative
        """
        fullg = DAG(self.obsVar, self.min_lag, self.max_lag, False)
        fullg.fully_connected_dag()
        fullscm = fullg.get_SCM()
        gt = self.get_SCM()
        gt_TN = copy.deepcopy(fullscm)
        
        # Build the True Negative graph [complementary graph of the ground-truth]
        for node in fullscm:
            for edge in fullscm[node]:
                if edge in gt[node]:
                    gt_TN[node].remove(edge)
                    
        counter = 0
        for node in gt_TN.keys():
            for edge in gt_TN[node]:
                if edge not in cm[node]: counter += 1
        return counter
    
    
    def get_FP(self, cm):
        """
        False positive number:
        edge present in the causal model 
        but absent in the groundtruth

        Args:
            cm (dict): estimated SCM

        Returns:
            int: false positive
        """
        gt = self.get_SCM()
        counter = 0
        for node in cm.keys():
            for edge in cm[node]:
                if edge not in gt[node]: counter += 1
        return counter


    def get_FN(self, cm):
        """
        False negative number:
        edge present in the groundtruth 
        but absent in the causal model
        
        Args:
            cm (dict): estimated SCM

        Returns:
            int: false negative
        """
        gt = self.get_SCM()
        counter = 0
        for node in gt.keys():
            for edge in gt[node]:
                if edge not in cm[node]: counter += 1
        return counter
    
    


    def shd(self, cm):
        """
        Computes Structural Hamming Distance between ground-truth causal graph and the estimated one

        Args:
            cm (dict): estimated SCM

        Returns:
            int: shd
        """
        fn = self.get_FN(cm)
        fp = self.get_FP(cm)
        return fn + fp


    def precision(self, cm):
        """
        Computes Precision between ground-truth causal graph and the estimated one

        Args:
            cm (dict): estimated SCM

        Returns:
            float: precision
        """
        tp = self.get_TP(cm)
        fp = self.get_FP(cm)
        if tp + fp == 0: return 0
        return tp/(tp + fp)

        
    def recall(self, cm):
        """
        Computes Recall between ground-truth causal graph and the estimated one

        Args:
            cm (dict): estimated SCM

        Returns:
            float: recall
        """
        tp = self.get_TP(cm)
        fn = self.get_FN(cm)
        if tp + fn == 0: return 0
        return tp/(tp + fn)


    def f1_score(self, cm):
        """
        Computes F1-score between ground-truth causal graph and the estimated one

        Args:
            cm (dict): estimated SCM

        Returns:
            float: f1-score
        """
        p = self.precision(cm)
        r = self.recall(cm)
        if p + r == 0: return 0
        return (2 * p * r) / (p + r)
    
    
    def FPR(self, cm):
        """
        Computes False Positve Rate between ground-truth causal graph and the estimated one

        Args:
            cm (dict): estimated SCM

        Returns:
            float: false positive rate
        """
        fp = self.get_FP(cm)
        tn = self.get_TN(cm)
        if tn + fp == 0: return 0
        return fp / (tn + fp)

