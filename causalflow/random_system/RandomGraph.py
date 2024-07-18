import copy
from enum import Enum
import math
import random
import numpy as np
from causalflow.basics.constants import ImageExt, LinkType
from causalflow.graph.DAG import DAG
from causalflow.graph.PAG import PAG
from causalflow.preprocessing.data import Data
import networkx as nx

NO_CYCLES_THRESHOLD = 10


class NoiseType(Enum):
    Uniform = 'uniform'
    Gaussian = 'gaussian'
    Weibull = 'weibull'
    
    
class PriorityOp(Enum):
    M = '*'
    D = '/'
    

class RandomGraph:
    def __init__(self, nvars, nsamples, link_density, coeff_range: tuple, 
                 min_lag, max_lag, max_exp = None, noise_config: tuple = None, 
                 operators = ['+', '-', '*'], 
                 functions = ['','sin', 'cos', 'exp', 'abs', 'pow'],
                 n_hidden_confounders = 0,
                 n_confounded_vars = None):
        """
        RandomDAG constructor

        Args:
            nvars (int): Number of variable
            nsamples (int): Number of samples
            link_density (int): Max number of parents per variable
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
               
        self.T = nsamples
        self.link_density = link_density
        self.coeff_range = coeff_range
        self.exponents = list(range(0, max_exp))
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.n_hidden_confounders = n_hidden_confounders
        self.n_confounded = n_confounded_vars
        
        self.obsVar = ['X_' + str(i) for i in range(nvars)]
        self.hiddenVar = ['H_' + str(i) for i in range(n_hidden_confounders)]
        self.operators = operators
        self.functions = functions
        self.equations = {var: list() for var in self.obsVar + self.hiddenVar}
        self.confounders = {h: list() for h in self.hiddenVar}
        # self.potentialIntervention = {h: {'type': None, 'vars': list()} for h in self.hiddenVar}
        self.dependency_graph = {var: set() for var in self.obsVar + self.hiddenVar}
        self.PAG = None

        self.noise_config = noise_config
        self.noise = None
        if noise_config is not None:
            if noise_config[0] is NoiseType.Uniform:
                self.noise = np.random.uniform(noise_config[1], noise_config[2], (self.T, self.N))
            elif noise_config[0] is NoiseType.Gaussian:
                self.noise = np.random.normal(noise_config[1], noise_config[2], (self.T, self.N))
            elif noise_config[0] is NoiseType.Weibull:
                self.noise = np.random.weibull(noise_config[1], (self.T, self.N)) * noise_config[2]
    
    
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
       
                
    def __build_equation(self, var_lagged_choice: list, var_contemp_choice: list, target_var):
        """
        Generates random equations

        Args:
            var_choice (list): list of possible parents for the target variable

        Returns:
            list: equation (list of tuple)
        """
        no_cycles_attempt = 0
        equation = []
        n_parents = random.randint(1, self.link_density)
        while len(equation) < n_parents:
            coefficient = random.uniform(self.coeff_range[0], self.coeff_range[1])
            lag = random.randint(self.min_lag, self.max_lag)
            if lag != 0:
                variable = random.choice(var_lagged_choice)
                var_lagged_choice.remove(variable)
            else:
                variable = random.choice(var_contemp_choice)
                var_contemp_choice.remove(variable)
                
            if not self.__creates_cycle((target_var, 0), (variable, lag)):
                operator = random.choice(self.operators)
                function = random.choice(self.functions)
                if function == 'pow':
                    exponent = random.choice(self.exponents)
                    term = (operator, coefficient, function, variable, lag, exponent)
                else:
                    term = (operator, coefficient, function, variable, lag)
                equation.append(term)
            else:
                no_cycles_attempt += 1
                if no_cycles_attempt >= NO_CYCLES_THRESHOLD:
                    raise ValueError("Cycle configuration impossible to be avoided!")
                
        return equation
    
    
    def __creates_cycle(self, target_var_lag, variable_lag):
        """
        Checks if adding an edge from variable_lag to target_var_lag would create a cycle
        considering only the same time lag
        """
        target_var, target_lag = target_var_lag
        variable, lag = variable_lag

        visited = set()
        stack = [(variable, lag, [(variable, lag)], lag - target_lag)]
        while stack:
            current_var, current_lag, path, initial_lag_diff = stack.pop()
            if (current_var, current_lag) == target_var_lag:
                print(f"Cycle path: {' -> '.join([f'{var} (lag {l})' for var, l in [target_var_lag] + path])}")
                return True
            if (current_var, current_lag) not in visited:
                visited.add((current_var, current_lag))
                for neighbor_var, neighbor_lag in self.dependency_graph.get(current_var, []):
                    if (neighbor_var, neighbor_lag) not in visited:
                        # Check if the lag difference is the same as the initial lag difference
                        if (neighbor_lag - current_lag) == initial_lag_diff:
                            stack.append((neighbor_var, neighbor_lag, path + [(neighbor_var, neighbor_lag)], initial_lag_diff))
        # Update dependency graph
        self.dependency_graph[target_var].add(variable_lag)
        return False


    def gen_equations(self):
        """
        Generates random equations using the operator and function lists provided in the constructor 
        """
        for var in self.obsVar:
            var_lagged_choice = copy.deepcopy(self.obsVar)
            var_contemp_choice = copy.deepcopy(var_lagged_choice)
            var_contemp_choice.remove(var)
            self.equations[var] = self.__build_equation(var_lagged_choice, var_contemp_choice, var)
            
        for hid in self.hiddenVar:
            var_lagged_choice = copy.deepcopy(self.obsVar + self.hiddenVar)
            var_contemp_choice = copy.deepcopy(var_lagged_choice)
            var_contemp_choice.remove(hid)
            self.equations[hid] = self.__build_equation(var_lagged_choice, var_contemp_choice, hid)
            
        self.__add_conf_links()
               
        
    def __add_conf_links(self):
        """
        Adds confounder links to a predefined causal model
        """
        no_cycles_attempt = 0
        self.expected_bidirected_links = list()
        firstvar_choice = copy.deepcopy(self.obsVar)
        for hid in self.hiddenVar:
            tmp_n_confounded = 0
            isContemporaneous = random.choice([True, False])
            n_confounded = random.randint(2, self.Nobs) if self.n_confounded is None else self.n_confounded 
            
            if isContemporaneous:
                # self.potentialIntervention[hid]['type'] = 'contemporaneous'
                lag = random.randint(self.min_lag, self.max_lag)
                confVar = list()
                while tmp_n_confounded < n_confounded:
                    variable = random.choice(firstvar_choice)
                    
                    if not self.__creates_cycle((variable, 0), (hid, lag)):
                        tmp_n_confounded += 1
                        firstvar_choice.remove(variable)
                        confVar.append(variable)
                                            
                        # self.potentialIntervention[hid]['vars'].append(variable)
                        
                        function = random.choice(self.functions)
                        coefficient = random.uniform(self.coeff_range[0], self.coeff_range[1])
                        operator = random.choice(self.operators)
                        if function == 'pow':
                            exponent = random.choice(self.exponents)
                            term = (operator, coefficient, function, hid, lag, exponent)
                        else:
                            term = (operator, coefficient, function, hid, lag)
                            
                        # NOTE: This is to remove the true link between confounded variable for ensuring 
                        # that the link due to the confounder is classified as spurious
                        if len(confVar) > 1:
                            for source in confVar:
                                tmp = copy.deepcopy(confVar)
                                tmp.remove(source)
                                for target in tmp:
                                    if (source, 0) in self.get_Adj()[target]:
                                        self.equations[target] = list(filter(lambda item: item[3] != source and item[3] != 0, self.equations[target]))
                        self.equations[variable].append(term)
                    
                        self.confounders[hid].append((variable, lag))
                    else:
                        no_cycles_attempt += 1
                        if no_cycles_attempt >= NO_CYCLES_THRESHOLD:
                            raise ValueError("Impossible to avoid the cycle configuration!")
                for source in confVar:
                    tmp = copy.deepcopy(confVar)
                    tmp.remove(source)
                    for target in tmp:
                        if not (source, 0) in self.get_Adj()[target]:
                            self.expected_bidirected_links.append({target: (source, 0)})
            else:    
                # self.potentialIntervention[hid]['type'] = 'lagged'  
                var_choice = copy.deepcopy(self.obsVar)
                firstConf = True
                source = None
                sourceLag = None
                targets = list()
                
                while tmp_n_confounded < n_confounded:
                    if firstConf:
                        variable = random.choice(firstvar_choice)
                        lag = random.randint(self.min_lag, self.max_lag - 1)
                        sourceLag = lag
                    else:
                        variable = random.choice(var_choice)
                        lag = random.randint(sourceLag + 1, self.max_lag)
                     
                    if not self.__creates_cycle((variable, 0), (hid, lag)):
                        tmp_n_confounded += 1
                        var_choice.remove(variable)
                        if firstConf:
                            firstvar_choice.remove(variable)
                            # self.potentialIntervention[hid]['vars'].append(variable)
                            firstConf = False
                            source = variable
                        else:
                            targets.append((variable, lag))
                            
                            # NOTE: This is to remove the true link between confounded variable for ensuring 
                            # that the link due to the confounder is classified as spurious
                            if (source, lag - sourceLag) in self.get_Adj()[variable]:
                                self.equations[variable] = list(filter(lambda item: item[3] != source and item[3] != lag - sourceLag, self.equations[variable]))
                        
                        function = random.choice(self.functions)
                        coefficient = random.uniform(self.coeff_range[0], self.coeff_range[1])
                        operator = random.choice(self.operators)
                        if function == 'pow':
                            exponent = random.choice(self.exponents)
                            term = (operator, coefficient, function, hid, lag, exponent)
                        else:
                            term = (operator, coefficient, function, hid, lag)
                        self.equations[variable].append(term)
                    
                        self.confounders[hid].append((variable, lag))
                    else:
                        no_cycles_attempt += 1
                        if no_cycles_attempt >= NO_CYCLES_THRESHOLD:
                            raise ValueError("Cycle configuration impossible to be avoided!")
                        
                # Lagged bidirected links
                for v in targets:
                    if not (source, v[1] - sourceLag) in self.get_Adj()[v[0]]:
                        self.expected_bidirected_links.append({v[0]: (source, v[1] - sourceLag)})
                # Contemporaneous bidirected links
                for source in targets:
                    tmp = copy.deepcopy(targets)
                    tmp.remove(source)
                    for target in tmp:
                        if not (source, 0) in self.get_Adj()[target[0]]:
                            self.expected_bidirected_links.append({target[0]: (source[0], 0)})


    def print_equations(self):
        """
        Prints the generated equations
        """
        toprint = list()
        for target, eq in self.equations.items():
            equation_str = target + '(t) = '
            for i, term in enumerate(eq):
                if len(term) == 6:
                    operator, coefficient, function, variable, lag, exponent = term
                    coefficient = round(coefficient, 2)
                    if i != 0: 
                        term_str = f"{operator} {coefficient} * {function}({variable}, {exponent})(t-{lag}) "
                    else:
                        term_str = f"{coefficient} * {function}({variable}, {exponent})(t-{lag}) "
                else:
                    operator, coefficient, function, variable, lag = term
                    coefficient = round(coefficient, 2)
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
            toprint.append(equation_str)
        eq = "\n".join(toprint)
        print(eq)
        return eq
            
            
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
            lag, exponent = args
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
        for t in range(self.T):
            if t < self.max_lag:
                for target, eq in self.equations.items():
                    np_data[t, self.variables.index(target)] = self.noise[t, self.variables.index(target)]
            else:
                for target, eq in self.equations.items():
                    np_data[t, self.variables.index(target)] = self.__evaluate_equation(eq, t, np_data)
                    if self.noise is not None: np_data[t, self.variables.index(target)] += self.noise[t, self.variables.index(target)]
                    
        data = Data(np_data, self.variables)
        data.shrink(self.obsVar)
        return data
    
    
    def gen_interv_ts(self, interventions, obs):
        """
        Generates time-series corresponding to intervention(s)

        Args:
            interventions (dict): dictionary {INT_VAR : {INT_LEN: int_len, INT_VAL: int_val}}

        Returns:
            Data: interventional time-series data
        """
        starting_point = obs.values
        int_data = dict()
        for int_var in interventions:
            T = int(interventions[int_var]["T"])
            if self.noise_config is not None:
                if self.noise_config[0] is NoiseType.Uniform:
                    int_noise = np.random.uniform(self.noise_config[1], self.noise_config[2], (T, self.N))
                elif self.noise_config[0] is NoiseType.Gaussian:
                    int_noise = np.random.normal(self.noise_config[1], self.noise_config[2], (T, self.N))
                elif self.noise_config[0] is NoiseType.Weibull:
                    int_noise= np.random.weibull(self.noise_config[1], (self.T, self.N)) * self.noise_config[2]
            np_data = np.zeros((T, self.N))
            np_data[0:self.max_lag, :] = starting_point[len(starting_point)-self.max_lag:,:]

            for t in range(self.max_lag, T):
                # if t < self.max_lag:
                #     for target, eq in self.equations.items():
                #         np_data[t, self.variables.index(target)] = self.noise[t, self.variables.index(target)]
                # else:
                for target, eq in self.equations.items():
                    if target != int_var:
                        np_data[t, self.variables.index(target)] = self.__evaluate_equation(eq, t, np_data)
                        if self.noise_config is not None: np_data[t, self.variables.index(target)] += int_noise[t, self.variables.index(target)]
                    else:
                        np_data[t, self.variables.index(target)] = interventions[int_var]["VAL"]
                        
            int_data[int_var] = Data(np_data, self.variables)
            int_data[int_var].shrink(self.obsVar)
            starting_point = np_data
        return int_data
    
    
    def get_DPAG(self):
        """
        Outputs the PAG starting from a DAG

        Returns:
            dict: scm
        """
        if self.PAG is None:
            scm = self.get_Adj(withHidden=True)
            self.PAG = PAG(scm, self.max_lag, self.hiddenVar)
        return self.PAG.convert2Graph()
    
    
    def get_Adj(self, withHidden = False):
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
                    _, _, _, variable, lag, _ = term
                else:
                    _, _, _, variable, lag = term
                if variable not in scm.keys(): continue # NOTE: this is needed to avoid adding hidden vars
                scm[target].append((variable, -abs(lag)))
        return scm
    
    
    def print_SCM(self, withHidden = False):
        """
        Prints the Structural Causal Model
        """
        scm = self.get_Adj(withHidden)
        for t in scm: print(t + ' : ' + str(scm[t]))    
          
        
    def intervene(self, int_var, int_len, int_value, obs):
        """
        Generates intervention on a single variable

        Args:
            int_var (str): variable name
            int_len (int): intervention length
            int_value (float): intervention value

        Returns:
            Data: interventional time-series data
        """
        return self.gen_interv_ts({int_var: {"T": int_len, "VAL": int_value}}, obs)
    
    
    def ts_dag(self, withHidden = False, save_name = None):
        """
        Draws a Time-seris DAG

        Args:
            withHidden (bool, optional): bit to decide whether to output the SCM including the hidden variables or not. Defaults to False.
            save_name (str, optional): figure path. Defaults to None.
        """
        gt = self.get_Adj(withHidden) if withHidden else self.get_DPAG()
        var = self.variables if withHidden else self.obsVar
        g = DAG(var, self.min_lag, self.max_lag, False, gt)
        
        node_color = 'orange'
        # node_c = ['tab:blue', 'tab:orange', 'tab:green']
        edge_color = 'grey'
        if withHidden:
            
            # Nodes color definition
            node_color = dict()
            tmpG = nx.grid_2d_graph(self.max_lag + 1, len(g.g.keys()))
            for n in tmpG.nodes():
                if var[abs(n[1] - (self.N - 1))] in self.hiddenVar:
                    node_color[n] = 'peachpuff'
                else:
                    # node_color[n] = node_c[abs(n[1] - (self.N - 1))]
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
                            
                        s_lag -= 1
                        t_lag -= 1
                        
        g.ts_dag(save_name = save_name, node_color = node_color, edge_color = edge_color, min_width=2, max_width=5, x_disp=0.5, node_size=6)

