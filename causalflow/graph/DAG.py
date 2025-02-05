"""
This module provides the DAG class.

Classes:
    DAG: class for facilitating the handling and the creation of DAGs.
"""
    
import copy
from itertools import combinations
import pickle
import numpy as np
from causalflow.graph.Node import Node
from causalflow.basics.constants import *
from matplotlib import pyplot as plt
import networkx as nx
from causalflow.graph.netgraph import Graph
import re
from pgmpy.models import BayesianNetwork

class DAG():
    """DAG class."""
    
    def __init__(self, var_names, min_lag, max_lag, neglect_autodep = False, scm = None):
        """
        DAG constructor.

        Args:
            var_names (list): variable list.
            min_lag (int): minimum time lag.
            max_lag (int): maximum time lag.
            neglect_autodep (bool, optional): bit to neglect nodes when they are only autodependent. Defaults to False.
            scm (dict, optional): Build the DAG for SCM. Defaults to None.
        """
        self.g = {var: Node(var, neglect_autodep) for var in var_names}
        self.neglect_autodep = neglect_autodep
        self.sys_context = dict()
        self.min_lag = min_lag
        self.max_lag = max_lag
        
        if scm is not None:
            for t in scm:
                    for s in scm[t]: 
                        if len(s) == 2:
                            self.add_source(t, s[0], 0.3, 0, s[1])
                        elif len(s) == 3:
                            self.add_source(t, s[0], 0.3, 0, s[1], s[2])
                            
        self.dbn = None


    @property
    def features(self) -> list:
        """
        Return features list.

        Returns:
            list: Features list.
        """
        return list(self.g.keys())
    
    
    @property
    def pretty_features(self) -> list:
        """
        Return list of features with LaTeX symbols.
                
        Returns:
            list(str): list of feature names.
        """
        return [r'$' + str(v) + '$' for v in self.g.keys()]

    
    @property
    def autodep_nodes(self) -> list:
        """
        Return the autodependent nodes list.

        Returns:
            list: Autodependent nodes list.
        """
        autodeps = list()
        for t in self.g:
            # NOTE: I commented this because I want to check all the auto-dep nodes with obs data
            # if self.g[t].is_autodependent and self.g[t].intervention_node: autodeps.append(t)
            if self.g[t].is_autodependent: autodeps.append(t)
        return autodeps
    
    
    @property
    def interventions_links(self) -> list:
        """
        Return the intervention links list.

        Returns:
            list: Intervention link list.
        """
        int_links = list()
        for t in self.g:
            for s in self.g[t].sources:
                if self.g[s[0]].intervention_node:
                    int_links.append((s[0], s[1], t))
        return int_links
    
    
    @property
    def max_auto_score(self) -> float:
        """
        Return maximum score of an auto-dependency link.

        Returns:
            float: maximum score of an auto-dependency link.
        """
        return max([self.g[t].sources[self.g[t].get_max_autodependent][SCORE] for t in self.g if self.g[t].is_autodependent])
    
    
    @property
    def max_cross_score(self) -> float:
        """
        Return maximum score of an cross-dependency link.

        Returns:
            float: maximum score of an cross-dependency link.
        """
        return max([self.g[t].sources[s][SCORE] if self.g[t].sources[s][SCORE] != float('inf') else 1 for t in self.g for s in self.g[t].sources if t != s[0]])
      
      
    @classmethod
    def load(cls, pkl):
        """
        Load a DAG object from a pickle file.

        Args:
            pkl (pickle): pickle file.

        Returns:
            DAG: loaded DAG object.
        """
        if 'neglect_autodep' not in pkl: 
            cm = cls(list(pkl['causal_model'].features), pkl['causal_model'].min_lag, pkl['causal_model'].max_lag)
        else:
            cm = cls(list(pkl['causal_model'].features), pkl['causal_model'].min_lag, pkl['causal_model'].max_lag, pkl['neglect_autodep'])
            
        cm.g = pkl['causal_model'].g

        return cm
    

    def save(self, respath):
        """
        Save DAG object as pickle file at respath.

        Args:
            respath (str): path where to save the DAG object.
        """
        res = dict()
        res['causal_model'] = self
        res['var_names'] = self.features
        res['min_lag'] = self.min_lag
        res['max_lag'] = self.max_lag
        res['neglect_autodep'] = self.neglect_autodep
        with open(respath, 'wb') as resfile:
            pickle.dump(res, resfile)
    
    
    def filter_alpha(self, alpha):
        """
        Filter the causal model by a certain alpha level.

        Args:
            alpha (float): dependency p-value.
            
        Returns:
            DAG: filtered DAG.
        """
        cm = copy.deepcopy(self)
        for t in self.g:
            for s in self.g[t].sources:
                if self.g[t].sources[s][PVAL] > alpha:
                    cm.del_source(t, s[0], s[1])
        return cm        
        
        
    def add_source(self, t, s, score, pval, lag, mode = LinkType.Directed.value):
        """
        Add source node to a target node.

        Args:
            t (str): target node name.
            s (str): source node name.
            score (float): dependency score.
            pval (float): dependency p-value.
            lag (int): dependency lag.
            mode (LinkType): link type. E.g., Directed -->
        """
        self.g[t].sources[(s, abs(lag))] = {SCORE: score, PVAL: pval, TYPE: mode}
        if t not in self.g[s].children: self.g[s].children.append(t)
       
        
    def del_source(self, t, s, lag):
        """
        Remove source node from a target node.

        Args:
            t (str): target node name.
            s (str): source node name.
            lag (int): dependency lag.
        """
        del self.g[t].sources[(s, lag)]
        if t not in self.g[s].children: self.g[s].children.remove(t)
        
        
    def remove_unneeded_features(self):
        """Remove isolated nodes."""
        tmp = copy.deepcopy(self.g)
        for t in self.g.keys():
            if self.g[t].is_isolated: 
                if self.g[t].intervention_node: del tmp[self.g[t].associated_context]
                del tmp[t]
        self.g = tmp
                          
    
    def add_context(self):
        """Add context variables."""
        for sys_var, context_var in self.sys_context.items():
            if sys_var in self.features:
                
                # Adding context var to the graph
                self.g[context_var] = Node(context_var, self.neglect_autodep)
                
                # Adding context var to sys var
                self.g[sys_var].intervention_node = True
                self.g[sys_var].associated_context = context_var
                self.add_source(sys_var, context_var, 1, 0, 0)
                
        # NOTE: bi-directed link contemporanous link between context vars
        for sys_var, context_var in self.sys_context.items():
            if sys_var in self.features:
                other_context = [value for value in self.sys_context.values() if value != context_var and value in self.features]
                for other in other_context: self.add_source(context_var, other, 1, 0, 0)
        
                    
    def remove_context(self):
        """Remove context variables."""
        for sys_var, context_var in self.sys_context.items():
            if sys_var in self.g:
                # Removing context var from sys var
                # self.g[sys_var].intervention_node = False
                self.g[sys_var].associated_context = None
                self.del_source(sys_var, context_var, 0)
                    
                # Removing context var from dag
                del self.g[context_var]
                
                
    def get_anchestors(self, t, _anchestors = None, include_lag = False):
        """
        Return node ancestors.

        Args:
            t (str): node name.

        Returns:
            list: node ancestors.
        """
        if _anchestors is None: _anchestors = set()
        for s in self.g[t].sources:
            if not include_lag:
                if s[0] not in _anchestors:
                    _anchestors.add(s[0])
                    _anchestors.update(self.get_anchestors(s[0], _anchestors))
            else:
                if s not in _anchestors:
                    _anchestors.add(s)
                    _anchestors.update(self.get_anchestors(s[0], _anchestors, include_lag=True))
        return list(_anchestors)                                                                  
                
    def get_link_assumptions(self, autodep_ok = False) -> dict:
        """
        Return link assumption dictionary.

        Args:
            autodep_ok (bool, optional): If true, autodependecy link assumption = -->. Otherwise -?>. Defaults to False.

        Returns:
            dict: link assumption dictionary.
        """
        link_assump = {self.features.index(f): dict() for f in self.features}
        for t in self.g:
            for s in self.g[t].sources:
                if autodep_ok and s[0] == t: # NOTE: new condition added in order to not control twice the autodependency links
                    link_assump[self.features.index(t)][(self.features.index(s[0]), -abs(s[1]))] = '-->'
                    
                elif s[0] not in list(self.sys_context.values()):
                    if s[1] == 0 and (t, 0) in self.g[s[0]].sources:
                        link_assump[self.features.index(t)][(self.features.index(s[0]), 0)] = 'o-o'
                    elif s[1] == 0 and (t, 0) not in self.g[s[0]].sources:
                        link_assump[self.features.index(t)][(self.features.index(s[0]),0)] = '-?>'
                        link_assump[self.features.index(s[0])][(self.features.index(t), 0)] = '<?-'
                    elif s[1] > 0:
                        link_assump[self.features.index(t)][(self.features.index(s[0]), -abs(s[1]))] = '-?>'
                    
                elif t in self.sys_context.keys() and s[0] == self.sys_context[t]:
                    link_assump[self.features.index(t)][(self.features.index(s[0]), -abs(s[1]))] = '-->'
                    
        return link_assump
   
    @staticmethod
    def prettify(name: str):
        """
        Turn a string in LaTeX-style.

        Args:
            name (str): string to convert.

        Returns:
            str: converted string.
        """
        # Check if the name is already in LaTeX-style format
        if name.startswith('$') and name.endswith('$') and re.search(r'_\{\w+\}', name):
            return name
        return '$' + re.sub(r'_(\w+)', r'_{\1}', name) + '$'
        
    
    def make_pretty(self) -> dict:
        """
        Make variables' names pretty, i.e. $ varname $ with '{' after '_' and '}' at the end of the string.

        Returns:
            dict: pretty DAG.
        """
        pretty = dict()
        for t in self.g:
            p_t = DAG.prettify(t)
            pretty[p_t] = copy.deepcopy(self.g[t])
            pretty[p_t].name = p_t
            pretty[p_t].children = [DAG.prettify(c) for c in self.g[t].children]
            for s in self.g[t].sources:
                del pretty[p_t].sources[s]
                p_s = DAG.prettify(s[0])
                pretty[p_t].sources[(p_s, s[1])] = {
                    SCORE: self.g[t].sources[s][SCORE],
                    PVAL: self.g[t].sources[s][PVAL],
                    TYPE: self.g[t].sources[s][TYPE]
                }
        return pretty
    
    
    def __add_edge(self, min_width, max_width, min_score, max_score, edges, edge_width, arrows, r, t, s, s_node, t_node):
        """
        Add edge to a graph. Support method for dag and ts_dag.

        Args:
            min_width (int): minimum linewidth. Defaults to 1.
            max_width (int): maximum linewidth. Defaults to 5.
            min_score (int): minimum score range. Defaults to 0.
            max_score (int): maximum score range. Defaults to 1.
            edges (list): list of edges.
            edge_width (dict): dictionary containing the width for each edge of the graph.
            arrows (dict): dictionary specifying the head and tail edge markers. E.g., {'h':'>', 't':'o'}.
            r (DAG): DAG.
            t (str or tuple): target node.
            s (str or tuple): source node.
            s_node (str): source node.
            t_node (str): target node.

        Raises:
            ValueError: link type associated to this edge not included in our LinkType list.
        """
        edges.append((s_node, t_node))
        score = r.g[t].sources[s][SCORE] if r.g[t].sources[s][SCORE] != float('inf') else 1
        edge_width[(s_node, t_node)] = DAG.__scale(score, min_width, max_width, min_score, max_score)
        
        if r.g[t].sources[s][TYPE] == LinkType.Directed.value:
            arrows[(s_node, t_node)] = {'h':'>', 't':''}
            
        elif r.g[t].sources[s][TYPE] == LinkType.Bidirected.value:
            edges.append((t_node, s_node))
            edge_width[(t_node, s_node)] = DAG.__scale(score, min_width, max_width, min_score, max_score)
            arrows[(t_node, s_node)] = {'h':'>', 't':''}
            arrows[(s_node, t_node)] = {'h':'>', 't':''}
            
        elif r.g[t].sources[s][TYPE] == LinkType.HalfUncertain.value:
            arrows[(s_node, t_node)] = {'h':'>', 't':'o'}
            
        elif r.g[t].sources[s][TYPE] == LinkType.Uncertain.value:
            arrows[(s_node, t_node)] = {'h':'o', 't':'o'}
        
        else:
            raise ValueError(f"{r.g[t].sources[s][TYPE]} not included in LinkType")
             
    
    def dag(self,
        node_layout='dot',
        min_auto_width=0.25, 
        max_auto_width=0.75,
        min_cross_width=1, 
        max_cross_width=5,
        node_size=8, 
        node_color='orange',
        edge_color='grey',
        tail_color='black',
        font_size=8,
        label_type=LabelType.Lag,
        save_name=None,
        img_extention=ImageExt.PNG):
        """
        Build a dag, first with contemporaneous links, then lagged links.

        Args:
            node_layout (str, optional): Node layout. Defaults to 'dot'.
            min_auto_width (float, optional): minimum border linewidth. Defaults to 0.25.
            max_auto_width (float, optional): maximum border linewidth. Defaults to 0.75.
            min_cross_width (float, optional): minimum edge linewidth. Defaults to 1.
            max_cross_width (float, optional): maximum edge linewidth. Defaults to 5.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str/dict, optional): node color. 
                                             If a string, all the nodes will have the same colour. 
                                             If a dict, each node will have its specified colour.
                                             Defaults to 'orange'.
            edge_color (str, optional): edge color for contemporaneous links. Defaults to 'grey'.
            tail_color (str, optional): tail color. Defaults to 'black'.
            font_size (int, optional): font size. Defaults to 8.
            label_type (LabelType, optional): Show the lag time (LabelType.Lag), the strength (LabelType.Score), or no labels (LabelType.NoLabels). Default LabelType.Lag.
            save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
            img_extention (ImageExt, optional): Image Extension. Defaults to PNG.
        """
        r = copy.deepcopy(self)
        r.g = r.make_pretty()
        node_color = copy.deepcopy(node_color)
        if isinstance(node_color, dict): node_color = {DAG.prettify(f): node_color.pop(f) for f in list(node_color)}

        Gcont = nx.DiGraph()
        Glag = nx.DiGraph()

        # 1. Nodes definition
        Gcont.add_nodes_from(r.g.keys())
        Glag.add_nodes_from(r.g.keys())
        
        # 2. Nodes border definition
        border = dict()
        for t in r.g:
            border[t] = 0
            if r.g[t].is_autodependent:
                border[t] = max(DAG.__scale(r.g[t].sources[r.g[t].get_max_autodependent][SCORE], 
                                             min_auto_width, max_auto_width, 
                                             0, r.max_auto_score), 
                                border[t])
        
        # 3. Nodes border label definition
        node_label = None
        if label_type == LabelType.Lag or label_type == LabelType.Score:
            node_label = {t: [] for t in r.g.keys()}
            for t in r.g:
                if r.g[t].is_autodependent:
                    for s in r.g[t].sources:
                        if s[0] == t:
                            if label_type == LabelType.Lag:
                                node_label[t].append(s[1])
                            elif label_type == LabelType.Score:
                                node_label[t].append(round(r.g[t].sources[s][SCORE], 3))
                node_label[t] = ",".join(str(s) for s in node_label[t])

        # 3. Edges definition
        cont_edges = []
        cont_edge_width = dict()
        cont_arrows = {}
        lagged_edges = []
        lagged_edge_width = dict()
        lagged_arrows = {}
        
        for t in r.g:
            for s in r.g[t].sources:
                if t != s[0]:  # skip self-loops
                    if s[1] == 0:  # Contemporaneous link (no lag)
                        self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score,
                                        cont_edges, cont_edge_width, cont_arrows, r, t, s, s[0], t)
                    else:  # Lagged link
                        self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score,
                                        lagged_edges, lagged_edge_width, lagged_arrows, r, t, s, s[0], t)
                        
        Gcont.add_edges_from(cont_edges)
        Glag.add_edges_from(lagged_edges)


        fig, ax = plt.subplots(figsize=(8, 6))

        # 4. Edges label definition
        cont_edge_label = None
        lagged_edge_label = None
        if label_type == LabelType.Lag or label_type == LabelType.Score:
            cont_edge_label = {(s[0], t): [] for t in r.g for s in r.g[t].sources if t != s[0] and s[1] == 0}
            lagged_edge_label = {(s[0], t): [] for t in r.g for s in r.g[t].sources if t != s[0] and s[1] != 0}
            for t in r.g:
                for s in r.g[t].sources:
                    if t != s[0]:
                        if s[1] == 0:  # Contemporaneous
                            if label_type == LabelType.Lag:
                                cont_edge_label[(s[0], t)].append(s[1])
                            elif label_type == LabelType.Score:
                                cont_edge_label[(s[0], t)].append(round(r.g[t].sources[s][SCORE], 3))
                        else:  # Lagged
                            if label_type == LabelType.Lag:
                                lagged_edge_label[(s[0], t)].append(s[1])
                            elif label_type == LabelType.Score:
                                lagged_edge_label[(s[0], t)].append(round(r.g[t].sources[s][SCORE], 3))
            for k in cont_edge_label.keys():
                cont_edge_label[k] = ",".join(str(s) for s in cont_edge_label[k])
            for k in lagged_edge_label.keys():
                lagged_edge_label[k] = ",".join(str(s) for s in lagged_edge_label[k])

        # 5. Draw graph - contemporaneous
        if cont_edges:
            # node_layout = {'$WP$': np.array([0.05, 0.95]), 
            #                '$TOD$': np.array([0.45, 0.95  ]), 
            #                '$OBS$': np.array([0.64375, 0.95   ]), 
            #                '$C_{S}$': np.array([0.88125, 0.95   ]), 
            #                '$PD$': np.array([0.45, 0.475 ]), 
            #                '$R_{V}$': np.array([0.7625, 0.475 ]), 
            #                '$R_{B}$': np.array([1.   , 0.475]), 
            #                '$ELT$': np.array([ 0.525, -0.   ])}
            a = Graph(Gcont,
                    node_layout=node_layout,
                    node_size=node_size,
                    node_color=node_color,
                    node_labels=None,
                    node_edge_width=border,
                    node_label_fontdict=dict(size=font_size),
                    node_edge_color=edge_color,
                    node_label_offset=0.05,
                    node_alpha=1,

                    arrows=cont_arrows,
                    edge_layout='straight',
                    edge_label=label_type != LabelType.NoLabels,
                    edge_labels=cont_edge_label,
                    edge_label_fontdict=dict(size=font_size),
                    edge_color=edge_color,
                    tail_color=tail_color,
                    edge_width=cont_edge_width,
                    edge_alpha=1,
                    edge_zorder=1,
                    edge_label_position=0.35)

            nx.draw_networkx_labels(Gcont,
                                    pos=a.node_positions,
                                    labels={n: n for n in Gcont},
                                    font_size=font_size)

        # 6. Draw graph - lagged
        if lagged_edges:
            a = Graph(Glag,
                    node_layout=a.node_positions if cont_edges else node_layout,
                    node_size=node_size,
                    node_color=node_color,
                    node_labels=node_label,
                    node_edge_width=border,
                    node_label_fontdict=dict(size=font_size),
                    node_edge_color=edge_color,
                    node_label_offset=0.05,
                    node_alpha=1,

                    arrows=lagged_arrows,
                    # edge_layout='straight',
                    edge_layout='curved',
                    edge_label=label_type != LabelType.NoLabels,
                    edge_labels=lagged_edge_label,
                    edge_label_fontdict=dict(size=font_size),
                    edge_color=edge_color,
                    tail_color=tail_color,
                    edge_width=lagged_edge_width,
                    edge_alpha=1,
                    edge_zorder=1,
                    edge_label_position=0.35)
            
            if not cont_edges:
                nx.draw_networkx_labels(Glag,
                                        pos=a.node_positions,
                                        labels={n: n for n in Glag},
                                        font_size=font_size)

        # 7. Plot or save
        if save_name is not None:
            plt.savefig(save_name + img_extention.value, dpi=300)
        else:
            plt.show()
          
   
    def ts_dag(self,
               min_cross_width = 1, 
               max_cross_width = 5,
               node_size = 8,
               x_disp = 1.5,
               y_disp = 0.2,
               node_color = 'orange',
               edge_color = 'grey',
               tail_color = 'black',
               font_size = 8,
               save_name = None,
               img_extention = ImageExt.PNG):
        """
        Build a timeseries dag.

        Args:
            min_cross_width (int, optional): minimum linewidth. Defaults to 1.
            max_cross_width (int, optional): maximum linewidth. Defaults to 5.
            node_size (int, optional): node size. Defaults to 8.
            x_disp (float, optional): node displacement along x. Defaults to 1.5.
            y_disp (float, optional): node displacement along y. Defaults to 0.2.
            node_color (str/dict, optional): node color. 
                                             If a string, all the nodes will have the same colour. 
                                             If a dict, each node will have its specified colour.
                                             Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            tail_color (str, optional): tail color. Defaults to 'black'.
            font_size (int, optional): font size. Defaults to 8.
            save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
            img_extention (ImageExt, optional): Image Extension. Defaults to PNG.
        """
        r = copy.deepcopy(self)
        r.g = r.make_pretty()

        Gcont = nx.DiGraph()
        Glagcross = nx.DiGraph()
        Glagauto = nx.DiGraph()

        # 1. Nodes definition
        if isinstance(node_color, dict):
            node_c = dict()
        else:
            node_c = node_color
        for i in range(len(self.features)):
            for j in range(self.max_lag + 1):
                Glagauto.add_node((j, i))
                Glagcross.add_node((j, i))
                Gcont.add_node((j, i))
                if isinstance(node_color, dict): node_c[(j, i)] = node_color[self.features[abs(i - (len(r.g.keys()) - 1))]]
                
        pos = {n : (n[0]*x_disp, n[1]*y_disp) for n in Glagauto.nodes()}
        scale = max(pos.values())

        # 2. Edges definition
        cont_edges = list()
        cont_edge_width = dict()
        cont_arrows = dict()
        lagged_cross_edges = list()
        lagged_cross_edge_width = dict()
        lagged_cross_arrows = dict()
        lagged_auto_edges = list()
        lagged_auto_edge_width = dict()
        lagged_auto_arrows = dict()

        for t in r.g:
            for s in r.g[t].sources:
                s_index = len(r.g.keys())-1 - list(r.g.keys()).index(s[0])
                t_index = len(r.g.keys())-1 - list(r.g.keys()).index(t)
                
                # 2.1. Contemporaneous edges definition
                if s[1] == 0:
                    for i in range(self.max_lag + 1):
                        s_node = (i, s_index)
                        t_node = (i, t_index)
                        self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score, 
                                        cont_edges, cont_edge_width, cont_arrows, r, t, s, 
                                        s_node, t_node)
                        
                else:
                    s_lag = self.max_lag - s[1]
                    t_lag = self.max_lag
                    while s_lag >= 0:
                        s_node = (s_lag, s_index)
                        t_node = (t_lag, t_index)
                        # 2.2. Lagged cross edges definition
                        if s[0] != t:
                            self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score, 
                                            lagged_cross_edges, lagged_cross_edge_width, lagged_cross_arrows, r, t, s, 
                                            s_node, t_node)
                        # 2.3. Lagged auto edges definition
                        else:
                            self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score, 
                                            lagged_auto_edges, lagged_auto_edge_width, lagged_auto_arrows, r, t, s, 
                                            s_node, t_node)
                        s_lag -= 1
                        t_lag -= 1
                    
        Gcont.add_edges_from(cont_edges)
        Glagcross.add_edges_from(lagged_cross_edges)
        Glagauto.add_edges_from(lagged_auto_edges)

        fig, ax = plt.subplots(figsize=(8,6))
        edge_layout = self.__get_fixed_edges(ax, x_disp, Gcont, node_size, pos, node_c, font_size, 
                                             cont_arrows, edge_color, tail_color, cont_edge_width, scale)
        
        # 3. Label definition
        for n in Gcont.nodes():
            if n[0] == 0:
                ax.text(pos[n][0]-0.1, pos[n][1], list(r.g.keys())[len(r.g.keys()) - 1 - n[1]], horizontalalignment='center', verticalalignment='center', fontsize=font_size)

        # 4. Time line text drawing
        pos_tau = set([pos[p][0] for p in pos])
        max_y = max([pos[p][1] for p in pos])
        for p in pos_tau:
            if abs(int(p/x_disp) - self.max_lag) == 0:
                ax.text(p, max_y + 0.1, r"$t$", horizontalalignment='center', fontsize=font_size)
            else:
                ax.text(p, max_y + 0.1, r"$t-" + str(abs(int(p/x_disp) - self.max_lag)) + "$", horizontalalignment='center', fontsize=font_size)
        
        # 5. Draw graph - contemporaneous
        if cont_edges:
            a = Graph(Gcont,
                    node_layout={p : np.array(pos[p]) for p in pos},
                    node_size=node_size,
                    node_color=node_c,
                    node_edge_width=0,
                    node_label_fontdict=dict(size=font_size),
                    node_label_offset=0,
                    node_alpha=1,

                    arrows=cont_arrows,
                    edge_layout=edge_layout,
                    edge_label=False,
                    edge_color=edge_color,
                    tail_color=tail_color,
                    edge_width=cont_edge_width,
                    edge_alpha=1,
                    edge_zorder=1,
                    scale = (scale[0] + 2, scale[1] + 2))

        # 6. Draw graph - lagged cross
        if lagged_cross_edges:
            a = Graph(Glagcross,
                    node_layout={p : np.array(pos[p]) for p in pos},
                    node_size=node_size,
                    node_color=node_c,
                    node_edge_width=0,
                    node_label_fontdict=dict(size=font_size),
                    node_label_offset=0,
                    node_alpha=1,

                    arrows=lagged_cross_arrows,
                    edge_layout='curved',
                    edge_label=False,
                    edge_color=edge_color,
                    tail_color=tail_color,
                    edge_width=lagged_cross_edge_width,
                    edge_alpha=1,
                    edge_zorder=1,
                    scale = (scale[0] + 2, scale[1] + 2))
            
        # 7. Draw graph - lagged auto
        if lagged_auto_edges:
            a = Graph(Glagauto,
                    node_layout={p : np.array(pos[p]) for p in pos},
                    node_size=node_size,
                    node_color=node_c,
                    node_edge_width=0,
                    node_label_fontdict=dict(size=font_size),
                    node_label_offset=0,
                    node_alpha=1,

                    arrows=lagged_auto_arrows,
                    edge_layout='straight',
                    edge_label=False,
                    edge_color=edge_color,
                    tail_color=tail_color,
                    edge_width=lagged_auto_edge_width,
                    edge_alpha=1,
                    edge_zorder=1,
                    scale = (scale[0] + 2, scale[1] + 2))
        
        # 7. Plot or save
        if save_name is not None:
            plt.savefig(save_name + img_extention.value, dpi = 300)
        else:
            plt.show()
            
            
    def __get_fixed_edges(self, ax, x_disp, Gcont, node_size, pos, node_c, font_size, cont_arrows, edge_color, tail_color, cont_edge_width, scale) -> dict:
        """
        Fix edge paths at t-tau_max.

        Args:
            ax (Axes): figure axis.
            x_disp (float): node displacement along x. Defaults to 1.5.
            Gcont (DiGraph): Direct Graph containing only contemporaneous links.
            node_size (int): node size.
            pos (dict): node layout.
            node_c (str/list, optional): node color. 
                                         If a string, all the nodes will have the same colour. 
                                         If a list (same dimension of features), each colour will have the specified colour.
            font_size (int): font size.
            cont_arrows (dict): edge-arrows dictionary .
            edge_color (str): edge color.
            tail_color (str): tail color.
            cont_edge_width (dict): edge-width dictionary.
            scale (tuple): graph scale.

        Returns:
            dict: new edge paths
        """
        a = Graph(Gcont,
                  node_layout={p : np.array(pos[p]) for p in pos},
                  node_size=node_size,
                  node_color=node_c,
                  node_edge_width=0,
                  node_label_fontdict=dict(size=font_size),
                  node_label_offset=0,
                  node_alpha=1,

                  arrows=cont_arrows,
                  edge_layout='curved',
                  edge_label=False,
                  edge_color=edge_color,
                  tail_color=tail_color,
                  edge_width=cont_edge_width,
                  edge_alpha=1,
                  edge_zorder=1,
                  scale = (scale[0] + 2, scale[1] + 2))
        res = copy.deepcopy(a.edge_layout.edge_paths)
        for edge, edge_path in a.edge_layout.edge_paths.items():
            if edge[0][0] == self.max_lag and edge[1][0] == self.max_lag: # t
                for t in range(0, self.max_lag):
                    for fixed_edge in a.edge_layout.edge_paths.keys():
                        if fixed_edge == edge: continue
                        if fixed_edge[0][0] == t and fixed_edge[0][1] == edge[0][1] and fixed_edge[1][0] == t and fixed_edge[1][1] == edge[1][1]:
                            res[fixed_edge] = edge_path - np.array([(self.max_lag - t)*x_disp,0])*np.ones_like(a.edge_layout.edge_paths[edge])
            # if edge[0][0] == 0 and edge[1][0] == 0: # t-tau_max
            #     for shifted_edge, shifted_edge_path in a.edge_layout.edge_paths.items():
            #         if shifted_edge == edge: continue
            #         if shifted_edge[0][0] == self.max_lag and shifted_edge[0][1] == edge[0][1] and shifted_edge[1][0] == self.max_lag and shifted_edge[1][1] == edge[1][1]:
            #             res[edge] = shifted_edge_path - np.array([x_disp,0])*np.ones_like(a.edge_layout.edge_paths[shifted_edge])
        ax.clear()              
        return res
    
    @staticmethod
    def __scale(score, min_width, max_width, min_score = 0, max_score = 1):
        """
        Scale the score of the cause-effect relationship strength to a linewitdth.

        Args:
            score (float): score to scale.
            min_width (float): minimum linewidth.
            max_width (float): maximum linewidth.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.

        Returns:
            (float): scaled score.
        """
        return ((score - min_score) / (max_score - min_score)) * (max_width - min_width) + min_width


    def get_skeleton(self) -> np.array:
        """
        Return skeleton matrix.
        
        Skeleton matrix is composed by 0 and 1.
        1 <- if there is a link from source to target 
        0 <- if there is not a link from source to target 

        Returns:
            np.array: skeleton matrix
        """
        r = np.full((len(self.features), len(self.features), self.max_lag + 1), '', dtype=object)
        for t in self.g.keys():
            for s in self.g[t].sources:
                r[self.features.index(t), self.features.index(s[0])][s[1]] = 1
        return np.array(r)
    
    
    def get_val_matrix(self) -> np.array:
        """
        Return val matrix.
        
        Val matrix contains information about the strength of the links componing the causal model.

        Returns:
            np.array: val matrix.
        """
        r = np.zeros((len(self.features), len(self.features), self.max_lag + 1))
        for t in self.g.keys():
            for s, info in self.g[t].sources.items():
                    r[self.features.index(t), self.features.index(s[0])][s[1]] = info[SCORE]
        return np.array(r)


    def get_pval_matrix(self) -> np.array:
        """
        Return pval matrix.
        
        Pval matrix contains information about the pval of the links componing the causal model.
        
        Returns:
            np.array: pval matrix
        """
        r = np.zeros((len(self.features), len(self.features), self.max_lag + 1))
        for t in self.g.keys():
            for s, info in self.g[t].sources.items():
                r[self.features.index(t), self.features.index(s[0])][s[1]] = info[PVAL]
        return np.array(r)
    
    
    def get_graph_matrix(self) -> np.array:
        """
        Return graph matrix.
        
        Graph matrix contains information about the link type. E.g., -->, <->, ..
        
        Returns:
            np.array: graph matrix.
        """
        r = np.full((len(self.features), len(self.features), self.max_lag + 1), '', dtype=object)
        for t in self.g.keys():
            for s, info in self.g[t].sources.items():
                r[self.features.index(t), self.features.index(s[0])][s[1]] = info[TYPE]
        return np.array(r)
    
    
    def get_Adj(self, indexed = False) -> dict:   
        """
        Return Adjacency dictionary.
        
        If indexed = True: example {0: [(0, -1), (1, -2)], 1: [], ...}
        If indexed = False: example {"X_0": [(X_0, -1), (X_1, -2)], "X_1": [], ...}
        
        Args:
            indexed (bool, optional): If true, returns the SCM with index instead of variables' names. Otherwise it uses variables' names. Defaults to False.
        
        Returns:
            dict: SCM.
        """
        if not indexed:
            scm = {v: list() for v in self.features}
            for t in self.g:
                for s in self.g[t].sources:
                    scm[t].append((s[0], -abs(s[1]))) 
        else:
            scm = {self.features.index(v): list() for v in self.features}
            for t in self.g:
                for s in self.g[t].sources:
                    scm[self.features.index(t)].append((self.features.index(s[0]), -abs(s[1]))) 
        return scm
    
    
    def get_Graph(self) -> dict:
        """
        Return Graph dictionary. E.g. {X1: {(X2, -2): '-->'}, X2: {(X3, -1): '-?>'}, X3: {(X3, -1): '-->'}}.

        Returns:
            dict: graph dictionary.
        """
        scm = {v: dict() for v in self.features}
        for t in self.g:
            for s in self.g[t].sources:
                scm[t][(s[0], -abs(s[1]))] = self.g[t].sources[s][TYPE] 
        return scm
    
    
    def DAG2NX(self) -> nx.DiGraph:
        G = nx.DiGraph()

        # 1. Nodes definition
        for i in range(len(self.features)):
            for j in range(self.max_lag, -1, -1):
                G.add_node((i, -j))

        # 2. Edges definition
        edges = list()
        for t in self.g:
            for s in self.g[t].sources:
                s_index = self.features.index(s[0])
                t_index = self.features.index(t)
                
                if s[1] == 0:
                    for j in range(self.max_lag, -1, -1):
                        s_node = (s_index, -j)
                        t_node = (t_index, -j)
                        edges.append((s_node, t_node))
                        
                else:
                    s_lag = -s[1]
                    t_lag = 0
                    while s_lag >= -self.max_lag:
                        s_node = (s_index, s_lag)
                        t_node = (t_index, t_lag)
                        edges.append((s_node, t_node))
                        s_lag -= 1
                        t_lag -= 1
                    
        G.add_edges_from(edges)        
        return G
    
    
    def get_topological_order(self) -> list:
        return [(self.features[node[0]], node[1]) for node in list(nx.topological_sort(self.DAG2NX()))]
            
    
    
    
    
    
    
    
    
    
    
    @staticmethod
    def get_DBN(link_assumptions, tau_max) -> BayesianNetwork:
        """
        Create a DAG represented by a Baysian Network.

        Args:
            link_assumptions (dict): DAG link assumptions.
            tau_max (int): max time lag.

        Raises:
            ValueError: source not well defined.

        Returns:
            BayesianNetwork: DAG represented by a Baysian Network.
        """
        DBN = BayesianNetwork()
        DBN.add_nodes_from([(t, -l) for t in link_assumptions.keys() for l in range(0, tau_max + 1)])

        # Edges
        edges = []
        for t in link_assumptions.keys():
            for source in link_assumptions[t]:
                if len(source) == 0: continue
                elif len(source) == 2: s, l = source
                elif len(source) == 3: s, l, _ = source
                else: raise ValueError("Source not well defined")
                edges.append(((s, l), (t, 0)))
                # Add edges across time slices from -1 to -tau_max
                for lag in range(1, tau_max + 1):
                    if l - lag >= -tau_max:
                        edges.append(((s, l - lag), (t, -lag)))
        DBN.add_edges_from(edges)
        return DBN

    
    def find_all_paths(dbn: BayesianNetwork, treatment, outcome, path=[]) -> list:
        """
        Find all path from start to goal.
        Args:
            dbn (BayesianNetwork): Directed Acyclic Graph (DAG) as a Bayesian Network.
            treatment (str): Treatment variable.
            outcome (str): Outcome variable.
            paths (list): All paths between treatment and outcome.
        Returns:
            list: paths
        """
        path = path + [treatment]
        if treatment == outcome:
            return [path]
        paths = []
        for node in dbn.successors(treatment):
            if node not in path:
                new_paths = DAG.find_all_paths(dbn, node, outcome, path)
                for new_path in new_paths:
                    paths.append(new_path)
        for node in dbn.predecessors(treatment):
            if node not in path:
                new_paths = DAG.find_all_paths(dbn, node, outcome, path)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths   
    
    def _find_backdoor_paths(dbn: BayesianNetwork, treatment, paths):
        """
        Filter backdoor paths from all paths based on backdoor rules.

        Args:
            dbn (BayesianNetwork): Directed Acyclic Graph (DAG) as a Bayesian Network.
            treatment (str): Treatment variable.
            paths (list): All paths between treatment and outcome.

        Returns:
            list: Backdoor paths.
        """
        backdoor_paths = []
        for path in paths:
            # A path is a backdoor path if it doesn't start with T -> ... (direct causal link)
            if path[1] not in dbn.successors(treatment):  # Check first edge
                backdoor_paths.append(path)
        return backdoor_paths
    
    
    # def get_open_backdoors_paths(self, treatment: str, outcome: str):
    #     """
    #     Get backdoor paths between treatment and outcome, considering temporal dependencies.

    #     Args:
    #         treatment (str): Treatment variable.
    #         outcome (str): Outcome variable.
        
    #     Returns:
    #         list: List of backdoor paths, where each path is a list of tuples (variable, lag).
    #     """        
    #     # Convert adjacency matrix to a Bayesian Network using PAG
    #     bn = DAG.get_DBN(self.get_Adj(), self.max_lag)  # BayesianNetwork object from pgmpy
        
    #     # Find all paths from treatment to outcome
    #     all_paths = DAG.find_all_paths(bn, treatment, outcome, [])

    #     # Filter backdoor paths
    #     backdoor_paths = DAG._find_backdoor_paths(bn, treatment, all_paths)
    #     if not backdoor_paths:
    #         return []  # No backdoor paths found
        
    #     def is_blocked_path(path, bn: BayesianNetwork):
    #         for i in range(1, len(path) - 1):  # We exclude the first and last node in the path (treatment and outcome)
    #             node = path[i]
                    
    #             # Get the parents of the current node (we only check parents to detect colliders)
    #             parents = bn.get_parents(node)

    #             # Check if the current node has a parent in the previous node and a parent in the next node (collider condition)
    #             if any(parent == path[i-1] for parent in parents) and any(parent == path[i+1] for parent in parents):
    #                 # print(f"Collider found: {path[i-1]} -> {node} <- {path[i+1]}")
    #                 return True
                    
    #         # If no colliders are found, the path is open
    #         return False
        
    #     open_backdoor_paths = [path for path in backdoor_paths if not is_blocked_path(path, bn)]
        
    #     return open_backdoor_paths
    
    def get_open_backdoors_paths(self, treatment: str, outcome: str, conditioned: list = None):
        """
        Get backdoor paths between treatment and outcome, considering temporal dependencies and conditioning variables.

        Args:
            treatment (str): Treatment variable.
            outcome (str): Outcome variable.
            conditioned (list): Variables to condition on. These variables block paths if encountered (default: None).
        
        Returns:
            list: List of backdoor paths, where each path is a list of tuples (variable, lag).
        """
        # Initialize conditioned list if not provided
        # conditioned = [conditioned] if not isinstance(conditioned, list) else conditioned
        
        # Convert adjacency matrix to a Bayesian Network using PAG
        bn = DAG.get_DBN(self.get_Adj(), self.max_lag)  # BayesianNetwork object from pgmpy
        
        # Find all paths from treatment to outcome
        all_paths = DAG.find_all_paths(bn, treatment, outcome, [])
        
        # Filter backdoor paths
        backdoor_paths = DAG._find_backdoor_paths(bn, treatment, all_paths)
        if not backdoor_paths:
            return []  # No backdoor paths found
        
        # Function to determine if a path is blocked
        def is_blocked_path(path, bn: BayesianNetwork, conditioned):
            for i in range(1, len(path) - 1):  # Exclude treatment and outcome nodes
                node = path[i]
                
                # Collider check
                parents = bn.get_parents(node)
                is_collider = (
                    len(parents) >= 2 and
                    any(parent == path[i - 1] for parent in parents) and
                    any(parent == path[i + 1] for parent in parents)
                )
                
                def get_descendants(bn, node):
                    descendants = set()

                    def add_descendants(n):
                        for child in bn.get_children(n):
                            if child not in descendants:
                                descendants.add(child)
                                add_descendants(child)
                    
                    add_descendants(node)
                    return descendants
                
                if is_collider:
                    # Colliders block the path unless they or their descendants are conditioned on
                    if node not in conditioned and not any(descendant in conditioned for descendant in get_descendants(bn, node)):
                        return True  # Path blocked due to an unconditioned collider
                else:
                    # Non-colliders block the path if they are conditioned on
                    if node in conditioned:
                        return True  # Path blocked due to a non-collider being conditioned on
            
            # If no blockers are encountered, the path is open
            return False

        # Identify and return open backdoor paths
        open_backdoor_paths = [path for path in backdoor_paths if not is_blocked_path(path, bn, conditioned)]
        
        return open_backdoor_paths

                         
    
    def find_d_separators(self, treatment: str, outcome: str, paths) -> set:
        """
        Find D-Separation set.

        Args:
            treatment (str): treatment node.
            outcome (str): outcome node.

        Returns:
            (bool, set): (True, separation set) if treatment and outcome are d-separated. Otherwise (False, empty set). 
        """
        bn = DAG.get_DBN(self.get_Adj(), self.max_lag)
        bn.remove_edge(treatment, outcome)
        
        if paths:
            nodes = {node for path in paths for node in path if node not in {treatment, outcome}}
                               
            for r in range(len(nodes) + 1):
                for subset in combinations(nodes, r):
                    subset_set = set(subset)
                    if not bn.is_dconnected(treatment, outcome, subset_set):
                        return subset_set
            
        return set()
    
    
    # def find_all_d_separators(self, treatment: str, outcome: str, paths) -> list:
    #     """
    #     Find all D-Separation sets.

    #     Args:
    #         treatment (str): treatment node.
    #         outcome (str): outcome node.

    #     Returns:
    #         list: all possible adjustment sets.
    #     """
    #     # Step 1: Get the Bayesian Network
    #     bn = DAG.get_DBN(self.get_Adj(), self.max_lag)
        
    #     # Step 2: Remove the direct causal path (including mediators)
    #     visited = set()

    #     def dfs_path(node, target):
    #         """Find causal path using DFS from 'node' to 'target'."""
    #         if node == target:
    #             return [node]
    #         visited.add(node)
    #         for child in bn.get_children(node):  # Assume 'get_children' fetches child nodes
    #             if child not in visited:
    #                 path = dfs_path(child, target)
    #                 if path:
    #                     return [node] + path
    #         return None

    #     # Find causal path from treatment to outcome
    #     causal_path = dfs_path(treatment, outcome)
    #     if causal_path:
    #         # Remove all edges along this direct causal chain
    #         for i in range(len(causal_path) - 1):
    #             bn.remove_edge(causal_path[i], causal_path[i + 1])
        
    #     # Step 3: Identify potential backdoor nodes
    #     if paths:
    #         nodes = {node for path in paths for node in path if node not in {treatment, outcome}}
    #         all_adjustment_sets = []
    #         for r in range(len(nodes) + 1):
    #             for subset in combinations(nodes, r):
    #                 subset_set = set(subset)
    #                 if not bn.is_dconnected(treatment, outcome, subset_set):
    #                     all_adjustment_sets.append(subset_set)
    #         return [adj for adj in all_adjustment_sets if len(adj) <= 2]
    #         #! return all_adjustment_sets
    #     else:
    #         return []
    def find_all_d_separators(self, treatment: str, outcome: str, paths, conditioned = None, max_adj_size = 2) -> list:
        """
        Find all D-Separation sets.

        Args:
            treatment (str): treatment node.
            outcome (str): outcome node.
            conditioned (list, None): variables to condition on. These variables block paths if encountered.

        Returns:
            list: all possible adjustment sets.
        """
        # Step 1: Get the Bayesian Network
        bn = DAG.get_DBN(self.get_Adj(), self.max_lag)
        
        # Step 2: Remove the direct causal path (including mediators)
        visited = set()
        all_causal_paths = []
        
        def dfs_path(node, target, path):
            """Find all causal paths using DFS from 'node' to 'target'."""
            if node == target:
                all_causal_paths.append(path)
                return
            visited.add(node)
            for child in bn.get_children(node):
                if child not in visited:
                    dfs_path(child, target, path + [child])
        
        dfs_path(treatment, outcome, [treatment])  # Start DFS from treatment to outcome

        # Find causal path from treatment to outcome
        if all_causal_paths:
            # Remove all edges along this direct causal chain
            for causal_path in all_causal_paths:
                bn.remove_edge(causal_path[0], causal_path[1])
        
        # Step 3: Identify potential backdoor nodes
        if paths:
            cond_set = set(conditioned) if conditioned is not None else set()
            nodes = {node for path in paths for node in path if node not in {treatment, outcome} | cond_set}
            all_adjustment_sets = []
            for r in range(0, len(nodes) + 1):
                for subset in combinations(nodes, r):
                    subset_set = set(subset) 
                    if not bn.is_dconnected(treatment, outcome, subset_set | cond_set):
                        all_adjustment_sets.append(subset_set)
            return [adj for adj in all_adjustment_sets if len(adj) <= max_adj_size]
            #! return all_adjustment_sets
        else:
            return []

