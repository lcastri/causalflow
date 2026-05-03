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
from collections import defaultdict

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
        """
        pretty = {}
        
        for t, node in self.g.items():
            p_t = DAG.prettify(t)
            
            new_node = copy.copy(node)
            new_node.name = p_t
            new_node.children = [DAG.prettify(c) for c in node.children]
            
            new_node.sources = {
                (DAG.prettify(s[0]), s[1]): {
                    SCORE: s_data[SCORE],
                    PVAL: s_data[PVAL],
                    TYPE: s_data[TYPE]
                }
                for s, s_data in node.sources.items()
            }
            
            pretty[p_t] = new_node
            
        return pretty
    
    
    def __add_edge(self, min_width, max_width, min_score, max_score, edges, edge_width, arrows, s_node, t_node, score, link_type):
        """
        Add edge to a graph. Support method.
        """
        edges.append((s_node, t_node))
        
        # Handle infinite scores efficiently
        safe_score = score if score != float('inf') else 1
        width = DAG.__scale(safe_score, min_width, max_width, min_score, max_score)
        edge_width[(s_node, t_node)] = width
        
        if link_type == LinkType.Directed.value:
            arrows[(s_node, t_node)] = {'h': '>', 't': ''}
            
        elif link_type == LinkType.Bidirected.value:
            edges.append((t_node, s_node))
            edge_width[(t_node, s_node)] = width
            arrows[(t_node, s_node)] = {'h': '>', 't': ''}
            arrows[(s_node, t_node)] = {'h': '>', 't': ''}
            
        elif link_type == LinkType.HalfUncertain.value:
            arrows[(s_node, t_node)] = {'h': '>', 't': 'o'}
            
        elif link_type == LinkType.Uncertain.value:
            arrows[(s_node, t_node)] = {'h': 'o', 't': 'o'}
        
        else:
            raise ValueError(f"{link_type} not included in LinkType")
    
    
    @staticmethod
    def __gen_label(label_type, s_lag, score):
        """
        Generate edge/node labels based on the specified label type.

        Args:
            label_type (LabelType): Type of label to generate (Lag, Score, NoLabels, OnlyLagged).
            s_lag (int): Lag value for the edge.
            score (float): Score value for the edge.

        Returns:
            str: Generated edge label.
        """
        if label_type in [LabelType.Lag, LabelType.OnlyLagged]:
            return str(s_lag)
        elif label_type == LabelType.Score:
            return str(round(score, 3))
        return None


    def plot_graph(self,
                   node_layout='dot', min_auto_width=0.25, max_auto_width=0.75,
                   min_cross_width=1, max_cross_width=5, node_size=8, 
                   node_color='orange', edge_color='grey', tail_color='black',
                   font_size=8, label_type=LabelType.Lag, save_name=None,
                   img_extention=ImageExt.PNG):
        """Build a dag, first with contemporaneous links, then lagged links."""
        
        r = copy.copy(self) 
        r.g = r.make_pretty()

        # Handle node colors
        if not isinstance(node_color, str):
            node_color = {DAG.prettify(f): color for f, color in node_color.items()}

        Gcont, Glag = nx.DiGraph(), nx.DiGraph()
        Gcont.add_nodes_from(r.g.keys())
        Glag.add_nodes_from(r.g.keys())

        # Initialize label dicts based on whether we need labels at all
        border = {}
        needs_labels = label_type != LabelType.NoLabels
        node_label = defaultdict(list) if needs_labels else None
        cont_edge_label = defaultdict(list) if needs_labels else None
        lagged_edge_label = defaultdict(list) if needs_labels else None

        cont_edges, cont_edge_width, cont_arrows = [], {}, {}
        lagged_edges, lagged_edge_width, lagged_arrows = [], {}, {}

        for t, node_data in r.g.items():
            # 1. Node Borders
            border[t] = 0
            if node_data.is_autodependent:
                max_score = node_data.sources[node_data.get_max_autodependent][SCORE]
                border[t] = DAG.__scale(max_score, min_auto_width, max_auto_width, 0, r.max_auto_score)
                
            for s_key, s_data in node_data.sources.items():
                s_node, s_lag = s_key[0], s_key[1]
                score = s_data[SCORE]
                link_type = s_data[TYPE]
                
                # 2. Node Labels (Autodependencies / Self-Loops)
                if s_node == t:
                    if needs_labels:
                        val = DAG.__gen_label(label_type, s_lag, score)
                        if val is not None: node_label[t].append(val)
                    continue # Skip edge creation for self-loops

                # 3. Edges & Edge Labels
                if s_lag == 0:  # Contemporaneous
                    self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score,
                                    cont_edges, cont_edge_width, cont_arrows, s_node, t, score, link_type)
                    
                    if needs_labels and label_type != LabelType.OnlyLagged:
                        val = DAG.__gen_label(label_type, s_lag, score)
                        if val is not None: cont_edge_label[(s_node, t)].append(val)
                        
                else:  # Lagged
                    self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score,
                                    lagged_edges, lagged_edge_width, lagged_arrows, s_node, t, score, link_type)
                    
                    if needs_labels:
                        val = DAG.__gen_label(label_type, s_lag, score)
                        if val is not None: lagged_edge_label[(s_node, t)].append(val)

        Gcont.add_edges_from(cont_edges)
        Glag.add_edges_from(lagged_edges)

        # 4. Finalize Label Formatting (Join strings)
        if needs_labels:
            node_label = {k: ",".join(v) for k, v in node_label.items()}
            cont_edge_label = {k: ",".join(v) for k, v in cont_edge_label.items()}
            lagged_edge_label = {k: ",".join(v) for k, v in lagged_edge_label.items()}

        # 5 & 6. Drawing Graphs
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Safe variable initialization for layout
        current_node_layout = node_layout 

        if cont_edges:
            a_cont = Graph(Gcont,
                    node_layout=current_node_layout,
                    node_size=node_size, node_color=node_color,
                    node_labels=None, node_edge_width=border,
                    node_label_fontdict=dict(size=font_size),
                    node_edge_color=edge_color, node_label_offset=0.05, node_alpha=1,
                    arrows=cont_arrows, edge_layout='straight',
                    edge_label=(label_type != LabelType.NoLabels),
                    edge_labels=cont_edge_label, edge_label_fontdict=dict(size=font_size),
                    edge_color=edge_color, tail_color=tail_color,
                    edge_width=cont_edge_width, edge_alpha=1, edge_zorder=1, edge_label_position=0.35)

            nx.draw_networkx_labels(Gcont, pos=a_cont.node_positions, labels={n: n for n in Glag}, font_size=font_size)
            current_node_layout = a_cont.node_positions # Safely pass layout to lagged graph

        if lagged_edges:
            a_lag = Graph(Glag,
                    node_layout=current_node_layout,
                    node_size=node_size, node_color=node_color,
                    node_labels=node_label, node_edge_width=border,
                    node_label_fontdict=dict(size=font_size),
                    node_edge_color=edge_color, node_label_offset=0.05, node_alpha=1,
                    arrows=lagged_arrows, edge_layout='curved',
                    edge_label=(label_type != LabelType.NoLabels),
                    edge_labels=lagged_edge_label, edge_label_fontdict=dict(size=font_size),
                    edge_color=edge_color, tail_color=tail_color,
                    edge_width=lagged_edge_width, edge_alpha=1, edge_zorder=1, edge_label_position=0.35)
            
            if not cont_edges:
                nx.draw_networkx_labels(Gcont, pos=a_lag.node_positions, labels={n: n for n in Glag}, font_size=font_size)

        # 7. Plot or save
        if save_name is not None:
            plt.savefig(save_name + img_extention.value, dpi=300)
        else:
            plt.show()
    

    def plot_ts_graph(self,
                      min_cross_width=1, max_cross_width=5, node_size=8,
                      x_disp=1.5, y_disp=0.2, node_color='orange',
                      edge_color='grey', tail_color='black',
                      font_size=8, save_name=None, img_extention=ImageExt.PNG):
        """
        Build a timeseries dag.

        Args:
            min_cross_width (int, optional): Minimum width for cross edges. Defaults to 1.
            max_cross_width (int, optional): Maximum width for cross edges. Defaults to 5.
            node_size (int, optional): Size of the nodes. Defaults to 8.
            x_disp (float, optional): Node displacement along the x-axis (time). Defaults to 1.5.
            y_disp (float, optional): Node displacement along the y-axis (variables). Defaults to 0.2.
            node_color (str or dict, optional): Color of the nodes. Can be a single color string or a dictionary mapping node names to colors. Defaults to 'orange'.
            edge_color (str, optional): Color of the edges. Defaults to 'grey'.
            tail_color (str, optional): Color of the edge tails. Defaults to 'black'.
            font_size (int, optional): Font size for labels. Defaults to 8.
            save_name (str, optional): Path to save the figure without extension. If None, the plot is displayed. Defaults to None.
            img_extention (ImageExt, optional): Image extension for saving. Defaults to ImageExt.PNG.
        """
        
        r = copy.copy(self)
        r.g = self.make_pretty()

        Gcont, Glagcross, Glagauto = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()

        # Pre-compute indices to eliminate O(N) lookups inside the loop
        node_keys = list(r.g.keys())
        num_nodes = len(node_keys)
        node_to_idx = {name: (num_nodes - 1 - idx) for idx, name in enumerate(node_keys)}
        idx_to_node = {idx: name for name, idx in node_to_idx.items()}

        # 1. Single-pass Nodes, Positions, and Colors definition
        pos = {}
        node_c = {} if isinstance(node_color, dict) else node_color
        
        for i in range(len(self.features)):
            # Fast color mapping mapping avoiding recalculations
            feat_name = self.features[num_nodes - 1 - i] if isinstance(node_color, dict) else None
            
            for j in range(self.max_lag + 1):
                node = (j, i)
                pos[node] = np.array([j * x_disp, i * y_disp])
                
                if feat_name is not None:
                    node_c[node] = node_color[feat_name]

        Gcont.add_nodes_from(pos.keys())
        Glagcross.add_nodes_from(pos.keys())
        Glagauto.add_nodes_from(pos.keys())

        # Safely determine scale without iterating over values multiple times
        scale = (self.max_lag * x_disp, (len(self.features) - 1) * y_disp)

        # 2. Edges definition
        cont_edges, cont_edge_width, cont_arrows = [], {}, {}
        lagged_cross_edges, lagged_cross_edge_width, lagged_cross_arrows = [], {}, {}
        lagged_auto_edges, lagged_auto_edge_width, lagged_auto_arrows = [], {}, {}

        for t, node_data in r.g.items():
            t_index = node_to_idx[t]
            
            for s_key, s_data in node_data.sources.items():
                s_name, s_lag_val = s_key[0], s_key[1]
                s_index = node_to_idx[s_name]
                
                # Extract once for the optimized __add_edge signature
                score = s_data[SCORE]
                link_type = s_data[TYPE]

                # 2.1. Contemporaneous edges definition
                if s_lag_val == 0:
                    for i in range(self.max_lag + 1):
                        self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score, 
                                        cont_edges, cont_edge_width, cont_arrows, 
                                        (i, s_index), (i, t_index), score, link_type)
                # 2.2 & 2.3. Lagged edges definition
                else:
                    s_lag_iter = self.max_lag - s_lag_val
                    t_lag_iter = self.max_lag
                    
                    # Determine target dictionaries O(1) outside the while loop
                    is_auto = (s_name == t)
                    target_edges = lagged_auto_edges if is_auto else lagged_cross_edges
                    target_width = lagged_auto_edge_width if is_auto else lagged_cross_edge_width
                    target_arrows = lagged_auto_arrows if is_auto else lagged_cross_arrows

                    while s_lag_iter >= 0:
                        self.__add_edge(min_cross_width, max_cross_width, 0, self.max_cross_score, 
                                        target_edges, target_width, target_arrows, 
                                        (s_lag_iter, s_index), (t_lag_iter, t_index), score, link_type)
                        s_lag_iter -= 1
                        t_lag_iter -= 1
                    
        Gcont.add_edges_from(cont_edges)
        Glagcross.add_edges_from(lagged_cross_edges)
        Glagauto.add_edges_from(lagged_auto_edges)

        # 3 & 4. Plotting Initialization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # NOTE: Make sure __get_fixed_edges handles numpy arrays in `pos` if needed.
        edge_layout = self.__get_fixed_edges(ax, x_disp, Gcont, node_size, pos, node_c, font_size, 
                                             cont_arrows, edge_color, tail_color, cont_edge_width, scale)
        
        # Label definition (Using O(1) dict lookup instead of list indexing)
        for n in Gcont.nodes():
            if n[0] == 0:
                ax.text(pos[n][0]-0.1, pos[n][1], idx_to_node[n[1]], 
                        horizontalalignment='center', verticalalignment='center', fontsize=font_size)

        # Time line text drawing (Optimized max_y calculation)
        max_y = scale[1] 
        pos_tau = set([pos[p][0] for p in pos])
        
        for p in pos_tau:
            lag_val = abs(int(p / x_disp) - self.max_lag)
            text_str = r"$t$" if lag_val == 0 else r"$t-" + str(lag_val) + r"$"
            ax.text(p, max_y + 0.1, text_str, horizontalalignment='center', fontsize=font_size)

        # 5, 6, 7. Draw graphs (Passing `pos` directly since it's already np.array mapping)
        common_kwargs = dict(
            node_layout=pos, node_size=node_size, node_color=node_c,
            node_edge_width=0, node_label_fontdict=dict(size=font_size),
            node_label_offset=0, node_alpha=1, edge_label=False,
            edge_color=edge_color, tail_color=tail_color, edge_alpha=1,
            edge_zorder=1, scale=(scale[0] + 2, scale[1] + 2)
        )

        if cont_edges:
            Graph(Gcont, arrows=cont_arrows, edge_layout=edge_layout, edge_width=cont_edge_width, **common_kwargs)

        if lagged_cross_edges:
            Graph(Glagcross, arrows=lagged_cross_arrows, edge_layout='curved', edge_width=lagged_cross_edge_width, **common_kwargs)
            
        if lagged_auto_edges:
            Graph(Glagauto, arrows=lagged_auto_arrows, edge_layout='straight', edge_width=lagged_auto_edge_width, **common_kwargs)
        
        # 8. Plot or save
        if save_name is not None:
            plt.savefig(save_name + img_extention.value, dpi=300)
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