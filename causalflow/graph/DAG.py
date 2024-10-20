"""
This module provides the DAG class.

Classes:
    DAG: class for facilitating the handling and the creation of DAGs.
"""
    
import copy
import numpy as np
from causalflow.graph.Node import Node
from causalflow.basics.constants import *
from matplotlib import pyplot as plt
import networkx as nx
from causalflow.graph.netgraph import Graph
import re

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
        self.g[s].children.append(t)
       
        
    def del_source(self, t, s, lag):
        """
        Remove source node from a target node.

        Args:
            t (str): target node name.
            s (str): source node name.
            lag (int): dependency lag.
        """
        del self.g[t].sources[(s, lag)]
        self.g[s].children.remove(t)
        
        
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
   
    
    def make_pretty(self) -> dict:
        """
        Make variables' names pretty, i.e. $ varname $ with '{' after '_' and '}' at the end of the string.

        Returns:
            dict: pretty DAG.
        """
        def prettify(name):
            return '$' + re.sub(r'_(\w+)', r'_{\1}', name) + '$'
        
        pretty = dict()
        for t in self.g:
            p_t = prettify(t)
            pretty[p_t] = copy.deepcopy(self.g[t])
            pretty[p_t].name = p_t
            pretty[p_t].children = [prettify(c) for c in self.g[t].children]
            for s in self.g[t].sources:
                del pretty[p_t].sources[s]
                p_s = prettify(s[0])
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
            arrows (dict): dictionary containing a bool for each edge of the graph describing if the edge is directed or not.
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
        edge_width[(s_node, t_node)] = self.__scale(score, min_width, max_width, min_score, max_score)
        
        if r.g[t].sources[s][TYPE] == LinkType.Directed.value:
            arrows[(s_node, t_node)] = {'h':'>', 't':''}
            
        elif r.g[t].sources[s][TYPE] == LinkType.Bidirected.value:
            edges.append((t_node, s_node))
            edge_width[(t_node, s_node)] = self.__scale(score, min_width, max_width, min_score, max_score)
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
        min_cross_width=0.5, 
        max_cross_width=1.5,
        node_size=4, 
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
            min_cross_width (float, optional): minimum edge linewidth. Defaults to 0.5.
            max_cross_width (float, optional): maximum edge linewidth. Defaults to 1.5.
            node_size (int, optional): node size. Defaults to 4.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color for contemporaneous links. Defaults to 'grey'.
            tail_color (str, optional): tail color. Defaults to 'black'.
            font_size (int, optional): font size. Defaults to 8.
            label_type (LabelType, optional): Show the lag time (LabelType.Lag), the strength (LabelType.Score), or no labels (LabelType.NoLabels). Default LabelType.Lag.
            save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
            img_extention (ImageExt, optional): Image Extension. Defaults to PNG.
        """
        r = copy.deepcopy(self)
        r.g = r.make_pretty()

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
                border[t] = max(self.__scale(r.g[t].sources[r.g[t].get_max_autodependent][SCORE], 
                                             min_auto_width, max_auto_width, 
                                             0, self.max_auto_score), 
                                border[t])
        
        # 3. Nodes border label definition
        node_label = None
        if label_type == LabelType.Lag or label_type == LabelType.Score:
            node_label = {t: [] for t in r.g.keys()}
            for t in r.g:
                if r.g[t].is_autodependent:
                    autodep = r.g[t].get_max_autodependent
                    if label_type == LabelType.Lag:
                        node_label[t].append(autodep[1])
                    elif label_type == LabelType.Score:
                        node_label[t].append(round(r.g[t].sources[autodep][SCORE], 3))
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
               text_disp = 0.1,
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
            text_disp (float, optional): text displacement along y. Defaults to 0.1.
            node_color (str/list, optional): node color. 
                                             If a string, all the nodes will have the same colour. 
                                             If a list (same dimension of features), each colour will have the specified colour.
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
        if isinstance(node_color, list):
            node_c = dict()
        else:
            node_c = node_color
        for i in range(len(self.features)):
            for j in range(self.max_lag + 1):
                Glagauto.add_node((j, i))
                Glagcross.add_node((j, i))
                Gcont.add_node((j, i))
                if isinstance(node_color, list): node_c[(j, i)] = node_color[abs(i - (len(r.g.keys()) - 1))]
                
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
                ax.text(pos[n][0] - text_disp, pos[n][1], list(r.g.keys())[len(r.g.keys()) - 1 - n[1]], horizontalalignment='center', verticalalignment='center', fontsize=font_size)

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
        
    def __scale(self, score, min_width, max_width, min_score = 0, max_score = 1):
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
    
        