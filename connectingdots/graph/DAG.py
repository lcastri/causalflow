import copy
import numpy as np
from connectingdots.graph.Node import Node
from connectingdots.basics.constants import *
from matplotlib import pyplot as plt
import networkx as nx
from netgraph import Graph


class DAG():
    def __init__(self, var_names, min_lag, max_lag, neglect_autodep = False, scm = None):
        """
        DAG constructor

        Args:
            var_names (list): variable list
            min_lag (int): minimum lag time
            max_lag (int): maximum time lag
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
                for s in scm[t]: self.add_source(t, s[0], 0.3, 0, s[1])


    @property
    def features(self) -> list:
        """
        Features list

        Returns:
            list: Features list
        """
        return list(self.g)

    
    @property
    def autodep_nodes(self) -> list:
        """
        Autodependent nodes list

        Returns:
            list: Autodependent nodes list
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
        Intervention links list

        Returns:
            list: Intervention link list
        """
        int_links = list()
        for t in self.g:
            for s in self.g[t].sources:
                if self.g[s[0]].intervention_node:
                    int_links.append((s[0], s[1], t))
        return int_links
    
    
    def fully_connected_dag(self):
        """
        Build a fully connected DAG
        """
        for t in self.g:
            for s in self.g:
                for l in range(1, self.max_lag + 1): self.add_source(t, s, 1, 0, l)
    
    
    def add_source(self, t, s, score, pval, lag, mode = LinkType.Directed.value):
        """
        Adds source node to a target node

        Args:
            t (str): target node name
            s (str): source node name
            score (float): dependency score
            pval (float): dependency p-value
            lag (int): dependency lag
        """
        self.g[t].sources[(s, abs(lag))] = {SCORE: score, PVAL: pval, TYPE: mode}
        self.g[s].children.append(t)
       
        
    def del_source(self, t, s, lag):
        """
        Removes source node from a target node

        Args:
            t (str): target node name
            s (str): source node name
            lag (int): dependency lag
        """
        del self.g[t].sources[(s, lag)]
        self.g[s].children.remove(t)
        
        
    def remove_unneeded_features(self):
        """
        Removes isolated nodes
        """
        tmp = copy.deepcopy(self.g)
        for t in self.g.keys():
            if self.g[t].is_isolated: 
                if self.g[t].intervention_node: del tmp[self.g[t].associated_context] # FIXME: last edit to be tested
                del tmp[t]
        self.g = tmp
            
            
    def add_context(self):
        """
        Adds context variables
        """
        for sys_var, context_var in self.sys_context.items():
            if sys_var in self.features:
                
                # Adding context var to the graph
                self.g[context_var] = Node(context_var, self.neglect_autodep)
                
                # Adding context var to sys var
                self.g[sys_var].intervention_node = True
                self.g[sys_var].associated_context = context_var
                self.add_source(sys_var, context_var, 1, 0, 1)
                
        # NOTE: bi-directed link contemporanous link between context vars
        for context_var in self.sys_context.values():
            other_context = [value for value in self.sys_context.values() if value != context_var]
            for other in other_context:
                self.add_source(context_var, other, 1, 0, 0)
                
    
    def remove_context(self):
        """
        Remove context variables
        """
        for sys_var, context_var in self.sys_context.items():
            if sys_var in self.g:
                
                # Removing context var from sys var
                # self.g[sys_var].intervention_node = False
                self.g[sys_var].associated_context = None
                self.del_source(sys_var, context_var, 1)
                
                # Removing context var from dag
                del self.g[context_var]
                
                
    def get_link_assumptions(self, autodep_ok = False) -> dict:
        """
        Returnes link assumption dictionary

        Args:
            autodep_ok (bool, optional): If true, autodependecy link assumption = -->. Otherwise -?>. Defaults to False.

        Returns:
            dict: link assumption dictionary
        """
        link_assump = {self.features.index(f): dict() for f in self.features}
        for t in self.g:
            for s in self.g[t].sources:
                if autodep_ok and s[0] == t: # NOTE: new condition added in order to not control twice the autodependency links
                    link_assump[self.features.index(t)][(self.features.index(s[0]), -abs(s[1]))] = '-->'
                    
                elif s[0] not in list(self.sys_context.values()):
                    link_assump[self.features.index(t)][(self.features.index(s[0]), -abs(s[1]))] = '-?>'
                    
                elif t in self.sys_context.keys() and s[0] == self.sys_context[t]:
                    link_assump[self.features.index(t)][(self.features.index(s[0]), -abs(s[1]))] = '-->'
                    
                elif t in self.sys_context.values() and s[0] in self.sys_context.values():
                    link_assump[self.features.index(t)][(self.features.index(s[0]), 0)] = 'o-o'
                    
        return link_assump


    def get_SCM(self) -> dict:   
        """
        Returns SCM

        Returns:
            dict: SCM
        """
        scm = {v: list() for v in self.features}
        for t in self.g:
            for s in self.g[t].sources:
                scm[t].append((s[0], -abs(s[1]))) 
        return scm
    
    
    def get_parents(self) -> dict:
        """
        Returns Parents dict

        Returns:
            dict: Parents dict
        """
        scm = {self.features.index(v): list() for v in self.features}
        for t in self.g:
            for s in self.g[t].sources:
                scm[self.features.index(t)].append((self.features.index(s[0]), -abs(s[1]))) 
        return scm
    
    
    def make_pretty(self) -> dict:
        """
        Makes variables' names pretty, i.e. $ varname $

        Returns:
            dict: pretty DAG
        """
        pretty = dict()
        for t in self.g:
            p_t = '$' + t + '$'
            pretty[p_t] = copy.deepcopy(self.g[t])
            pretty[p_t].name = p_t
            pretty[p_t].children = ['$' + c + '$' for c in self.g[t].children]
            for s in self.g[t].sources:
                del pretty[p_t].sources[s]
                p_s = '$' + s[0] + '$'
                pretty[p_t].sources[(p_s, s[1])] = {SCORE: self.g[t].sources[s][SCORE], PVAL: self.g[t].sources[s][PVAL], TYPE: self.g[t].sources[s][TYPE]}
        return pretty
        
    
    def dag(self,
            node_layout = 'dot',
            min_width = 1, max_width = 5,
            min_score = 0, max_score = 1,
            node_size = 8, node_color = 'orange',
            edge_color = 'grey',
            bundle_parallel_edges = True,
            font_size = 12,
            label_type = LabelType.Lag,
            save_name = None,
            img_extention = ImageExt.PNG):
        """
        build a dag

        Args:
            node_layout (str, optional): Node layout. Defaults to 'dot'.
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            bundle_parallel_edges (str, optional): bundle parallel edge bit. Defaults to True.
            font_size (int, optional): font size. Defaults to 12.
            label_type (LabelType, optional): enum to set whether to show the lag time (LabelType.Lag) or the strength (LabelType.Score) of the dependencies on each link/node or not showing the labels (LabelType.NoLabels). Default LabelType.Lag.
            save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
        """
        r = copy.deepcopy(self)
        r.g = r.make_pretty()

        G = nx.DiGraph()

        # NODES DEFINITION
        G.add_nodes_from(r.g.keys())
        
        # BORDER LINE
        border = dict()
        for t in r.g:
            border[t] = 0
            if r.g[t].is_autodependent:
                autodep = r.g[t].get_max_autodependent
                border[t] = max(self.__scale(r.g[t].sources[autodep][SCORE], min_width, max_width, min_score, max_score), border[t])
        
        # BORDER LABEL
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


        # EDGE DEFINITION
        edges = [(s[0], t) for t in r.g for s in r.g[t].sources if t != s[0]]
        arrows = {(s[0], t):True for t in r.g for s in r.g[t].sources if t != s[0]}
        G.add_edges_from(edges)
        
        # EDGE LINE
        edge_width = {(s[0], t): 0 for t in r.g for s in r.g[t].sources if t != s[0]}
        for t in r.g:
            for s in r.g[t].sources:
                if t != s[0]:
                    edge_width[(s[0], t)] = max(self.__scale(r.g[t].sources[s][SCORE], min_width, max_width, min_score, max_score), edge_width[(s[0], t)])
        
        # EDGE LABEL
        edge_label = None
        if label_type == LabelType.Lag or label_type == LabelType.Score:
            edge_label = {(s[0], t): [] for t in r.g for s in r.g[t].sources if t != s[0]}
            for t in r.g:
                for s in r.g[t].sources:
                    if t != s[0]:
                        if label_type == LabelType.Lag:
                            edge_label[(s[0], t)].append(s[1])
                        elif label_type == LabelType.Score:
                            edge_label[(s[0], t)].append(round(r.g[t].sources[s][SCORE], 3))
            for k in edge_label.keys():
                edge_label[k] = ",".join(str(s) for s in edge_label[k])

        fig, ax = plt.subplots(figsize=(8,6))

        if edges:
            a = Graph(G, 
                    node_layout = node_layout,
                    node_size = node_size,
                    node_color = node_color,
                    node_labels = node_label,
                    node_edge_width = border,
                    node_label_fontdict = dict(size=font_size),
                    node_edge_color = edge_color,
                    node_label_offset = 0.1,
                    node_alpha = 1,
                    
                    arrows = arrows,
                    edge_layout = 'curved',
                    edge_label = label_type != LabelType.NoLabels,
                    edge_labels = edge_label,
                    edge_label_fontdict = dict(size=font_size),
                    edge_color = edge_color, 
                    edge_width = edge_width,
                    edge_alpha = 1,
                    edge_zorder = 1,
                    edge_label_position = 0.35,
                    edge_layout_kwargs = dict(bundle_parallel_edges = bundle_parallel_edges, k = 0.05))
            
            nx.draw_networkx_labels(G, 
                                    pos = a.node_positions,
                                    labels = {n: n for n in G},
                                    font_size = font_size)

        if save_name is not None:
            plt.savefig(save_name + img_extention.value, dpi = 300)
        else:
            plt.show()
                
            
    
    def ts_dag(self,
               tau,
               min_width = 1, max_width = 5,
               min_score = 0, max_score = 1,
               node_size = 8,
               node_proximity = 2,
               node_color = 'orange',
               edge_color = 'grey',
               font_size = 12,
               save_name = None,
               img_extention = ImageExt.PNG):
        """
        build a timeseries dag

        Args:
            tau (int): max time lag
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_proximity (int, optional): node proximity. Defaults to 2.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
            save_name (str, optional): Filename path. If None, plot is shown and not saved. Defaults to None.
        """
        
        r = copy.deepcopy(self)
        r.g = r.make_pretty()

        G = nx.DiGraph()

        # Add nodes to the graph
        for i in range(len(self.features)):
            for j in range(tau + 1):
                G.add_node((j, i))
                
        pos = {n : (n[0], n[1]/node_proximity) for n in G.nodes()}
        scale = max(pos.values())
        
        # Nodes color definition
        # node_c = ['tab:blue', 'tab:orange','tab:red', 'tab:purple']
        # node_c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        # node_color = dict()
        # tmpG = nx.grid_2d_graph(self.max_lag + 1, len(r.g.keys()))
        # for n in tmpG.nodes():
        #     node_color[n] = node_c[abs(n[1] - (len(r.g.keys()) - 1))]

        # edges definition
        edges = list()
        edge_width = dict()
        arrows = dict()
        for t in r.g:
            for s in r.g[t].sources:
                s_index = len(r.g.keys())-1 - list(r.g.keys()).index(s[0])
                t_index = len(r.g.keys())-1 - list(r.g.keys()).index(t)
                
                # Contemporaneous dependecies
                if s[1] == 0:
                    for i in range(tau + 1):
                        s_node = (i, s_index)
                        t_node = (i, t_index)
                        if (t_node, s_node) not in edges: # this is to avoit double edge for undirected contemporaneous link
                            edges.append((s_node, t_node))
                            edge_width[(s_node, t_node)] = self.__scale(r.g[t].sources[s][SCORE], min_width, max_width, min_score, max_score)
                            arrows[(s_node, t_node)] = r.g[t].sources[s][TYPE] == LinkType.Directed.value
                # Lagged dependecies
                else:
                    s_lag = tau - s[1]
                    t_lag = tau
                    while s_lag >= 0:
                        s_node = (s_lag, s_index)
                        t_node = (t_lag, t_index)
                        edges.append((s_node, t_node))
                        edge_width[(s_node, t_node)] = self.__scale(r.g[t].sources[s][SCORE], min_width, max_width, min_score, max_score)
                        arrows[(s_node, t_node)] = True
                        s_lag -= s[1]
                        t_lag -= s[1]
                    
        G.add_edges_from(edges)

        # label definition
        labeldict = {}
        for n in G.nodes():
            if n[0] == 0:
                labeldict[n] = list(r.g.keys())[len(r.g.keys()) - 1 - n[1]]

        fig, ax = plt.subplots(figsize=(8,6))

        # time line text drawing
        pos_tau = set([pos[p][0] for p in pos])
        max_y = max([pos[p][1] for p in pos])
        for p in pos_tau:
            if abs(int(p) - tau) == 0:
                ax.text(p, max_y + .3, r"$t$", horizontalalignment='center', fontsize=font_size)
            else:
                ax.text(p, max_y + .3, r"$t-" + str(abs(int(p) - tau)) + "$", horizontalalignment='center', fontsize=font_size)

        Graph(G,
            node_layout = {p : np.array(pos[p]) for p in pos},
            node_size = node_size,
            node_color = node_color,
            node_labels = labeldict,
            node_label_offset = 0,
            node_edge_width = 0,
            node_label_fontdict = dict(size=font_size),
            node_alpha = 1,
            
            arrows = arrows,
            edge_layout = 'curved',
            edge_label = False,
            edge_color = edge_color, 
            edge_width = edge_width,
            edge_alpha = 1,
            edge_zorder = 1,
            scale = (scale[0] + 2, scale[1] + 2))

        if save_name is not None:
            plt.savefig(save_name + img_extention.value, dpi = 300)
        else:
            plt.show()


    def __scale(self, score, min_width, max_width, min_score = 0, max_score = 1):
        """
        Scales the score of the cause-effect relationship strength to a linewitdth

        Args:
            score (float): score to scale
            min_width (float): minimum linewidth
            max_width (float): maximum linewidth
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.

        Returns:
            (float): scaled score
        """
        return ((score - min_score) / (max_score - min_score)) * (max_width - min_width) + min_width


    def get_skeleton(self) -> np.array:
        """
        Returns skeleton matrix.
        Skeleton matrix is composed by 0 and 1.
        1 <- if there is a link from source to target 
        0 <- if there is not a link from source to target 

        Returns:
            np.array: skeleton matrix
        """
        r = [np.zeros(shape=(len(self.features), len(self.features)), dtype = np.int32) for _ in range(self.min_lag, self.max_lag + 1)]
        for l in range(self.min_lag, self.max_lag + 1):
            for t in self.g.keys():
                for s in self.g[t].sources:
                    if s[1] == l: r[l - self.min_lag][self.features.index(t), self.features.index(s[0])] = 1
        return np.array(r)


    def get_val_matrix(self) -> np.array:
        """
        Returns val matrix.
        val matrix contains information about the strength of the links componing the causal model.

        Returns:
            np.array: val matrix
        """
        r = [np.zeros(shape=(len(self.features), len(self.features))) for _ in range(self.min_lag, self.max_lag + 1)]
        for l in range(self.min_lag, self.max_lag + 1):
            for t in self.g.keys():
                for s, info in self.g[t].sources.items():
                    if s[1] == l: r[l - self.min_lag][self.features.index(t), self.features.index(s[0])] = info['score']
        return np.array(r)


    def get_pval_matrix(self) -> np.array:
        """
        Returns pval matrix.
        pval matrix contains information about the pval of the links componing the causal model.=
        
        Returns:
            np.array: pval matrix
        """
        r = [np.zeros(shape=(len(self.features), len(self.features))) for _ in range(self.min_lag, self.max_lag + 1)]
        for l in range(self.min_lag, self.max_lag + 1):
            for t in self.g.keys():
                for s, info in self.g[t].sources.items():
                    if s[1] == l: r[l - self.min_lag][self.features.index(t), self.features.index(s[0])] = info['pval']
        return np.array(r)