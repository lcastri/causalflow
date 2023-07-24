from fpcmci.basics.constants import SCORE


class Node():
    
    def __init__(self, name, neglect_autodep):
        """
        Node class contructer

        Args:
            name (str): node name
            neglect_autodep (bool): flag to decide whether to to skip the node if it is only auto-dependent
        """
        self.name = name
        self.sources = dict()
        self.children = list()
        self.neglect_autodep = neglect_autodep
        self.intervention_node = False        
        self.associated_context = None        
    
    
    @property
    def is_autodependent(self):
        """
        Returns True if the node is autodependent

        Returns:
            bool: Returns True if the node is autodependent. Otherwise False
        """
        return self.name in self.sourcelist
    
    
    @property
    def is_isolated(self):
        """
        Returns True if the node is isolated

        Returns:
            bool: Returns True if the node is isolated. Otherwise False
        """
        if self.neglect_autodep:
            return (self.is_exogenous or self.is_only_autodep or self.is_only_autodep_context) and not self.has_child
        
        return (self.is_exogenous or self.has_only_context) and not self.has_child
    
    
    @property
    def is_only_autodep(self):
        """
        Returns True if the node is ONLY auto-dependent

        Returns:
            bool: Returns True if the node is ONLY auto-dependent. Otherwise False
        """
        return len(self.sources) == 1 and self.name in self.sourcelist
    
    
    @property
    def has_only_context(self):
        """
        Returns True if the node has ONLY the context variable as parent

        Returns:
            bool: Returns True if the node has ONLY the context variable as parent. Otherwise False
        """
        return len(self.sources) == 1 and self.associated_context in self.sourcelist
    
    
    @property
    def is_only_autodep_context(self):
        """
        Returns True if the node has ONLY the context variable and itself as parent

        Returns:
            bool: Returns True if the node has ONLY the context variable and itself as parent. Otherwise False
        """
        return len(self.sources) == 2 and self.name in self.sourcelist and self.associated_context in self.sourcelist
    
    
    @property
    def is_exogenous(self):
        """
        Returns True if the node has no parents

        Returns:
            bool: Returns True if the node has no parents. Otherwise False
        """
        return len(self.sources) == 0
        
        
    @property
    def has_child(self):
        """
        Returns True if the node has at least one child

        Returns:
            bool: Returns True if the node has at least one child. Otherwise False
        """
        return len(self.children) > 0
    
    
    @property
    def sourcelist(self):
        """
        Returns list of source names

        Returns:
            list(str): Returns list of source names
        """
        return [s[0] for s in self.sources]
    
    
    @property
    def autodependency_links(self):
        autodep_links = list()
        if self.is_autodependent:
            for s in self.sources: 
                if s[0] == self.name: 
                    autodep_links.append(s)
        return autodep_links
    
    
    @property
    def get_max_autodependent(self):
        max_score = 0
        max_s = None
        if self.is_autodependent:
            for s in self.sources: 
                if s[0] == self.name:
                    if self.sources[s][SCORE] > max_score: max_s = s
        return max_s