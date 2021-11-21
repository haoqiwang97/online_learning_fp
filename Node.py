import numpy as np

class Node(object):
    def __init__(self, index, name, depth):
        """
        construction of node 
        need to at least know the name
        """
        self.parent = None
        self.children = []
        self.name = name
        self.index = index
        self.depth = depth

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self, parent):
        self.parent = parent



