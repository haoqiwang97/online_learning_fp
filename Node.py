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
        self.index = index # node_id
        self.depth = depth # layer_id

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self, parent):
        self.parent = parent
        
    def __repr__(self):
        # return "layer_id=" + repr(self.layer_id) + ";\nnode_id=" + repr(self.node_id) + ";\nparent=" + repr(self.parent) + ";\nchildren=" + repr(self.children)
        return "layer_id=" + repr(self.depth) + ";\nnode_id=" + repr(self.index) + ";\nparent=" + repr(self.parent) + ";\nchildren=" + repr(self.children)

    def __str__(self):
        return self.__repr__()    

if __name__ == '__main__':
    node = Node(0, 'node_0', 5)
    print(node)