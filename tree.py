# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:41:34 2021

@author: hw9335
"""


class ExpTree():
    def __init__(self, distance_mt, b, n_layers):
        self.distance_mt = distance_mt
        self.b = b
        self.n_layers = n_layers # depth of tree
        # A, constant like 2 in UCB in square root

    def build_tree(self):
        pass

    def get_layer(self, layer_id):
        pass

    # TODO: NUMBER OF nodes at layer l
    def get_node(self, layer_id, node_id):
        pass
    
    def restart():
        pass
    
    

    
class Node():
    def __init__(self, layer_id, node_id):
        self.layer_id = layer_id
        self.node_id = node_id
        self.children = None
        self.parent = None
        self.n_childern = None
        self.items = [] # all the items it contains
        
        self.n_plays = None
        self.emp_means = None
        self.bound = None # index is upper bound value
        
    
