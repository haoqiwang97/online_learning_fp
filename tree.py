# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:41:34 2021

@author: hw9335
"""

class ExpTree():
    def __init__(self, distance_mt, b, n_layers):
        self.distance_mt = distance_mt
        self.b = b
        self.n_layers = n_layers
        # A, constant like 2 in UCB in square root
        
    
    def build_tree(self):
        pass
        
    
class Node():
    def __init__(self, layer_id, node_id):
        self.layer_id = layer_id
        self.node_id = node_id
        self.child = None
        self.parent = None
        
        self.n_plays = None
        self.emp_means = None
        self.index = None # index is upper bound value
        
    