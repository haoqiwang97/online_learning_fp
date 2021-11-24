# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:46:02 2021

@author: hw9335
"""

import numpy as np


class User(object):
    def __init__(self, age, BMI, race, ethnicity):
        self.age = age
        self.BMI = BMI
        self.race = race
        self.ethnicity = ethnicity
        
        

class AdaptiveRecommender(object):
    def __init__(self, dist_lookup, tree, time_horizon, user=None, ground_truth=None):
        self.dist_lookup = dist_lookup
        self.tree = tree
        # self.user = self.user
        # self.tree = self.filter_context()
        self.time_horizon = time_horizon
        
        self.n_epochs = min(self.tree.n_layers, np.log2(self.time_horizon)) #TODO: the tree has attribute: n_layers, the total number of layers
        
        self.ground_truth = ground_truth # ground_truth is know for testing, for real experiment with human, we do not know
        
    def restart(self):
        self.tree.restart() # TODO: set all n_plays, emp_mean, bound 0
        self.cum_regret = 0.0
        pass
    
    def filter_context(self):
        # TODO: future, return a trimmed tree, rule out impossible items
        # self.user and self.tree
        pass
    
    def update_stats(self, t, layer_id, node_selected_id, child_node_selected_id, reward):
        node_selected = self.tree.get_node(layer_id, node_selected_id)
        node_selected.n_plays += 1 # TODO: node has attribute n_plays, initial value is 0
        
        child_node_selected = self.tree.get_node(layer_id+1, child_node_selected_id)
        child_node_selected.n_plays += 1
        
        node_selected.emp_mean = (node_selected.emp_mean * max(1, node_selected.n_plays-1) + reward)/node_selected.n_plays # TODO: node has attribute emp_mean, initial value is 0
        child_node_selected.emp_mean = (child_node_selected.emp_mean * max(1, child_node_selected.n_plays-1) + reward)/child_node_selected.n_plays
        
        A_s = 2 # A_s is the exploration_exploitation trade-off factor
        node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(t) / node_selected.n_plays)# TODO: each node has attribute bound/ucb/index
        child_node_selected.bound = child_node_selected.emp_mean + np.sqrt(A_s * np.log(t) / child_node_selected.n_plays)
        
    def update_regret(self, item_recommended):
        # record cumulative regret
        # TODO: look-up table, initially set image and recommended image
        self.cum_regret += self.dist_lookup[self.ground_truth][item_recommended]
        
    def get_reward(self, item_recommended):
        return self.dist_lookup[self.ground_truth][item_recommended]
        
    def run(self):
        rng = np.random.default_rng(1)     
        for layer_id in range(0, self.n_epochs-1):
            partitions = self.tree.get_layer(layer_id) # TODO: the tree has function, input layer id, output all the nodes at layer id in a list
            if layer_id == 0: # first big cluster
                partitions[0].bound = 0 # TODO: each node has attribute bound/ucb/index
            
            bound_list = [partitions[i].bound for i in range(len(partitions))]
            for t in range(int(2**layer_id), int(2**(layer_id+1)-1)):
                # select cluster
                node_selected_id = np.argmax(bound_list)
                node_selected = self.tree.get_node(layer_id, node_selected_id) # TODO: the tree has function, input layer id and node id, output the nodes
                
                # randomly select a child node
                child_node_selected_id = rng.choice(node_selected.n_children)
                
                # randomly recommend an item in child node
                possible_items = node_selected.children[child_node_selected_id].items() # TODO: tree has function items, which return all the image items in this node
                item_recommended = rng.choice(possible_items)
                
                # TODO: get reward from look-up table, or human
                reward = self.get_reward(item_recommended) # reward or loss
                
                # update parameters
                self.update_stats(t, layer_id, node_selected_id, child_node_selected_id)
                
                self.update_regret(node_selected_id, child_node_selected_id, reward)
                
