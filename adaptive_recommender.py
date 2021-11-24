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
    def __init__(self, dist_lookup, tree, time_horizon, user=None, ground_truth=None, test=True):
        self.dist_lookup = dist_lookup
        self.tree = tree
        # self.user = self.user
        # self.tree = self.filter_context()
        self.time_horizon = time_horizon
        
        self.n_epochs = min(self.tree.n_layers, np.log2(self.time_horizon)) #TODO: the tree has attribute: n_layers, the total number of layers
        
        self.ground_truth = ground_truth # ground_truth is known for testing, for real experiment with human, we do not know
        
        self.test = test # whether it is testing or doing experiment with human
        
    def restart(self):
        self.tree.restart() # TODO: set all n_plays, emp_mean, bound 0
        self.cum_regret = 0.0
        pass
    
    def filter_context(self):
        # TODO: future, return a trimmed tree, rule out impossible items
        # self.user and self.tree
        pass
    
    def update_stats(self, t, layer_id, node_selected_id, child_node_selected_id, reward):
        A_s = 2 # A_s is the exploration_exploitation trade-off factor, here use UCB factor
        node_selected = self.tree.get_node(layer_id, node_selected_id)
        node_selected.n_plays += 1 # TODO: node has attribute n_plays, initial value is 0
        node_selected.emp_mean = (node_selected.emp_mean * max(1, node_selected.n_plays-1) + reward)/node_selected.n_plays # TODO: node has attribute emp_mean, initial value is 0
        node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(t) / node_selected.n_plays)# TODO: each node has attribute bound/ucb/index
        
        if child_node_selected_id:
            child_node_selected = self.tree.get_node(layer_id+1, child_node_selected_id)
            child_node_selected.n_plays += 1
            child_node_selected.emp_mean = (child_node_selected.emp_mean * max(1, child_node_selected.n_plays-1) + reward)/child_node_selected.n_plays
            child_node_selected.bound = child_node_selected.emp_mean + np.sqrt(A_s * np.log(t) / child_node_selected.n_plays)
            
    def update_regret(self, item_recommended):
        # record cumulative regret
        # TODO: look-up table, initially set image and recommended image
        self.cum_regret += self.dist_lookup[self.ground_truth][item_recommended]
        # TODO: for human, we have no ground_truth, maybe simply add all the rewards?
        
    def get_loss(self, item_recommended):
        if self.test:
            return self.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * 0.01
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)
        
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
                possible_items = node_selected.children[child_node_selected_id].get_items() # TODO: tree has function of get_items, which return all the image items in this node
                item_recommended = rng.choice(possible_items)
                
                # TODO: get reward from look-up table, or human
                reward = 1 - self.get_loss(item_recommended) # reward or loss
                
                # update parameters
                self.update_stats(t, layer_id, node_selected_id, child_node_selected_id)
                
                self.update_regret(item_recommended)
                
        layer_id = self.n_epochs
        partitions = self.tree.get_layer(layer_id)
        bound_list = [partitions[i].bound for i in range(len(partitions))]
        for t in range(2**layer_id, self.time_horizon):
            # should reach the L-1 layer
            # select cluster
            node_selected_id = np.argmax(bound_list)
            node_selected = self.tree.get_node(layer_id, node_selected_id)
            # randomly recommend item
            possible_items = node_selected.get_items()
            item_recommended = rng.choice(possible_items)
            
            reward = 1 - self.get_loss(item_recommended)

            self.update_stats(t, layer_id, node_selected_id, None, reward)
                
            self.update_regret(item_recommended)            

