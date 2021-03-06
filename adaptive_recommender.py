# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:46:02 2021

@author: hw9335
"""

import numpy as np
import matplotlib.pyplot as plt


class User(object):
    def __init__(self, age, BMI, race, ethnicity):
        self.age = age
        self.BMI = BMI
        self.race = race
        self.ethnicity = ethnicity
        

class Item(object):
    def __init__(self, name):
        self.name = name
        self.n_plays = 0
        self.emp_mean = 0
        self.bound = 1e5
    
    def __repr__(self):
        return "name=" + repr(self.name)

    def __str__(self):
        return self.__repr__()
        

class UCB(object):
    def __init__(self, dist_lookup, time_horizon, ground_truth=None, test=True, noise=0.01):
        self.dist_lookup = dist_lookup
        
        self.time_horizon = time_horizon
        self.ground_truth = ground_truth
        self.test = test
        self.noise = noise
        
        self._restart()
        
    def _restart(self):
        item_names = sorted(list(self.dist_lookup.keys()))
        self.item_list = [Item(item_name) for item_name in item_names]
        
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        #self.cum_regret_list.append(self.cum_regret)

    def get_loss(self, item_recommended):
        if self.test:
            return self.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)

    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.dist_lookup[self.ground_truth][item_recommended.name]
        self.cum_regret_list.append(self.cum_regret)
        
    def update_stats(self, t, item_recommended, reward):
        item_recommended.n_plays += 1 # node has attribute n_plays, initial value is 0
        item_recommended.emp_mean = (item_recommended.emp_mean * max(1, item_recommended.n_plays-1) + reward)/item_recommended.n_plays # node has attribute emp_mean, initial value is 0
        
        if t >= len(self.item_list):
            # option 1, original UCB
            ft = 1 + t * np.log(t) * np.log(t)
            item_recommended.bound = item_recommended.emp_mean + np.sqrt(2 * np.log(ft) / item_recommended.n_plays)
            
            # option 2, Linqi Song
            # A_s = 2
            # item_recommended.bound = item_recommended.emp_mean + np.sqrt(A_s * np.log(t) / item_recommended.n_plays) # each node has attribute bound
            
    def run(self):
        for t in range(self.time_horizon):
            bound_list = [self.item_list[i].bound for i in range(len(self.item_list))]
            item_recommended_id = np.argmax(bound_list)
            item_recommended = self.item_list[item_recommended_id]
            # print("\niteration =", t, "\nrecommend item =", item_recommended.name)
            
            # get reward from look-up table, or human
            reward = 1 - self.get_loss(item_recommended.name) # reward or loss
            # print("reward =", round(reward, 3))

            self.update_stats(t, item_recommended, reward)
            self.update_regret(item_recommended)

    def plot_regret(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig


class GraphUCB(object):
    def __init__(self, dist_lookup, time_horizon, ground_truth=None, test=True, noise=0.01):
        self.dist_lookup = dist_lookup
        
        self.time_horizon = time_horizon
        self.ground_truth = ground_truth
        self.test = test
        self.noise = noise
        
        self._restart()
        
    def _restart(self):
        item_names = sorted(list(self.dist_lookup.keys()))
        self.item_list = [Item(item_name) for item_name in item_names]
        
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        #self.cum_regret_list.append(self.cum_regret)

    def get_loss(self, item_recommended):
        if self.test:
            return self.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)

    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.dist_lookup[self.ground_truth][item_recommended.name]
        self.cum_regret_list.append(self.cum_regret)
        
    def update_stats(self, t, item_recommended, reward):
        #item_recommended=self.item_list[item_recommended_id]
        #item_recommended.n_plays += 1 # node has attribute n_plays, initial value is 0
        #item_recommended.emp_mean = (item_recommended.emp_mean * max(1, item_recommended.n_plays-1) + reward)/item_recommended.n_plays # node has attribute emp_mean, initial value is 0
        for item in self.item_list:
            similarity=1-self.dist_lookup[item_recommended.name][item.name]
            if t<100 or similarity>0.9:
                item.n_plays += similarity
                item.emp_mean = (item.emp_mean * (item.n_plays-similarity) + similarity * reward)/item.n_plays  
        
        #if t >= len(self.item_list):
            # option 1, original UCB
            # ft = 1 + t * np.log(t) * np.log(t)
            # item_recommended.bound = item_recommended.emp_mean + np.sqrt(2 * np.log(ft) / item_recommended.n_plays)
            
            # option 2, Linqi Song
            #A_s = 2
            #item_recommended.bound = item_recommended.emp_mean + np.sqrt(A_s * np.log(t) / item_recommended.n_plays) # each node has attribute bound
            
        #option 3 update all
        A_s = 2
        if t>1:
            for item in self.item_list:
                if item.n_plays > 0:
                    item.bound=item.emp_mean + np.sqrt(A_s * np.log(t)/item.n_plays)


        
    def run(self):
        for t in range(self.time_horizon):
            bound_list = [self.item_list[i].bound for i in range(len(self.item_list))]
            item_recommended_id = np.argmax(bound_list)
            item_recommended = self.item_list[item_recommended_id]
            # print("\niteration =", t, "\nrecommend item =", item_recommended.name)
            
            # get reward from look-up table, or human
            reward = 1 - self.get_loss(item_recommended.name) # reward or loss
            # print("reward =", round(reward, 3))

            self.update_stats(t, item_recommended, reward)
            self.update_regret(item_recommended)

    def plot_regret(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig

class NearNeighborUCB(object):
    def __init__(self, dist_lookup, time_horizon, ground_truth=None, test=True, noise=0.01):
        self.dist_lookup = dist_lookup
        
        self.time_horizon = time_horizon
        self.ground_truth = ground_truth
        self.test = test
        self.noise = noise
        
        self._restart()
        
    def _restart(self):
        item_names = sorted(list(self.dist_lookup.keys()))
        self.item_list = [Item(item_name) for item_name in item_names]
        
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        #self.cum_regret_list.append(self.cum_regret)

    def get_loss(self, item_recommended):
        if self.test:
            return self.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)

    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.dist_lookup[self.ground_truth][item_recommended.name]
        self.cum_regret_list.append(self.cum_regret)
        
    def update_stats(self, t, item_recommended, reward):
        # option 1: update neighbors
        for item in self.item_list:
            similarity=1-self.dist_lookup[item_recommended.name][item.name]
            if t<20 or similarity>0.7:
                item.n_plays += similarity
                item.emp_mean = (item.emp_mean * (item.n_plays-similarity) + similarity * reward)/item.n_plays  

        # option 2: update all
        # for item in self.item_list:
        #     similarity=1-self.dist_lookup[item_recommended.name][item.name]
        #     item.n_plays += similarity
        #     item.emp_mean = (item.emp_mean * (item.n_plays-similarity) + similarity * reward)/item.n_plays  
        
        #if t >= len(self.item_list):
            # option 1, original UCB
            # ft = 1 + t * np.log(t) * np.log(t)
            # item_recommended.bound = item_recommended.emp_mean + np.sqrt(2 * np.log(ft) / item_recommended.n_plays)
            
            # option 2, Linqi Song
            #A_s = 2
            #item_recommended.bound = item_recommended.emp_mean + np.sqrt(A_s * np.log(t) / item_recommended.n_plays) # each node has attribute bound
            
        A_s = 2
        if t>1:
            for item in self.item_list:
                if item.n_plays > 0:
                    item.bound=item.emp_mean + np.sqrt(A_s * np.log(t)/item.n_plays)


        
    def run(self):
        for t in range(self.time_horizon):
            bound_list = [self.item_list[i].bound for i in range(len(self.item_list))]
            item_recommended_id = np.argmax(bound_list)
            item_recommended = self.item_list[item_recommended_id]
            if item_recommended.name == self.ground_truth:
                print("\niteration =", t, "\nrecommend item =", item_recommended.name)
            
            # get reward from look-up table, or human
            reward = 1 - self.get_loss(item_recommended.name) # reward or loss
            # print("reward =", round(reward, 3))

            self.update_stats(t, item_recommended, reward)
            self.update_regret(item_recommended)

    def plot_regret(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig        


class AdaptiveRecommenderSong(object):
    def __init__(self, exptree, time_horizon, user=None, ground_truth=None, test=True, noise = 0.01):
        # self.dist_lookup = dist_lookup
        self.tree = exptree
        # self.user = self.user
        # self.tree = self.filter_context()
        self.time_horizon = time_horizon
        
        self.n_epochs = min(self.tree.n_layers, int(np.log2(self.time_horizon))) # the tree has attribute: n_layers, the total number of layers, -1 so that start from 0
        
        self.ground_truth = ground_truth # ground_truth is known for testing, for real experiment with human, we do not know
        
        self.test = test # whether it is testing or doing experiment with human
        self.noise = noise
        
        self._restart()
        
    def _restart(self):
        self.tree._restart_tree() # set all n_plays, emp_mean, bound 0
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        self.cum_regret_list.append(self.cum_regret)
    
    def filter_context(self):
        # TODO: future, return a trimmed tree, rule out impossible items
        # use self.user and self.tree
        pass
    
    def update_stats(self, t, layer_id, node_selected_id, child_node_selected_id, reward):
        A_s = 2 # TODO: test A_s, A_s is the exploration_exploitation trade-off factor, here use UCB factor
        node_selected = self.tree.get_node(layer_id, node_selected_id)
        node_selected.n_plays += 1 # node has attribute n_plays, initial value is 0
        node_selected.emp_mean = (node_selected.emp_mean * max(1, node_selected.n_plays-1) + reward)/node_selected.n_plays # node has attribute emp_mean, initial value is 0
        
        # option 1, Linqi Song
        node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(t) / node_selected.n_plays) # each node has attribute bound
        
        # option 2, UCB
        # ft = 1 + t * np.log(t) * np.log(t)
        # node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(ft) / node_selected.n_plays)
        
        # option 3
        # node_selected.bound = node_selected.emp_mean
        
        # only update the child level
        if child_node_selected_id:
            child_node_selected = self.tree.get_node(layer_id+1, child_node_selected_id)
            child_node_selected.n_plays += 1
            child_node_selected.emp_mean = (child_node_selected.emp_mean * max(1, child_node_selected.n_plays-1) + reward)/child_node_selected.n_plays
            child_node_selected.bound = child_node_selected.emp_mean + np.sqrt(A_s * np.log(t) / child_node_selected.n_plays)
            
        # update all children
        
    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.tree.dist_lookup[self.ground_truth][item_recommended]
        self.cum_regret_list.append(self.cum_regret)
        # TODO: for human, we have no ground_truth, maybe simply add all the rewards?
        
    def get_loss(self, item_recommended):
        if self.test:
            return self.tree.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)
        
    def run(self):
        rng = np.random.default_rng()#np.random.default_rng(1)
        for layer_id in range(0, self.n_epochs-1):
            partitions = self.tree.get_layer(layer_id) # the tree has function, input layer id, output all the nodes at layer id in a list
            # if layer_id == 0: # first big cluster
            #     partitions[0].bound = 0 # each node has attribute bound
            for t in range(int(2**layer_id), int(2**(layer_id+1))):
                # select cluster
                bound_list = [partitions[i].bound for i in range(len(partitions))]
                node_selected_id = np.argmax(bound_list)
                node_selected = self.tree.get_node(layer_id, node_selected_id) # the tree has function, input layer id and node id, output the node
                
                # randomly select a child node
                child_node_selected_id = rng.choice(node_selected.n_children)
                
                # randomly recommend an item in child node
                possible_items = node_selected.children[child_node_selected_id].items # tree has attribute of items, which return all the image items in this node
                item_recommended = rng.choice(possible_items) # TODO: just recommend this node item
                # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)

                if item_recommended == self.ground_truth:
                    print("\niteration =", t, "\nrecommend item =", item_recommended)
                           
                # get reward from look-up table, or human
                reward = 1 - self.get_loss(item_recommended) # reward or loss
                # print("reward =", round(reward, 3))
                # update parameters
                self.update_stats(t, layer_id, node_selected_id, child_node_selected_id, reward)
                self.update_regret(item_recommended)
                # print("regret =", self.cum_regret)
                
        layer_id = self.n_epochs - 1
        partitions = self.tree.get_layer(layer_id)
        
        for t in range(2**layer_id, self.time_horizon):
            # should reach the last layer
            # select cluster
            bound_list = [partitions[i].bound for i in range(len(partitions))]
            
            node_selected_id = np.argmax(bound_list)
            node_selected = self.tree.get_node(layer_id, node_selected_id)
            # randomly recommend item
            possible_items = node_selected.items
            item_recommended = rng.choice(possible_items)
            # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
            if item_recommended == self.ground_truth:
                    print("\niteration =", t, "\nrecommend item =", item_recommended)
              
            
            reward = 1 - self.get_loss(item_recommended)
            #print("reward =", round(reward, 3))
            
            self.update_stats(t, layer_id, node_selected_id, None, reward)
            self.update_regret(item_recommended)
            
    def plot_regret(self):
        # TODO: two plots
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig


class AdaptiveRecommender(object):
    def __init__(self, exptree, time_horizon, user=None, ground_truth=None, test=True, noise=0.01):
        # self.dist_lookup = dist_lookup
        self.tree = exptree
        # self.user = self.user
        # self.tree = self.filter_context()
        self.time_horizon = time_horizon
        
        self.n_epochs = min(self.tree.n_layers, np.log2(self.time_horizon)) # the tree has attribute: n_layers, the total number of layers, -1 so that start from 0
        
        self.ground_truth = ground_truth # ground_truth is known for testing, for real experiment with human, we do not know
        
        self.test = test # whether it is testing or doing experiment with human
        self.noise = noise
        self._restart()
        
    def _restart(self):
        self.tree._restart_tree() # set all n_plays, emp_mean, bound 0
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        self.cum_regret_list.append(self.cum_regret)
    
    def filter_context(self):
        # TODO: future, return a trimmed tree, rule out impossible items
        # use self.user and self.tree
        pass
    
    def update_stats(self, t, item_recommended_node, reward):
        A_s = 2 # A_s is the exploration_exploitation trade-off factor, here use UCB factor
        # node_selected = self.tree.get_node(layer_id, node_selected_id)
        # node_selected.n_plays += 1 # node has attribute n_plays, initial value is 0
        # node_selected.emp_mean = (node_selected.emp_mean * max(1, node_selected.n_plays-1) + reward)/node_selected.n_plays # node has attribute emp_mean, initial value is 0
        
        # option 1, Linqi Song
        # node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(t) / node_selected.n_plays) # each node has attribute bound
        
        # option 2, UCB
        # ft = 1 + t * np.log(t) * np.log(t)
        # node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(ft) / node_selected.n_plays)
        
        # option 3
        # node_selected.bound = node_selected.emp_mean
        
        # update all parents
        def helper(leaf):
            if not leaf:
                return []
            leaf.n_plays += 1
            leaf.emp_mean = (leaf.emp_mean * max(1, leaf.n_plays-1) + reward)/leaf.n_plays
            leaf.bound = leaf.emp_mean + np.sqrt(A_s * np.log(t) / leaf.n_plays)
            
            # update siblings
            if leaf.layer_id > 0:
                for sibling in leaf.parent.children:
                    # print(sibling.layer_id)
                    similarity = 1-self.tree.dist_lookup[sibling.name][leaf.name]
                    if sibling.name != leaf.name:
                        sibling.n_plays += similarity
                        sibling.emp_mean = (sibling.emp_mean * (sibling.n_plays-similarity) + similarity * reward)/sibling.n_plays
                        if sibling.n_plays > 0:
                            sibling.bound = sibling.emp_mean + np.sqrt(A_s * np.log(t)/sibling.n_plays)
                        
            helper(leaf.parent)
        # node_selected = self.tree.get_node(layer_id, node_selected_id)
        helper(item_recommended_node)
        
    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.tree.dist_lookup[self.ground_truth][item_recommended]
        self.cum_regret_list.append(self.cum_regret)        
        
    def get_loss(self, item_recommended):
        if self.test:
            return self.tree.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)
        
    def run(self):
        rng = np.random.default_rng() #np.random.default_rng(1)
        for layer_id in range(0, self.n_epochs-1):
            partitions = self.tree.get_layer(layer_id) # the tree has function, input layer id, output all the nodes at layer id in a list
            # if layer_id == 0: # first big cluster
            #     partitions[0].bound = 0 # each node has attribute bound
            for t in range(int(2**layer_id), int(2**(layer_id+1))):
                # select cluster
                bound_list = [partitions[i].bound for i in range(len(partitions))]
                node_selected_id = np.argmax(bound_list)
                node_selected = self.tree.get_node(layer_id, node_selected_id) # the tree has function, input layer id and node id, output the node
                
                # randomly recommend an item in node
                possible_items_node = node_selected.items_node # tree has attribute of items, which return all the image items in this node
                item_recommended_node = rng.choice(possible_items_node)
                item_recommended = item_recommended_node.name
                # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
                # if item_recommended.name == self.ground_truth:
                #     print("\niteration =", t, "\nrecommend item =", item_recommended.name)
                                
                # get reward from look-up table, or human
                reward = 1 - self.get_loss(item_recommended) # reward or loss
                # print("reward =", round(reward, 3))
                # update parameters
                self.update_stats(t, item_recommended_node, reward)
                self.update_regret(item_recommended)
                
                # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
                # print("reward =", round(reward, 3))
                # print("regret =", self.cum_regret)
                
        layer_id = self.n_epochs - 1
        partitions = self.tree.get_layer(layer_id)
        
        for t in range(2**layer_id, self.time_horizon):
            # should reach the last layer
            # select cluster
            bound_list = [partitions[i].bound for i in range(len(partitions))]
            
            node_selected_id = np.argmax(bound_list)
            node_selected = self.tree.get_node(layer_id, node_selected_id)
            # randomly recommend item
            possible_items_node = node_selected.items_node # tree has attribute of items, which return all the image items in this node
            item_recommended_node = rng.choice(possible_items_node)
            item_recommended = item_recommended_node.name
            # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
            # if item_recommended.name == self.ground_truth:
            #     print("\niteration =", t, "\nrecommend item =", item_recommended.name)            
                # get reward from look-up table, or human
            reward = 1 - self.get_loss(item_recommended) # reward or loss
            # print("reward =", round(reward, 3))
            # update parameters
            self.update_stats(t, item_recommended_node, reward)
            self.update_regret(item_recommended)
            
            # if t%1000 == 0:
            #     print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
            #     print("reward =", round(reward, 3))
            #     print("regret =", self.cum_regret)

    def plot_regret(self):
        # TODO: two plots
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig


class AdaptiveRecommenderAll(object):
    # recommend all top level representative items first
    def __init__(self, exptree, time_horizon, user=None, ground_truth=None, test=True, noise=0.01):
        # self.dist_lookup = dist_lookup
        self.tree = exptree
        # self.user = self.user
        # self.tree = self.filter_context()
        self.time_horizon = time_horizon
        
        self.n_epochs = min(self.tree.n_layers, np.log2(self.time_horizon)) # the tree has attribute: n_layers, the total number of layers, -1 so that start from 0
        
        self.ground_truth = ground_truth # ground_truth is known for testing, for real experiment with human, we do not know
        
        self.test = test # whether it is testing or doing experiment with human
        self.noise = noise
        self._restart()
        
    def _restart(self):
        self.tree._restart_tree() # set all n_plays, emp_mean, bound 0
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
    
    def filter_context(self):
        # TODO: future, return a trimmed tree, rule out impossible items
        # use self.user and self.tree
        pass
    
    def update_stats(self, t, item_recommended_node, reward):
        A_s = 2 # A_s is the exploration_exploitation trade-off factor, here use UCB factor
        # update all parents
        def helper(leaf):
            if not leaf:
                return []
            leaf.n_plays += 1
            leaf.emp_mean = (leaf.emp_mean * max(1, leaf.n_plays-1) + reward)/leaf.n_plays
            leaf.bound = leaf.emp_mean + np.sqrt(A_s * np.log(t) / leaf.n_plays)
            
            # update siblings
            if leaf.layer_id > 0:
                for sibling in leaf.parent.children:
                    # print(sibling.layer_id)
                    similarity = 1-self.tree.dist_lookup[sibling.name][leaf.name]
                    if sibling.name != leaf.name:
                        sibling.n_plays += similarity
                        sibling.emp_mean = (sibling.emp_mean * (sibling.n_plays-similarity) + similarity * reward)/sibling.n_plays
                        if sibling.n_plays > 0:
                            sibling.bound = sibling.emp_mean + np.sqrt(A_s * np.log(t)/sibling.n_plays)
                        
            helper(leaf.parent)
        if t > 0:
            helper(item_recommended_node)
        
    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.tree.dist_lookup[self.ground_truth][item_recommended]
        self.cum_regret_list.append(self.cum_regret)        
        
    def get_loss(self, item_recommended):
        if self.test:
            return self.tree.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)
        
    def run(self):
        rng = np.random.default_rng() #np.random.default_rng(1)
        t = 0
        for layer_id in range(0, self.n_epochs-1):
            partitions = self.tree.get_layer(layer_id) # the tree has function, input layer id, output all the nodes at layer id in a list
            for node_id in range(len(partitions)):
                node_selected = self.tree.get_node(layer_id, node_id)
                
                for node in self.tree.get_layer(self.tree.n_layers-1):
                    if node.name == node_selected.name:
                        break
                # node_selected's name is the same as node_selected_bottom's name
                # node_selected is on top layer, but node_selected_bottom is at the bottom layer
                
                node_selected_bottom = node # find the node at the bottom
                # recommend the representitive node
                item_recommended = node_selected.name
                # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
                
                # get reward from look-up table, or human
                reward = 1 - self.get_loss(item_recommended) # reward or loss
                # update parameters
                self.update_stats(t, node_selected_bottom, reward)
                self.update_regret(item_recommended)
                t += 1
                
        layer_id = self.n_epochs - 1
        partitions = self.tree.get_layer(layer_id)
        
        while t < self.time_horizon:
            # should reach the last layer
            # select cluster
            bound_list = [partitions[i].bound for i in range(len(partitions))]
            
            node_selected_id = np.argmax(bound_list)
            node_selected = self.tree.get_node(layer_id, node_selected_id)
            # randomly recommend item
            possible_items_node = node_selected.items_node # tree has attribute of items, which return all the image items in this node
            item_recommended_node = rng.choice(possible_items_node)
            item_recommended = item_recommended_node.name
            if item_recommended == self.ground_truth:
                print("\niteration =", t, "\nrecommend item =", item_recommended)   
            # get reward from look-up table, or human
            reward = 1 - self.get_loss(item_recommended) # reward or loss
            # print("reward =", round(reward, 3))
            # update parameters
            self.update_stats(t, item_recommended_node, reward)
            self.update_regret(item_recommended)
            t += 1

    def plot_regret(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig
    

class AdaptiveRecommenderRe(object):
    # recommend top levels representative items
    def __init__(self, exptree, time_horizon, user=None, ground_truth=None, test=True, noise=0.01):
        # self.dist_lookup = dist_lookup
        self.tree = exptree
        # self.user = self.user
        # self.tree = self.filter_context()
        self.time_horizon = time_horizon
        
        self.n_epochs = min(self.tree.n_layers, np.log2(self.time_horizon)) # the tree has attribute: n_layers, the total number of layers, -1 so that start from 0
        
        self.ground_truth = ground_truth # ground_truth is known for testing, for real experiment with human, we do not know
        
        self.test = test # whether it is testing or doing experiment with human
        self.noise = noise
        self._restart()
        
    def _restart(self):
        self.tree._restart_tree() # set all n_plays, emp_mean, bound 0
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        self.cum_regret_list.append(self.cum_regret)
    
    def filter_context(self):
        # TODO: future, return a trimmed tree, rule out impossible items
        # use self.user and self.tree
        pass
    
    def update_stats(self, t, item_recommended_node, reward):
        A_s = 1 # A_s is the exploration_exploitation trade-off factor, here use UCB factor
        
        # update all parents
        def helper(leaf):
            if not leaf:
                return []
            leaf.n_plays += 1
            leaf.emp_mean = (leaf.emp_mean * max(1, leaf.n_plays-1) + reward)/leaf.n_plays
            leaf.bound = leaf.emp_mean + np.sqrt(A_s * np.log(t) / leaf.n_plays)
            
            # update siblings
            if leaf.layer_id > 0:
                for sibling in leaf.parent.children:
                    # print(sibling.layer_id)
                    similarity = 1-self.tree.dist_lookup[sibling.name][leaf.name]
                    if sibling.name != leaf.name:
                        sibling.n_plays += similarity
                        sibling.emp_mean = (sibling.emp_mean * (sibling.n_plays-similarity) + similarity * reward)/sibling.n_plays
                        if sibling.n_plays > 0:
                            sibling.bound = sibling.emp_mean + np.sqrt(A_s * np.log(t)/sibling.n_plays)
                        
            helper(leaf.parent)
        # node_selected = self.tree.get_node(layer_id, node_selected_id)
        helper(item_recommended_node)
        
    def update_regret(self, item_recommended):
        # record cumulative regret
        self.cum_regret += self.tree.dist_lookup[self.ground_truth][item_recommended]
        self.cum_regret_list.append(self.cum_regret)        
        
    def get_loss(self, item_recommended):
        if self.test:
            return self.tree.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
            # return self.tree.dist_lookup[self.ground_truth][item_recommended]
        else:
            loss = input("How close is this image to your thought: ")
            # larger distance = bad prediction = larger loss
            return float(loss)
        
    def run(self):
        rng = np.random.default_rng() #np.random.default_rng(1)
        for layer_id in range(0, self.n_epochs-1):
            partitions = self.tree.get_layer(layer_id) # the tree has function, input layer id, output all the nodes at layer id in a list
            for t in range(int(2**layer_id), int(2**(layer_id+1))):
                # select cluster
                bound_list = [partitions[i].bound for i in range(len(partitions))]
                node_selected_id = np.argmax(bound_list)
                node_selected = self.tree.get_node(layer_id, node_selected_id) # the tree has function, input layer id and node id, output the node
                
                for node in self.tree.get_layer(self.tree.n_layers-1):
                    if node.name == node_selected.name:
                        break
                # node_selected's name is the same as node_selected_bottom's name
                # node_selected is on top layer, but node_selected_bottom is at the bottom layer
                
                node_selected_bottom = node # find the node at the bottom
                # recommend the representitive node
                item_recommended = node_selected.name
                # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
                if item_recommended == self.ground_truth:
                    print("\niteration =", t, "\nrecommend item =", item_recommended)                
                # get reward from look-up table, or human
                reward = 1 - self.get_loss(item_recommended) # reward or loss
                # update parameters
                self.update_stats(t, node_selected_bottom, reward)
                self.update_regret(item_recommended)
                
        layer_id = self.n_epochs - 1
        partitions = self.tree.get_layer(layer_id)
        
        for t in range(2**layer_id, self.time_horizon):
            # should reach the last layer
            # select cluster
            bound_list = [partitions[i].bound for i in range(len(partitions))]
            
            node_selected_id = np.argmax(bound_list)
            node_selected = self.tree.get_node(layer_id, node_selected_id)
            # randomly recommend item
            possible_items_node = node_selected.items_node # tree has attribute of items, which return all the image items in this node
            item_recommended_node = rng.choice(possible_items_node)
            item_recommended = item_recommended_node.name
            # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
            if item_recommended == self.ground_truth:
                print("\niteration =", t, "\nrecommend item =", item_recommended)                
        
            # get reward from look-up table, or human
            reward = 1 - self.get_loss(item_recommended) # reward or loss
            # print("reward =", round(reward, 3))
            # update parameters
            self.update_stats(t, item_recommended_node, reward)
            self.update_regret(item_recommended)

    def plot_regret(self):
        # TODO: two plots
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.time_horizon), self.cum_regret_list)
        ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
        return fig