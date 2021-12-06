# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:46:02 2021

@author: hw9335
"""
import support_func
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
    def __init__(self, dist_lookup, time_horizon, rho=0.0001, delta=0.0001, eps=0, ground_truth=None, test=True, noise=0.01):
        self.dist_lookup = dist_lookup
        self.L = self.cal_laplacian(self.dist_lookup) 
        self.time_horizon = time_horizon
        self.ground_truth = ground_truth
        self.test = test
        self.noise = noise
        self.rho = rho
        self.delta = delta 
        self.dim = len(self.L)
        self.eps = eps
        self.eta = 0.0
        self.remaining_nodes = [i for i in range(self.dim)]
        self.L_rho = self.eta * self.L + self.rho * np.identity(self.dim)
        self.counter = np.zeros((self.dim, self.dim))
        self.conf_width = np.zeros(self.dim)
        self.total_reward = np.zeros(self.dim)
        self.mean_estimate = np.zeros(self.dim)
        #self.clusters = support_func.get_clusters(self.A)
        #self.jumping_index = np.array(support_func.jumping_list(self.clusters, self.dim))

        self.beta_tracker = 0.0
        self.inverse_tracker = np.zeros((self.dim, self.dim))
        self.picking_order = []
        # self.global_tracker_conf_width = []

        self.initialize_conf_width()
        self._restart()
    
    def cal_laplacian(self, dist_lookup):
        nodes = sorted(self.dist_lookup.keys())
        U = []
        W_diag = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                W_diag.append(self.dist_lookup[nodes[i]][nodes[j]])
                ue = [0 for x in range(len(nodes))]
                if i!=j:
                    ue[i] = 1
                    ue[j] = -1
                U.append(ue)
        U=np.array(U)
        W=np.diag(W_diag)
        L=np.matmul(np.matmul(U.T,W),U)
        return L    

    def initialize_conf_width(self):
        """
        Initialize confidence width of all arms.
        """
        v_t_inverse = np.linalg.inv(self.counter + self.L_rho)
        self.inverse_tracker = v_t_inverse
        self.update_conf_width()
    
    def compute_imperfect_info(self):
        """
        Computing the error cause of imperfect graph information
        Returns
        -------
        <x, Lx> : quadratic error value
        """
        return support_func.matrix_norm(self.means, self.L)

    def _restart(self):
        item_names = sorted(list(self.dist_lookup.keys()))
        self.item_list = [Item(item_name) for item_name in item_names]
        
        self.counter = np.zeros((self.dim, self.dim))
        self.cum_regret = 0.0
        
        self.cum_regret_list = []
        #self.cum_regret_list.append(self.cum_regret)
    
    def required_reset(self):
        """
        Reset all the arm-counter to 0.
        """
        if self.reset:
            self.counter = np.zeros((self.dim, self.dim))
    
    def update_conf_width(self):
        """
        Update confidence width of all arms.
        """
        for i in range(self.dim):
            self.conf_width[i] = np.sqrt(self.inverse_tracker[i, i])
    
    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index

    def play_arm(self, index, reward):
        """
        Update counter and reward based on arm played.
        Parameters
        ----------
        index : Arm being played in the current round.
        """
        self.picking_order.append(index)
        counter_vec = np.zeros(self.dim)
        counter_vec[index] = 1
        old_v_t_inverse = self.inverse_tracker
        v_t_inverse = support_func.sherman_morrison_inverse(counter_vec, old_v_t_inverse)
        self.inverse_tracker = v_t_inverse
        self.update_conf_width()

        # FIXME : Testing to remove function "increment_count"
        # #self.increment_count(index)
        self.counter[index, index] += 1
        current_counter = np.array(self.counter)
        # self.counter_tracker.append(current_counter)
        # self.update_conf_width()

        #reward = support_func.gaussian_reward(self.means[index])
        self.total_reward[index] = self.total_reward[index] + reward

    def estimate_mean(self):
        """
        Estimate mean using quadratic Laplacian closed form expression.
        """

        self.mean_estimate = np.dot(self.inverse_tracker, self.total_reward)

    def eliminate_arms(self):
        """
        Eliminate arms based on UCB-style argument.
        """

        # TODO : Need to change log(T) to  log(|A_i|)

        beta = 2 * np.sqrt(14 * np.log2(2 * self.dim * np.trace(self.counter) / self.delta)) + 0.5 * self.eta * self.eps
        self.beta_tracker = beta
        temp_array = np.zeros(self.dim)

        # FIXME : Testing commented out code with alternative.
        for i in self.remaining_nodes:
            temp_array[i] = self.mean_estimate[i] - beta * self.conf_width[i]

        max_value = max(temp_array)
        self.remaining_nodes = [i for i in self.remaining_nodes if
                                self.mean_estimate[i] + beta * self.conf_width[i] >= max_value]
    
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
            item_recommended_id = self.select_arm()
            item_recommended = self.item_list[item_recommended_id]
            
            reward = 1 - self.get_loss(item_recommended.name) # reward or loss
            self.play_arm(item_recommended_id, reward)
            self.estimate_mean()
            self.eliminate_arms()
            #from IPython import embed; embed()
            #self.update_stats(t, item_recommended, reward)
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
            #print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)

            
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
# TODO: update selected child's children? update selected item?

# update all children, cannot find global optimal, HW 2021-12-05
# class AdaptiveRecommender(object):
#     def __init__(self, exptree, time_horizon, user=None, ground_truth=None, test=True, noise=0.01):
#         # self.dist_lookup = dist_lookup
#         self.tree = exptree
#         # self.user = self.user
#         # self.tree = self.filter_context()
#         self.time_horizon = time_horizon
        
#         self.n_epochs = min(self.tree.n_layers, np.log2(self.time_horizon)) # the tree has attribute: n_layers, the total number of layers, -1 so that start from 0
        
#         self.ground_truth = ground_truth # ground_truth is known for testing, for real experiment with human, we do not know
        
#         self.test = test # whether it is testing or doing experiment with human
#         self.noise = noise
#         self._restart()
        
#     def _restart(self):
#         self.tree._restart_tree() # set all n_plays, emp_mean, bound 0
#         self.cum_regret = 0.0
    
#     def filter_context(self):
#         # TODO: future, return a trimmed tree, rule out impossible items
#         # use self.user and self.tree
#         pass
    
#     def update_stats(self, t, layer_id, node_selected_id, reward):
#         A_s = 2 # A_s is the exploration_exploitation trade-off factor, here use UCB factor
#         # node_selected = self.tree.get_node(layer_id, node_selected_id)
#         # node_selected.n_plays += 1 # node has attribute n_plays, initial value is 0
#         # node_selected.emp_mean = (node_selected.emp_mean * max(1, node_selected.n_plays-1) + reward)/node_selected.n_plays # node has attribute emp_mean, initial value is 0
        
#         # option 1, Linqi Song
#         # node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(t) / node_selected.n_plays) # each node has attribute bound
        
#         # option 2, UCB
#         # ft = 1 + t * np.log(t) * np.log(t)
#         # node_selected.bound = node_selected.emp_mean + np.sqrt(A_s * np.log(ft) / node_selected.n_plays)
        
#         # option 3
#         # node_selected.bound = node_selected.emp_mean
        
#         # update all children
#         def helper(root, layer_id):
#             if not root:
#                 return []
#             root.n_plays += 1
#             root.emp_mean = (root.emp_mean * max(1, root.n_plays-1) + reward)/root.n_plays
#             root.bound = root.emp_mean + np.sqrt(A_s * np.log(t) / root.n_plays)
#             for child in root.children:
#                 helper(child, layer_id + 1)
#         node_selected = self.tree.get_node(layer_id, node_selected_id)
#         helper(node_selected, 0)
        
        
        
        
#     def update_regret(self, item_recommended):
#         # record cumulative regret
#         self.cum_regret += self.tree.dist_lookup[self.ground_truth][item_recommended]
#         # TODO: for human, we have no ground_truth, maybe simply add all the rewards?
        
#     def get_loss(self, item_recommended):
#         if self.test:
#             return self.tree.dist_lookup[self.ground_truth][item_recommended] + np.random.default_rng().standard_normal() * self.noise
#             # return self.tree.dist_lookup[self.ground_truth][item_recommended]
#         else:
#             loss = input("How close is this image to your thought: ")
#             # larger distance = bad prediction = larger loss
#             return float(loss)
        
#     def run(self):
#         rng = np.random.default_rng() #np.random.default_rng(1)
#         for layer_id in range(0, self.n_epochs-1):
#             partitions = self.tree.get_layer(layer_id) # the tree has function, input layer id, output all the nodes at layer id in a list
#             # if layer_id == 0: # first big cluster
#             #     partitions[0].bound = 0 # each node has attribute bound
#             for t in range(int(2**layer_id), int(2**(layer_id+1))):
#                 # select cluster
#                 bound_list = [partitions[i].bound for i in range(len(partitions))]
#                 node_selected_id = np.argmax(bound_list)
#                 node_selected = self.tree.get_node(layer_id, node_selected_id) # the tree has function, input layer id and node id, output the node
                
#                 # randomly recommend an item in node
#                 possible_items = node_selected.items # tree has attribute of items, which return all the image items in this node
#                 item_recommended = rng.choice(possible_items)
#                 # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)
                
#                 # get reward from look-up table, or human
#                 reward = 1 - self.get_loss(item_recommended) # reward or loss
#                 # print("reward =", round(reward, 3))
#                 # update parameters
#                 self.update_stats(t, layer_id, node_selected_id, reward)
#                 self.update_regret(item_recommended)
#                 # print("regret =", self.cum_regret)
                
#         layer_id = self.n_epochs - 1
#         partitions = self.tree.get_layer(layer_id)
        
#         for t in range(2**layer_id, self.time_horizon):
#             # should reach the last layer
#             # select cluster
#             bound_list = [partitions[i].bound for i in range(len(partitions))]
            
#             node_selected_id = np.argmax(bound_list)
#             node_selected = self.tree.get_node(layer_id, node_selected_id)
#             # randomly recommend item
#             possible_items = node_selected.items
#             item_recommended = rng.choice(possible_items)
#             # print("epoch =", layer_id, "\niteration =", t, "\nrecommend node =", node_selected_id, "\nrecommend item =", item_recommended)

            
#             reward = 1 - self.get_loss(item_recommended)
#             # print("reward =", round(reward, 3))
            
#             self.update_stats(t, layer_id, node_selected_id, reward)
#             self.update_regret(item_recommended)
