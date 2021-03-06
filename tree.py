# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:41:34 2021

@author: hw9335
"""

import numpy as np


class Node(object):
    def __init__(self, layer_id, node_id, name=None):
        """
        construction of node 
        need to at least know the name
        """
        self.layer_id = layer_id  # layer_id
        self.node_id = node_id  # node_id
        self.parent = None
        self.children = []  # last layer does not have child
        self.name = name
        self.n_plays = 0
        self.emp_mean = 0
        self.bound = 1e5 #0  # TODO: Q1_UCB_EE381V.ipynb set 1e5*np.ones(self.num_arms), index is upper bound value

    def _restart_node(self):
        self.n_plays = 0
        self.emp_mean = 0
        self.bound = 1e5 #0  # TODO: Q1_UCB_EE381V.ipynb set 1e5*np.ones(self.num_arms), index is upper bound value

    def add_child(self, child):
        self.children.append(child)

    @property
    def n_children(self):
        return len(self.children)

    def set_parent(self, parent):
        self.parent = parent

    @property
    def items(self):
        # go to the bottom layer and return all items
        result = []

        def helper(root, layer_id):
            if not root:
                return []
            if len(result) == layer_id:
                result.append([])
            result[layer_id].append(root.name)
            for child in root.children:
                helper(child, layer_id + 1)
        helper(self, 0)
        return result[-1]

    @property
    def items_node(self):
        # go to the bottom layer and return all items
        result = []

        def helper(root, layer_id):
            if not root:
                return []
            if len(result) == layer_id:
                result.append([])
            result[layer_id].append(root)
            for child in root.children:
                helper(child, layer_id + 1)
        helper(self, 0)
        return result[-1]
    
    def __repr__(self):
        # return "layer_id=" + repr(self.layer_id) + ";\nnode_id=" + repr(self.node_id) + ";\nparent=" + repr(self.parent) + ";\nchildren=" + repr(self.children)
        # + ";\nparent=" + repr(self.parent) + ";\nchildren=" + repr(self.children)
        return "layer_id=" + repr(self.layer_id) + "\nnode_id=" + repr(self.node_id) + "\nn_plays=" + repr(self.n_plays) + "\nemp_mean=" + repr(self.emp_mean) + "\nbound=" + repr(self.bound)

    def __str__(self):
        return self.__repr__()


class ExpTree(object):
    def __init__(self, b, n_layers, dist_lookup):
        self.b = b
        self.n_layers = n_layers  # depth of tree
        self.dist_lookup = dist_lookup
        # A, constant like 2 in UCB in square root
    
    def select_lower(self, visited_lower, node_list, mode='random', radius=1.0):
        # input: available index, node list
        # output: index
        rng = np.random.default_rng(1)
        if mode == 'random':
            return rng.choice(visited_lower)
        if mode == 'farthest':
            node_neighbor_count={}
            for idx in visited_lower:
                if idx not in node_neighbor_count.keys():
                    node_neighbor_count[idx]=0
                    for neighbor_idx in visited_lower:
                        if self.dist_lookup[node_list[idx].name][node_list[neighbor_idx].name]<radius:
                            node_neighbor_count[idx]+=1
            farthest_idx=visited_lower[0]
            for key, value in node_neighbor_count.items():
                if value < node_neighbor_count[farthest_idx]:
                    farthest_idx = key
            return farthest_idx
        return visited_lower[0]

    def build_tree(self):
        """
        input: items(name), distance of items, b in (0.5, 1), depth L
        output: a hierchical tree
        """
        
        # item_names = sorted(list(dist_lookup.keys()))
        dist_lookup = self.dist_lookup
        item_names = sorted(list(dist_lookup.keys()))
        b = self.b
        n_layers = self.n_layers

        rng = np.random.default_rng(1)
        # TODO: find Lb
        # make a class of lookup table, only pass in the instance

        visited_lower = list(np.arange(len(item_names)))
        C_lower = []

        # build bottom layer
        for idx, item_name in enumerate(item_names):
            C_lower.append(Node(n_layers-1, idx, item_name))
            
        for i in range(n_layers-1):  # layer
            l = n_layers-i-1
            C_higher = []
            node_id = 0 # counter for the number of nodes at layer
            while len(visited_lower) > 0:  # node
                #idx = rng.choice(visited_lower)  # arbitrary select an element
                #do not randomly select function input:[layer nodes], output:index with least number of satisfied nodes
                idx = self.select_lower(visited_lower, C_lower, mode='farthest', radius=b**(l-1))
                new_node = Node(C_lower[idx].layer_id-1, # 1 layer up
                                node_id, # node_id start from 0
                                C_lower[idx].name) 
                node_id += 1

                for j in range(len(C_lower)):
                    c = C_lower[j]
                    #threshold = (1-b)*b**l/(1+b) used by the original paper, but it does not help return the only-one result in the highest layer
                    if dist_lookup[c.name][new_node.name] <= b**(l-1) and j in visited_lower:
                        # find the subset near to the arbitrary element
                        new_node.add_child(c)
                        c.set_parent(new_node)
                        visited_lower.remove(j)
                C_higher.append(new_node)
            C_lower = C_higher
            visited_lower = list(np.arange(len(C_lower)))
            # print("Level {}".format(l))
            # for c in C_higher:
            #     print(c.name)

        self.tree_stru = C_higher[0]
        self._get_all_layers()

    def get_layer(self, layer_id):
        # return list of nodes at layer_id
        return self._all_layers[layer_id]
    
    def _get_all_layers(self):
        result = []

        def helper(root, layer_id):
            if not root:
                return []
            if len(result) == layer_id:
                result.append([])
            result[layer_id].append(root)
            for child in root.children:
                helper(child, layer_id + 1)
        helper(self.tree_stru, 0)
        self._all_layers = result
        
    def get_node(self, layer_id, node_id):
        return self._all_layers[layer_id][node_id]

    def _restart_tree(self):
        # iterate all the nodes and restart
        def helper(root, layer_id):
            if not root:
                return []
            root._restart_node()
            for child in root.children:
                helper(child, layer_id + 1)
        helper(self.tree_stru, 0)
        
    def print_tree(self):
        def helper(root, layer_id):
            if not root:
                return []
            if layer_id == 0:
                print(' ' * 4 * layer_id + '->', root.name)
            if layer_id > 0:
                print(' ' * 4 * layer_id + '->', root.name, round(self.dist_lookup[root.name][root.parent.name], 3))
            for child in root.children:
                helper(child, layer_id + 1)
        helper(self.tree_stru, 0)        


def build_dist_lookup(data, normalization=True):
    """
    This function builds a distance look up table for two points
    It is undirected 
    In the form of [node1][node2][distance]
    """
    if normalization:
        max_dist = 0
        for d in data:
            max_dist = max(max_dist, float(d[2]))
    else:
        max_dist = 1.0

    dists = {}
    for d in data:
        if d[0] not in dists.keys():
            dists[d[0]] = {}
            dists[d[0]][d[0]] = 0
        if d[1] not in dists.keys():
            dists[d[1]] = {}
            dists[d[1]][d[1]] = 0
        dists[d[1]][d[0]] = float(d[2])/max_dist
        dists[d[0]][d[1]] = float(d[2])/max_dist
    return dists


if __name__ == '__main__':
    node = Node(0, 5, 'node_0')
    print(node)

    import pandas as pd
    data = pd.read_excel('data/NOUN_Sorting_Tables.xlsx', usecols=[1, 2, 3])
    data = data.values.tolist()[1:]
    dist_lookup = build_dist_lookup(data)

    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    exptree.get_layer(1)
    exptree.get_node(2, 1).children
    exptree.get_node(2, 1).n_children
    exptree._restart_tree()
    exptree.print_tree()
