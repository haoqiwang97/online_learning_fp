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
        self.children = []  # TODO: last layer does not have child
        self.name = name

    def _restart_node(self):
        self.n_plays = 0
        self.emp_mean = 0
        self.bound = 0  # index is upper bound value

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

    def __repr__(self):
        # return "layer_id=" + repr(self.layer_id) + ";\nnode_id=" + repr(self.node_id) + ";\nparent=" + repr(self.parent) + ";\nchildren=" + repr(self.children)
        # + ";\nparent=" + repr(self.parent) + ";\nchildren=" + repr(self.children)
        return "layer_id=" + repr(self.layer_id) + "\nnode_id=" + repr(self.node_id)

    def __str__(self):
        return self.__repr__()


class ExpTree(object):
    def __init__(self, b, n_layers):
        # self.distance_mt = distance_mt
        #self.dist_lookup = dist_lookup
        self.b = b
        self.n_layers = n_layers  # depth of tree
        # A, constant like 2 in UCB in square root

    def build_tree(self, dist_lookup):
        """
        input: items(name), distance of items, b in (0.5, 1), depth L
        output: a hierchical tree
        """
        # items = sorted(list(self.dist_lookup.keys()))
        items = sorted(list(dist_lookup.keys()))
        # dist_lookup = self.dist_lookup
        b = self.b
        n_layers = self.n_layers

        rng = np.random.default_rng(1)
        #print(dist_lookup)
        # TODO: find Lb
        # make a class of lookup table, only pass in the instance
        index_name = [(idx, item) for idx, item in enumerate(items)]
        visited_lower = list(np.arange(len(index_name)))
        C_lower = []
        for item in index_name:
            C_lower.append(Node(n_layers, item[0], item[1]))
        for i in range(n_layers-1):  # layer
            l = n_layers-i-1
            C_higher = []
            # TODO: k = 0
            while len(visited_lower) > 0:  # node
                idx = rng.choice(visited_lower)  # arbitrary select an element
                new_node = Node(C_lower[idx].layer_id-1,
                                C_lower[idx].node_id, C_lower[idx].name)
                #visited_lower.remove(idx)
                for j in range(len(C_lower)):  # TODO: map between name and index
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
            print("Level {}".format(l))
            for c in C_higher:
                print(c.name)

        self.tree_stru = C_higher[0]
        self._get_all_layers()

    def get_layer(self, layer_id):
        # TODO: name should be index, not item namesS
        # return list of nodes at layer_id
        return self._all_layers[layer_id]
    
    def _get_all_layers(self):
        # TODO: name should be index, not item namesS
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
        
    # TODO: NUMBER OF nodes at layer l
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

    exptree = ExpTree(b=0.6, n_layers=4)
    exptree.build_tree(dist_lookup)
    exptree.get_layer(1)
    exptree.get_node(2, 1).children
    exptree.get_node(2, 1).n_children
    exptree._restart_tree()
