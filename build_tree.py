#import random
import numpy as np
from Node import Node


def build_tree(items=None, dist_lookup=None, b=0, L=0):
    """
    input: items(name), distance of items, b in (0.5, 1), depth L
    output: a hierchical tree
    """
    rng = np.random.default_rng(1)
    #print(dist_lookup) 
    # TODO: find Lb
    # make a class of lookup table, only pass in the instance
    index_name = [(idx, item) for idx, item in enumerate(items)]
    visited_lower = list(np.arange(len(index_name)))
    C_lower = []
    for item in index_name:
        C_lower.append(Node(item[0], item[1], L))
    for i in range(L-1): # layer
        l = L-i-1
        C_higher = []
        # TODO: k = 0
        while len(visited_lower) > 0: # node
            idx = rng.choice(visited_lower) # arbitrary select an element
            new_node = Node(C_lower[idx].index,
                            C_lower[idx].name, 
                            C_lower[idx].depth-1)
            #visited_lower.remove(idx)
            for j in range(len(C_lower)): # TODO: map between name and index
                c = C_lower[j]
                #threshold = (1-b)*b**l/(1+b) used by the original paper, but it does not help return the only-one result in the highest layer
                if dist_lookup[c.name][new_node.name] <= b**(l-1) and j in visited_lower:
                    new_node.add_child(c) # find the subset near to the arbitrary element
                    c.set_parent(new_node)
                    visited_lower.remove(j)
            C_higher.append(new_node)
        C_lower = C_higher
        visited_lower = list(np.arange(len(C_lower)))
        print("Level {}".format(l))
        for c in C_higher:
            print(c.name)
    #from IPython import embed; embed()
    return C_higher

def build_dist_lookup(data, normalization=True):
    """
    This function builds a distance look up table for two points
    It is undirected 
    In the form of [node1][node2][distance]
    """
    dists = {}
    max_dist = 0
    for d in data:
        max_dist = max(max_dist, float(d[2]))

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
    import pandas as pd
    data = pd.read_excel('data/NOUN_Sorting_Tables.xlsx', usecols=[1,2,3])
    data = data.values.tolist()[1:] 
    dist_lookup = build_dist_lookup(data)
    items = sorted(list(dist_lookup.keys()))
    C_higher = build_tree(items, dist_lookup, b=0.6, L=4)

# TODO: given an item, knows its layer_id, node_id
# TODO: given layer_id, node_id, know its items



