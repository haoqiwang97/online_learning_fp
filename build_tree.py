import random
import numpy as np
from Node import Node

def build_tree(items=None, dist_lookup=None, b=0, L=0):
    """
    input: items(name), distance of items, b, depth L
    output: a hierchical tree
    """
    index_name = [(idx, item) for idx,item in enumerate(items)]
    visited_lower = list(np.arange(len(index_name)))
    C_lower = []
    for item in index_name:
        C_lower.append(Node(item[0], item[1], L))
    for i in range(L-1):
        l=L-i
        C_higher = []
        while len(visited_lower)>0:
            idx = random.choice(visited_lower)
            new_node = Node(C_lower[idx].index, C_lower[idx].name, C_lower[idx].depth-1)
            visited_lower.remove(idx)
            for i in range(len(C_lower)):
                c = C_lower[i]
                if c.name!=new_node.name and dist_lookup[c.name][new_node.name] < (1-b)*b**l/(1+b) and i in visited_lower:
                    new_Node.add_child(c)
                    c.add_parent(new_Node)
                    visited_lower.remove(i)
            C_higher.append(new_node)
        C_lower=C_higher
        visited_lower=list(np.arange(len(C_lower)))
    
    return C_higher 

def build_dist_lookup(data):
    """
    This function builds a distance look up table for two points
    It is undirected 
    In the form of [node1][node2][distance]
    """
    dists = {}
    for d in data:
        if d[0] not in dists.keys():
            dists[d[0]] = {}
        if d[1] not in dists.keys():
            dists[d[1]] = {}
        dists[d[1]][d[0]] = float(d[2])
        dists[d[0]][d[1]] = float(d[2])

    return dists

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_excel('data/NOUN_Sorting_Tables.xlsx', usecols=[1,2,3])
    data = data.values.tolist()[1:] 
    dist_lookup = build_dist_lookup(data)
    items = sorted(list(dist_lookup.keys()))
    build_tree(items, dist_lookup, b=0.6, L=3)
