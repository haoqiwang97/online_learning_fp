import numpy as np
from Node import Node

def build_tree(items=None, dist_lookup=None, b=0, L=0):
    """
    input: items(name), distance of items, b, depth L
    output: a hierchical tree
    """

    return 

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
        else:
            dists[d[0]][d[1]] = float(d[2])
        # undirected, so add distance from another direction
        if d[1] not in dists.keys():
            dists[d[1]] = {}
        else:
            dists[d[1]][d[0]] = float(d[2])

    return dists

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_excel('data/NOUN_Sorting_Tables.xlsx', usecols=[1,2,3])
    data = data.values.tolist()[1:] 
    dist_lookup = build_dist_lookup(data)
    items = sorted(list(dist_lookup.keys()))
    from IPython import embed; embed()
    build_tree(items, dist_lookup, b=0.6, L=3)
