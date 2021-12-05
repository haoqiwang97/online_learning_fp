# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:35:08 2021

@author: hw9335
"""

from tree import *
from adaptive_recommender import *

import numpy as np
import pandas as pd


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    
    parser.add_argument('--recommender', type=str, default="UCB", help="choose a recommender algorithm")
    parser.add_argument('--data_path', type=str, default="data/NOUN_Sorting_Tables.xlsx")
    
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

data = pd.read_excel(args.data_path, usecols=[1, 2, 3])
data = data.values.tolist()[1:]
dist_lookup = build_dist_lookup(data)

if args.recommender == "UCB":
    recommender = UCB(dist_lookup=dist_lookup, 
                      time_horizon=10000, 
                      ground_truth='I_2055', 
                      test=True)
    # item_list = [recommender.item_list[i].name for i in range(len(recommender.item_list))]
    # n_plays_list = [recommender.item_list[i].n_plays for i in range(len(recommender.item_list))]
    
elif args.recommender == "AdaptiveRecommenderSong":
    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    #exptree.print_tree()
    
    recommender = AdaptiveRecommenderSong(exptree=exptree,
                                          time_horizon=10000,
                                          user=None,
                                          ground_truth='I_2055',
                                          test=True)

    # exptree.tree_stru.children
    
    # partitions = recommender.tree.get_layer(3)
    # bound_list = [partitions[i].bound for i in range(len(partitions))]
    # n_plays_list = [partitions[i].n_plays for i in range(len(partitions))]
    # emp_mean_list = [partitions[i].emp_mean for i in range(len(partitions))]

elif args.recommender == "AdaptiveRecommender":
    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    recommender = AdaptiveRecommender(exptree=exptree,
                                      time_horizon=10000,
                                      user=None,
                                      ground_truth='I_2055',
                                      test=True)



recommender.run()
fig = recommender.plot_regret()


