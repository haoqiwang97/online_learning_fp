# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:35:08 2021

@author: hw9335
"""

from tree import *
from adaptive_recommender import *

import numpy as np
import pandas as pd


data = pd.read_excel('data/NOUN_Sorting_Tables.xlsx', usecols=[1, 2, 3])
data = data.values.tolist()[1:]
dist_lookup = build_dist_lookup(data)

exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
exptree.build_tree()
#exptree.print_tree()

recommender = AdaptiveRecommender(exptree=exptree,
                                  time_horizon=1000,
                                  user=None,
                                  ground_truth='I_2055',
                                  test=True)

recommender.run()

exptree.tree_stru.children

partitions = recommender.tree.get_layer(3)
bound_list = [partitions[i].bound for i in range(len(partitions))]
n_plays_list = [partitions[i].n_plays for i in range(len(partitions))]
emp_mean_list = [partitions[i].emp_mean for i in range(len(partitions))]
