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
exptree.print_tree()

recommender = AdaptiveRecommender(exptree=exptree,
                                  time_horizon=20,
                                  user=None,
                                  ground_truth='I_2055',
                                  test=True)

recommender.run()
