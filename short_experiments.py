# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:13:24 2021

@author: hw9335
"""

from tree import *
from adaptive_recommender import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    
    # parser.add_argument('--recommender', type=str, default="UCB", help="choose a recommender algorithm")
    parser.add_argument('--recommender', type=str, default="NearNeighborUCB", help="choose a recommender algorithm")
    # parser.add_argument('--recommender', type=str, default="AdaptiveRecommenderRe", help="choose a recommender algorithm")
    
    parser.add_argument('--data_path', type=str, default="data/NOUN_Sorting_Tables.xlsx")
    parser.add_argument('--time_horizon', type=int, default=50)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--ground_truth', type=str, default="I_2051")
    parser.add_argument('--test', type=bool, default=True, help="test or not")
    
    parser.add_argument('--do_short_experiments', type=bool, default=False) # short experiments
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

data = pd.read_excel(args.data_path, usecols=[1, 2, 3])
data = data.values.tolist()[1:]
dist_lookup = build_dist_lookup(data)


if args.recommender == "NearNeighborUCB":
    recommender = NearNeighborUCB(dist_lookup=dist_lookup,
                                  time_horizon=args.time_horizon,
                                  ground_truth=args.ground_truth,
                                  test=True)
    
elif args.recommender == "AdaptiveRecommenderRe":
    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    recommender = AdaptiveRecommenderRe(exptree=exptree,
                                      time_horizon=args.time_horizon,
                                      user=None,
                                      ground_truth=args.ground_truth,
                                      test=True,
                                      noise=args.noise)
# recommender.run()
# fig = recommender.plot_regret()

# counter = 0
# results_len = [len(value) for key, value in results.items()]
# for length in results_len:
#     if length > 0:
#         counter += 1
# print("counter", counter)

# do short experiments
def run_short_experiments(recommender, results):
    # how many times ground truth recommended in 50 steps, and what step
    for groud_truth in item_names:
        # recommender = NearNeighborUCB(dist_lookup=dist_lookup,
        #                               time_horizon=args.time_horizon,
        #                               ground_truth=groud_truth,
        #                               test=True)
        recommender.ground_truth = groud_truth
        recommender._restart()
        recommender.run()
        results[groud_truth].append(recommender.play_ground_truth)
    return results

item_names = sorted(list(dist_lookup.keys()))
results = {key: [] for key in item_names}

n_instances = 50
for i in range(n_instances):
    results = run_short_experiments(recommender, results)

#%% plot results
positions = []
D = np.zeros((n_instances, len(item_names))) # 50 * 64
i = 0
for key, value in results.items():
    position = int(key[-2:])
    positions.append(position)
    
    for j, number in enumerate(value):
        if len(number) == 0:
            D[j, i] = 50  # TODO: deal with this
        else:
            D[j, i] = number[0]
    i += 1


# plot
fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot()
VP = ax.boxplot(D, positions=positions, widths=0.8, patch_artist=True,
                showmeans=True, showfliers=True, vert=False)

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()

fig.savefig("short_experiments.pdf")
