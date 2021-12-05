# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:35:08 2021

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
    
    parser.add_argument('--recommender', type=str, default="AdaptiveRecommenderSong", help="choose a recommender algorithm")
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
                      time_horizon=100000, 
                      ground_truth='I_2055', 
                      test=True)
    # item_list = [recommender.item_list[i].name for i in range(len(recommender.item_list))]
    # n_plays_list = [recommender.item_list[i].n_plays for i in range(len(recommender.item_list))]
    
elif args.recommender == "AdaptiveRecommenderSong":
    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    #exptree.print_tree()
    
    recommender = AdaptiveRecommenderSong(exptree=exptree,
                                          time_horizon=100000,
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

def run_algo(recommender, n_instances):
    regret_lists = []
    for i in range(n_instances):
        # if (i+1)%10 == 0:
        print('Instance number =', i)
        recommender._restart()
        recommender.run()
        regret_lists.append(recommender.cum_regret_list)
    return regret_lists

do_experiments = False
if do_experiments:
    # compare different algorithms
    n_instances = 30
    
    results = {}
    recommender = UCB(dist_lookup=dist_lookup, 
                      time_horizon=100000, 
                      ground_truth='I_2055', 
                      test=True)
    results['UCB'] = run_algo(recommender, n_instances)
    
    
    
    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    recommender = AdaptiveRecommenderSong(exptree=exptree,
                                          time_horizon=100000,
                                          user=None,
                                          ground_truth='I_2055',
                                          test=True,
                                          noise=0.1)
    results['AdaptiveRecommenderSong'] = run_algo(recommender, n_instances)
    
    
    
    exptree = ExpTree(b=0.6, n_layers=4, dist_lookup=dist_lookup)
    exptree.build_tree()
    recommender = AdaptiveRecommender(exptree=exptree,
                                      time_horizon=10000,
                                      user=None,
                                      ground_truth='I_2055',
                                      test=True,
                                      noise=0.1)
    results['AdaptiveRecommender'] = run_algo(recommender, n_instances)

    # only plot one regret result
    reg = np.array(results['AdaptiveRecommenderSong'])
    
    mean_reg = np.mean(reg, axis=0)
    std_reg = np.std(reg, axis=0)
    x = np.arange(len(mean_reg))
    
    plt.plot(x, mean_reg)
    plt.fill_between(x, 
                     mean_reg-std_reg, mean_reg+std_reg,
                     alpha=0.3)
    plt.xlabel("Time");
    plt.ylabel("Cumulative Regret");
    plt.title('Cumulative Regret with AdaptiveRecommenderSong');
    
    
    

    
    # plot all regret results and compare
    # TODO: write to a function?
    fig, ax = plt.subplots()
    for key, value in results.items():
        
        reg = np.array(value)
        mean_reg = np.mean(reg, axis=0)
        std_reg = np.std(reg, axis=0)
        
        x = np.arange(len(mean_reg))
        ax.plot(x, mean_reg, label=key)
        ax.fill_between(x, mean_reg-std_reg, mean_reg+std_reg, alpha=0.3)
    ax.set(xlabel='Time', ylabel='Cumulative Regret', title='Cumulative Regret')
    ax.legend()
