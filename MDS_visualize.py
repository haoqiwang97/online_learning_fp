# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:27:45 2021

@author: hw9335
"""
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('data/NOUN_Sorting_Tables.xlsx', usecols=[1,2,3], skiprows=1)
distance_mt = df.pivot(index='Stim_1', columns='Stim_2', values='Distance')

#distance_mt['I_2001'] = 0
distance_mt.insert(0, 'I_2001', 0)
distance_mt.loc[len(distance_mt)] = 0
distance_mt = distance_mt.rename(index={63: 'I_2064'}).fillna(0)

plt.imshow(distance_mt, zorder=2, cmap='Blues', interpolation='nearest')
distance_mt = distance_mt + distance_mt.T

from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
model = MDS(n_components=3, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(distance_mt)

#plt.scatter(out[:, 0], out[:, 1])

ax = plt.axes(projection='3d')
# for s in model.feature_names_in_:
for s in ['I_2002', 'I_2031', 'I_2037']:
    ax.scatter3D(out[:, 0], out[:, 1], out[:, 2], label=s)

model.stress_

def find_items(root):
    result = []
    def helper(root, depth):
        if not root:
            return []
        if len(result) == depth:
            result.append([])
        
        if depth == 0:
            result[depth].append(root.name)
            print('no parent')
        else:
            result[depth].append(root.parent.name + '_' + root.name)
        for child in root.children:
            helper(child, depth + 1)
    helper(root, 0)
    return result
    
result = find_items(C_higher[0])

for i in range(4):
    C_higher[0].children[0].children 