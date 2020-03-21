# %%
# Importing the required libraries
import csv
import os
import pickle
import random
from itertools import count
from operator import itemgetter

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pylab

from src import constants, utils
from src.snowball import Snowball

# os.chdir('/home/afonso/Projects/ARSI/SNAPReddit_link_prediction')

# %%
G_reddit = nx.read_edgelist(constants.PROC_DATA_DIR /
                            'edge_list.txt', create_using=nx.DiGraph(), nodetype=int)

# %%
nb_nodes = len(G_reddit)
nb_edges = len(G_reddit.edges())
degree_sequence = list(G_reddit.degree())
(largest_hub, degree) = sorted(degree_sequence, key=itemgetter(1))[-1]
avg_degree = np.around(np.mean(np.array(degree_sequence)[
                       :, 1].astype(np.float)), decimals=3)
med_degree = np.median(np.array(degree_sequence)[:, 1].astype(np.float))
max_degree = max(np.array(degree_sequence)[:, 1].astype(np.float))
min_degree = np.min(np.array(degree_sequence)[:, 1].astype(np.float))
num_weak_conn = nx.number_weakly_connected_components(G_reddit)

weak_components = utils.get_weakly_connected_components(G_reddit)
strong_components = utils.get_strongly_connected_components(G_reddit)
giant_weak = max(weak_components, key=len)
giant_strong = max(strong_components, key=len)


print("Number of nodes: " + str(nb_nodes))
print("Number of edges: " + str(nb_edges))
print("Maximum degree: " + str(max_degree))
print("Minimum degree: " + str(min_degree))
print("Average degree: " + str(avg_degree))
print("Median degree: " + str(med_degree))
print(f'Has {num_weak_conn} weakly connected components.')
print(f'Size of weak giant component: {len(giant_weak)}')
print(f'Size of strong giant component: {len(giant_strong)}')

# %%
# Create ego graph of main hub
hub_ego = nx.ego_graph(G_reddit, largest_hub)

# %%
# Draw graph
pos = nx.spring_layout(hub_ego, k=0.30, iterations=50)
plt.figure(figsize=(14, 12))
nx.draw_networkx_edges(hub_ego, pos, alpha=0.2)
nx.draw_networkx_nodes(hub_ego, pos, node_color='b',
                            with_labels=False, node_size=50, cmap=plt.cm.jet)
# Draw ego as large and red
nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
plt.axis('off')
plt.savefig(constants.ROOT_DIR / 'images' /
            'ego_graph.png', bbox_inches='tight')

# %%
subreddit_names = utils.load_obj(constants.PROC_DATA_DIR, 'subreddit_names')

# %%
subreddit_names[largest_hub]


# %%
"""
# %%
G_sample = Snowball().snowball(G_reddit, 200, 20)

# %%
#pos = nx.kamada_kawai_layout(G_sample)
pos = nx.spring_layout(G_sample, k=0.30, iterations=50)
bet_cent = nx.betweenness_centrality(G_sample, normalized=True, endpoints=True)
node_color = [G_sample.degree(v) for v in G_sample]
node_size = [v * 1e5 for v in bet_cent.values()]
plt.figure(figsize=(14, 12))

ec = nx.draw_networkx_edges(G_sample, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(G_sample, pos, node_color=node_color,
                            with_labels=False, node_size=node_size, cmap=plt.cm.jet)
plt.colorbar(nc)
plt.axis('off')
#plt.savefig(constants.ROOT_DIR / 'images' /
#            'bet_cent_degree.png', bbox_inches='tight')

# %%
rnd_sample = random.sample(list(G_reddit.nodes()), 200)

# %%
strong_components = utils.get_strongly_connected_components(G_reddit)
sample_strong = max(len(g) for g in strong_components if len(g) < 12000) 
"""

