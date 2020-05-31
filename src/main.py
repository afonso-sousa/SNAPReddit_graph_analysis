# %%
# Import required libraries
import datetime
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from community import community_louvain

from SNAPReddit_graph_analysis.src.core.visualisation import plotting
from SNAPReddit_graph_analysis.src.core.visualisation.community_layout import \
    community_layout
from SNAPReddit_graph_analysis.src.core.visualisation.temporal_snapshots import \
    temporal_snapshots
from src import constants, utils
from src.core import bt_parallel, drawing, sampling
from src.snowball import Snowball

# %%
# Load data
df = pd.read_csv(constants.PROC_DATA_DIR / 'temporal_sentiment_edgelist.csv',
                 sep='\t', parse_dates=['TIMESTAMP'])

G = nx.from_pandas_edgelist(df, source='SOURCE_SUBREDDIT',
                            target='TARGET_SUBREDDIT', edge_attr=True, create_using=nx.DiGraph())

subreddit_names = utils.load_obj(constants.PROC_DATA_DIR, 'subreddit_names')

# %%
plotting.frequency_plot(df, 'LINK_SENTIMENT')

# %%
plotting.node_frequency_time_bins(G)

# %%
degree_sequence = list(G.degree())
(largest_hub, degree) = sorted(degree_sequence, key=itemgetter(1))[-1]
avg_degree = np.around(np.mean(np.array(degree_sequence)[
                       :, 1].astype(np.float)), decimals=3)
med_degree = np.median(np.array(degree_sequence)[:, 1].astype(np.float))
max_degree = max(np.array(degree_sequence)[:, 1].astype(np.float))
min_degree = np.min(np.array(degree_sequence)[:, 1].astype(np.float))
num_weak_conn = nx.number_weakly_connected_components(G)

weak_components = utils.get_weakly_connected_components(G)
strong_components = utils.get_strongly_connected_components(G)
giant_weak = max(weak_components, key=len)
giant_strong = max(strong_components, key=len)

print("Number of nodes: " + str(len(G)))
print("Number of edges: " + str(len(G.edges())))
print("Maximum degree: " + str(max_degree))
print("Minimum degree: " + str(min_degree))
print("Average degree: " + str(avg_degree))
print("Median degree: " + str(med_degree))
print(f'Has {num_weak_conn} weakly connected components.')
print(f'Size of weak giant component: {len(giant_weak)}')
print(f'Size of strong giant component: {len(giant_strong)}')

# %%
G_undir = G.to_undirected(reciprocal=True)
G_undir.remove_nodes_from(list(nx.isolates(G_undir)))

#####################################################
##################### SENTIMENT #####################
#####################################################

# %%
drawing.draw_sentiment_network(
    G, 200, names=subreddit_names, with_degree=True, savefig=True)


# %%
triangle_types = utils.sentiment_triangles(G_undir)

# %%
balanced = {}
unbalanced = {}
for (triangle, sentiment) in triangle_types.items():
    if sentiment == 1:
        balanced[triangle] = sentiment
    if sentiment == -1:
        unbalanced[triangle] = sentiment

# %%
drawing.unbalanced_triangles_in_net(G_undir, subreddit_names, 800)

# %%
drawing.draw_balance_triangle(G_undir, unbalanced, subreddit_names)

# %%
# tri = utils.get_uneven_triangle(G_undir, 1)
tri = utils.get_full_negative_triangle(G_undir)
labels = {node: subreddit_names[node] for node in list(tri)}
print(labels)

sentiments = nx.get_edge_attributes(tri, 'LINK_SENTIMENT')
print(sentiments)


# %%
sample_small = sampling.small_graph(G_undir)

# %%
sample_bridge = sampling.small_bridge(G_undir)

#####################################################
#################### COMMUNITIES ####################
#####################################################

# %%
sample = Snowball().snowball(G_undir, 300, 5)
# sample = sample.to_undirected(reciprocal=True)
sample = nx.Graph(sample)  # unfreeze
sample.remove_nodes_from(list(nx.isolates(sample)))

# nx.write_edgelist(sample, constants.PROC_DATA_DIR /
#                  'sample_smaller.edgelist', data=['LINK_SENTIMENT'])

# %%


partition = community_louvain.best_partition(sample)
pos = community_layout(sample, partition)

nx.draw(sample, pos, node_size=100, node_color=list(partition.values()))

# %%
communities = defaultdict(list)

for key, value in sorted(partition.items()):
    communities[value].append(key)

largest_group_idx = sorted(
    communities, key=lambda x: len(communities[x]), reverse=True)[2]
largest_group = communities[largest_group_idx]

sec_large_group_idx = sorted(
    communities, key=lambda x: len(communities[x]), reverse=True)[3]

# %%
# Bridges
largest_subG = G_undir.subgraph(largest_group)
bridges = list(nx.bridges(largest_subG))
drawing.print_simple_network(
    largest_subG, names=subreddit_names, bridges=bridges)

# %%
new_part = {key: value for (key, value) in partition.items() if (
    value == largest_group_idx) or (value == sec_large_group_idx)}
new_part_subG = sample.subgraph(list(new_part.keys()))

# %%
drawing.print_simple_network(new_part_subG, names=subreddit_names)

# %%
partition2 = community_louvain.best_partition(new_part_subG)
pos2 = community_layout(new_part_subG, partition2)

nx.draw(new_part_subG, pos2, node_size=100,
        node_color=list(partition2.values()))

#####################################################
################# TRIADIC ANIMATION #################
#####################################################

# %%
edges = sorted(G.edges(data=True), key=lambda t: t[2].get('TIMESTAMP', 1))
edges = edges[850:999]

# %%
# get fixed positions to draw
G_aux = nx.DiGraph()
G_aux.add_edges_from(edges)
pos = nx.spring_layout(G_aux, k=0.30, iterations=50)

# %%
drawing.triad_animation(edges, pos, subreddit_names,
                        save_path=constants.ROOT_DIR / 'images' / 'triadic_closure.gif')



# %%
sample = Snowball().snowball(G, 300, 5)
sample = nx.DiGraph(sample)  # unfreeze
sample.remove_nodes_from(list(nx.isolates(sample)))
# nx.write_edgelist(sample, constants.PROC_DATA_DIR /
#                  'directed_sample.edgelist', data=['LINK_SENTIMENT'])


# %%
utils.add_toxicity_node_attribute(sample)
drawing.toxicity_per_centrality(sample)

# %%
toxic_nodes = sorted(sample.nodes(data=True),
                     key=lambda n: n[1].get('TOXICITY', 1))


# %%
bt = bt_parallel.betweenness_centrality_parallel(sample)

# %%
max_bt_node, max_bt = max(bt.items(), key=itemgetter(1))

# %%
power_sample = sampling.small_graph_with_central_node(G, max_bt_node)
drawing.print_simple_network(power_sample, names=subreddit_names)

# %%
hub_ego = nx.ego_graph(sample, max_bt_node)
drawing.print_simple_network(hub_ego, names=subreddit_names)

# %%
node_toxicity = {node: utils.toxicity_out(node) for node in G.nodes()}

toxic_nodes_sorted = {k: v for k, v in sorted(
    node_toxicity.items(), key=lambda item: item[1], reverse=True)}

# %%
out_degree = {node: G.out_degree(node) for node in G.nodes()}
out_degree_sorted = [out_degree.get(i) for i in toxic_nodes_sorted.keys()]

# %%
plt.figure()
toxicity_values = list(toxic_nodes_sorted.values())
#toxic_perc = np.array(toxicity_values) / len(toxicity_values) * 100
# plt.plot(np.cumsum(toxic_perc))
plt.plot(toxicity_values, out_degree_sorted)
#degrees = list(G.degree(toxic_nodes_sorted.keys()))
# plt.plot(np.cumsum(list(zip(*degrees))[1]))

plt.xlabel('Number of toxic out-edges')
plt.ylabel('Total number of out-edges')
# plt.yscale('log')
# plt.xscale('log')

# %%
plotting.in_degree_freq(G)

# %%
fit = powerlaw.Fit(np.array(degrees)+1, xmin=1, discrete=True)
fit.power_law.plot_pdf(color='b', linestyle='--',
                       label='power law distribution')
fit.plot_pdf(color='b', label='in-degree distribution')
plt.legend()
print('alpha= ', fit.power_law.alpha, '  sigma= ', fit.power_law.sigma)
plt.savefig(constants.ROOT_DIR / 'images' /
            'in_degree_powerlaw.png', bbox_inches='tight')

# %%
max(dict(G.edges).items(), key=lambda x: x[1]['TIMESTAMP'])

# %%
min(dict(G.edges).items(), key=lambda x: x[1]['TIMESTAMP'])

# %%
# temporal_split = np.array_split(G.edges, 7)

# %%
sample = Snowball().snowball(G, 200, 5)
sample = nx.DiGraph(sample)  # unfreeze
sample.remove_nodes_from(list(nx.isolates(sample)))


# %%
# Threshold dates
thresholds = []
year = 2014
for i in range(7):
    if i % 2 == 0:
        # middle of year
        thresholds.append(datetime.datetime(year, 6, 1))
    else:
        year += 1
        # beginning of year
        thresholds.append(datetime.datetime(year, 1, 1))

temporal_snapshots(sample, subreddit_names, thresholds)

# %%
bt = bt_parallel.betweenness_centrality_parallel(sample)

# %%
plotting.six_separation(sample)

# %%
