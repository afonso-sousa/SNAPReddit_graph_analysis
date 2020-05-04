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
from matplotlib import dates as mdates

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
nb_nodes = len(G)
nb_edges = len(G.edges())
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
drawing.draw_balance_triangle(G_undir, unbalanced, subreddit_names)

# %%
# tri = utils.get_uneven_triangle(G_undir, 1)
tri = utils.get_full_negative_triangle(G_undir)
labels = {node: subreddit_names[node] for node in list(tri)}
print(labels)

sentiments = nx.get_edge_attributes(tri, 'LINK_SENTIMENT')
print(sentiments)

# %%
nx.average_clustering(G_undir)

# %%
sample_small = sampling.small_graph(G_undir)

# %%
sample_medium = sampling.small_graph(G_undir, size='medium')

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

nx.write_edgelist(sample, constants.PROC_DATA_DIR /
                  'sample_smaller.edgelist', data=['LINK_SENTIMENT'])


# %%


def community_layout(g, partition):
    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):
    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


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
sample_medium = sampling.small_graph(G, size='medium')

# %%
drawing.print_simple_network(sample_medium, names=subreddit_names)


# %%
sample = Snowball().snowball(G, 300, 5)
sample = nx.DiGraph(sample)  # unfreeze
sample.remove_nodes_from(list(nx.isolates(sample)))

# %%
nx.write_edgelist(sample, constants.PROC_DATA_DIR /
                  'directed_sample.edgelist', data=['LINK_SENTIMENT'])

# %%
utils.add_toxicity_node_attribute(sample)

pos = nx.spring_layout(sample, k=1, iterations=50)
bet_cent = nx.betweenness_centrality(sample, normalized=True, endpoints=True)
node_color = [node[1]['TOXICITY'] for node in list(sample.nodes(data=True))]
node_size = [v * 1e4 for v in bet_cent.values()]
plt.figure(figsize=(16, 14))

ec = nx.draw_networkx_edges(sample, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(sample, pos, node_color=node_color,
                            with_labels=False, node_size=node_size, cmap=plt.cm.Reds)
plt.colorbar(nc)
plt.axis('off')
plt.savefig(constants.ROOT_DIR / 'images' /
            'bet_cent_toxicity.png', bbox_inches='tight')

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

# %%


def toxicity_out(node):
    out_edges = G.out_edges(node)
    return len([(source, target) for (source, target) in out_edges if G[source][target]['LINK_SENTIMENT'] == -1])


node_toxicity = {node: toxicity_out(node) for node in G.nodes()}
# toxic_nodes = sorted(G.nodes(), key=lambda i:toxicity_out(i), reverse=True)

# %%
toxic_nodes_sorted = {k: v for k, v in sorted(
    node_toxicity.items(), key=lambda item: item[1], reverse=True)}

# %%
plt.figure()
toxicity_values = list(toxic_nodes_sorted.values())
toxic_perc = np.array(toxicity_values) / len(toxicity_values) * 100
plt.plot(np.cumsum(toxic_perc))
degrees = list(G.degree(toxic_nodes_sorted.keys()))
plt.plot(np.cumsum(list(zip(*degrees))[1]))

plt.xlabel('Number of subreddits')
plt.ylabel('Number of toxic out-edges')
plt.yscale('log')


# %%
degrees = [G.in_degree(n) for n in G.nodes()]
plt.hist(degrees)
plt.xlabel('Degree of nodes')
plt.ylabel('Frequency')
plt.yscale('log')
plt.savefig(constants.ROOT_DIR / 'images' /
            'degree_dist.png', bbox_inches='tight')

# %%
degrees = [G.in_degree(n) for n in G.nodes()]
plt.hist(degrees)
plt.plot(1/np.array(degrees)**2)
plt.xlabel('In-degree of nodes')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.savefig(constants.ROOT_DIR / 'images' /
            'in_degree_dist.png', bbox_inches='tight')

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
timestamps = nx.get_edge_attributes(sample, 'TIMESTAMP').values()
timestamps = [ts.date() for ts in timestamps]

bin_nr = 10
fig, ax = plt.subplots(1, 1)
_counts, bins, _patches = ax.hist(timestamps, bins=bin_nr)
plt.xticks(bins)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xticks(rotation=70)
plt.ylabel('Frequency')


# %%
# Threshold dates

thresholds = []
thresholds.append(datetime.datetime(2014, 6, 1))
thresholds.append(datetime.datetime(2015, 1, 1))
thresholds.append(datetime.datetime(2015, 6, 1))
thresholds.append(datetime.datetime(2016, 1, 1))
thresholds.append(datetime.datetime(2016, 6, 1))
thresholds.append(datetime.datetime(2017, 1, 1))
thresholds.append(datetime.datetime(2017, 6, 1))

# %%
snapshots = []
fig = plt.figure(figsize=(20, 20))
for index, thresh in enumerate(thresholds):
    snapshot = nx.DiGraph(((source, target, attr) for source, target,
                           attr in sample.edges(data=True) if attr['TIMESTAMP'] < thresh))
    snapshots.append(snapshot)
    fig.add_subplot(3, 3, index + 1)
    # temporal_1 = G.subgraph(list(map(tuple, temporal_split[0])))

    pos = nx.spring_layout(snapshot, k=1, iterations=50)

    degrees = list(snapshot.degree())
    node_size = [v * 5 for (k, v) in degrees]

    # Labels
    (top_degree, _) = sorted(degrees,
                             key=itemgetter(1), reverse=True)[0]
    # top10degree = [i[0] for i in top10degree]
    # labels = {node: subreddit_names[node] for node in top10degree}
    nx.draw_networkx_labels(
        snapshot, pos, {top_degree: subreddit_names[top_degree]}, font_size=16, font_color='black')

    ec = nx.draw_networkx_edges(
        snapshot, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(snapshot, pos,
                                with_labels=False, node_size=node_size, cmap=plt.cm.Reds)

    if index > 0:
        new_edges = set(snapshots[index].edges()) - \
            set(snapshots[index - 1].edges())
        new_nodes = set(snapshots[index]) - set(snapshots[index - 1])

        ec = nx.draw_networkx_edges(snapshot,
            pos, edgelist=new_edges, edge_color='g', alpha=0.2)

        nc = nx.draw_networkx_nodes(snapshot,
            pos, nodelist=new_nodes, node_color='g',
                                    with_labels=False, node_size=node_size, cmap=plt.cm.Reds)

    plt.title(thresh.strftime("%B, %Y"))
    plt.axis('off')


# %%
