# %%
# Importing the required libraries
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from community import community_louvain
from networkx.algorithms import community

from src import constants, drawing, sampling, utils
from src.snowball import Snowball

# %%
df = pd.read_csv(constants.PROC_DATA_DIR / 'combined.csv',
                 sep='\t', parse_dates=['TIMESTAMP'])

G_reddit = nx.from_pandas_edgelist(df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT']],
                                   source='SOURCE_SUBREDDIT', target='TARGET_SUBREDDIT', edge_attr=True, create_using=nx.DiGraph())

subreddit_names = utils.load_obj(constants.PROC_DATA_DIR, 'subreddit_names')

#####################################################
##################### SENTIMENT #####################
#####################################################

# %%
drawing.draw_sentiment_network(
    G_reddit, 200, names=subreddit_names, with_degree=True, savefig=True)

# %%
G_undir = G_reddit.to_undirected(reciprocal=True)
G_undir.remove_nodes_from(list(nx.isolates(G_undir)))

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
#tri = utils.get_uneven_triangle(G_undir, 1)
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
sample = Snowball().snowball(G_undir, 1000, 5)
sample = sample.to_undirected(reciprocal=True)
sample.remove_nodes_from(list(nx.isolates(sample)))


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

# %%
largest_group_idx = sorted(
    communities, key=lambda x: len(communities[x]), reverse=True)[2]
largest_group = communities[largest_group_idx]

sec_large_group_idx = sorted(
    communities, key=lambda x: len(communities[x]), reverse=True)[3]

# %%
largest_subG = G_undir.subgraph(largest_group)
bridges = list(nx.bridges(largest_subG))
drawing.print_simple_network(
    largest_subG, names=subreddit_names, bridges=bridges)

# %%
new_part = {key: value for (key, value) in partition.items() if (value == largest_group_idx) or (value == sec_large_group_idx)}
new_part_subG = sample.subgraph(list(new_part.keys()))

# %%
drawing.print_simple_network(new_part_subG, names=subreddit_names)


# %%
partition2 = community_louvain.best_partition(new_part_subG)
pos2 = community_layout(new_part_subG, partition2)

nx.draw(new_part_subG, pos2, node_size=100, node_color=list(partition2.values()))


# %%
