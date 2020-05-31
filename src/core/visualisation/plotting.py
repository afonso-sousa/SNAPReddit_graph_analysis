from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx


def node_frequency_time_bins(G):
    timestamps = nx.get_edge_attributes(G, 'TIMESTAMP').values()
    timestamps = [ts.date() for ts in timestamps]

    bin_nr = 10
    _, ax = plt.subplots(1, 1)
    _counts, bins, _patches = ax.hist(timestamps, bins=bin_nr)
    plt.xticks(bins)
    plt.xticks(rotation=70)
    plt.ylabel('Frequency')

def frequency_plot(df, column):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(ax=ax, kind='bar')
    # fix bug where labels are already rotated
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')

def in_degree_freq(G):
    degrees = [G.in_degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.xlabel('In-Degree of nodes')
    plt.ylabel('Frequency')
    plt.yscale('log')
    # plt.savefig(constants.ROOT_DIR / 'images' /
    #            'in_degree_dist.png', bbox_inches='tight')


def six_separation(G):
    sp_len = nx.shortest_path_length(G)
    largest_sp = {}
    for (key, value) in sp_len:
        largest_sp[key] = max(value.items(), key=itemgetter(1))[1]

    grouped_lsp = defaultdict(list)
    for key, value in sorted(largest_sp.items()):
        grouped_lsp[value].append(key)
    grouped_lsp = {key: len(value) for key, value in grouped_lsp.items()}

    _, ax = plt.subplots(1, 1)
    ax.bar(list(grouped_lsp.keys()), list(grouped_lsp.values()))
    plt.ylabel('Frequency')
    plt.xlabel('Longest shortest path length')
