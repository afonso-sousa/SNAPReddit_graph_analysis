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
    #sp_len = nx.shortest_path_length(G)
    #largest_sp = {}
    #for (key, value) in sp_len:
    #    largest_sp[key] = max(value.items(), key=itemgetter(1))[1]

    #grouped_lsp = defaultdict(list)
    #for key, value in sorted(largest_sp.items()):
    #    grouped_lsp[value].append(key)
    #grouped_lsp = {key: len(value) for key, value in grouped_lsp.items()}

    #_, ax = plt.subplots(1, 1)
    #ax.bar(list(grouped_lsp.keys()), list(grouped_lsp.values()))
    #plt.ylabel('Frequency')
    #plt.xlabel('Longest shortest path length')

    sp_len = nx.shortest_path_length(G)
    freq_dict = defaultdict(int)
    for (_, value) in sp_len:
        for (_, v) in value.items():
            freq_dict[v] += 1

    plt.bar(list(freq_dict.keys()), list(freq_dict.values()))
    plt.xlabel('Shortest path length')
    plt.ylabel('Frequency')
    plt.yscale('log')

def in_out_degree(G):
    in_degrees = dict(G.in_degree()) # dictionary node:degree
    in_values = sorted(set(in_degrees.values()))
    in_hist = [list(in_degrees.values()).count(x) for x in in_values]

    out_degrees = dict(G.out_degree()) # dictionary node:degree
    out_values = sorted(set(out_degrees.values()))
    out_hist = [list(out_degrees.values()).count(x) for x in out_values]

    plt.figure() # you need to first do 'import pylab as plt'
    plt.grid(True)
    plt.loglog(in_values, in_hist, 'ro-') # in-degree
    plt.loglog(out_values, out_hist, 'bv-') # out-degree
    plt.legend(['In-degree', 'Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.xlim([0, 2*10**2]) 