import random
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx

from src import constants


def draw_sentiment_network(G, thresh, names=None, with_degree=False, with_hits=False, savefig=False):
    H = nx.DiGraph()

    for (source, target) in list(G.edges)[:thresh]:
        weight = G[source][target]['LINK_SENTIMENT']
        if weight == 1:
            H.add_edge(source, target, color='g')
        if weight == -1:
            H.add_edge(source, target, color='r')

    pos = nx.spring_layout(H, k=0.30, iterations=50)
    plt.figure(figsize=(14, 12))

    colors = [H[u][v]['color'] for u, v in H.edges]

    labels = {}
    if with_degree:
        degree_sequence = list(H.degree())
        (largest_hub, _) = sorted(degree_sequence, key=itemgetter(1))[-1]
        (second_largest, _) = sorted(degree_sequence, key=itemgetter(1))[-2]

        labels[largest_hub] = names[largest_hub]
        labels[second_largest] = names[second_largest]

    if with_hits:
        h, a = nx.hits(H)

        max_hub = max(h.items(), key=itemgetter(1))[0]
        max_authority = max(a.items(), key=itemgetter(1))[0]

        labels[max_hub] = names[max_hub]
        labels[max_authority] = names[max_authority]

    print(labels)
    nx.draw_networkx_labels(
        G, pos, labels, font_size=16, font_color='black')

    nx.draw_networkx_edges(H, pos, edge_color=colors)
    nx.draw_networkx_nodes(H, pos, node_color='b',
                           with_labels=False, node_size=50)
    plt.axis('off')

    if savefig:
        plt.savefig(constants.ROOT_DIR / 'images' /
                    f'sentiment_net_{thresh}.png', bbox_inches='tight')


def draw_balance_triangle(G, triangles, names):
    triangle, _ = random.choice(list(triangles.items()))

    H = nx.subgraph(G, triangle)

    pos = nx.spring_layout(H)
    plt.figure()

    labels = {node: names[node] for node in list(H)}
    print(labels)

    sentiments = nx.get_edge_attributes(H, 'LINK_SENTIMENT')
    print(sentiments)

    nx.draw_networkx_labels(
        H, pos, labels, font_size=16, font_color='black')

    nx.draw_networkx_edges(H, pos, edge_color='black')
    nx.draw_networkx_nodes(H, pos, node_color='b',
                           with_labels=False, node_size=50)
    plt.axis('off')

def print_simple_network(G, names=None, bridges=None):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_edges(G, pos, edge_color='black')
    nx.draw_networkx_nodes(G, pos, node_color='b',
                           with_labels=False, node_size=50)
    plt.axis('off')

    if names:
        labels = {node: names[node] for node in list(G)}
        print(labels)

        nx.draw_networkx_labels(
            G, pos, labels, font_size=16, font_color='black')

    if bridges:
        nx.draw_networkx_edges(
            G, pos=pos, edgelist=bridges, width=2, edge_color=['g'])

