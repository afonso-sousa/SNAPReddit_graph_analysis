import random
from itertools import combinations
from operator import itemgetter
from pathlib import Path

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import Image

from SNAPReddit_graph_analysis.src.core import utils
from SNAPReddit_graph_analysis.src.core.config import cfg


def draw_sentiment_network(G, num_nodes, names=None, with_degree=False, with_hits=False, savefig=False):
    H = nx.DiGraph()

    sample = random.sample(list(G.edges), num_nodes)

    for (source, target) in sample:
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
        plt.savefig(Path(cfg.DATA.IMAGES_DIR) /
                    f'sentiment_net_{num_nodes}.png', bbox_inches='tight')


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


def triad_animation(edges, end_pos, names, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')

    G_100 = nx.Graph()

    def animation_cicle(frame):
        ax.clear()
        i = frame % 98

        source = edges[i][0]
        target = edges[i][1]

        new_nodes = []
        if source not in G_100:
            new_nodes.append(source)

        if target not in G_100:
            new_nodes.append(target)

        G_100.add_edge(source, target)

        curr_pos = {n: p for n, p in end_pos.items() if n in list(G_100.nodes)}

        # cliques
        cliques = list(nx.enumerate_all_cliques(G_100))
        clique_len = [len(x) for x in cliques]
        length3 = list(np.where(np.array(clique_len) == 3)[0])
        cliques3 = [cliques[i] for i in length3]

        _, subG = utils.find_open_triads(G_100)
        open_triads_edges = subG.edges

        # nodes
        nx.draw_networkx_nodes(G_100, pos=curr_pos, ax=ax, node_color='b',
                               node_size=50)
        if new_nodes:
            nx.draw_networkx_nodes(
                G_100, pos=curr_pos, nodelist=new_nodes, ax=ax, node_size=50, node_color='g')

        # edges
        nx.draw_networkx_edges(G_100, pos=curr_pos, ax=ax, alpha=0.2)

        if open_triads_edges:
            nx.draw_networkx_edges(G_100, pos=curr_pos, edgelist=open_triads_edges,
                                   ax=ax, width=2, edge_color=['r'], alpha=0.2)

        if length3:
            edges3 = list(combinations(cliques3[0], 2))
            nx.draw_networkx_edges(
                G_100, pos=curr_pos, edgelist=edges3, ax=ax, width=2, edge_color=['g'])

            node_labels = {id: name for (
                id, name) in names.items() if id in cliques3[0]}
            nx.draw_networkx_labels(G_100, pos=curr_pos,
                                    labels=node_labels, ax=ax, font_size=16)
            plt.pause(4)

    animation = ani.FuncAnimation(
        fig, animation_cicle, frames=100, interval=1000)

    if save_path:
        animation.save(save_path, writer='imagemagick', fps=6)

    Image(url=save_path)


def unbalanced_triangles_in_net(G_undir, names, num_edges):
    sample = random.sample(list(G_undir.edges), num_edges)
    H = G_undir.edge_subgraph(sample)

    pos = nx.spring_layout(H, k=0.30, iterations=50)
    plt.figure(figsize=(14, 12))

    labels = {}
    degree_sequence = list(H.degree())
    (largest_hub, _) = sorted(degree_sequence, key=itemgetter(1))[-1]
    (second_largest, _) = sorted(degree_sequence, key=itemgetter(1))[-2]

    labels[largest_hub] = names[largest_hub]
    labels[second_largest] = names[second_largest]

    print(labels)
    nx.draw_networkx_labels(
        H, pos, labels, font_size=16, font_color='black')

    nx.draw_networkx_edges(H, pos, edge_color='g', alpha=0.3)
    nx.draw_networkx_nodes(H, pos, node_color='b',
                           with_labels=False, node_size=50)

    triangle_types = utils.sentiment_triangles(H)
    for (triangle, sentiment) in triangle_types.items():
        if sentiment == -1:
            print('Found unbalanced triangle')
            unb_tri = H.subgraph(triangle)
            nx.draw_networkx_edges(H, pos, edgelist=list(
                unb_tri.edges), edge_color='r')

    plt.axis('off')


def toxicity_per_centrality(G, save_path=None):
    pos = nx.spring_layout(G, k=1, iterations=50)
    bet_cent = nx.betweenness_centrality(
        G, normalized=True, endpoints=True)
    node_color = [node[1]['TOXICITY']
                  for node in list(G.nodes(data=True))]
    node_size = [v * 1e4 for v in bet_cent.values()]
    plt.figure(figsize=(16, 14))

    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_color,
                                with_labels=False, node_size=node_size, cmap=plt.cm.Reds)
    plt.colorbar(nc)
    plt.axis('off')
    if save_path:
        plt.savefig(Path(cfg.DATA.IMAGES_DIR) /
                    'bet_cent_toxicity.png', bbox_inches='tight')
