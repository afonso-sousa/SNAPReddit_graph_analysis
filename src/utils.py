import itertools
import pickle
import random

import networkx as nx
import numpy as np


def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def get_strongly_connected_components(G):
    for c in nx.strongly_connected_components(G):
        yield G.subgraph(c)


def get_weakly_connected_components(G):
    for c in nx.weakly_connected_components(G):
        yield G.subgraph(c)


def save_obj(obj, filepath, filename):
    with open(filepath / f'{filename}.pkl', 'wb') as handle:
        pickle.dump(obj, handle, pickle.HIGHEST_PROTOCOL)


def load_obj(filepath, filename):
    with open(filepath / f'{filename}.pkl', 'rb') as handle:
        return pickle.load(handle)

def nodes_connected(G, u, v):
    return u in G.neighbors(v)


def sentiment_triangles(G):
    triangles = [c for c in nx.cycle_basis(G) if len(c) == 3]
    triangle_types = {}
    for triangle in triangles:
        tri = nx.subgraph(G, triangle)
        # take the product of the edge relationships.
        # If there are an odd number of -1s, the triangle is unbalanced.
        triangle_types[tuple(tri.nodes())] = np.product(
            [x[2]['LINK_SENTIMENT'] for x in tri.edges(data=True)])
    return triangle_types


def get_uneven_triangle(G, balance):
    triangles = [c for c in nx.cycle_basis(G) if len(c) == 3]
    uneven = []
    for triangle in triangles:
        tri = nx.subgraph(G, triangle)
        if (np.product([x[2]['LINK_SENTIMENT'] for x in tri.edges(data=True)])) == balance:
            for edge in tri.edges(data=True):
                if edge[2]['LINK_SENTIMENT'] == -balance:
                    uneven.append(tri)
    return random.choice(uneven)


def get_full_negative_triangle(G):
    triangles = [c for c in nx.cycle_basis(G) if len(c) == 3]
    full_negative = []
    for triangle in triangles:
        tri = nx.subgraph(G, triangle)
        if (np.sum([x[2]['LINK_SENTIMENT'] for x in tri.edges(data=True)])) == -3:
            full_negative.append(tri)
    return random.choice(full_negative)


def find_open_triads(G):
    open_triads = []
    nodes = set()
    for i in G.nodes():
        neigh = [i] + list(G.neighbors(i))
        a = tuple(neigh)
        if len(a) == 3:
            # number of edges of subgraph of node i and neighbours being less than 3
            if len(G.subgraph(neigh).edges()) != 3:
                open_triads.append(a)
                nodes.update(a)
        elif len(a) > 3:
            triad_neig = list(itertools.combinations(
                G.subgraph(neigh).nodes(), 3))
            triad_neig = [x for x in triad_neig if i in x]
            for k in triad_neig:
                if len(G.subgraph(k).edges()) == 2:
                    if k not in open_triads:
                        open_triads.append(k)
                        nodes.update(k)

    return open_triads, G.subgraph(list(nodes))

def add_toxicity_node_attribute(G):
    node_toxicity = {}
    for node in list(G):
        out_edges = list(G.out_edges(node, data=True))
        toxicity = sum(edge[2]['LINK_SENTIMENT'] for edge in out_edges if edge[2]['LINK_SENTIMENT'] < 0)
        node_toxicity[node] = toxicity

    nx.set_node_attributes(G, node_toxicity, 'TOXICITY')

def add_toxicity_perc_node_attribute(G):
    node_perc_tox = {}
    for node in list(G):
        out_edges = list(G.out_edges(node, data=True))
        if len(out_edges) == 0:
            node_perc_tox[node] = 0
        else:
            toxicity = sum(edge[2]['LINK_SENTIMENT'] for edge in out_edges if edge[2]['LINK_SENTIMENT'] < 0)
            node_perc_tox[node] = toxicity / len(out_edges)

    nx.set_node_attributes(G, node_perc_tox, 'PERC_TOXICITY')


def toxicity_out(node):
    out_edges = G.out_edges(node)
    return len([(source, target) for (source, target) in out_edges if G[source][target]['LINK_SENTIMENT'] == -1])
