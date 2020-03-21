import pickle

import networkx as nx


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
