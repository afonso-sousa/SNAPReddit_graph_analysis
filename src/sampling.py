
import random
from operator import itemgetter

import networkx as nx

from src import utils


def small_graph(G, size='small'):
    degree_sequence = list(G.degree())
    (largest_hub, _) = sorted(degree_sequence, key=itemgetter(1))[-1]

    sample_nodes = [largest_hub]

    if size == 'small':
        thresh = 2
        num_neigh = 1
    if size == 'medium':
        thresh = 4
        num_neigh = random.randint(1, 2)

        outcast = random.choice(list(G))
        sample_nodes.append(outcast)
        sample_nodes.append(list(G.neighbors(outcast))[0])

    distance1 = list(G.neighbors(largest_hub))[:thresh]
    sample_nodes.extend(distance1)
    for d in distance1:
        nodes = list(G.neighbors(d))[:num_neigh]
        sample_nodes.extend(nodes)

    return G.subgraph(sample_nodes)


def small_bridge(G):
    bridges = list(nx.bridges(G))
    node1, node2 = random.choice(bridges)

    sample_nodes = [node1]

    neighs1 = [node for node in list(G.neighbors(node1)) if not utils.nodes_connected(G, node, node2)]
    assert len(neighs1) > 1

    if len(neighs1) > 4:
        neighs1 = random.sample(neighs1, 4)

    sample_nodes.extend(neighs1)
    
    H = G.subgraph(sample_nodes)

    neighs2 = [node for node in list(G.neighbors(node2)) if not H.has_node(node)]
    assert len(neighs2) > 1

    if len(neighs2) > 4:
        neighs2 = random.sample(neighs2, 4)

    sample_nodes.extend(neighs2)

    return G.subgraph(sample_nodes)

