
import networkx as nx
import numpy as np

from SNAPReddit_graph_analysis.src.core import utils

def main_info(G):
    degree_sequence = list(G.degree())
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


def bet_clo_eig_cen(G, names, top):
    strong_components = utils.get_strongly_connected_components(G)
    giant_strong = max(strong_components, key=len)

    # Betweenness centrality
    bet_cen = nx.betweenness_centrality(giant_strong)

    # Closeness centrality
    clo_cen = nx.closeness_centrality(giant_strong)

    # Eigenvector centrality
    eig_cen = nx.eigenvector_centrality(giant_strong)

    def get_top_keys(dictionary):
        items = dictionary.items()
        sorted(items, reverse=True, key=lambda x: x[1])
        return map(lambda x: x[0], list(items)[:top])

    top_bet_cen = get_top_keys(bet_cen)
    top_clo_cen = get_top_keys(clo_cen)
    top_eig_cent = get_top_keys(eig_cen)

    print('Top 10 places for betweenness centrality:')
    for node_id in top_bet_cen:
        print(names[node_id])
    print('Top 10 places for closeness centrality:')
    for node_id in top_clo_cen:
        print(names[node_id])
    print('Top 10 places for eigenvector centrality:')
    for node_id in top_eig_cent:
        print(names[node_id])
