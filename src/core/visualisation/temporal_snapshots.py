from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx


def temporal_snapshots(G, labels, thresholds):
    snapshots = []
    fig = plt.figure(figsize=(20, 20))
    for index, thresh in enumerate(thresholds):
        snapshot = nx.DiGraph(((source, target, attr) for source, target,
                               attr in G.edges(data=True) if attr['TIMESTAMP'] < thresh))
        snapshots.append(snapshot)
        fig.add_subplot(3, 3, index + 1)

        pos = nx.spring_layout(snapshot, k=1, iterations=50)

        degrees = list(snapshot.degree())
        node_size = [v * 5 for (k, v) in degrees]

        # Labels
        (top_degree, _) = sorted(degrees,
                                 key=itemgetter(1), reverse=True)[0]
        nx.draw_networkx_labels(
            snapshot, pos, {top_degree: labels[top_degree]}, font_size=16, font_color='black')

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
