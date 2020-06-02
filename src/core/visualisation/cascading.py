from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def propagate_toxicity(G):
    choices = {}
    for (id, _) in G.nodes(data=True):
        toxic_neigh = 0
        non_toxic_neigh = 0
        for n in G.neighbors(id):
            if G.nodes[n]['TOXICITY'] < 0:
                toxic_neigh += 1
            else:
                non_toxic_neigh += 1
        proba = non_toxic_neigh / \
            (toxic_neigh + non_toxic_neigh) if (toxic_neigh +
                                                non_toxic_neigh) != 0 else 0
        choice = np.random.choice(
            ['non_toxic', 'toxic'], 1, p=[proba, 1 - proba])[0]
        choices[id] = choice
    return choices


def cascading_effect(G, names):
    fig = plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1, iterations=50)
    for index in range(9):
        fig.add_subplot(3, 3, index + 1)

        degrees = list(G.degree())
        # Labels
        (top_degree, _) = sorted(degrees,
                                 key=itemgetter(1), reverse=True)[0]
        nx.draw_networkx_labels(
            G, pos, {top_degree: names[top_degree]}, font_size=16, font_color='black')

        nx.draw_networkx_edges(
            G, pos, alpha=0.2)

        toxicity = nx.get_node_attributes(G, 'TOXICITY')
        colours = ['r' if value < 0 else 'g' for node,
                   value in toxicity.items()]

        #print(f'Toxic: {colours.count('r')}. Non-toxic: {colours.count('g')}.')
        print('Toxic: {}'.format(colours.count('r')))

        nx.draw_networkx_nodes(G, pos,
                               with_labels=False, node_color=colours)

        choices = propagate_toxicity(G)

        for (id, tox_label) in choices.items():
            if tox_label == 'non_toxic':
                G.nodes[id]['TOXICITY'] = 0
            else:
                G.nodes[id]['TOXICITY'] = -1

        plt.title(f'Snapshot {index + 1}')
        plt.axis('off')
