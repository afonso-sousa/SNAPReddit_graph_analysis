# %%
# Importing the required libraries
from itertools import combinations

import matplotlib.animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Image

from src import constants, utils

# %%
df = pd.read_csv(constants.PROC_DATA_DIR / 'combined.csv',
                 sep='\t', parse_dates=['TIMESTAMP'])

# %%
G_reddit = nx.from_pandas_edgelist(df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']],
                                   source='SOURCE_SUBREDDIT', target='TARGET_SUBREDDIT', edge_attr=None, create_using=nx.DiGraph())

# %%
df.set_index('TIMESTAMP', inplace=True)

# %%
first_100 = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].head(98)
edges_100 = [tuple(x) for x in first_100.values]

# %%
subreddit_names = utils.load_obj(constants.PROC_DATA_DIR, 'subreddit_names')

# %%
# get fixed positions to draw
G_aux = nx.DiGraph()
G_aux.add_edges_from(edges_100)
pos_100 = nx.spring_layout(G_aux, k=0.30, iterations=50)


# %%
fig, ax = plt.subplots(figsize=(12, 10))
plt.axis('off')

G_100 = nx.Graph()


def animate(frame):
    ax.clear()
    i = frame % 98

    source = edges_100[i][0]
    target = edges_100[i][1]

    new_nodes = []
    if source not in G_100:
        new_nodes.append(source)

    if target not in G_100:
        new_nodes.append(target)

    G_100.add_edge(source, target)

    curr_pos = {n: p for n, p in pos_100.items() if n in list(G_100.nodes)}

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
            id, name) in subreddit_names.items() if id in cliques3[0]}
        nx.draw_networkx_labels(G_100, pos=curr_pos,
                                labels=node_labels, ax=ax, font_size=16)
        plt.pause(4)


ani = matplotlib.animation.FuncAnimation(
    fig, animate, frames=100, interval=1000)
ani.save(constants.ROOT_DIR / 'images' / 'triadic_closure.gif', writer='imagemagick', fps=6)

Image(url=constants.ROOT_DIR / 'images' / 'triadic_closure.gif')


# %%
