"""
with open(constants.ROOT_DIR / 'input' / 'edge_list.txt', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    f.close()

edge_list = [element[0].split(' ') for element in data]
np.shape(edge_list)

# Full graph
nx_reddit = nx.DiGraph()
nx_reddit.add_edges_from(edge_list)
"""


"""

def sample_graph(edge_list, perc_keep):
    to_keep = random.sample(range(len(edge_list)), k=int(
        round(len(edge_list) * perc_keep)))
    sample_graph = [edge_list[i] for i in to_keep]

    nx_sample_graph = nx.DiGraph()
    nx_sample_graph.add_edges_from(sample_graph)

    return nx_sample_graph


def save_graph(graph, file_name):
    # initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig


degree_freq = np.array(nx.degree_histogram(nx_reddit)).astype('float')
log_degree_freq = np.log(degree_freq + 1)

plt.figure(figsize=(12, 8))
plt.stem(log_degree_freq)
plt.ylabel("Frequence")
plt.xlabel("Degre")
plt.show()


import random
k = 100
sampled_edges = random.sample(nx_reddit.edges, k)
nx_sample_reddit = nx.DiGraph()
nx_sample_reddit.add_edges_from(sampled_edges)


sampled_nodes = random.sample(nx_reddit.nodes, k)
nx_sample_reddit = nx_reddit.subgraph(sampled_nodes)


pos = nx.spring_layout(G_sample)
plt.figure(figsize=(8, 8))
plt.axis('off')
nx.draw_networkx_nodes(G_sample, pos, node_size=600,
                       cmap=plt.cm.RdYlBu)
nx.draw_networkx_edges(G_sample, pos, alpha=1)
plt.show(G_sample)
"""


# %%
"""
# get fixed positions to draw
G_base = nx.DiGraph()
G_base.add_edges_from(edges_100)
all_pos = nx.spring_layout(G_base)
for n, p in all_pos.items():
    G_base._node[n]['pos'] = p
"""
# %%
"""
plt.figure(figsize=(14, 12))
nx.draw_networkx_edges(G_100, pos, alpha=0.2)
nx.draw_networkx_nodes(G_100, pos, node_color='b',
                            with_labels=False, node_size=50, cmap=plt.cm.jet)
plt.axis('off')
"""
"""
# %%
fig, ax = plt.subplots()
plt.axis('off')

G_100 = nx.DiGraph()

# %%


def animate(num):
    ax.clear()
    i = num % 100

    source = edges_100[i][0]
    target = edges_100[i][1]
    G_100.add_edge(source, target)

    curr_pos = {n: p for n, p in all_pos.items() if n in list(G_100.nodes)}

    nx.draw_networkx_nodes(G_100, pos=curr_pos, ax=ax, node_color='b',
                           with_labels=False, node_size=50, cmap=plt.cm.jet)
    nx.draw_networkx_edges(G_100, pos=curr_pos, ax=ax, alpha=0.2)


ani = matplotlib.animation.FuncAnimation(
    fig, animate, frames=100, interval=1000, repeat=True)
ani.save('graph100.gif', writer='imagemagick', fps=1)

Image(url='graph100.gif')



# %%
first_20 = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].head(5)
edges_20 = [tuple(x) for x in first_20.values]

# %%
# get fixed positions to draw
G_base_20 = nx.DiGraph()
G_base_20.add_edges_from(edges_20)
pos_20 = nx.spring_layout(G_base_20)
for n, p in pos_20.items():
    G_base_20._node[n]['pos'] = p


# %%
fig, ax = plt.subplots()
plt.axis('off')

G_20 = nx.DiGraph()
num_nodes = len(G_base_20)
curr_node_idx = 0
red_edges = set()
green_edges = set()
new_nodes = set()

# %%


def animate(num):
    ax.clear()
    i = num % 5

    source = edges_20[i][0]
    target = edges_20[i][1]

    print(source, target)

    # list(G_base._node.items())[0]

    if source in G_20:
        print('source existe')
        G_20.add_edge(source, target)
        source_edges = list(G_20.edges(source))
        red_edges.update(source_edges)
        print(red_edges)
    else:
        G_20.add_edge(source, target)
        new_nodes.add(source)

    if target in G_20:
        print('target existe')
        G_20.add_edge(source, target)
        target_edges = list(G_20.edges(target))
        red_edges.update(target_edges)
        print(red_edges)
    else:
        G_20.add_edge(source, target)
        new_nodes.add(target)

    curr_pos = {n: p for n, p in pos_20.items() if n in list(G_20.nodes)}

    nx.draw_networkx_nodes(G_20, pos=curr_pos, ax=ax, node_color='b',
                           with_labels=False, node_size=50, cmap=plt.cm.jet)
    nx.draw_networkx_edges(G_20, pos=curr_pos, ax=ax, alpha=0.2)
    if red_edges:
        # red edges
        nx.draw_networkx_edges(G_20, pos=curr_pos, edge_list=list(red_edges), ax=ax, edge_color='r', alpha=0.2)
    if green_edges:
        # green edges
        nx.draw_networkx_edges(G_20, pos=curr_pos, edge_list=list(green_edges), ax=ax, edge_color='g', alpha=0.2)
    if new_nodes:
        # new node
        nx.draw_networkx_nodes(G_20, pos=curr_pos, nodelist=list(new_nodes), ax=ax, node_size=300, node_color='r')

    #curr_node_idx += 1


ani = matplotlib.animation.FuncAnimation(
    fig, animate, frames=100, interval=1000, repeat=True)
ani.save('graph20.gif', writer='imagemagick', fps=1)

Image(url='graph20.gif')
"""


# %%
"""
first_20 = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].head(20)
edges_20 = [tuple(x) for x in first_20.values]
"""