# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn.manifold import TSNE

from src import constants, utils

# %%
# Glove format to Word2Vec format to load with Gensim package
glove_file = datapath(constants.PROC_DATA_DIR / 'subreddit_embeddings.txt')
tmp_file = get_tmpfile(constants.PROC_DATA_DIR / 'emb_word2vec.txt')

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

# %%


def cosine_distance(model, word, target_list, num):
    cosine_dict = {}
    word_list = []
    a = model[word]
    for item in target_list:
        if item != word:
            b = model[item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    # in Descedning order
    dist_sort = sorted(cosine_dict.items(),
                       key=lambda dist: dist[1], reverse=True)
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]


all_nodes = list(model.vocab.keys())
# Show the most similar to 'askreddit' by cosine distance
cosine_distance(model, 'askreddit', all_nodes, 5)

# %%


def display_closestwords_tsnescatterplot(model, word, size):
    arr = np.empty((0, size), dtype='f')
    word_labels = [word]

    close_words = model.most_similar(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(
            0, 0), textcoords='offset points')

    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


display_closestwords_tsnescatterplot(model, 'askreddit', 300)



# %%
result = model.most_similar(positive=['programming', 'datascience'])
print("{}: {:.4f}".format(*result[0]))
# machinelearning: 0.8797

# %%
result = model.most_similar_cosmul(positive=['mensrights', 'feminism'])
print("{}: {:.4f}".format(*result[0]))
# egalitarianism: 0.8056

# %%
print(model.doesnt_match("vinylIdeals voicework techtheatre baking".split()))

# %%
similarity = model.similarity('cooking', 'baking')
similarity > 0.8

# %%
result = model.similar_by_word('cooking')
print("{}: {:.4f}".format(*result[0]))

# %%
distance = model.distance("cooking", "askculinary")
print("{:.1f}".format(distance))

# %%
sim = model.n_similarity(['slowcooking', 'bartenders'], [
                         'homebrewing', 'beer'])
print("{:.4f}".format(sim))

############################################

# %%
sample = nx.read_edgelist(constants.PROC_DATA_DIR / 'sample_smaller.edgelist',
                          nodetype=int, data=(('LINK_SENTIMENT', int),))


# %%
subreddit_names = utils.load_obj(constants.PROC_DATA_DIR, 'subreddit_names')


# %%
named_nodes = [subreddit_names[node] for node in sample.nodes()]

# %%
arr = np.empty((0, 300), dtype='f')

for named_node in named_nodes:
    node_vector = model[named_node]
    arr = np.append(arr, np.array([node_vector]), axis=0)

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(arr)

x_coords = Y[:, 0]
y_coords = Y[:, 1]
plt.scatter(x_coords, y_coords)

for label, x, y in zip(named_nodes, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(
        0, 0), textcoords='offset points')

plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.axis('off')
plt.show()

# %%
pos = dict(zip(sample.nodes(), np.array(list(zip(x_coords, y_coords)))))

plt.figure(figsize=(10, 8))
plt.axis('off')

edge_colors = []
widths = []
for (source, target) in list(sample.edges):
    sentiment = sample[source][target]['LINK_SENTIMENT']
    if sentiment == 1:
        edge_colors.append('g')
        widths.append(.3)
    if sentiment == -1:
        edge_colors.append('r')
        widths.append(3)

partition = community_louvain.best_partition(sample)

nx.draw_networkx_edges(sample, pos, edge_color=edge_colors, width=widths)
nx.draw_networkx_nodes(sample, pos, node_color=list(partition.values()),
                       with_labels=False, node_size=50, cmap=plt.cm.jet)

nx.draw_networkx_labels(
    sample, pos, dict(zip(sample.nodes, named_nodes)), font_size=16, font_color='black')

# DIRECTED GRAPH #
# %%
sample = nx.read_edgelist(constants.PROC_DATA_DIR / 'directed_sample.edgelist',
                          nodetype=int, data=(('LINK_SENTIMENT', int),), create_using=nx.DiGraph())

# %%
named_nodes = [subreddit_names[node] for node in sample.nodes()]

# %%
utils.add_toxicity_perc_node_attribute(sample)

# Find position base on embeddings
arr = np.empty((0, 300), dtype='f')
for named_node in named_nodes:
    node_vector = model[named_node]
    arr = np.append(arr, np.array([node_vector]), axis=0)

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(arr)

x_coords = Y[:, 0]
y_coords = Y[:, 1]

pos = dict(zip(sample.nodes(), np.array(list(zip(x_coords, y_coords)))))
bet_cent = nx.betweenness_centrality(sample, normalized=True, endpoints=True)
node_color = [node[1]['PERC_TOXICITY'] for node in list(sample.nodes(data=True))]
node_size = [v * 1e4 for v in bet_cent.values()]

plt.figure(figsize=(16, 14))

ec = nx.draw_networkx_edges(sample, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(sample, pos, node_color=node_color, node_size=node_size,
                            with_labels=False, cmap=plt.cm.Reds)
nx.draw_networkx_labels(
    sample, pos, dict(zip(sample.nodes, named_nodes)), font_size=16, font_color='black')

plt.colorbar(nc)
plt.axis('off')

plt.savefig(constants.ROOT_DIR / 'images' /
            'bet_cent_perc_toxicity.png', bbox_inches='tight')

# %%
