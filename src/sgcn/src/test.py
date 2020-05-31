# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn.manifold import TSNE

from core.config import cfg

# %%
embs = pd.read_csv(Path(cfg.DATA.EMB_DIR) / 'reddit_sgcn.csv')

# %%
np.savetxt(Path(cfg.DATA.EMB_DIR) / 'reddit_sgcn.txt', embs.reset_index().values,
           delimiter=" ",
           header="{} {}".format(len(embs), len(embs.columns)),
           comments="",
           fmt=["%s"] + ["%.18e"]*len(embs.columns))

# %%
word_vectors = KeyedVectors.load_word2vec_format(Path(cfg.DATA.EMB_DIR) / 'reddit_sgcn.txt', binary=False)

# %%
with open(Path(cfg.DATA.INPUT_DIR) / 'sample_subreddit_names.pkl', 'rb') as handle:
    subreddit_names = pickle.load(handle)


# %%
arr = np.empty((0, 65), dtype='f')

for node in list(word_vectors.wv.vocab.keys()):
    node_vector = word_vectors[node]
    arr = np.append(arr, np.array([node_vector]), axis=0)

# %%
import random
#emb_names = list(map(int, map(float, list(word_vectors.wv.vocab.keys()))))
emb_names = list(range(word_vectors.wv.vectors.shape[0] - 1))

thresh = 15
rnd_sample = random.sample(emb_names, len(emb_names))

named_nodes = [subreddit_names[node] for node in rnd_sample]

# %%
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(arr)

x_coords = Y[:, 0][rnd_sample]
y_coords = Y[:, 1][rnd_sample]
plt.scatter(x_coords, y_coords)

for label, x, y in zip(named_nodes, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(
        0, 0), textcoords='offset points')

plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.axis('off')
plt.show()

# %%
result = word_vectors.most_similar(positive=['213.0'])
print("{}: {:.4f}".format(*result[0]))

# %%
df = pd.read_csv(Path(cfg.DATA.INPUT_DIR) / 'temporal_sentiment_edgelist.csv',
                 sep='\t', parse_dates=['TIMESTAMP'])
                 
G = nx.from_pandas_edgelist(df, source='SOURCE_SUBREDDIT',
                            target='TARGET_SUBREDDIT', edge_attr=True, create_using=nx.DiGraph())

#with open(Path(cfg.DATA.INPUT_DIR) / 'subreddit_names.pkl', 'rb') as handle:
#    full_names = pickle.load(handle)

# %%
G_sample = G.subgraph(emb_names)

# %%
pos = dict(zip(G_sample.nodes(), np.array(list(zip(x_coords, y_coords)))))
pos_edges = [(source, target) for (source, target) in G_sample.edges() if G_sample[source][target]['LINK_SENTIMENT'] == 1]
neg_edges = [(source, target) for (source, target) in G_sample.edges() if G_sample[source][target]['LINK_SENTIMENT'] == -1]

rnd = random.sample(pos_edges, 5)
flat_pos = list(sum(rnd, ()))
flat_neg = list(sum(neg_edges, ()))

G_neg = G_sample.subgraph(set(flat_pos + flat_neg))

pos_edges = [(source, target) for (source, target) in G_neg.edges() if G_neg[source][target]['LINK_SENTIMENT'] == 1]
neg_edges = [(source, target) for (source, target) in G_neg.edges() if G_neg[source][target]['LINK_SENTIMENT'] == -1]


plt.figure(figsize=(16, 14))

nx.draw_networkx_edges(G_neg, pos, edgelist=pos_edges, edge_color='g', alpha=0.2)
nx.draw_networkx_edges(G_neg, pos, edgelist=neg_edges, edge_color='r')

nx.draw_networkx_nodes(G_neg, pos,
                            with_labels=False, cmap=plt.cm.Reds)
nx.draw_networkx_labels(
    G_neg, pos, dict(zip(G_neg.nodes, named_nodes)), font_size=16, font_color='black')

plt.axis('off')

# %%


# %%
