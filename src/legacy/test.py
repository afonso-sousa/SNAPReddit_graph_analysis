# %%
import os
import pickle
from argparse import Namespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from GraphN2V import GraphN2V
from src import constants

default_params = {
    'log2p': 0,                     # Parameter p, p = 2**log2p
    'log2q': 0,                     # Parameter q, q = 2**log2q
    'log2d': 7,                     # Feature size, dimensions = 2**log2d
    'num_walks': 10,                # Number of walks from each node
    'walk_length': 80,              # Walk length
    'window_size': 10,              # Context size for word2vec
    'edge_function': "hadamard",    # Default edge function to use
    "prop_pos": 0.5,                # Proportion of edges to remove nad use as positive samples
    "prop_neg": 0.5,                # Number of non-edges to use as negative samples
                                    #  (as a proportion of existing edges, same as prop_pos)
}

parameter_searches = {
    'log2p': (np.arange(-2, 3), '$\log_2 p$'),
    'log2q': (np.arange(-2, 3), '$\log_2 q$'),
    'log2d': (np.arange(4, 9), '$\log_2 d$'),
    'num_walks': (np.arange(6, 21, 2), 'Number of walks, r'),
    'walk_length': (np.arange(40, 101, 10), 'Walk length, l'),
    'window_size': (np.arange(8, 21, 2), 'Context size, k'),
}

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

# %%
args = Namespace(input=constants.ROOT_DIR / 'graph.txt',
            regen=False,
            workers=4,
            weighted=False,
            directed=False,
            )

# %%
# Remove half the edges, and the same number of "negative" edges
prop_pos = default_params['prop_pos']
prop_neg = default_params['prop_neg']

# Create random training and test graphs with different random edge selections
cached_fn = "%s.graph" % (os.path.basename(args.input))
if os.path.exists(cached_fn) and not args.regen:
    print("Loading link prediction graphs from %s" % cached_fn)
    with open(cached_fn, 'rb') as f:
        cache_data = pickle.load(f)
    Gtrain = cache_data['g_train']
    Gtest = cache_data['g_test']

# %%
else:
    print("Regenerating link prediction graphs")
    # Train graph embeddings on graph with random links
    Gtrain = GraphN2V(is_directed=False,
                    prop_pos=prop_pos,
                    prop_neg=prop_neg,
                    workers=args.workers)
    Gtrain.read_graph(args.input,
                    weighted=args.weighted,
                    directed=args.directed)
    Gtrain.generate_pos_neg_links()

    # Generate a different random graph for testing
    Gtest = GraphN2V(is_directed=False,
                    prop_pos=prop_pos,
                    prop_neg=prop_neg,
                    workers = args.workers)
    Gtest.read_graph(args.input,
                    weighted=args.weighted,
                    directed=args.directed)
    Gtest.generate_pos_neg_links()

    # Cache generated  graph
    cache_data = {'g_train': Gtrain, 'g_test': Gtest}
    with open(cached_fn, 'wb') as f:
        pickle.dump(cache_data, f)


# %%
