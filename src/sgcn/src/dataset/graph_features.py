import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from core.config import cfg


def setup_features(positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    if cfg.SPECTRAL_FEATURES:
        X = create_spectral_features(
            positive_edges, negative_edges, node_count)
    else:
        X = np.array(pd.read_csv(cfg.FEATURES_PATH))
    return X


def create_spectral_features(positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                   shape=shaping,
                                                   dtype=np.float32))

    svd = TruncatedSVD(n_components=cfg.SVD.REDUCTION_DIMS,
                       n_iter=cfg.SVD.REDUCTION_ITER,
                       random_state=cfg.SEED)
    svd.fit(signed_A)
    X = svd.components_.T
    return X
