import pandas as pd

from core.config import cfg


def read_graph():
    dataset = pd.read_csv(cfg.DATA.EDGE_PATH).values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(
        set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges
