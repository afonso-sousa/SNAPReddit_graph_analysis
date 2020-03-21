'''
'''
from __future__ import division, print_function

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import constants
from src.tasks import (create_train_test_graphs, grid_search,
                   plot_parameter_sensitivity, test_edge_functions)


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', type=str,
                        help="Task to run, one of 'gridsearch', 'edgeencoding', and 'sensitivity'")

    parser.add_argument('--input', nargs='?', default=constants.ROOT_DIR / 'graph.txt',
                        help='Input graph path')

    parser.add_argument('--regen', dest='regen', action='store_true',
                        help='Regenerate random positive/negative links')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--num_experiments', type=int, default=5,
                        help='Number of experiments to average. Default is 5.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.task is None:
        print("Specify task to run: edgeembedding, sensitivity, gridsearch")
        exit()

    if args.task.startswith("grid"):
        grid_search(args)

    elif args.task.startswith("edge"):
        test_edge_functions(args)

    elif args.task.startswith("sens"):
        plot_parameter_sensitivity(args)
