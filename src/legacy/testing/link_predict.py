import time

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.testing.rcnn_layer import BaseRGCN, RGCNBlockLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        # Create an embedding value [0,1] for each node of the graph
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g):
        # Retrieve all nodes from the graph in the training data.
        # Consider only the nodes resultin from the sampling process
        node_id = g.ndata['id'].squeeze()
        # Assign a specific embedding to each of this node
        g.ndata['h'] = self.embedding(node_id)


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim).to(torch.device('cuda'))

    def build_hidden_layer(self, idx):
        # Build a number of hidden layer according to the parameter
        # Add a relu activation function, until I create the last layer
        act = F.relu if idx < self.num_hidden_layers - 1 else None

        return RGCNBlockLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                              activation=act, self_loop=True, dropout=self.dropout)


class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()

        # Build RGCN Layers
        # Num rels is doubled, because we consider both directions
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)

        self.rgcn = self.rgcn.to(torch.device('cuda'))

        # Define regularization
        self.reg_param = reg_param

        # Define relations and normalize them
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # Apply DistMult for scoring
        # embedding contains the embedding values of the node after the propagation
        # within the RGCN Block layer
        # triplets contains all triples resulting from the negative sampling process
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g):
        return self.rgcn.forward(g)

    def evaluate(self, g):
        # Get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets, labels):
        # Triplets is a list of data samples (positive and negative),
        # because we train the network using negative sampling
        # Each row in the triplets is a 3-tuple of (source, relation, destination)
        embedding = self.forward(g)

        # The score is computed with the value-by-value multiplication of
        # the embedding values of data produced by the negative sampling process
        # and sum them using the vertical dimension
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)

        return predict_loss + self.reg_param * reg_loss
