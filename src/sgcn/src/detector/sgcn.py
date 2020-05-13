"""SGCN runner."""

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

from core.config import cfg
from detector.layers import (ListModule, SignedSAGEConvolutionBase,
                             SignedSAGEConvolutionDeep)


class SignedGraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, device, shape):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param X: Node features.
        """
        torch.manual_seed(cfg.SEED)
        self.device = device
        self.shape = shape
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.shape[0])
        self.neurons = cfg.LAYERS
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.shape[1]*2,
                                                                  self.neurons[0]).to(self.device)

        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.shape[1]*2,
                                                                  self.neurons[0]).to(self.device)
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1, self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1],
                                                                       self.neurons[i]).to(self.device))

            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1],
                                                                       self.neurons[i]).to(self.device))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        self.regression_weights = Parameter(
            torch.Tensor(4*self.neurons[-1], 3))
        init.xavier_normal_(self.regression_weights)

    def calculate_regression_loss(self, z, target):
        """
        Calculating the regression loss for all pairs of nodes.
        :param z: Hidden vertex representations.
        :param target: Target vector.
        :return loss_term: Regression loss.
        :return predictions_soft: Predictions for each vertex pair.
        """
        pos = torch.cat((self.positive_z_i, self.positive_z_j), 1)
        neg = torch.cat((self.negative_z_i, self.negative_z_j), 1)

        surr_neg_i = torch.cat((self.negative_z_i, self.negative_z_k), 1)
        surr_neg_j = torch.cat((self.negative_z_j, self.negative_z_k), 1)
        surr_pos_i = torch.cat((self.positive_z_i, self.positive_z_k), 1)
        surr_pos_j = torch.cat((self.positive_z_j, self.positive_z_k), 1)

        features = torch.cat(
            (pos, neg, surr_neg_i, surr_neg_j, surr_pos_i, surr_pos_j))
        predictions = torch.mm(features, self.regression_weights)
        predictions_soft = F.log_softmax(predictions, dim=1)
        loss_term = F.nll_loss(predictions_soft, target)
        return loss_term, predictions_soft

    def calculate_positive_embedding_loss(self, z, positive_edges):
        """
        Calculating the loss on the positive edge embedding distances
        :param z: Hidden vertex representation.
        :param positive_edges: Positive training edges.
        :return loss_term: Loss value on positive edge embedding.
        """
        self.positive_surrogates = [random.choice(
            self.nodes) for node in range(positive_edges.shape[1])]
        self.positive_surrogates = torch.from_numpy(
            np.array(self.positive_surrogates, dtype=np.int64).T)
        self.positive_surrogates = self.positive_surrogates.type(
            torch.long).to(self.device)
        positive_edges = torch.t(positive_edges)
        self.positive_z_i = z[positive_edges[:, 0], :]
        self.positive_z_j = z[positive_edges[:, 1], :]
        self.positive_z_k = z[self.positive_surrogates, :]
        norm_i_j = torch.norm(
            self.positive_z_i-self.positive_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(
            self.positive_z_i-self.positive_z_k, 2, 1, True).pow(2)
        term = norm_i_j-norm_i_k
        term[term < 0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_negative_embedding_loss(self, z, negative_edges):
        """
        Calculating the loss on the negative edge embedding distances
        :param z: Hidden vertex representation.
        :param negative_edges: Negative training edges.
        :return loss_term: Loss value on negative edge embedding.
        """
        self.negative_surrogates = [random.choice(
            self.nodes) for node in range(negative_edges.shape[1])]
        self.negative_surrogates = torch.from_numpy(
            np.array(self.negative_surrogates, dtype=np.int64).T)
        self.negative_surrogates = self.negative_surrogates.type(
            torch.long).to(self.device)
        negative_edges = torch.t(negative_edges)
        self.negative_z_i = z[negative_edges[:, 0], :]
        self.negative_z_j = z[negative_edges[:, 1], :]
        self.negative_z_k = z[self.negative_surrogates, :]
        norm_i_j = torch.norm(
            self.negative_z_i-self.negative_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(
            self.negative_z_i-self.negative_z_k, 2, 1, True).pow(2)
        term = norm_i_k-norm_i_j
        term[term < 0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_loss_function(self, z, positive_edges, negative_edges, target):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param positive_edges: Positive edge pairs.
        :param negative_edges: Negative edge pairs.
        :param target: Target vector.
        :return loss: Value of loss.
        """
        loss_term_1 = self.calculate_positive_embedding_loss(z, positive_edges)
        loss_term_2 = self.calculate_negative_embedding_loss(z, negative_edges)
        regression_loss, self.predictions = self.calculate_regression_loss(
            z, target)
        loss_term = regression_loss+cfg.TRAIN.LAMBDA*(loss_term_1+loss_term_2)
        return loss_term

    def forward(self, X, positive_edges, negative_edges, target):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h_pos, self.h_neg = [], []
        self.h_pos.append(torch.tanh(
            self.positive_base_aggregator(X, positive_edges)))
        self.h_neg.append(torch.tanh(
            self.negative_base_aggregator(X, negative_edges)))
        for i in range(1, self.layers):
            self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](
                self.h_pos[i-1], self.h_neg[i-1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(self.negative_aggregators[i-1](
                self.h_neg[i-1], self.h_pos[i-1], positive_edges, negative_edges)))
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        loss = self.calculate_loss_function(
            self.z, positive_edges, negative_edges, target)
        return loss, self.z
