import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def model_predictions(model, X, positive_edges, negative_edges, y, test_positive_edges, test_negative_edges, device):
    with torch.no_grad():
        _, train_z = model(X,
                           positive_edges, negative_edges, y)

        score_positive_edges = torch.from_numpy(np.array(
            test_positive_edges, dtype=np.int64).T).type(torch.long).to(device)
        score_negative_edges = torch.from_numpy(np.array(
            test_negative_edges, dtype=np.int64).T).type(torch.long).to(device)

        test_positive_z = torch.cat(
            (train_z[score_positive_edges[0, :], :], train_z[score_positive_edges[1, :], :]), 1)
        test_negative_z = torch.cat(
            (train_z[score_negative_edges[0, :], :], train_z[score_negative_edges[1, :], :]), 1)

        scores = torch.mm(torch.cat((test_positive_z, test_negative_z), 0),
                          model.regression_weights.to(device))
        probability_scores = torch.exp(F.softmax(scores, dim=1))
        predictions = probability_scores[:,
                                         0]/probability_scores[:, 0:2].sum(1)
        predictions = predictions.cpu().detach().numpy()
        targets = [0]*len(test_positive_edges) + \
            [1]*len(test_negative_edges)
        auc = calculate_auc(targets, predictions)
        return predictions, targets, auc


def calculate_auc(targets, predictions):
    targets = [0 if target == 1 else 1 for target in targets]
    auc = roc_auc_score(targets, predictions)
    return auc
