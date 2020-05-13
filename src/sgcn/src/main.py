# %%
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import trange

from core.config import cfg
from core.eval_utils import model_predictions
from core.log_utils import save_logs, score_printer
from dataset.graph_features import setup_features
from dataset.preprocess import read_graph
from detector.sgcn import SignedGraphConvolutionalNetwork

# %%
edges = read_graph()

# %%
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

# %%
logs = {}
logs["performance"] = [["Epoch", "AUC"]]
logs["training_time"] = [["Epoch", "Seconds"]]

positive_edges, test_positive_edges = train_test_split(edges["positive_edges"],
                                                       test_size=cfg.TEST_SIZE)

negative_edges, test_negative_edges = train_test_split(edges["negative_edges"],
                                                       test_size=cfg.TEST_SIZE)
ecount = len(positive_edges + negative_edges)

# %%
X = setup_features(positive_edges,
                   negative_edges,
                   edges["ncount"])

# %%
positive_edges = torch.from_numpy(np.array(positive_edges,
                                           dtype=np.int64).T).type(torch.long).to(device)

negative_edges = torch.from_numpy(np.array(negative_edges,
                                           dtype=np.int64).T).type(torch.long).to(device)

y = np.array([0 if i < int(
    ecount/2) else 1 for i in range(ecount)]+[2]*(ecount*2))
y = torch.from_numpy(y).type(
    torch.LongTensor).to(device)
X = torch.from_numpy(X).float().to(device)

# %%
# Setup model
model = SignedGraphConvolutionalNetwork(
    device, X.shape).to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=cfg.TRAIN.LR,
                             weight_decay=cfg.TRAIN.WEIGHT_DECAY)


# %%
# Train
best_val_auc = 0
model.train()
epoch_iter = trange(1, cfg.TRAIN.EPOCHS + 1, desc="Loss")
for epoch in epoch_iter:
    start_time = time.time()
    optimizer.zero_grad()
    loss, _ = model(X, positive_edges,
                    negative_edges, y)
    loss.backward()
    epoch_iter.set_description(
        "SGCN (Loss=%g)" % round(loss.item(), 4))
    optimizer.step()
    logs["training_time"].append(
        [epoch, time.time()-start_time])
    if cfg.TEST_SIZE > 0:
        predictions, targets, auc_val = model_predictions(
            model, X, positive_edges, negative_edges, y, test_positive_edges, test_negative_edges, device)
        logs["performance"].append([epoch, auc_val])
        if auc_val > best_val_auc:
            best_val_auc = auc_val
            print('best record: [epoch %d], [val loss %.5f], [val auc %.5f]' % (
                epoch, loss, auc_val))

            # save model
            checkpoint_filename = "checkpoint_epoch_{}.pth".format(epoch + 1)
            checkpoint_filepath = os.path.join(
                cfg.DATA.CHECKPOINT_DIR, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_filepath)


# %%
score_printer(logs)
save_logs(logs)

# %%
#neg_ratio = len(edges["negative_edges"])/edges["ecount"]
pred = [1 if p > .5 else 0 for p in predictions]

# %%


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


confusion_mtx = confusion_matrix(targets, pred)
plot_confusion_matrix(confusion_mtx, ['positive', 'negative'])


# %%
