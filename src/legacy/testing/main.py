# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import constants
from src.testing import evaluation, link_predict, rcnn_layer, utils

# %%
data = np.loadtxt(constants.PROC_DATA_DIR /
                  'reddit_edgelist.csv', delimiter=',').astype(np.int64)
data[:, [1, 2]] = data[:, [2, 1]]
num_nodes = data[:, 0:2].max() - data[:, 0:2].min() + 1
train_data, remaining = np.split(data,
                                 [int(0.8 * len(data))])
valid_data, test_data = np.split(remaining, [int(0.5 * len(remaining))])
num_rels = 2

# %%
# build test graph
test_graph, test_rel, test_norm = utils.build_test_graph(
    num_nodes, num_rels, train_data)
test_deg = test_graph.in_degrees(
    range(test_graph.number_of_nodes())).float().view(-1, 1)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel).view(-1, 1)
# test_rel = torch.from_numpy(test_rel)
test_norm = torch.from_numpy(test_norm).view(-1, 1)
test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
test_graph.edata['type'] = test_rel

# %%
args = {'dropout': 0.2,
        'n_hidden': 500,
        'gpu': 1,
        'lr': 1e-2,
        'n_bases': 100,
        'n_layers': 2,
        'n_epochs': 10,
        'dataset': 'FB15k-237',
        'eval_batch_size': 500,
        'regularization': 0.01,
        'grad_norm': 1.0,
        'graph_batch_size': 3000,
        'graph_split_size': 0.5,
        'negative_sample': 10,
        'evaluate_every': 10}
# check cuda
use_cuda = args['gpu'] >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.device('cuda')

# %%
# create model
model = link_predict.LinkPredict(num_nodes,
                    args['n_hidden'],
                    num_rels,
                    num_bases=args['n_bases'],
                    num_hidden_layers=args['n_layers'],
                    dropout=args['dropout'],
                    use_cuda=use_cuda,
                    reg_param=args['regularization'])



# %%
# build adj list and calculate degrees for sampling
adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

# %%
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

# %%
model_state_file = 'model_state.pth'
forward_time = []
backward_time = []
loss_values = []
train_start = time.time()

# %%
# Training loop
print("start training...")

epoch = 0
best_mrr = 0
while True:
    model.train()
    epoch += 1

    # Perform edge neighborhood sampling to generate training graph and data
    # The training stage is performed on a sample graph (not the entire graph)
    g, node_id, edge_type, node_norm, data, labels = \
        utils.generate_sampled_graph_and_labels(
            train_data, 
            args['graph_batch_size'], # Useful for sampling
            args['graph_split_size'], # Useful for sampling
            num_rels, 
            adj_list,
            degrees,
            args['negative_sample'])
    
    print("Done edge sampling")

    # Set node/edge feature
    # Reminder of returning values of the generate_sampled_graph_and_labels func
    # g -- DGL graph
    # node_id -- (vector) ids of unique nodes (result of the sampling process)
    # edge_type -- (vector) ids of relations (result of the sampling process)
    # node_norm -- (vector) normalized degree values of each node (same dim of node_id)
    # data -- (matrix) triple samples considering positive and negative samples
    # labels -- (vector) size(, positive + negative samples)
    
    node_id = torch.from_numpy(node_id).view(-1, 1)
    edge_type = torch.from_numpy(edge_type)
    node_norm = torch.from_numpy(node_norm).view(-1, 1)
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1) # same dim of node_id
    
    # Use cuda to store the tensors
    if use_cuda:
        node_id, deg = node_id.cuda(), deg.cuda()
        edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
        data, labels = data.cuda(), labels.cuda()
    
    # The DGL graph is generated, we need to bind the data on the graph elements
    g.ndata.update({'id': node_id, 'norm': node_norm})    
    g.edata['type'] = edge_type

    t0 = time.time()
    
    # Compute loss
    loss = model.get_loss(g, data, labels)
    t1 = time.time()
    
    # Compute back propagation
    loss.backward()
    
    # Clip gradients before optimizing to avoid gradients explosion problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_norm'])
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    loss_values.append(loss.item())
    print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
          format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

    optimizer.zero_grad()

    # Perform the evaluation of the model when the epoch is multiple of evaluate_every
    if epoch % args['evaluate_every'] == 0:
        
        # Perform validation on CPU because full graph is too large
        if use_cuda:
            model.cpu()
        model.eval()
        
        print("start eval")
        mrr = evaluation.evaluate(test_graph, model, valid_data, num_nodes,
                             hits=[1, 3, 10], eval_bz=args['eval_batch_size'])
        
        # save best model
        if mrr < best_mrr:
            if epoch >= args['n_epochs']:
                break
        else:
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       model_state_file)
        if use_cuda:
            model.cuda()

print("training done")
train_done = time.time()
print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))
print("Training time: {:4f}min").format((train_done - train_start)/60)

# %%
# Testing
print("\nstart testing:")
# use best model checkpoint
checkpoint = torch.load(model_state_file)
if use_cuda:
    model.cpu() # test on CPU
model.eval()
model.load_state_dict(checkpoint['state_dict'])
print("Using best epoch: {}".format(checkpoint['epoch']))
eval = evaluation.evaluate(test_graph, model, test_data, num_nodes, hits=[1, 3, 10],
               eval_bz=args['eval_batch_size'])

# Print loss values
plt.plot(loss_values)
plt.show()
