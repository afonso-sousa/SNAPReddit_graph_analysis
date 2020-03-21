# %%
import os
os.chdir('/home/afonso/Projects/ARSI/SNAPReddit_link_prediction')

# %%
#Importing the required libraries
import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import pickle
import csv

from src import constants

# %%
with open(constants.ROOT_DIR / 'input' / 'graph.txt', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)
  f.close()
  
graph = [element[0].split(' ') for element in data]
np.shape(graph)

# %%
valid = nx.DiGraph()
valid.add_edges_from(graph)

# %%
from networkit import linkprediction as lp, nxadapter

#Covert networkx graph to networkit graph
valid_it = nxadapter.nx2nk(valid)

#Training and Test graph generation
test_it = lp.RandomLinkSampler.byPercentage(valid_it, 0.9)
train_it = lp.RandomLinkSampler.byPercentage(test_it, 0.7)

# %%
testing_set = lp.MissingLinksFinder(test_it).findAtDistance(2)
training_set = lp.MissingLinksFinder(train_it).findAtDistance(2)

# %%
from functools import partial

def assign_label(pair, graph):
  u, v = pair[0], pair[1]
  return (int(graph.hasEdge(u,v)))

def concatenate(node_set, label):
  dataset = pd.DataFrame({'nodes': node_set, 'label': label})
  return dataset

# Label creation
y_train = list(map(partial(assign_label, graph = test_it), training_set))
y_test = list(map(partial(assign_label, graph = valid_it), testing_set))

# %%

train = concatenate(training_set, y_train)
test = concatenate(testing_set, y_test)

train.head()

# %%
train[['src', 'tar']] = pd.DataFrame(train['nodes'].tolist(), index = train.index)
train.drop(labels = ['nodes'], axis = 1, inplace = True)

test[['src', 'tar']] = pd.DataFrame(test['nodes'].tolist(), index = test.index)
test.drop(labels = ['nodes'], axis = 1, inplace = True)

train.head()

# %%

cols = ['src', 'tar', 'label']

train = train[cols]
test = test[cols]

# %%
train.to_csv(constants.ROOT_DIR / 'input' / 'train.csv', header = True, index = False)
test.to_csv(constants.ROOT_DIR / 'input' / 'test.csv', header = True, index = False)

#Converting NetworKit graphs to NetworkX graphs and pickling them

valid_nx = nxadapter.nk2nx(valid_it)
test_nx = nxadapter.nk2nx(test_it)
train_nx = nxadapter.nk2nx(train_it)

nx.write_gpickle(valid_nx, constants.ROOT_DIR / 'input' / 'valid_graph.gpickle')
nx.write_gpickle(test_nx, constants.ROOT_DIR / 'input' / 'test_graph.gpickle')
nx.write_gpickle(train_nx, constants.ROOT_DIR / 'input' / 'train_graph.gpickle')

# %%
test['label'].value_counts()

# %%
train['label'].value_counts()

# %%
test_dict = test.to_dict(orient = 'series')
train_dict = train.to_dict(orient = 'series')


# Feature Engineering

# %%
test_dict['cos_sim'] = {}

for i in tqdm.tqdm(range(len(test_dict['src']))):
  test_dict['cos_sim'][i] = sum(np.array(list(embeddings[test_dict['src'][i]].values())) * np.array(list(embeddings[test_dict['tar'][i]].values())))

# %%
train_dict['cos_sim'] = {}
for i in tqdm.tqdm(range(len(train['src']))):
  train_dict['cos_sim'][i] = sum(np.array(list(embeddings[train_dict['src'][i]].values())) * np.array(list(embeddings[train_dict['tar'][i]].values())))

# %%
valid_graph = nx.read_gpickle(constants.ROOT_DIR / 'input' / 'valid_graph.gpickle')
test_graph = nx.read_gpickle(constants.ROOT_DIR / 'input' / 'test_graph.gpickle')
train_graph = nx.read_gpickle(constants.ROOT_DIR / 'input' / 'train_graph.gpickle')


# %%
# Deductive Metric (DED)
#for test
test_dict['DED'] = {}

for i in tqdm.tqdm(range(len(test_dict['src']))):
  try:
    test_dict['DED'][i] = len(set(test_graph.successors(test_dict['src'][i])).intersection(set(test_graph.predecessors(test_dict['tar'][i]))))/len(set(test_graph.successors(test_dict['src'][i])))
  except ZeroDivisionError:
    test_dict['DED'][i] = 0

#for train
train_dict['DED'] = {}

for i in tqdm.tqdm(range(len(train_dict['src']))):
  try:
    train_dict['DED'][i] = len(set(train_graph.successors(train_dict['src'][i])).intersection(set(train_graph.predecessors(train_dict['tar'][i]))))/len(set(train_graph.successors(train_dict['src'][i])))
  except ZeroDivisionError:
    train_dict['DED'][i] = 0

# %%
# Inductive Metric (IND)
#for test
test_dict['IND'] = {}

for i in tqdm.tqdm(range(len(test_dict['src']))):
  try:
    test_dict['IND'][i] = len(set(test_graph.predecessors(test_dict['src'][i])).intersection(set(test_graph.predecessors(test_dict['tar'][i]))))/len(set(test_graph.predecessors(test_dict['src'][i])))
  except ZeroDivisionError:
    test_dict['IND'][i] = 0

#for train
train_dict['IND'] = {}

for i in tqdm.tqdm(range(len(train_dict['src']))):
  try:
    train_dict['IND'][i] = len(set(train_graph.predecessors(train_dict['src'][i])).intersection(set(train_graph.predecessors(train_dict['tar'][i]))))/len(set(train_graph.predecessors(train_dict['src'][i])))
  except ZeroDivisionError:
    train_dict['IND'][i] = 0

# %%
# DED LOG
#for test
test_dict['DED_LOG'] = {}

for i in tqdm.tqdm(range(len(test_dict['src']))):
  DED_SCORE = test_dict['DED'][i]
  if(DED_SCORE):
    test_dict['DED_LOG'][i] = DED_SCORE * np.log(len(set(test_graph.successors(test_dict['src'][i]))))
  else:
    test_dict['DED_LOG'][i] = 0

#for train
train_dict['DED_LOG'] = {}

for i in tqdm.tqdm(range(len(train_dict['src']))):
  DED_SCORE = train_dict['DED'][i]
  if(DED_SCORE):
    train_dict['DED_LOG'][i] = DED_SCORE * np.log(len(set(train_graph.successors(train_dict['src'][i]))))
  else:
    train_dict['DED_LOG'][i] = 0

# %%
# IND LOG
#for test
test_dict['IND_LOG'] = {}

for i in tqdm.tqdm(range(len(test_dict['src']))):
  IND_SCORE = test_dict['IND'][i]
  if(IND_SCORE):
    test_dict['IND_LOG'][i] = IND_SCORE * np.log(len(set(test_graph.predecessors(test_dict['src'][i]))))
  else:
    test_dict['IND_LOG'][i] = 0

#for train
train_dict['IND_LOG'] = {}

for i in tqdm.tqdm(range(len(train_dict['src']))):
  IND_SCORE = train_dict['IND'][i]
  if(IND_SCORE):
    train_dict['IND_LOG'][i] = IND_SCORE * np.log(len(set(train_graph.predecessors(train_dict['src'][i]))))
  else:
    train_dict['IND_LOG'][i] = 0 