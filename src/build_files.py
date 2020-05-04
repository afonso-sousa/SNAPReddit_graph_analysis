# %%
# Importing the required libraries
import random

import networkx as nx
import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from src import constants, utils

# %%
# Subreddit embeddings
subreddits = pd.read_csv(constants.RAW_DATA_DIR /
                         'web-redditEmbeddings-subreddits.csv', header=None)

# hyperlinks extracted from hyperlinks in the body of the post
body = pd.read_csv(constants.RAW_DATA_DIR /
                   'soc-redditHyperlinks-body.tsv', delimiter='\t')

# hyperlinks extracted from hyperlinks in the title of the post
title = pd.read_csv(constants.RAW_DATA_DIR /
                    'soc-redditHyperlinks-title.tsv', delimiter='\t')

# %%
# Changing column names
subreddits.columns = ['name'] + \
    ['e' + str(i) for i in subreddits.loc[:, 1:].columns]
subreddits.head()

# %%
"""Storing dictionary of normalised subreddit embeddings
to lower the computational expense of cosine similarities
between pairs of subreddit embeddings"""
temp_normalised_embeddings = np.divide(subreddits.loc[:, 'e1':], np.array(
    np.sqrt(np.square(subreddits.loc[:, 'e1':]).sum(axis=1))).reshape((-1, 1)),)
# null embeddings
to_remove = temp_normalised_embeddings[temp_normalised_embeddings.isnull().any(
    axis=1)].index.values

subreddits.loc[to_remove, :]
subreddits = subreddits.drop(to_remove, axis=0).reset_index(drop=True)
temp_normalised_embeddings = temp_normalised_embeddings.drop(
    labels=to_remove, axis=0).reset_index(drop=True)

# %%
# Storing raw embedding representation
subreddits.to_csv(constants.PROC_DATA_DIR / 'subreddit_embeddings.txt',
                  header=None, index=None, sep=' ')

# %%
# Storing embeddings dictionary
embeddings = temp_normalised_embeddings.to_dict(orient='index')
utils.save_obj(embeddings, constants.PROC_DATA_DIR, 'norm_embeddings')

# %%
# Combine body and title files
combined = pd.concat([title, body], axis=0).reset_index(drop=True)
combined.head()

# %%
# removing source nodes without embedding representation
combined = combined[combined.SOURCE_SUBREDDIT.isin(subreddits['name'])]
# removing target nodes without embedding representation
combined = combined[combined.TARGET_SUBREDDIT.isin(subreddits['name'])]

# %%
# Mapper from name to index
unique_subreddits = np.unique(
    combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].values)
mapper = {k: v for v, k in enumerate(unique_subreddits)}

mapper_to_save = {v: k for k, v in mapper.items()}

utils.save_obj(mapper_to_save, constants.PROC_DATA_DIR, 'subreddit_names')

# %%
# Replace name strings by indexes in map
combined.SOURCE_SUBREDDIT = combined.SOURCE_SUBREDDIT.map(mapper)
combined.TARGET_SUBREDDIT = combined.TARGET_SUBREDDIT.map(mapper)

# %%
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']] = combined[[
    'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].astype(np.uint16)
combined.reset_index(drop=True, inplace=True)

# %%
"""
factor = 0.5
to_keep = random.sample(range(len(combined)),
                        k=int(round(len(combined) * factor)))
sampled = combined.loc[to_keep, :].reset_index(drop=True)

sampled[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT']].to_csv(constants.PROC_DATA_DIR / 'sampled_reddit_edgelist.csv',
                                                          header=None, index=None)
"""

# %%
# Simple edge list
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].to_csv(constants.PROC_DATA_DIR / 'simple_edgelist.txt',
                                                          header=None, index=None, sep=' ')

# %%
# Sentiment edge list
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT']].to_csv(constants.PROC_DATA_DIR / 'sentiment_edgelist.csv',
                                                                            header=None, index=None)

# %%
# Temporal sentiment edge list
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT', 'TIMESTAMP']].to_csv(constants.PROC_DATA_DIR / 'temporal_sentiment_edgelist.csv',
                                                                            index=None, sep='\t')

# %%
# Full data
combined.to_csv(constants.PROC_DATA_DIR / 'combined_data.csv',
                index=False, sep='\t')



############################################
######## GLOVE AND WORD2VEC FORMATS ########
############################################

# %%
# Sentiment edge list Glove formatted
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT']].to_csv(constants.PROC_DATA_DIR / 'glove_edgelist.txt',
                                                                            header=None, index=None, sep=' ')

# %%
glove_file = datapath(constants.PROC_DATA_DIR / 'glove_edgelist.txt')
tmp_file = get_tmpfile(constants.PROC_DATA_DIR / 'w2v_edgelist.txt')
_ = glove2word2vec(glove_file, tmp_file)

# %%
data = np.loadtxt(constants.PROC_DATA_DIR /
                  'glove_edgelist.txt', delimiter=' ').astype(np.int64)
train_data, remaining = np.split(data,
                                 [int(0.8 * len(data))])
valid_data, test_data = np.split(remaining, [int(0.5 * len(remaining))])

np.savetxt(constants.PROC_DATA_DIR / 'glove_reddit_train.txt',
           train_data.astype(np.int64), fmt='%d', delimiter=' ')
np.savetxt(constants.PROC_DATA_DIR / 'glove_reddit_val.txt',
           valid_data.astype(np.int64), fmt='%d', delimiter=' ')
np.savetxt(constants.PROC_DATA_DIR / 'glove_reddit_test.txt',
           test_data.astype(np.int64), fmt='%d', delimiter=' ')

glove_file = datapath(constants.PROC_DATA_DIR / 'glove_reddit_train.txt')
tmp_file = get_tmpfile(constants.PROC_DATA_DIR / 'reddit_train.txt')
_ = glove2word2vec(glove_file, tmp_file)

glove_file = datapath(constants.PROC_DATA_DIR / 'glove_reddit_val.txt')
tmp_file = get_tmpfile(constants.PROC_DATA_DIR / 'reddit_val.txt')
_ = glove2word2vec(glove_file, tmp_file)

glove_file = datapath(constants.PROC_DATA_DIR / 'glove_reddit_test.txt')
tmp_file = get_tmpfile(constants.PROC_DATA_DIR / 'reddit_test.txt')
_ = glove2word2vec(glove_file, tmp_file)


# %%
