# %%
# Importing the required libraries
import random

import networkx as nx
import numpy as np
import pandas as pd

from src import constants, utils

# %%
# Subreddit embeddings
subreddits = pd.read_csv(constants.RAW_DATA_DIR /
                         'web-redditEmbeddings-subreddits.csv', header=None)

# hyperlinks extracted from hyperlinks in the body of the post.
body = pd.read_csv(constants.RAW_DATA_DIR /
                   'soc-redditHyperlinks-body.tsv', delimiter='\t')

# hyperlinks extracted from hyperlinks in the title of the post.
title = pd.read_csv(constants.RAW_DATA_DIR /
                    'soc-redditHyperlinks-title.tsv', delimiter='\t')

# %%
subreddits.head()

# %%
subreddits.columns = ['name'] + \
    ['e' + str(i) for i in subreddits.loc[:, 1:].columns]
subreddits.head()

# %%
body.head()

# %%
title.head()

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
subreddits.to_csv(constants.PROC_DATA_DIR / 'subreddit_embeddings.txt',
                                                          header=None, index=None, sep=' ')


# %%
# Storing embeddings dictionary
embeddings = temp_normalised_embeddings.to_dict(orient='index')
utils.save_obj(embeddings, constants.PROC_DATA_DIR, 'embeddings')

# %%
# Combine body and title files
combined = pd.concat([title, body], axis=0).reset_index(drop=True)
# combined.drop(labels=['POST_ID', 'TIMESTAMP',
#                      'LINK_SENTIMENT', 'PROPERTIES'], axis=1, inplace=True)
combined.head()

# %%
temp_mapper = subreddits['name'].to_dict()
mapper = {v: k for k, v in temp_mapper.items()}

combined.SOURCE_SUBREDDIT = combined.SOURCE_SUBREDDIT.map(mapper)
combined.TARGET_SUBREDDIT = combined.TARGET_SUBREDDIT.map(mapper)

utils.save_obj(temp_mapper, constants.PROC_DATA_DIR, 'subreddit_names')

# %%
combined.head(n=25)

# %%
source_null = combined[combined.SOURCE_SUBREDDIT.isnull()].index.values
target_null = combined[combined.TARGET_SUBREDDIT.isnull()].index.values

# %%
to_remove = np.sort(list(set(np.concatenate([source_null, target_null]))))
combined = combined.drop(labels=to_remove, axis=0)
#combined = combined.astype(np.uint16).reset_index(drop=True)
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']] = combined[[
    'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].astype(np.uint16)
combined.reset_index(drop=True, inplace=True)

# %%
factor = 0.5
to_keep = random.sample(range(len(combined)),
                        k=int(round(len(combined) * factor)))
sampled = combined.loc[to_keep, :].reset_index(drop=True)

sampled[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT']].to_csv(constants.PROC_DATA_DIR / 'sampled_reddit_edgelist.csv',
                                                          header=None, index=None)



# %%
# SOURCE TARGET edge list
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].to_csv(constants.PROC_DATA_DIR / 'edge_list.txt',
                                                          header=None, index=None, sep=' ')

# %%
# SOURCE TARGET LABEL edge list
combined[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'LINK_SENTIMENT']].to_csv(constants.PROC_DATA_DIR / 'reddit_edgelist.csv',
                                                          header=None, index=None)



# %%
# Full data
combined.to_csv(constants.PROC_DATA_DIR / 'combined.csv',
                index=False, sep='\t')


# %%
