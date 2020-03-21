'''Preparing the graph for Subreddit hyperlinks.'''

import random

import pandas as pd

from src import constants

if __name__ == "__main__":

    fraction = 0.1
    
    body = pd.read_csv(constants.ROOT_DIR / 'input' / 'soc-redditHyperlinks-body.tsv', delimiter = '\t')
    title = pd.read_csv(constants.ROOT_DIR / 'input' / 'soc-redditHyperlinks-title.tsv', delimiter = '\t')

    combined = pd.concat([title, body], axis = 0).reset_index(drop = True)
    source = combined.SOURCE_SUBREDDIT.unique()
    target = combined.TARGET_SUBREDDIT.unique()

    unique_subreddits = list(set(combined['SOURCE_SUBREDDIT'].values.tolist() + combined['TARGET_SUBREDDIT'].values.tolist()))
    mapper = {v:k for (k,v) in enumerate(unique_subreddits)}
    
    combined.SOURCE_SUBREDDIT = combined.SOURCE_SUBREDDIT.map(mapper)
    combined.TARGET_SUBREDDIT = combined.TARGET_SUBREDDIT.map(mapper)
    combined.drop(labels = ['POST_ID', 'TIMESTAMP', 'LINK_SENTIMENT', 'PROPERTIES'], axis = 1, inplace = True)

    to_keep = random.sample(range(len(combined)), k = int(round(len(combined) * fraction)))
    combined = combined.loc[to_keep, :].reset_index(drop = True)
    combined.to_csv(constants.ROOT_DIR / 'graph.txt', header = None, index=None, sep=' ')
