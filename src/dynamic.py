# %%
# Importing the required libraries
import networkx as nx
import pandas as pd

from src import constants, utils, drawing

# %%
df = pd.read_csv(constants.PROC_DATA_DIR / 'combined.csv',
                 sep='\t', parse_dates=['TIMESTAMP'])

# %%
G_reddit = nx.from_pandas_edgelist(df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']],
                                   source='SOURCE_SUBREDDIT', target='TARGET_SUBREDDIT', edge_attr=None, create_using=nx.DiGraph())

# %%
df.set_index('TIMESTAMP', inplace=True)

# %%
first_100 = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].head(98)
edges_100 = [tuple(x) for x in first_100.values]

# %%
subreddit_names = utils.load_obj(constants.PROC_DATA_DIR, 'subreddit_names')

# %%
# get fixed positions to draw
G_aux = nx.DiGraph()
G_aux.add_edges_from(edges_100)
pos_100 = nx.spring_layout(G_aux, k=0.30, iterations=50)

# %%
drawing.triad_animation(edges_100, pos_100, subreddit_names,
                        save_path=constants.ROOT_DIR / 'images' / 'triadic_closure.gif')

# %%
