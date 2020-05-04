# %%
import os

from src.SGCN.sgcn import SignedGCNTrainer
from src.SGCN.utils import read_graph, save_logs, score_printer, tab_printer

from src import constants

os.chdir('/home/afonso/Projects/ARSI/SNAPReddit_graph_analysis/src/SGCN')


class Args:
    def __init__(self):
        self.edge_path = constants.PROC_DATA_DIR / 'reddit_edgelist.csv'
        self.features_path = constants.PROC_DATA_DIR / 'reddit_edgelist.csv'
        self.embedding_path = constants.ROOT_DIR / 'output/embedding/reddit_sgcn.csv'
        self.regression_weights_path = constants.ROOT_DIR / 'output/weights/reddit_sgcn.csv'
        self.log_path = constants.ROOT_DIR / 'logs/reddit_logs.json'
        self.epochs = 100
        self.reduction_iterations = 30
        self.reduction_dimensions = 64
        self.seed = 42
        self.lamb = 1.0
        self.test_size = 0.2
        self.learning_rate = 0.01
        self.weight_decay = 10**-5
        self.layers = [32, 32]
        self.spectral_features = False


args = Args()

tab_printer(args)
edges = read_graph(args)

# %%
trainer = SignedGCNTrainer(args, edges)

# %%
trainer.setup_dataset()

# %%
trainer.create_and_train_model()

# %%
if args.test_size > 0:
    trainer.save_model()
    score_printer(trainer.logs)
    save_logs(args, trainer.logs)

# %%
