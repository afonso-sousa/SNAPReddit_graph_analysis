import os
from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

# Dataset
_C.DATA = CN()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_C.DATA.INPUT_DIR = (ROOT_DIR / 'input').as_posix()
_C.DATA.OUTPUT_DIR = (ROOT_DIR / 'output').as_posix()

_C.DATA.EMB_DIR = os.path.join(_C.DATA.OUTPUT_DIR, 'embedding/')

_C.DATA.EDGE_PATH = os.path.join(_C.DATA.INPUT_DIR, 'sample_sentiment_edgelist.csv')
#_C.DATA.EDGE_PATH = '../input/reddit_edgelist.csv'
_C.DATA.FEATURES_PATH = '../input/reddit_edgelist.csv'
_C.DATA.EMBEDDING_PATH = '../output/embedding/reddit_sgcn.csv'
_C.DATA.REGRESSION_WEIGHTS_PATH = '../output/weights/reddit_sgcn.csv'
_C.DATA.LOG_PATH = '../logs/reddit_logs.json'
_C.DATA.CHECKPOINT_DIR = '../output/checkpoint'

# SVD
_C.SVD = CN()
_C.SVD.REDUCTION_ITER = 30
_C.SVD.REDUCTION_DIMS = 64

# Train
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-2
_C.TRAIN.LAMBDA = 1.0
_C.TRAIN.EPOCHS = 150
_C.TRAIN.WEIGHT_DECAY = 1e-5

_C.SEED = 42
_C.TEST_SIZE = 0.2
_C.SPECTRAL_FEATURES = True
_C.LAYERS = [32, 64, 32]

cfg = _C
