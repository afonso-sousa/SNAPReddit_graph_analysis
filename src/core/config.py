from yacs.config import CfgNode as CN
from pathlib import Path
import os

_C = CN()

# Dataset
_C.DATA = CN()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_C.DATA.DATA_DIR = (ROOT_DIR / 'input').as_posix()
_C.DATA.RAW_DATA_DIR =  os.path.join(_C.DATA.DATA_DIR, 'raw/')
_C.DATA.PROC_DATA_DIR = os.path.join(_C.DATA.DATA_DIR, 'processed/')
_C.DATA.ARTIFACTS_DIR = (ROOT_DIR / 'artifacts').as_posix()
_C.DATA.TRAIN_DATA_DIR = os.path.join(_C.DATA.ARTIFACTS_DIR, 'bev_train_data/')
_C.DATA.VALIDATION_DATA_DIR = os.path.join(_C.DATA.ARTIFACTS_DIR, 'bev_validation_data/')

cfg = _C
