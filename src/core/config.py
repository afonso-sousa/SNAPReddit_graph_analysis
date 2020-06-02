import os
from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

# Dataset
_C.DATA = CN()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_C.DATA.IMAGES_DIR = (ROOT_DIR / 'images').as_posix()
_C.DATA.DATA_DIR = (ROOT_DIR / 'data').as_posix()
_C.DATA.RAW_DATA_DIR = os.path.join(_C.DATA.DATA_DIR, 'raw/')
_C.DATA.PROC_DATA_DIR = os.path.join(_C.DATA.DATA_DIR, 'processed/')

cfg = _C
