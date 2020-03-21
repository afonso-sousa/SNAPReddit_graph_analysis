from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent  # SNAPReddit
SRC_DIR = ROOT_DIR / 'src'
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROC_DATA_DIR = DATA_DIR / 'processed'