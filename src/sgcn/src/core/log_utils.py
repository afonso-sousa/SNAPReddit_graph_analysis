import json

from texttable import Texttable

from core.config import cfg


def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(
        logs["performance"]) if i % 10 == 0])
    print(t.draw())


def save_logs(logs):
    with open(cfg.DATA.LOG_PATH, "w") as f:
        json.dump(logs, f)
