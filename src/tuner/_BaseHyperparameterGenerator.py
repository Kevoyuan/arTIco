import sys
from logging import Logger
from pathlib import Path
from typing import Union

import optuna

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log


class BaseHyperparameterGenerator:
    def __init__(self, log: Union[Logger, None] = None) -> None:
        """Base class for optuna hyperparameter generator

        Args:
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # logger
        self.log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log

    def suggest_hyperparameter(self, trial: optuna.Trial) -> dict:
        """Draw new hyperparameter for each trial, suggestions have to be compatible with chosen sampler

        Args:
            trial (optuna.Trial): current trial

        Returns:
            dict: hyperparameter
        """
        params = {"example": trial.suggest_categorical("example", [None])}

        return params
