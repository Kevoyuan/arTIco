import sys
from logging import Logger
from pathlib import Path

import optuna

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator


class ExampleHyperparameterGenerator(BaseHyperparameterGenerator):
    def __init__(self, constant_mode: bool, log: Logger | None = None) -> None:
        super().__init__(log)
        self.__constant_mode = constant_mode

    def suggest_hyperparameter(self, trial: optuna.Trial) -> dict:
        """Draw new hyperparameter for each trial, suggestions have to be compatible with chosen sampler

        Args:
            trial (optuna.Trial): current trial

        Returns:
            dict: hyperparameter
        """
        if self.__constant_mode:
            return self.__suggest_single(trial=trial)
        else:
            return self.__suggest_space(trial=trial)

    def __suggest_single(self, trial: optuna.Trial) -> dict:
        """Example for constant hyperparameter
        Useful for testing purpose only, e.g. with RandomSampler

        Args:
            trial (optuna.Trial): trial object

        Returns:
            dict: parameters for current trial
        """
        params = {
            "constant": trial.suggest_categorical("constant", choices=[None]),
            "random_state": trial.suggest_categorical("random_state", choices=[42]),
            "strategy": trial.suggest_categorical("strategy", choices=["most_frequent"]),
        }

        return params

    def __suggest_space(self, trial: optuna.Trial) -> dict:
        """Example for variable hyperparameters

        Args:
            trial (optuna.Trial): trial object

        Returns:
            dict: parameters for current trial
        """
        params = {
            "constant": trial.suggest_categorical("constant", [None]),
            "random_state": trial.suggest_int("random_state", low=10, high=50, step=2),
            "strategy": trial.suggest_categorical("strategy", choices=["most_frequent", "prior", "stratified", "uniform"]),
        }

        return params
