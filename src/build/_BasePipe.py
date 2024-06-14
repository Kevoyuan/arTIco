import sys
from pathlib import Path
from typing import Union, Tuple, List
from logging import Logger
import numpy as np

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log


class BasePipe:
    def __init__(
        self,
        work_dir: Path,
        log: Union[Logger, None] = None,
    ) -> None:
        """Parent class for data transformation and machine learning wrapper

        Args:
            work_dir (Path): directory to store extended results in
            log (Union[Logger, None], optional): logger. Defaults to None.
        """

        # set log
        self.log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.work_dir = work_dir
        self.channel_names: List[str] = []
        self.tmsp_names: List[str] = []
        self.feature_2d_names: List[str] = []

        # required result objects
        self._loss_per_epoch: Tuple[List[float], List[float]] = ([0], [0])

    def set_params(self, **parameters):
        """Set hyperparameters

        Returns:
            _type_: _description_
        """
        for parameter, value in parameters.items():
            if parameter in self.__dict__ and self.__valid_para_name(parameter):
                self.log.debug("Set parameter %s to value %s", parameter, value)
                setattr(self, parameter, value)
            else:
                self.log.debug("Parameter %s not in namespace  -IGNORE")
        return self

    def __valid_para_name(self, para: str) -> bool:
        """Check naming convention for names of parameters

        Args:
            para (str): name of parameter object

        Returns:
            bool: check result
        """
        return not para.startswith("_")

    def fill_names(self, channel_names: List[str], tmsp_names: List[str], feature_2d_names: List[str]):
        """Store names of channels and targets for the expected data shape

        Args:
            channel_names (List[str]): channel names of shape (n_channels)
            tmsp_names (List[str]): time stamp names of shape (n_time_stamps)
            feature_2d_names (List[str]): 2D feature names of shape (n_2D_features)
        """
        self.channel_names = channel_names
        self.tmsp_names = tmsp_names
        self.feature_2d_names = feature_2d_names
        self.log.debug(
            "Store names of n_channels %s and n_time_stamps %s and n_2D_features %s",
            len(channel_names),
            len(tmsp_names),
            len(feature_2d_names),
        )

    def fit(self, x: np.ndarray, x_2d: np.ndarray, y: np.ndarray) -> None:
        """Fit pipeline

        Args:
            x (np.ndarray): feature tensor, array-like of shape (n_samples, n_channels, n_time_stamps)
            x_2d (np.ndarray): feature tabular, array-like of shape (n_samples, n_features)
            y (np.ndarray): target vector, array-like of shape (n_samples, 1)
        """
        # update loss per epoch if possible
        self._loss_per_epoch: Tuple[List[float], List[float]] = ([0], [0])

    def predict(self, x: np.ndarray, x_2d: np.ndarray) -> np.ndarray:
        """Predict from fitted pipeline

        Args:
            x (np.ndarray): feature tensor, array-like of shape (n_samples, n_channels, n_time_stamps)
            x_2d (np.ndarray): feature tabular, array-like of shape (n_samples, n_features)
        Returns:
            np.ndarray: predictions in shape (n_samples, 1)
        """
        return np.zeros((x.shape[0], 1))
