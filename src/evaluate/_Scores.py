import sys
from logging import Logger
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src._StandardNames import StandardNames


class Scores:
    def __init__(self, log: Union[Logger, None] = None) -> None:
        """Object to store scores

        Args:
            log (Union[Logger, None], optional): Logging. Defaults to None.
        """
        # set log
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.str = StandardNames()

        # scores
        scores = (
            self.str.artico,
            self.str.f1,
            self.str.recall,
            self.str.precision,
            self.str.accuracy,
        )
        self.test_sc_types = (
            self.str.test_conf_lo,
            self.str.test_median,
            self.str.test_conf_up,
        )

        # init training mean scores
        self._training_scores: Dict[str, float] = {sc: 0 for sc in scores}

        # init testing bootstrap scores
        self._testing_scores: Dict[str, Dict[str, float]] = {
            sc: {te: 0 for te in self.test_sc_types} for sc in scores
        }

    def update_train_score(self, sc_name: str, sc_val: float) -> None:
        """Update training score

        Args:
            sc_name (str): name of metric
            sc_val (float): score
        """
        if sc_name in self._training_scores:
            self.__log.debug("Store %s training score", sc_name)
            self._training_scores[sc_name] = sc_val
        else:
            self.__log.warning("%s is no valid training metric - SKIP", sc_name)

    def update_test_score(
        self, sc_name: str, sc_vals: Union[Tuple[float, float, float], np.ndarray]
    ) -> None:
        """Update testing score

        Args:
            sc_name (str): name of metric
            sc_vals (Union[Tuple[float, float, float], np.ndarray]): score in shape (lower confidence boarder, median, upper confidence boarder)
                                                                     or confusion matrix in shape (n_classes, n_classes)
        """
        self.__log.debug("Store %s testing score", sc_name)
        if sc_name in self._testing_scores:
            self._testing_scores[sc_name] = dict(zip(self.test_sc_types, sc_vals))
        elif sc_name == self.str.confusion:
            self._testing_scores[sc_name] = sc_vals
        else:
            self.__log.warning("%s is no valid testing metric - SKIP", sc_name)

    def get_scores(
        self,
    ) -> Dict[str, Union[Dict[str, float], Dict[str, Dict[str, float]]]]:
        """Scores

        Returns:
            Dict[str, Union[Dict[str, float], Dict[str, Dict[str, float]]]]: scores, shallow dict for training, folded dict for testing
        """
        return {
            self.str.training_metrics: self._training_scores,
            self.str.testing_metrics: self._testing_scores,
        }


def test():
    log = custom_log.init_logger(log_lvl=10)
    sc = Scores(log=log)
    log.debug("%s", sc.__dict__)

    log.info("DONE")


if __name__ == "__main__":
    test()
