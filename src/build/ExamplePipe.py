from typing import Union
from logging import Logger
from sklearn.dummy import DummyClassifier
import numpy as np
import sys
from pathlib import Path

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe


class ExamplePipe(BasePipe):
    def __init__(self, work_dir: Path, log: Union[Logger, None] = None) -> None:
        """Example User defined pipeline, can contain data transformation and learning

        Args:
            work_dir (Path): directory to store extended results in, not used in this example
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # get from parent
        super().__init__(work_dir=work_dir, log=log)

        # classifier
        self.__estimator = DummyClassifier()

        # parameters
        self.constant = None
        self.random_state = None
        self.strategy = "prior"

    def fit(self, x: np.ndarray, x_2d: np.ndarray, y: np.ndarray) -> None:
        self.log.info("Fit estimator with feature shape %s and target shape %s", x.shape, y.shape)
        x_ = x[:, :, 0].copy()
        self.__estimator.fit(x_, y)

    def predict(self, x: np.ndarray, x_2d: np.ndarray) -> np.ndarray:
        self.log.info("Predict with estimator from feature shape %s", x.shape)
        x_ = x[:, :, 0].copy()
        return self.__estimator.predict(x_)

    def set_params(self, **parameters):
        self.__estimator.set_params(**parameters["Estimator"])
        self.log.debug("Estimator parameters are: %s", self.__estimator.get_params())


def test():
    # dummy data
    gen = np.random.default_rng(42)
    feature = gen.random((10, 3, 2))
    feature_2d = gen.random((10, 4))
    target = gen.random((feature.shape[0], 1)) > 0.5

    # estimator
    esti = ExamplePipe(work_dir=Path())
    esti.set_params(**{"Estimator": {}})
    esti.fit(x=feature, x_2d=feature_2d, y=target)
    print(esti.predict(x=feature, x_2d=feature_2d))


if __name__ == "__main__":
    test()
