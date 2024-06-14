from typing import Union
from logging import Logger
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import sys
from pathlib import Path
import keras

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
import src.utils._custom_log as custom_log


class _Model:
    def __init__(self, log: Union[Logger, None] = None) -> None:
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.__model = keras.models.Sequential()
        self.is_binary: Union[bool, None] = None

        self.__encoder = OneHotEncoder(sparse_output=False)
        self.__labels = np.array([["Good"], ["Acceptable"], ["Marginal"], ["Poor"]])

    def __build(self, input_dim: int):
        self.__log.debug("Build model with %s input dimensions", input_dim)
        # input layer
        self.__model.add(keras.layers.Input(shape=(input_dim,)))

        # hidden layers
        self.__model.add(keras.layers.Dense(100, activation="relu"))
        self.__model.add(keras.layers.Dense(50, activation="relu"))

        # output layer
        if self.is_binary:
            self.__model.add(keras.layers.Dense(1, activation="sigmoid"))
        else:
            self.__model.add(keras.layers.Dense(self.__labels.shape[0], activation="softmax"))

        # Compile the model
        loss = "binary_crossentropy" if self.is_binary else "categorical_crossentropy"
        self.__model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        self.__log.debug("Model summary:\n%s", self.__model.summary())

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.__log.debug("Fit model on target shape %s and feature shape %s", y.shape, x.shape)
        self.__build(input_dim=x.shape[1])
        if not self.is_binary:
            self.__encoder.fit(self.__labels)
            y = self.__encoder.transform(y)
            self.__log.debug("Transformed target shape %s", y.shape)
        self.__model.fit(x, y, epochs=10, batch_size=32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.__log.debug("Predict with model on feature shape %s", x.shape)
        y = self.__model.predict(x)
        if not self.is_binary:
            y = self.__encoder.inverse_transform(y)
        self.__log.debug("Prediction shape %s", y.shape)
        return y


class ExampleKerasPipe(BasePipe):
    def __init__(self, work_dir: Path, log: Union[Logger, None] = None) -> None:
        """Example User defined pipeline, can contain data transformation and learning

        Args:
            work_dir (Path): directory to store extended results in, not used in this example
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # get from parent
        super().__init__(work_dir=work_dir, log=log)

        # classifier
        self.__estimator = _Model(log=self.log)

    def fit(self, x: np.ndarray, x_2d: np.ndarray, y: np.ndarray) -> None:
        self.log.info("Fit estimator with feature shape %s and target shape %s", x.shape, y.shape)
        x_ = x[:, 0, :].copy()
        self.__estimator.fit(x_, y)

    def predict(self, x: np.ndarray, x_2d: np.ndarray) -> np.ndarray:
        self.log.info("Predict with estimator from feature shape %s", x.shape)
        x_ = x[:, 0, :].copy()
        return self.__estimator.predict(x_)

    def set_params(self, **parameters) -> None:
        self.__estimator.is_binary = parameters["Estimator"]["is_binary"]
        self.log.debug("Parameters passed: %s", parameters)


def test():
    # dummy data
    gen = np.random.default_rng(42)
    feature = gen.random((10, 3, 2))
    feature_2d = gen.random((10, 3))
    target = gen.random((feature.shape[0], 1)) > 0.5

    # estimator
    esti = ExampleKerasPipe(work_dir=Path())
    esti.set_params(**{"Estimator": {"is_binary": True}})
    esti.fit(x=feature, x_2d=feature_2d, y=target)
    print(esti.predict(x=feature, x_2d=feature_2d))


if __name__ == "__main__":
    test()
