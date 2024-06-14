import sys
from pathlib import Path

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src.experiments.example import test as simple_test
from src.experiments.example_keras import test as keras_test
from src.experiments.example_optuna import test as optuna_test
from src.experiments.example_2d import test as example_2d_test
from src.experiments.example_keras_cnn_tab_mixed import test as example_keras_cnn_tab_mixed_test


def test_all():
    log = custom_log.init_logger(log_lvl=10)
    log.info("Start tests")

    log.info("Simple test")
    simple_test(log)

    log.info("Keras test")
    keras_test(log)

    log.info("Optuna test")
    optuna_test(log)

    log.info("Simple test 2D")
    example_2d_test(log)

    log.info("Keras test CNN + Tabular mixed")
    example_keras_cnn_tab_mixed_test(log)

    log.info("Done")


if __name__ == "__main__":
    test_all()
