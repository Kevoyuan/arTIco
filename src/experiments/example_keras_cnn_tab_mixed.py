import sys
from logging import Logger
from pathlib import Path
from typing import Union

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src.build.ExampleCnnMixedPipe import ExampleCnnMixedPipe
from src.experiments._Experiment import Experiment


def test(log: Union[Logger, None] = None):
    log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
    work_dir = Path("experiments") / "2024-01-26-14-42-00_Example_Keras_Cnn_mixed"

    exp = Experiment(
        work_dir=work_dir,
        user_pipeline=ExampleCnnMixedPipe,
        hyperparameter={"is_binary": False},
        shuffle_data=True,
        random_state_shuffle=42,
        pickle_pipe_dev_fitted=True,
        log=log,
    )
    exp.prepare()
    exp.run()

    log.info("Results:\n%s", exp.get_scores())

    log.info("Done")


if __name__ == "__main__":
    test()
