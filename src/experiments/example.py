import sys
from logging import Logger
from pathlib import Path
from typing import Union

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src.build.ExamplePipe import ExamplePipe
from src.experiments._Experiment import Experiment


def test(log: Union[Logger, None] = None) -> None:
    log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
    work_dir = Path("experiments") / "2023-10-18-09-48-00_Example"

    exp = Experiment(
        work_dir=work_dir,
        user_pipeline=ExamplePipe,
        hyperparameter={
            "constant": None,
            "random_state": 42,
            "strategy": "most_frequent",
        },
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
