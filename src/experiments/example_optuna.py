import sys
from logging import Logger
from pathlib import Path
from typing import Union

import optuna
import optuna.visualization as ov

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src.build.ExamplePipe import ExamplePipe
from src.tuner._Study import Study
from src.tuner.ExampleHyperparameterGenerator import \
    ExampleHyperparameterGenerator


def evaluate_study(study: optuna.Study, log: Logger):
    """Simple example of accessing study results
       Alternative to optuna dashboard (VSCode extension)

    Args:
        study (optuna.Study): filled study object
        log (Logger): logger
    """
    print(study.trials_dataframe())
    try:
        ov.plot_param_importances(study=study).show()
    except RuntimeError as er:
        log.warning("Data to simple for plot - %s - SKIP plotting", er)


def test(log: Union[Logger, None] = None):
    """Small example for optuna use
    load dashboard from cmd line: optuna-dashboard sqlite:///experiments/2023-11-15-17-00-00_Example_Optuna/study.sqlite3
    view dashboard: open internet explorer and copy URL from cmd line
    """
    log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
    work_dir = Path("experiments") / "2023-11-15-17-00-00_Example_Optuna"

    # init
    study = Study(
        work_dir=work_dir,
        user_pipeline=ExamplePipe,
        hyperparameter_generator=ExampleHyperparameterGenerator(constant_mode=False, log=log),
        sampler=optuna.samplers.RandomSampler(seed=42),
        random_state_shuffle=42,
        shuffle_data=True,
        load_study_if_exist=False,
        n_trials=4,
        n_jobs=1,  # only threading
        study_name="Example",
        multi_objective=True,
        pickle_pipe_dev_fitted=True,
        log=log,
    )

    # run
    results = study.run_study()

    # evaluate
    evaluate_study(study=results, log=log)

    log.info("Done")


if __name__ == "__main__":
    test()
