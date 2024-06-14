import shutil
import sys
from logging import Logger
from pathlib import Path
from typing import Tuple, Union, Type

import optuna

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.experiments._Experiment import Experiment
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator
from src.utils._ObjectChecker import ObjectChecker


class Study:
    def __init__(
        self,
        work_dir: Path,
        user_pipeline: Type[BasePipe],
        hyperparameter_generator: BaseHyperparameterGenerator,
        random_state_shuffle: Union[None, int] = 42,
        shuffle_data: bool = True,
        load_study_if_exist: bool = False,
        n_trials: int = 2,
        study_name: str = "Study",
        sampler: Union[optuna.samplers.BaseSampler, None] = None,
        pruner: Union[optuna.pruners.BasePruner, None] = None,
        multi_objective: bool = False,
        timeout: Union[float, None] = None,
        n_jobs: int = 1,
        pickle_pipe_dev_fitted: bool = False,
        log: Union[Logger, None] = None,
    ) -> None:
        """Prepare and run an optuna study

        Args:
            work_dir (Path): directory to perform study in
            user_pipeline (BasePipe): user pipeline object (NOT initialized)
            hyperparameter_generator (BaseHyperparameterGenerator): generator of hyperparameter dict for each trial
            random_state_shuffle (Union[None, int], optional): random state of KFOLD shuffle (only active if shuffle data True). Defaults to 42.
            shuffle_data (bool, optional): switch to shuffle data in KFOLD. Defaults to True.
            load_study_if_exist (bool, optional): continue existing study or start clean. Defaults to False.
            n_trials (int, optional): number of trials. Defaults to 2.
            study_name (str, optional): name of study. Defaults to "Study".
            sampler (Union[optuna.samplers.BaseSampler, None], optional): optuna sampler. Defaults to None.
            multi_objective (bool, optional): switch between single (artico test score) and multi objectives (train, test, conf scores). Default to False
            pruner (Union[optuna.pruners.BasePruner, None], optional): optuna pruner. Defaults to None.
            timeout (Union[float, None], optional): timeout for optuna optimizer. Defaults to None.
            n_jobs (int, optional): number of parallel processes. Defaults to 1.
            pickle_pipe_dev_fitted (bool, optional): pickle pipeline fitted on whole dev set. Defaults to False
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # logger
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log

        self.__str = StandardNames()

        # set working directory
        self.__work_dir = work_dir
        if not self.__work_dir.is_dir():
            self.__log.info("Create study directory %s", self.__work_dir)
            self.__work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.__log.info("Run study in %s", self.__work_dir)

        # user pipeline
        self.__pipeline = user_pipeline
        self.__generator = ObjectChecker(log=self.__log).hyperparameter_generator(generator=hyperparameter_generator)
        self._random_state_shuffle = random_state_shuffle
        self._shuffle_data = shuffle_data

        # optuna
        self.__load_study_if_exist = load_study_if_exist
        self.__n_trials = n_trials
        self.__study_name = study_name
        self.__sampler = sampler
        self.__pruner = pruner
        self.__multi_objective = multi_objective
        self.__timeout = timeout
        self.__n_jobs = n_jobs
        self.pickle_pipe_dev_fitted: bool = pickle_pipe_dev_fitted

    def run_trial(self, trial: optuna.Trial) -> Union[float, Tuple[float, float, float, float]]:
        """Run trial

        Args:
            trial (optuna.Trial): trial object

        Returns:
            float: objective value for optimization
        """
        # set experiments directory name
        trial_dir = self.__work_dir / f"trial_{trial.number}"

        # init experiment
        experiment_handler = Experiment(
            work_dir=trial_dir,
            hyperparameter=self.__generator.suggest_hyperparameter(trial=trial),
            user_pipeline=self.__pipeline,
            random_state_shuffle=self._random_state_shuffle,
            shuffle_data=self._shuffle_data,
            pickle_pipe_dev_fitted=self.pickle_pipe_dev_fitted,
            log=self.__log,
        )

        # create experiment
        experiment_handler.prepare()

        # run
        experiment_handler.run()

        # eval
        metric_test = experiment_handler.get_artico_score()
        if self.__multi_objective:
            metric_train = experiment_handler.get_artico_train_score()
            delta = abs(metric_train - metric_test)
            conf = experiment_handler.get_confidence_interval()
            return metric_test, metric_train, delta, conf
        else:
            return metric_test

    def run_study(self) -> optuna.Study:
        """Run optuna study

        Returns:
            optuna.Study: filled study object
        """
        # prepare directory
        if not self.__load_study_if_exist and self.__work_dir.is_dir():
            for obj in self.__work_dir.glob("*"):
                if obj.is_dir():
                    shutil.rmtree(obj)
                else:
                    obj.unlink()

        # set URL of data base
        db_url = f"sqlite:///{self.__work_dir.absolute() / 'study.sqlite3'}"
        self.__log.info("URL of Optuna Database is %s", db_url)

        # set direction
        if self.__multi_objective:
            direction = None
            directions = [
                optuna.study.StudyDirection.MAXIMIZE,
                optuna.study.StudyDirection.MAXIMIZE,
                optuna.study.StudyDirection.MINIMIZE,
                optuna.study.StudyDirection.MINIMIZE,
            ]
        else:
            direction = optuna.study.StudyDirection.MAXIMIZE
            directions = None

        # init study
        study = optuna.create_study(
            study_name=self.__study_name,
            direction=direction,
            storage=db_url,
            load_if_exists=self.__load_study_if_exist,
            sampler=self.__sampler,
            pruner=self.__pruner,
            directions=directions,
        )
        if self.__multi_objective:
            study.set_metric_names(
                metric_names=[
                    self.__str.opt_test_sc,
                    self.__str.opt_train_sc,
                    self.__str.opt_delta,
                    self.__str.opt_test_conf,
                ]
            )
        else:
            study.set_metric_names(metric_names=[self.__str.opt_test_sc])

        # run
        study.optimize(func=self.run_trial, n_trials=self.__n_trials, timeout=self.__timeout, n_jobs=self.__n_jobs)

        return study
