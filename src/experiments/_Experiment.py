import shutil
import sys
from logging import Logger
from pathlib import Path
from typing import Type, Union

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
import src.utils._json_util as json_util
from src._Pipeline import Pipeline
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.experiments._Parameters import Parameters
from src.utils._ObjectChecker import ObjectChecker
from src.utils._Tee import Tee


class Experiment:
    def __init__(
        self,
        work_dir: Path,
        user_pipeline: Type[BasePipe],
        hyperparameter: dict,
        random_state_shuffle: Union[None, int] = 42,
        shuffle_data: bool = True,
        pickle_pipe_dev_fitted: bool = False,
        log: Union[Logger, None] = None,
    ) -> None:
        """Utility to create and run an experiment

        Args:
            work_dir (Path): directory of unique experiment (will be cleaned before run)
            user_pipeline (Type[BasePipe]): BasePipe like user pipeline (NOT initialized)
            hyperparameter (dict): hyperparameter of user pipeline
            random_state_shuffle (Union[None, int], optional): random state of data shuffle in KFOLD, only active if shuffle_data=True. Defaults to 42.
            shuffle_data (bool, optional): shuffle data in KFOLD. Defaults to True.
            pickle_pipe_dev_fitted (bool, optional): pickle pipeline fitted on whole dev set. Defaults to False
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # logger
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log

        # environment
        self._current_dir = work_dir
        self._hyperparameter = hyperparameter
        self.__str = StandardNames()

        # init pipeline
        self.__user_pipeline = ObjectChecker(log=self.__log).pipeline(
            pipe=user_pipeline(work_dir=self._current_dir, log=self.__log)
        )
        self.pickle_pipe_dev_fitted: bool = pickle_pipe_dev_fitted

        # shuffle KFOLD
        self._random_state_shuffle = random_state_shuffle
        self._shuffle_data = shuffle_data

    def prepare(self) -> None:
        """Prepare experiment"""
        # clean
        if self._current_dir.is_dir():
            shutil.rmtree(self._current_dir)
        self._current_dir.mkdir()

        # prepare directory
        Parameters(log=self.__log).create(
            exp_dir=self._current_dir,
            data_dir=self.__str.dir_processed_data,
            pipeline_paras={"Estimator": self._hyperparameter},
        )

    def run(self) -> None:
        """Run main pipeline"""
        # init main pipeline
        self.__log.info("Process Main Pipe")
        with Tee(fpath=self._current_dir / "run.log", log=self.__log):
            self.__log.debug("Init Main Pipe")
            head_pipe = Pipeline(
                exp_dir=self._current_dir,
                model_pipe=self.__user_pipeline,
                shuffle_data=self._shuffle_data,
                random_state_shuffle=self._random_state_shuffle,
                pickle_pipe_dev_fitted=self.pickle_pipe_dev_fitted,
                log=self.__log,
            )

            # run
            self.__log.info("Run Main Pipe")
            head_pipe.run()

    def get_artico_score(self) -> float:
        """Get artico test score from results.json

        Returns:
            float: artico test score
        """
        results = self.get_scores()
        metric = results[self.__str.metrics][self.__str.testing_metrics][self.__str.artico][self.__str.test_median]

        return metric

    def get_artico_train_score(self) -> float:
        """Get artico train score from results.json

        Returns:
            float: artico train score
        """
        results = self.get_scores()

        return results[self.__str.metrics][self.__str.training_metrics][self.__str.artico]

    def get_confidence_interval(self) -> float:
        results = self.get_scores()
        upper = results[self.__str.metrics][self.__str.testing_metrics][self.__str.artico][self.__str.test_conf_up]
        lower = results[self.__str.metrics][self.__str.testing_metrics][self.__str.artico][self.__str.test_conf_lo]

        return abs(upper - lower)

    def get_scores(self) -> dict:
        """Get results from results.json

        Returns:
            dict: results
        """
        # load result
        results_json = json_util.load(f_path=self._current_dir / self.__str.fname_results, log=self.__log)

        # extract metric
        results = results_json[self.__str.result]

        return results
