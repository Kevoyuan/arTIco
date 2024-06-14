import datetime
import pickle
import sys
from logging import Logger
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

src_dir = str(Path(__file__).absolute().parents[1])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
import src.utils._json_util as json_util
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.build.ExamplePipe import ExamplePipe
from src.evaluate._Evaluate import Evaluate
from src.experiments._Parameters import Parameters
from src.utils._Csv import Csv
from src.utils._hash_file import hash_file
from src.utils._ObjectChecker import ObjectChecker


class Pipeline:
    def __init__(
        self,
        exp_dir: Path,
        model_pipe: BasePipe,
        random_state_shuffle: Union[None, int] = 42,
        shuffle_data: bool = True,
        pickle_pipe_dev_fitted: bool = False,
        log: Union[Logger, None] = None,
    ) -> None:
        """Wrapper for user pipeline to run training and evaluation

        Args:
            exp_dir (Path): working directory
            model_pipe (BasePipe): initialized user pipeline
            random_state (Union[None, int], optional): random state of KFOLD shuffle. Defaults to 42.
            shuffle_data (bool, optional): controls KFOLD shuffle. Defaults to True
            pickle_pipe_dev_fitted (bool, optional): pickle pipeline fitted on whole dev set. Defaults to False
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # set log
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.str = StandardNames()

        # set environment
        self.experiment_dpath: Path = self.__check_directory(dir_path=exp_dir)
        self.parameter_fpath: Path = self.__check_file(fpath=self.experiment_dpath / self.str.fname_para)
        self.parameter_hash: str = hash_file(fpath=self.parameter_fpath, log=self.__log)
        self.results_fpath: Path = self.__check_results_file(fpath=self.experiment_dpath / self.str.fname_results)
        self.results_csv_fpath: Path = self.__check_results_file(fpath=self.experiment_dpath / self.str.fname_results_csv)
        self.in_data_dir: Path = Path()
        self.feature_path: Path = Path()
        self.feature_path_2d: Path = Path()
        self.target_path: Path = Path()
        self.info_path: Path = Path()
        self.info_path_2d: Path = Path()

        # get pipe
        self.pipe: BasePipe = ObjectChecker(log=self.__log).pipeline(pipe=model_pipe)
        self.pipe_paras: dict = {}
        self.random_state_shuffle: Union[None, int] = random_state_shuffle
        self.shuffle_data: bool = shuffle_data

        # init data
        n_samples, n_features, n_channels = 4, 2, 1
        self.feature: np.array = np.zeros((n_samples, n_features, n_channels))
        self.feature_hash: str = ""
        self.feature_2d: np.array = np.zeros((n_samples, n_features))
        self.feature_2d_hash: str = ""
        self.target: np.array = np.zeros((n_samples, 1))
        self.target_hash: str = ""
        self.evaluator: Union[Evaluate, None] = None
        self.labels_sample_id: List[str] = []
        self.labels_tmsp: List[str] = []
        self.labels_channels: List[str] = []
        self.labels_2d_features: List[str] = []
        self.labels_target: List[str] = []

        # store pipe
        self.pickle_pipe_dev_fitted: bool = pickle_pipe_dev_fitted
        self.pipe_pickle_fpath = self.__check_results_file(fpath=self.experiment_dpath / self.str.fname_pipe_pickle)

    def run(self) -> None:
        """Run Pipeline"""
        self.__log.info("Start Experiment")
        self.__log.info("Load Parameters")
        self.__load_parameters()
        self.__log.info("Load Data")
        self.__load_data()
        self.__log.info("Evaluate")
        self.__evaluate()
        self.__log.info("Store")
        self.__store()
        self.__log.info("Experiment End")

    def __check_directory(self, dir_path: Path) -> Path:
        """Check if directory exists

        Args:
            dir_path (Path): path to be checked

        Returns:
            Path: checked path
        """
        if dir_path.is_dir():
            self.__log.debug("Directory is %s", dir_path)
        else:
            self.__log.critical("Directory %s does not exist - EXIT", dir_path)
            sys.exit()

        return dir_path

    def __check_file(self, fpath: Path) -> Path:
        """Check is file exists

        Args:
            fpath (Path): path of file to be checked

        Returns:
            Path: checked path
        """
        if fpath.is_file():
            self.__log.debug("File is %s", fpath)
        else:
            self.__log.critical("File %s does not exist - EXIT", fpath)
            sys.exit()

        return fpath

    def __check_results_file(self, fpath: Path) -> Path:
        """Check if result file exist - delete if True

        Args:
            fpath (Path): path to result file

        Returns:
            Path: path of result clean file
        """
        if fpath.is_file():
            self.__log.warning("Results file %s already exist - REMOVE", fpath)
            fpath.unlink()
        else:
            self.__log.debug("Results file in %s", fpath)

        return fpath

    def __load_parameters(self) -> None:
        """Load parameters from json file to model pipeline"""
        self.__log.info("Load parameters to model pipeline")
        # read json
        para = Parameters(log=self.__log)
        paras = para.read(exp_dir=self.experiment_dpath)

        # store paths
        self.in_data_dir = self.__check_directory(Path(paras[self.str.data][self.str.input][self.str.dir]))
        self.feature_path = self.__check_file(self.in_data_dir / paras[self.str.data][self.str.input][self.str.feature])
        self.feature_path_2d = self.__check_file(self.in_data_dir / paras[self.str.data][self.str.input][self.str.feature_2d])
        self.target_path = self.__check_file(self.in_data_dir / paras[self.str.data][self.str.input][self.str.target])
        self.info_path = self.__check_file(self.in_data_dir / paras[self.str.data][self.str.input][self.str.info])
        self.info_path_2d = self.__check_file(self.in_data_dir / paras[self.str.data][self.str.input][self.str.info_2d])

        # store parameters for model pipeline
        self.pipe_paras = paras[self.str.pipeline]

    def __load_data(self) -> None:
        """Load data into the pipeline"""
        # load info
        info: dict = json_util.load(f_path=self.info_path, log=self.__log)
        self.feature_hash = info[self.str.output][self.str.feature][self.str.hash]
        self.target_hash = info[self.str.output][self.str.target][self.str.hash]
        self.labels_sample_id = info[self.str.id][self.str.samples]
        self.labels_tmsp = info[self.str.id][self.str.tsps]
        self.labels_channels = info[self.str.id][self.str.channels]
        self.labels_target = info[self.str.id][self.str.labels]

        # load info 2D
        info_2d: dict = json_util.load(f_path=self.info_path_2d, log=self.__log)
        self.feature_2d_hash = info_2d[self.str.output][self.str.feature][self.str.hash]
        self.labels_2d_features = info_2d[self.str.id][self.str.feature]

        # store channel & tmsp in user pipe
        self.pipe.fill_names(
            channel_names=self.labels_channels, tmsp_names=self.labels_tmsp, feature_2d_names=self.labels_2d_features
        )

        # load feature
        self.__load_feature_multichannel()

        # load 2D feature
        self.__load_feature_tabular()

        # load target
        self.__load_target()

    def __load_feature_multichannel(self) -> None:
        self.__log.info("Read feature data from %s", self.feature_path)
        feature_hash = hash_file(fpath=self.feature_path, log=self.__log)
        if self.feature_hash != feature_hash:
            self.__log.warning(
                "Hash of feature data not equal to %s - not tracked changes likely",
                self.info_path,
            )
            self.feature_hash = feature_hash
        else:
            self.__log.debug("Feature hash consistent")
        self.feature: np.ndarray = np.load(self.feature_path, allow_pickle=True).astype('float32')
        self.__log.info("Feature has shape %s", self.feature.shape)

    def __load_feature_tabular(self) -> None:
        self.__log.info("Read 2D feature data from %s", self.feature_path_2d)
        feature_hash = hash_file(fpath=self.feature_path_2d, log=self.__log)
        if self.feature_2d_hash != feature_hash:
            self.__log.warning(
                "Hash of 2D feature data not equal to %s - not tracked changes likely",
                self.info_path_2d,
            )
            self.feature_2d_hash = feature_hash
        else:
            self.__log.debug("2D Feature hash consistent")
        self.feature_2d: np.ndarray = np.load(self.feature_path_2d, allow_pickle=True).astype('float32')
        self.__log.info("2D Feature has shape %s", self.feature_2d.shape)

    def __load_target(self) -> None:
        """load target data"""
        self.__log.info("Read target data from %s", self.target_path)
        target_hash = hash_file(fpath=self.target_path, log=self.__log)
        if self.target_hash != target_hash:
            self.__log.warning(
                "Hash of target data not equal to %s - not tracked changes likely",
                self.info_path,
            )
            self.target_hash = target_hash
        else:
            self.__log.debug("Target hash consistent")
        self.target: np.ndarray = np.load(self.target_path, allow_pickle=True)
        self.__log.info("Target has shape %s", self.target.shape)

    def __evaluate(self) -> None:
        """Train and Evaluate pipeline"""
        self.__log.debug("Init Evaluation")
        self.evaluator = Evaluate(
            x=self.feature,
            x_2d=self.feature_2d,
            y=self.target,
            pipe=self.pipe,
            pipe_paras=self.pipe_paras,
            labels=self.labels_target,
            shuffle=self.shuffle_data,
            random_state=self.random_state_shuffle,
            pickle_pipe_dev_fitted=self.pickle_pipe_dev_fitted,
            log=self.__log,
        )

        self.__log.debug("Run Evaluation")
        self.evaluator.run()

    def __store(self) -> None:
        """Store results"""
        self.__log.info("Store results")

        # store predictions
        predictions = pd.DataFrame(
            {
                self.str.true: self.evaluator.y_true_test,
                self.str.predicted: self.evaluator.y_pred_test,
                self.str.samples: np.array(self.labels_sample_id)[self.evaluator.idx_test],
            }
        )
        predictions.set_index(keys=self.str.samples, inplace=True)
        predictions.sort_index(inplace=True)
        csv = Csv(csv_path=self.results_csv_fpath, log=self.__log)
        csv.write(predictions)
        csv_hash = hash_file(fpath=self.results_csv_fpath, log=self.__log)

        # store full dev set fitted pipeline
        if self.pickle_pipe_dev_fitted:
            self.__log.info("Store pipeline in %s", self.pipe_pickle_fpath)
            with open(self.pipe_pickle_fpath, "wb") as handle:
                pickle.dump(self.evaluator.final_pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_hash = hash_file(fpath=self.pipe_pickle_fpath, log=self.__log) if self.pickle_pipe_dev_fitted else None

        # get epoch loss
        epoch_loss = {f"Fold {i}": pipe._loss_per_epoch for i, pipe in enumerate(self.evaluator.pipes)}

        # store results
        results = {
            self.str.creation: str(datetime.datetime.now()),
            self.str.data: {
                self.str.input: {
                    self.str.dir: self.experiment_dpath,
                    self.str.feature: {
                        self.str.path: self.feature_path,
                        self.str.hash: self.feature_hash,
                    },
                    self.str.feature_2d: {
                        self.str.path: self.feature_path_2d,
                        self.str.hash: self.feature_2d_hash,
                    },
                    self.str.target: {
                        self.str.path: self.target_path,
                        self.str.hash: self.target_hash,
                        "Pickle_Protocol": pickle.HIGHEST_PROTOCOL,
                    },
                    self.str.para: {
                        self.str.path: self.parameter_fpath,
                        self.str.hash: self.parameter_hash,
                        "shuffle": self.shuffle_data,
                        "random_state": self.random_state_shuffle,
                    },
                },
                self.str.output: {
                    self.str.dir: self.experiment_dpath,
                    self.str.test: {
                        self.str.path: self.results_csv_fpath,
                        self.str.hash: csv_hash,
                    },
                    self.str.dev: {self.str.path: self.pipe_pickle_fpath, self.str.hash: pickle_hash},
                },
                self.str.labels: self.labels_target,
            },
            self.str.result: {
                self.str.comp_time: np.median(self.evaluator.comp_times_fit_s),
                self.str.dev_comp_time: self.evaluator.comp_times_fit_final_s,
                self.str.metrics: self.evaluator.scores.get_scores(),
                self.str.epoch: epoch_loss,
            },
        }

        json_util.dump(obj=results, f_path=self.results_fpath, log=self.__log)


def test() -> None:
    """Test pipeline"""
    log = custom_log.init_logger()
    bpath = Path(__file__).absolute().parent.parent
    example = bpath / "experiments" / "2023-10-18-09-48-00_Example"

    pipeline = Pipeline(
        exp_dir=example,
        model_pipe=ExamplePipe(work_dir=example, log=log),
        log=log,
    )
    pipeline.run()

    log.info("DONE")


if __name__ == "__main__":
    test()
