import sys
import time
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.evaluate._Metrics import Metrics
from src.evaluate._Scores import Scores


class Evaluate:
    def __init__(
        self,
        x: np.ndarray,
        x_2d: np.ndarray,
        y: np.ndarray,
        pipe: BasePipe,
        pipe_paras: dict,
        labels: Union[None, List[Union[str, int]]] = None,
        random_state: Union[None, int] = 42,
        shuffle: bool = True,
        pickle_pipe_dev_fitted: bool = False,
        log: Union[Logger, None] = None,
    ) -> None:
        """Evaluate with n fold cross validation

        Args:
            x (np.ndarray): feature tensor, array-like of shape (n_samples, n_channels, n_time_stamps)
            x_2d (np.ndarray): tabular feature, array-like of shape (n_samples, n_features)
            y (np.ndarray): target vector, array-like of shape (n_samples, 1)target vector, array-like of shape (n_samples, 1)
            pipe (BasePipe): initialized user pipeline
            pipe_paras (dict): parameters of user pipeline
            labels (Union[None, List[Union[str, int]]], optional): target class labels. Defaults to None.
            random_state (Union[None, int], optional): random state of KFOLD. Defaults to 42.
            shuffle (bool, optional): controls KFOLD shuffle. Defaults to True
            pickle_pipe_dev_fitted (bool, optional): pickle pipeline fitted on whole dev set. Defaults to False
            log (Union[Logger, None], optional): _description_. Defaults to None.
        """
        # set log
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        self.str = StandardNames()

        # dev set
        self.x, self.x_2d, self.y = self.__check_feature_target(feature=x, feature_2d=x_2d, target=y)

        # pipeline
        self.pipe: BasePipe = pipe
        self.pipe_paras: Dict[str, Union[str, int, float, bool]] = pipe_paras
        self.pipes: List[BasePipe] = []

        # process
        self.shuffle: bool = shuffle
        self.random_state: Union[None, int] = random_state
        self.n_splits: int = 5
        self.n_bootstraps: int = 1000

        # cross fold validation
        self.fold_idxs: List[Tuple[np.ndarray, np.ndarray]] = []
        self.y_pred: List[Tuple[np.ndarray, np.ndarray]] = []

        # evaluation
        self.metrics = Metrics(labels=list(set(self.y.flatten())) if labels is None else labels)
        self.scores: Scores = Scores(log=self.__log)
        self.y_pred_test = np.ndarray([])
        self.y_true_test = np.ndarray([])
        self.idx_test = np.ndarray([])

        self.comp_times_fit_s: List[float] = []

        self.pickle_pipe_dev_fitted: bool = pickle_pipe_dev_fitted
        self.comp_times_fit_final_s: Union[float, None] = 0 if self.pickle_pipe_dev_fitted else None
        self.final_pipe: Union[BasePipe, None] = deepcopy(pipe) if self.pickle_pipe_dev_fitted else None

    def __check_feature_target(
        self, feature: np.ndarray, feature_2d: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Checks if feature target shapes are consistent and as expected,
        EXIT if false

        Args:
            feature (np.ndarray): feature tensor, array-like of shape (n_samples, n_channels, n_time_stamps)
            feature_2d (np.ndarray): feature table, array-like of shape (n_samples, n_features)
            target (np.ndarray): target vector, array-like of shape (n_samples, 1)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: feature, feature table, target unchanged
        """
        checks_passed = (
            len(feature.shape) == 3
            and len(feature_2d.shape) == 2
            and len(target.shape) == 2
            and target.shape[0] == feature.shape[0]
            and target.shape[0] == feature_2d.shape[0]
            and target.shape[1] == 1
        )

        if not checks_passed:
            p1 = "Feature Target inconsistent:"
            p2 = "expected feature shape (n_samples, n_channels, n_time_stamps)"
            p2_2d = "expected tabular feature shape (n_samples, n_features)"
            p3 = "expected target shape (n samples, 1)"
            self.__log.critical(
                "%s %s and %s and %s - got %s and %s and %s - EXIT",
                p1,
                p2,
                p2_2d,
                p3,
                feature.shape,
                feature_2d.shape,
                target.shape,
            )
            sys.exit()
        else:
            self.__log.info(
                "Feature Shape %s, Tabular Feature Shape %s, Target Shape %s", feature.shape, feature_2d.shape, target.shape
            )

        return feature, feature_2d, target

    def run(self):
        """Run evaluation process of estimator"""
        self.__log.info("Split")
        self.__generate_fold_idxs()
        self.__log.info("Fit")
        self.__fit_estimator()
        self.__log.info("Predict")
        self.__predict()
        self.__log.info("Score")
        self.__score()

    def __generate_fold_idxs(self):
        """Generate indices for folds"""
        self.__log.info("Split Dev Set")
        # initialize
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        # generate folds indices
        x = np.zeros(self.y.shape[0])  # actual features irrelevant for split
        self.fold_idxs = [(train_idx, test_idx) for train_idx, test_idx in skf.split(x, self.y)]
        self.__log.info("Generated %s folds", len(self.fold_idxs))

    def __fit_estimator(self):
        """Fit estimators"""
        # Fit estimators per fold
        for i, fold in enumerate(self.fold_idxs):
            self.__log.info("Fit fold %s with %s samples", i, fold[0].shape[0])
            # construct a new unfitted estimator with the same parameters
            self.pipes.append(deepcopy(self.pipe))
            self.pipes[-1].set_params(**self.pipe_paras)

            # fit estimator on training data
            tic = time.perf_counter()
            self.pipes[-1].fit(x=self.x[fold[0]], x_2d=self.x_2d[fold[0]], y=self.y[fold[0]])
            toc = time.perf_counter()
            self.comp_times_fit_s.append(toc - tic)
            self.__log.info("Fold %s fitted in %ss", i, self.comp_times_fit_s[-1])

        # fit on whole dataset
        if self.pickle_pipe_dev_fitted:
            self.__log.info("Fit pipeline on whole dev set with %s samples", self.x.shape[0])
            self.final_pipe.set_params(**self.pipe_paras)
            tic = time.perf_counter()
            self.final_pipe.fit(x=self.x, x_2d=self.x_2d, y=self.y)
            toc = time.perf_counter()
            self.comp_times_fit_final_s = toc - tic
            self.__log.info("Fitted pipeline on %s samples in %ss", self.x.shape[0], self.comp_times_fit_final_s)

    def __predict(self):
        """Predict from train and test set of each fold"""
        for i, fold in enumerate(self.fold_idxs):
            fold_predicts = []
            for j, f in enumerate(fold):
                tt = "Train" if j == 0 else "Test"
                self.__log.info("%s Predict on fold %s for %s samples", tt, i, self.x[f].shape[0])
                fold_predicts.append(self.pipes[i].predict(x=self.x[f], x_2d=self.x_2d[f]))
            self.y_pred.append(tuple(fold_predicts))

    def __score(self):
        """Calculate train and test scores
        Train score is the mean of all folds
        Test score is bootstrapped from all unique predictions on unseen data from each fold
        """

        metrics = {
            self.str.artico: self.metrics.w_score,
            self.str.accuracy: self.metrics.balanced_accuracy_score,
            self.str.recall: self.metrics.recall_score,
            self.str.precision: self.metrics.precision_score,
            self.str.f1: self.metrics.f1_score,
        }

        # assemble test predictions
        self.y_pred_test = np.array([fold[1] for fold in self.y_pred]).flatten()
        self.idx_test = np.array([fold[1] for fold in self.fold_idxs]).flatten()
        self.y_true_test = self.y[self.idx_test].flatten()

        for metric_name, scorer in metrics.items():
            self.__log.debug("Evaluate Metric %s", metric_name)
            # evaluate performance on seen data
            self.__log.debug("Train Set - Get mean of all folds")
            train_scores = [scorer(y_true=self.y[fold[0]], y_pred=self.y_pred[i][0]) for i, fold in enumerate(self.fold_idxs)]
            self.scores.update_train_score(sc_name=metric_name, sc_val=np.mean(train_scores))

            # bootstrap
            self.__log.debug("Test Set concatenated - bootstrap ")
            boot_scores = [scorer(*resample(self.y_true_test, self.y_pred_test, replace=True)) for _ in range(self.n_bootstraps)]

            # calculate median and 95% confidence range
            self.scores.update_test_score(
                sc_name=metric_name,
                sc_vals=np.quantile(a=boot_scores, q=[0.025, 0.5, 0.975]),
            )

        # get overall test confusion matrix
        self.__log.debug("Test Set concatenated - confusion matrix")
        self.scores.update_test_score(
            sc_name=self.str.confusion,
            sc_vals=confusion_matrix(
                y_true=self.y_true_test,
                y_pred=self.y_pred_test,
                labels=self.metrics.labels,
            ),
        )
