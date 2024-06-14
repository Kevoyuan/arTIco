from typing import List, Union

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Metrics:
    def __init__(self, labels: List[Union[str, int]]) -> None:
        """A bunch of metrics to evaluate the performance of a classifier

        Args:
            labels (List[Union[str, int]]): class labels
        """
        self.labels: List[Union[str, int]] = labels

        # weight matrix for w score calculation, weights to be applied on normalized confusion matrix
        self.weight_m: np.ndarray = np.array([[0, 10, 20, 30], [1, 0, 10, 20], [5, 1, 0, 10], [20, 5, 1, 0]])

    def __w_score_from_conf(self, conf_matrix_norm: np.ndarray) -> float:
        """Calculate the W-Score from a normalized confusion matrix

        Args:
            conf_matrix_norm (np.ndarray): normalized confusion matrix of shape (n_classes, n_classes)

        Returns:
            float: score value [-1 - 1] where 1 is best
        """
        # normalize weights row wise
        # weight_m = (weight_m * conf_matrix_norm) / (conf_matrix_norm + 1e-10)
        weight_m_r = self.weight_m / (self.weight_m.sum() + 1e-10)
        np.fill_diagonal(weight_m_r, 0)

        # weighted sum
        score_accuracy = np.diagonal(conf_matrix_norm).mean()  # [0-1] where 1 is best
        score_errors = np.sum(conf_matrix_norm * weight_m_r)  # [0-1] where 0 is best

        # weighted average
        score_final = score_accuracy - score_errors

        return score_final

    def w_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the W-Score

        Args:
            y_true (np.ndarray): true values of shape (n_samples,)
            y_pred (np.ndarray): predicted values of shape (n_samples,)

        Returns:
            float: score value [-1 - 1] where 1 is best
        """
        # calculate normalized confusion matrix
        conf_m = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.labels, normalize="true")

        score = self.__w_score_from_conf(conf_matrix_norm=conf_m)

        return score

    def balanced_accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the balanced accuracy score

        Args:
            y_true (np.ndarray): true values of shape (n_samples,)
            y_pred (np.ndarray): predicted values of shape (n_samples,)

        Returns:
            float: balanced accuracy score
        """
        return balanced_accuracy_score(y_true=y_true, y_pred=y_pred, adjusted=True)

    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the recall score

        Args:
            y_true (np.ndarray): true values of shape (n_samples,)
            y_pred (np.ndarray): predicted values of shape (n_samples,)

        Returns:
            float: recall score
        """
        return recall_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)

    def precision_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the precision score

        Args:
            y_true (np.ndarray): true values of shape (n_samples,)
            y_pred (np.ndarray): predicted values of shape (n_samples,)

        Returns:
            float: precision score
        """
        return precision_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the F1 score

        Args:
            y_true (np.ndarray): true values of shape (n_samples,)
            y_pred (np.ndarray): predicted values of shape (n_samples,)

        Returns:
            float: f1 score
        """
        return f1_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)
