import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings

from .data_prep import logging_basic_config


class MLPModel:
    """
    A class for managing MLP models.
    Additionally, the class can be initialized with any number of keyword
    arguments. These will be added as attributes to the class.

    :param predictors: List of predictors used for modeling.
    :type predictors: list
    :param to_predict: Target variable for modeling.
    :type to_predict: str
    :param key_var: Key variable for modeling.
    :type key_var: str

    """

    def __init__(
        self,
        predictors: list,
        to_predict: str,
        key_var: str,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        logger = logging_basic_config(verbose, content_only=True)

        self.predictors = predictors
        self.to_predict = to_predict
        self.key_var = key_var
        self.verbose = verbose
        self.stats: Any

        valid_parameters = ["k", "n_repeats", "task", "param_grid"]

        for parameter in kwargs.keys():
            if parameter not in valid_parameters:
                print(
                    "Parameter '"
                    + parameter
                    + "' is not accepted for "
                    + "MLPModel. Ignoring..."
                )
                continue

            if parameter == "task":
                if kwargs[parameter] not in ["Classification", "Regression"]:
                    logger.error(
                        "Only 'Classification' and 'Regression' "
                        + "tasks are supported."
                    )
                    raise ValueError(
                        "Only 'Classification' and 'Regression' "
                        + "tasks are supported."
                    )
                else:
                    self.task = kwargs[parameter]
                continue

            self.__dict__.update({parameter: kwargs[parameter]})

        # Set default values for the parameters if not provided

        if "k" not in kwargs.keys():
            self.k = 5

        if "n_repeats" not in kwargs.keys():
            self.n_repeats = 1

        if "task" not in kwargs.keys():
            self.task = "Regression"

        if "param_grid" not in kwargs.keys() or kwargs["param_grid"] is None:

            self.param_grid = {
                "mlp__hidden_layer_sizes": [
                    (10, 30, 10),
                    (10, 20, 10),
                    (10, 30, 30, 10),
                    (100,),
                    (200,),
                ],
                "mlp__activation": ["tanh", "relu"],
                "mlp__solver": ["sgd", "adam"],
                "mlp__alpha": [0.001, 0.01, 0.05, 0.1],
                "mlp__learning_rate": ["constant", "adaptive"],
                "mlp__early_stopping": [True],
                "mlp__max_iter": [500],
            }

    def set_parameters(self, **parameters: Any) -> None:
        self.__dict__.update(parameters)

    @ignore_warnings(category=RuntimeWarning)  # type: ignore
    def _fit(self, df: pd.DataFrame) -> None:

        X = df[self.predictors].astype("float64")
        y = df[self.to_predict].astype("float64")

        if self.task == "Regression":
            mlp = MLPRegressor(early_stopping=True, max_iter=500)
            scoring = "neg_mean_absolute_error"
            metrics = ["MAE", "RMSE", "R2"]
        else:
            mlp = MLPClassifier(early_stopping=True, max_iter=500)
            scoring = "balanced_accuracy"
            metrics = [
                "AUC",
                "Accuracy",
                "Sensitivity",
                "Specificity",
                "Precision",
                "Recall",
                "F1",
            ]

        pipeline = [("scaler", StandardScaler()), ("mlp", mlp)]

        pipeline_obj = Pipeline(pipeline)

        grid_search = GridSearchCV(
            pipeline_obj,
            self.param_grid,
            scoring=scoring,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            refit=True,
        )
        grid_search.fit(X, y)

        self.mdl = grid_search.best_estimator_["mlp"]
        self.scaler = grid_search.best_estimator_["scaler"]
        self.params = grid_search.best_params_

        X_scale = self.scaler.transform(X)
        self.y_hat = self.mdl.predict(X_scale)

        self.stats = {metric: [] for metric in metrics}

        self.get_stats(y, self.y_hat)

    @ignore_warnings(category=(ConvergenceWarning, UserWarning))  # type: ignore
    def fit(self, df: pd.DataFrame, verbose: int = 1) -> dict:
        """
        Trains the model using the provided dataframe and default parameters.

        :param df: the provided dataframe.
        :type df: pandas.DataFrame
        :param verbose: the verbosity level
        :type verbose: int

        :return: A dictionary with the results from training.
        :rtype: dict

        """
        logger = logging_basic_config(verbose, content_only=True)

        # Time the training:
        start_time = time.time()

        logger.info("Training the MLP model...")

        self._fit(df)

        training_time = time.time() - start_time
        self.stats["training_time"] = round(training_time, 4)

        result = {
            "predicted": self.y_hat,
            "model": self.mdl,
            "stats": self.stats,
            "best_params": self.params,
            "CV_folds": None,
            "scaler": self.scaler,
        }

        if self.task == "Regression":
            print(">>MAE = ", self.stats["MAE"][0])
            print(">>RMSE = ", self.stats["RMSE"][0])
            print(">>R2 = ", self.stats["R2"][0])

        else:
            print(">>AUC = ", self.stats["AUC"][0])
            print(">>Accuracy = ", self.stats["Accuracy"][0])
            print(">>Sensityvity = ", self.stats["Sensitivity"][0])
            print(">>Specificity = ", self.stats["Specificity"][0])
            print(">>Precision = ", self.stats["Precision"][0])
            print(">>Recall = ", self.stats["Recall"][0])
            print(">>F1 = ", self.stats["F1"][0])

        return result

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts the result of the provided dataframe using the trained model.

        :param df: the provided dataframe.
        :type df: pandas.DataFrame

        :return: The predictions from the trained model regarding the provided dataframe.
        :rtype: np.ndarray

        """

        X = df[self.predictors]
        X_transformed = self.scaler.transform(X)

        y_pred = (
            self.mdl.predict(X_transformed)
            if self.task == "Regression"
            else self.mdl.predict_proba(X_transformed)[:, 1]
        )

        return y_pred

    def get_stats(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Return the stats from the training

        :param y: original labels
        :type y: np.ndarray
        :param y_hat: predicted values
        :type y_hat: np.ndarray

        """

        if len(y.unique()) == 2:
            fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=1)
            self.stats["AUC"].append(metrics.auc(fpr, tpr))

            # tn, fp, fn, tp = metrics.confusion_matrix(y_test, (y_score >= thresholds[np.argmax(tpr - fpr)])*2-1).ravel()
            tn, fp, fn, tp = metrics.confusion_matrix(y, y_hat).ravel()
            self.stats["Accuracy"].append((tp + tn) / (tp + tn + fp + fn))
            self.stats["Sensitivity"].append(tp / (tp + fp))
            self.stats["Specificity"].append(tn / (tn + fn))
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            self.stats["Precision"].append(precision)
            self.stats["Recall"].append(recall)
            self.stats["F1"].append(2 * precision * recall / (precision + recall))
        else:
            self.stats["MAE"].append(metrics.mean_absolute_error(y, y_hat))
            self.stats["RMSE"].append(
                metrics.mean_squared_error(y, y_hat, squared=False)
            )
            self.stats["R2"].append(metrics.r2_score(y, y_hat))

    def output_stats(self) -> None:
        for key, value in self.stats.items():
            logging.info(
                f">> {key} = {np.mean(value): #.4f} \u00B1 {np.std(value): #.4f}"
            )
