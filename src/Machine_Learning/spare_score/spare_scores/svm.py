import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, LinearSVR

from .data_prep import logging_basic_config
from .util import expspace


class SVMModel:
    """
    A class for managing SVM models.
    Additionally, the class can be initialized with any number of keyword
    arguments. These will be added as attributes to the class.

    Methods:
        train_model(df, **kwargs):
            Trains the model using the provided dataframe.

        apply_model(df):
            Applies the trained model on the provided dataframe and returns
            the predictions.

        set_parameters(**parameters):
            Updates the model's parameters with the provided values. This also
            changes the model's attributes, while retaining the original ones.

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
        self.params: Any
        self.stats: Any
        self.y_hat: Any
        self.mdl: Any
        self.cv_results: Any
        self.cv_folds: Any
        self.classify: Any

        valid_parameters = ["kernel", "k", "n_repeats", "task", "param_grid"]

        for parameter in kwargs.keys():
            if parameter not in valid_parameters:
                print(
                    "Parameter '"
                    + parameter
                    + "' is not accepted for "
                    + "SVMModel. Ignoring..."
                )
                continue

            if parameter == "kernel":
                if kwargs[parameter] not in ["linear", "rbf"]:
                    logger.error("Only 'linear' and 'rbf' kernels are " + "supported.")
                    raise ValueError(
                        "Only 'linear' and 'rbf' kernels are " "supported."
                    )
                else:
                    self.kernel = kwargs[parameter]
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
        if "kernel" not in kwargs.keys():
            self.kernel = "linear"

        if "k" not in kwargs.keys():
            self.k = 5

        if "n_repeats" not in kwargs.keys():
            self.n_repeats = 1

        if "task" not in kwargs.keys():
            self.task = "Regression"

        if "param_grid" not in kwargs.keys() or kwargs["param_grid"] is None:

            if self.task == "Classification":
                if self.kernel == "linear":
                    self.param_grid = {"C": expspace([-9, 5])}
                elif self.kernel == "rbf":
                    self.param_grid = {
                        "C": expspace([-9, 5]),
                        "gamma": expspace([-5, 5]),
                    }

            elif self.task == "Regression":
                self.param_grid = {"C": expspace([-5, 5]), "epsilon": expspace([-5, 5])}

    def set_parameters(self, **parameters: Any) -> None:
        self.__dict__.update(parameters)

    # @ignore_warnings(category=ConvergenceWarning)
    def fit(self, df: pd.DataFrame, verbose: int = 1, **kwargs: Any) -> dict:
        logger = logging_basic_config(verbose, content_only=True)

        # Make sure that only the relevant columns are passed
        df = df[self.predictors + [self.to_predict] + [self.key_var]]

        # Time the training:
        start_time = time.time()

        # If the model is too big, optimize the parameters from a sample
        if len(df.index) > 1000:
            logger.info(
                "Due to large dataset, first performing parameter "
                + "tuning with 500 randomly sampled data points."
            )
            sampled_df = df.sample(n=500, random_state=2023)
            sampled_df = sampled_df.reset_index(drop=True)
            self.train_initialize(sampled_df, self.to_predict)
            self.run_CV(sampled_df)
            # Use the optimal parameters to train the model on the full data
            param_grid = {
                par: expspace(
                    [
                        np.min(self.params[f"{par}_optimal"]),
                        np.max(self.params[f"{par}_optimal"]),
                    ]
                )
                for par in self.param_grid
            }
            self.param_grid = param_grid

        # Train the model on the full data, with the optimal parameters
        logger.info("Training SVM model...")
        self.train_initialize(df, self.to_predict)
        self.run_CV(df)
        training_time = time.time() - start_time
        self.stats["training_time"] = round(training_time, 4)

        result = {
            "predicted": self.y_hat,
            "model": self.mdl,
            "stats": self.stats,
            "best_params": self.params,
            "CV_folds": [a[1] for a in self.folds],
        }

        return result

    def predict(self, df: pd.DataFrame, verbose: int = 1) -> np.ndarray:
        # Unpack the model
        self.scaler = self.mdl["scaler"]
        if "bias_correct" in self.mdl.keys():
            self.bias_correct = self.mdl["bias_correct"]
        self.mdl = self.mdl["mdl"]

        # Predict
        n_ensemble = len(self.scaler)
        ss = np.zeros([len(df.index), n_ensemble])
        for i in range(n_ensemble):
            X = self.scaler[i].transform(df[self.predictors])
            if self.task == "Regression":
                ss[:, i] = self.mdl[i].predict(X)
                ss[:, i] = (ss[:, i] - self.bias_correct["int"][i]) / self.bias_correct[
                    "slope"
                ][i]
            else:
                ss[:, i] = self.mdl[i].decision_function(X)

            if self.key_var in df.columns:
                index_to_nan = df[self.key_var].isin(
                    self.cv_results[self.key_var].drop(self.cv_folds[i])
                )
                ss[index_to_nan, i] = np.nan
        ss_mean = np.nanmean(ss, axis=1)
        ss_mean[np.all(np.isnan(ss), axis=1)] = np.nan

        return ss_mean

    def train_initialize(self, df: pd.DataFrame, to_predict: str) -> None:
        id_unique = df[self.key_var].unique()
        self.folds = list(
            RepeatedKFold(
                n_splits=self.k, n_repeats=self.n_repeats, random_state=2022
            ).split(id_unique)
        )
        if len(id_unique) < len(df):
            self.folds = [
                [
                    np.array(df.index[df[self.key_var].isin(id_unique[a])])
                    for a in self.folds[b]
                ]
                for b in range(len(self.folds))
            ]
        self.scaler = [StandardScaler()] * len(self.folds)
        self.params = self.param_grid.copy()
        self.params.update(
            {
                f"{par}_optimal": np.zeros(len(self.folds))
                for par in self.param_grid.keys()
            }
        )
        self.y_hat = np.zeros(len(df))
        if self.task == "Classification":
            self.type, self.scoring, metrics = (
                "SVC",
                "roc_auc",
                [
                    "AUC",
                    "Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "Recall",
                    "F1",
                ],
            )
            self.to_predict, self.classify = to_predict, list(df[to_predict].unique())
            self.mdl = (
                [LinearSVC(max_iter=100000, dual="auto")]
                if self.kernel == "linear"
                else [SVC(max_iter=100000, kernel=self.kernel)]
            ) * len(self.folds)
        elif self.task == "Regression":
            self.type, self.scoring, metrics = (
                "SVR",
                "neg_mean_absolute_error",
                ["MAE", "RMSE", "R2"],
            )
            self.to_predict, self.classify = to_predict, None
            self.mdl = [LinearSVR(max_iter=100000, dual="auto")] * len(self.folds)
            self.bias_correct = {
                "slope": np.zeros((len(self.folds),)),
                "int": np.zeros((len(self.folds),)),
            }
        self.stats = {metric: [] for metric in metrics}
        logging.info(
            f"Training a SPARE model ({self.type}) with {len(df.index)} participants"
        )

    def run_CV(self, df: pd.DataFrame) -> None:
        for i, fold in enumerate(self.folds):
            if i % self.n_repeats == 0:
                logging.info(f"  FOLD {int(i / self.n_repeats + 1)}...")
            X_train, X_test, y_train, y_test = self.prepare_sample(
                df, fold, self.scaler[i], classify=self.classify
            )
            self.mdl[i] = self.param_search(
                self.mdl[i], X_train, y_train, scoring=self.scoring
            )
            # for par in self.param_grid.keys():
            #   self.params[f'{par}_optimal'][i] = np.round(np.log(self.mdl[i].best_params_[par]), 0)
            if self.type == "SVC":
                self.y_hat[fold[1]] = self.mdl[i].decision_function(X_test)
            if self.type == "SVR":
                self.y_hat[fold[1]] = self.mdl[i].predict(X_test)
                self.bias_correct["slope"][i], self.bias_correct["int"][i] = (
                    self.correct_reg_bias(fold, y_test)
                )
            self.get_stats(y_test, self.y_hat[fold[1]])
        self.output_stats()
        self.mdl = {"mdl": self.mdl, "scaler": self.scaler}
        if self.type == "SVR":
            self.mdl["bias_correct"] = self.bias_correct

    def prepare_sample(
        self, df: pd.DataFrame, fold: Any, scaler: Any, classify: Any = None
    ) -> Any:
        X_train, X_test = scaler.fit_transform(
            df.loc[fold[0], self.predictors]
        ), scaler.transform(df.loc[fold[1], self.predictors])
        y_train, y_test = (
            df.loc[fold[0], self.to_predict],
            df.loc[fold[1], self.to_predict],
        )
        if classify is not None:
            y_train, y_test = y_train.map(dict(zip(classify, [-1, 1]))), y_test.map(
                dict(zip(classify, [-1, 1]))
            )
        return X_train, X_test, y_train, y_test

    def param_search(
        self, mdl_i: Any, X_train: list, y_train: list, scoring: Any
    ) -> Any:
        gs = GridSearchCV(
            mdl_i,
            self.param_grid,
            scoring=scoring,
            cv=self.k,
            return_train_score=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        return gs.best_estimator_.fit(X_train, y_train)

    def get_stats(self, y_test: np.ndarray, y_score: np.ndarray) -> None:
        if len(y_test.unique()) == 2:
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
            self.stats["AUC"].append(metrics.auc(fpr, tpr))
            tn, fp, fn, tp = metrics.confusion_matrix(
                y_test, (y_score >= thresholds[np.argmax(tpr - fpr)]) * 2 - 1
            ).ravel()
            self.stats["Accuracy"].append((tp + tn) / (tp + tn + fp + fn))
            self.stats["Sensitivity"].append(tp / (tp + fp))
            self.stats["Specificity"].append(tn / (tn + fn))
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            self.stats["Precision"].append(precision)
            self.stats["Recall"].append(recall)
            self.stats["F1"].append(2 * precision * recall / (precision + recall))
        else:
            self.stats["MAE"].append(metrics.mean_absolute_error(y_test, y_score))
            self.stats["RMSE"].append(
                metrics.mean_squared_error(y_test, y_score, squared=False)
            )
            self.stats["R2"].append(metrics.r2_score(y_test, y_score))
        logging.debug(
            "   > "
            + " / ".join(
                [f"{key}={value[-1]:#.4f}" for key, value in self.stats.items()]
            )
        )

    def correct_reg_bias(self, fold: Any, y_test: list) -> Any:
        slope, interc = np.polyfit(y_test, self.y_hat[fold[1]], 1)
        if slope != 0:
            self.y_hat[fold[1]] = (self.y_hat[fold[1]] - interc) / slope
        return slope, interc

    def output_stats(self) -> None:
        for key, value in self.stats.items():
            logging.info(
                f">> {key} = {np.mean(value): #.4f} \u00B1 {np.std(value): #.4f}"
            )
