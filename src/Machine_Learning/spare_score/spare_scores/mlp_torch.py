import logging
import os
import time
from typing import Any, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from torch.utils.data import DataLoader, Dataset

from .data_prep import logging_basic_config

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # for MPS backend
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class MLPDataset(Dataset):
    """
    A class for managing datasets that will be used for MLP training

    :param X: the first dimension of the provided data(input)
    :type X: list
    :param y: the second dimension of the provided data(output)
    :type y: list

    """

    def __init__(self, X: list, y: list):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self) -> int:
        """
        returns the length of the provided dataset
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[np.float32, np.float32]:
        """
        (getter)returns the index of both X and y at index: idx(X[idx], y[idx])
        :param idx: the index
        :type idx: int
        """
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    """
    A class to create a simple MLP model.

    :param num_features: total number of features. Default value = 147.
    :type num_features: int
    :param hidden_size: number of features that will be passed to normalization layers of the model. Default value = 256.
    :type hidden_size: int
    :param classification: If set to True, then the model will perform classification, otherwise, regression. Default value = True.
    :type classification: bool
    :param dropout: the dropout value.
    :type dropout: float
    :param use_bn: if set to True, then the model will use the normalization layers, otherwise, the model will use the linear layers.
    :type use_bn: bool
    :param bn: if set to 'bn' the model will use BatchNorm1d() for the hidden layers, otherwise, it will use InstanceNorm1d().
    :type bn: str

    """

    def __init__(
        self,
        hidden_size: int = 256,
        classification: bool = True,
        dropout: float = 0.2,
        use_bn: bool = False,
        bn: str = "bn",
    ) -> None:
        super(SimpleMLP, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.classification = classification
        self.use_bn = use_bn

        def MLPLayer(hidden_size: int) -> nn.Module:
            """
            Our model contains 2 MLPLayers(see bellow)
            """
            return nn.Sequential(
                nn.LazyLinear(hidden_size),
                (
                    nn.InstanceNorm1d(hidden_size, eps=1e-15)
                    if bn != "bn"
                    else nn.BatchNorm1d(hidden_size, eps=1e-15)
                ),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

        self.model = nn.Sequential(
            MLPLayer(self.hidden_size),
            MLPLayer(self.hidden_size // 2),
            nn.LazyLinear(1),
            nn.Sigmoid() if self.classification else nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()


class MLPTorchModel:
    """
    A class for managing MLP models.

    :param predictors: List of predictors used for modeling.
    :type predictors: list
    :param to_predict: Target variable for modeling.
    :type to_predict: str
    :param key_var: Key variable for modeling.
    :type key_var: str

    Additionally, the class can be initialized with any number of keyword
    arguments. These will be added as attributes to the class.
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

        valid_parameters = ["task", "bs", "num_epoches"]

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

            if parameter == "bs":
                try:
                    self.batch_size = int(kwargs[parameter])
                except ValueError:
                    print("Parameter: # of bs should be integer")

            if parameter == "num_epoches":
                try:
                    self.num_epochs = int(kwargs[parameter])
                except ValueError:
                    print("Parameter: # of num_epoches should be integer")

            self.__dict__.update({parameter: kwargs[parameter]})

        # Set default values for the parameters if not provided

        if "task" not in kwargs.keys():
            self.task = "Regression"

        if "batch_size" not in kwargs.keys():
            self.batch_size = 128

        if "num_epochs" not in kwargs.keys():
            self.num_epochs = 100
        if device != "cuda" and device != "mps":
            print("You are not using the GPU! Check your device")

        # Model settings
        self.classification = True if self.task == "Classification" else False
        self.mdl: Any
        self.scaler: Any
        self.stats: Any
        self.param: Any
        self.train_dl: Any
        self.val_dl: Any

    def find_best_threshold(self, y_hat: list, y: list) -> Any:
        """
        Returns best threshold value using the roc_curve

        :param y_hat: predicted values
        :type y_hat: list
        :param y: original labels
        :type y: list

        :return: the best threshold value
        :rtype: Any

        """

        fpr, tpr, thresholds_roc = roc_curve(y, y_hat, pos_label=1)
        youden_index = tpr - fpr
        best_threshold_youden = thresholds_roc[np.argmax(youden_index)]

        return best_threshold_youden

    def get_all_stats(self, y_hat: list, y: list, classification: bool = True) -> dict:
        """
        Returns all stats from training in a dictionary

        :param y: ground truth y (1: AD, 0: CN) -> numpy
        :type y: list
        :param y_hat:predicted y -> numpy, notice y_hat is predicted value [0.2, 0.8, 0.1 ...]
        :type y_hat: list

        :return: A dictionary with the Accuracy, F1 score, Sensitivity, Specificity, Balanced Accuracy, Precision, Recall
        :rtype: dict

        """
        y = np.array(y)
        y_hat = np.array(y_hat)

        if classification:
            auc = roc_auc_score(y, y_hat) if len(set(y)) != 1 else 0.5

            self.threshold = self.find_best_threshold(y_hat, y)

            y_hat = np.where(y_hat >= self.threshold, 1, 0)

            res_dict = {}
            res_dict["Accuracy"] = accuracy_score(y, y_hat)
            res_dict["AUC"] = auc
            res_dict["Sensitivity"] = 0
            res_dict["Specificity"] = 0
            res_dict["Balanced Accuarcy"] = balanced_accuracy_score(y, y_hat)
            res_dict["Precision"] = precision_score(y, y_hat)
            res_dict["Recall"] = recall_score(y, y_hat)
            res_dict["F1"] = f1_score(y, y_hat)

            if len(set(y)) != 1:
                tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                res_dict["Sensitivity"] = sensitivity
                res_dict["Specificity"] = specificity
        else:
            res_dict = {}
            mae = mean_absolute_error(y, y_hat)
            mrse = mean_squared_error(y, y_hat, squared=False)
            r2 = r2_score(y, y_hat)
            res_dict["MAE"] = mae
            res_dict["RMSE"] = mrse
            res_dict["R2"] = r2

        return res_dict

    def object(self, trial: Any) -> float:
        evaluation_metric = (
            "Balanced Accuracy" if self.task == "Classification" else "MAE"
        )
        assert self.train_dl is not None
        assert self.val_dl is not None

        hidden_size = trial.suggest_categorical(
            "hidden_size", [x for x in range(32, 512, 32)]
        )
        dropout = trial.suggest_float("dropout", 0.1, 0.8, step=0.03)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        use_bn = trial.suggest_categorical("use_bn", [False, True])
        bn = trial.suggest_categorical("bn", ["bn", "in"])

        model = SimpleMLP(
            hidden_size=hidden_size,
            classification=self.classification,
            dropout=dropout,
            use_bn=use_bn,
            bn=bn,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss() if self.classification else nn.L1Loss()

        model.train()

        for epoch in range(self.num_epochs):

            step = 0
            for _, (x, y) in enumerate(self.train_dl):
                step += 1
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                optimizer.zero_grad()

                loss = loss_fn(output, y)

                loss.backward()

                optimizer.step()

            val_step = 0
            val_total_metric = 0.0
            val_total_loss = 0.0

            with torch.no_grad():
                for _, (x, y) in enumerate(self.val_dl):
                    val_step += 1
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x.float())

                    loss = loss_fn(output, y)
                    val_total_loss += loss.item()
                    metric = self.get_all_stats(
                        output.cpu().data.numpy(),
                        y.cpu().data.numpy(),
                        classification=self.classification,
                    )[evaluation_metric]
                    val_total_metric += metric

                val_total_loss /= val_step
                val_total_metric /= val_step

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # save checkpoint
            trial.report(val_total_loss, epoch)
            checkpoint = {
                "trial_params": trial.params,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": val_total_loss,
            }
            trial.set_user_attr("checkpoint", checkpoint)

        return val_total_metric

    def set_parameters(self, **parameters: Any) -> None:
        if "linear1.weight" in parameters.keys():
            self.param = parameters
        else:
            self.__dict__.update(parameters)

    @ignore_warnings(category=(ConvergenceWarning, UserWarning))  # type: ignore
    def fit(self, df: pd.DataFrame, verbose: int = 1, **kwargs: Any) -> dict:
        logger = logging_basic_config(verbose, content_only=True)

        # Time the training:
        start_time = time.time()

        logger.info("Training the MLP model...")

        # Start training model here
        X = df[self.predictors]
        y = df[self.to_predict].tolist()

        stratify = y if self.task == "Classification" else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)

        self.scaler = StandardScaler().fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)

        train_ds = MLPDataset(X_train, y_train)
        val_ds = MLPDataset(X_val, y_val)

        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)

        study = optuna.create_study(
            direction="maximize" if self.classification else "minimize"
        )
        study.optimize(self.object, n_trials=100)

        # Get the best trial and its checkpoint
        best_trial = study.best_trial
        best_checkpoint = best_trial.user_attrs["checkpoint"]

        best_hyperparams = best_checkpoint["trial_params"]
        best_model_state_dict = best_checkpoint["model_state_dict"]

        self.mdl = SimpleMLP(
            hidden_size=best_hyperparams["hidden_size"],
            classification=self.classification,
            dropout=best_hyperparams["dropout"],
            use_bn=best_hyperparams["use_bn"],
            bn=best_hyperparams["bn"],
        )

        self.mdl.load_state_dict(best_model_state_dict)
        self.mdl.to(device)
        self.mdl.eval()
        X_total = self.scaler.transform(np.array(X, dtype=np.float32))
        X_total = torch.tensor(X_total).to(device)

        self.y_pred = self.mdl(X_total).cpu().data.numpy()
        self.stats = self.get_all_stats(
            self.y_pred, y, classification=self.classification
        )

        self.param = best_model_state_dict

        training_time = time.time() - start_time
        self.stats["training_time"] = round(training_time, 4)

        result = {
            "predicted": self.y_pred,
            "model": self.mdl,
            "stats": self.stats,
            "best_params": self.param,
            "CV_folds": None,
            "scaler": self.scaler,
        }

        if self.task == "Regression":
            print(">>MAE = ", self.stats["MAE"])
            print(">>RMSE = ", self.stats["RMSE"])
            print(">>R2 = ", self.stats["R2"])

        else:
            print(">>AUC = ", self.stats["AUC"])
            print(">>Accuracy = ", self.stats["Accuracy"])
            print(">>Sensityvity = ", self.stats["Sensitivity"])
            print(">>Specificity = ", self.stats["Specificity"])
            print(">>Precision = ", self.stats["Precision"])
            print(">>Recall = ", self.stats["Recall"])
            print(">>F1 = ", self.stats["F1"])
            print(">>Threshold = ", self.threshold)

        return result

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.predictors]
        X = self.scaler.transform(np.array(X, dtype=np.float32))
        X = torch.tensor(X).to(device)

        checkpoint_dict = self.param
        self.mdl.load_state_dict(checkpoint_dict)
        self.mdl.eval()

        y_pred = self.mdl(X).cpu().data.numpy()

        return y_pred

    def output_stats(self) -> None:
        for key, value in self.stats.items():
            logging.info(
                f">> {key} = {np.mean(value): #.4f} \u00B1 {np.std(value): #.4f}"
            )
