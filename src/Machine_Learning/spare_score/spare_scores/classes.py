from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data_prep import logging_basic_config
from .mlp import MLPModel
from .mlp_torch import MLPTorchModel
from .svm import SVMModel


class SpareModel:
    """
    A class for managing different spare models.
    Additionally, the class can be initialized with any number of keyword
    arguments. These will be added as attributes to the class.

    Methods:
        train_model(df, **kwargs):
            Calls the model's train_model method and returns the result.

        apply_model(df):
            Calls the model's apply_model method and returns the result.

        set_parameters(**parameters):
            Updates the model's parameters with the provided values. This also
            changes the model's attributes, while retaining the original ones.

    :param model_type: Type of model to be used.
    :type model_type: str
    :predictors: List of predictors used for modeling.
    :type predictors: list
    :param target: Target variable for modeling.
    :type target: str
    :param key_var: key variable for modeling
    :type key_var: str
    :param verbose: Verbosity level.
    :type verbose: int
    :param parameters: Additional parameters for the model.
    :type parameters: dict

    """

    def __init__(
        self,
        model_type: str,
        predictors: list,
        target: str,
        key_var: str,
        verbose: int = 1,
        parameters: dict = {},
        **kwargs: Any,
    ) -> None:
        super().__init__()
        logger = logging_basic_config(verbose, content_only=True)

        self.model_type = model_type
        self.model: Any
        self.predictors = predictors
        self.target = target
        self.key_var = key_var
        self.verbose = verbose
        self.parameters = parameters

        if self.model_type == "SVM":
            self.model = SVMModel(
                predictors,
                target,
                key_var,
                verbose,
                **parameters,
            )

        elif self.model_type == "MLP":
            self.model = MLPModel(predictors, target, key_var, verbose, **parameters)
        elif self.model_type == "MLPTorch":
            self.model = MLPTorchModel(
                predictors, target, key_var, verbose, **parameters, **kwargs
            )
        else:
            logger.error(f"Model type {self.model_type} not supported.")
            raise NotImplementedError("Only SVM is supported currently.")

    def set_parameters(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.__dict__.update(parameters)
        self.model.set_parameters(**parameters)

    def get_parameters(self) -> Any:
        return self.__dict__.copy()

    def train_model(self, df: pd.DataFrame, **kwargs: Any) -> Any:

        logger = logging_basic_config(self.verbose, content_only=True)

        try:
            result = self.model.fit(
                df[self.predictors + [self.key_var] + [self.target]], self.verbose
            )
        except Exception as e:
            err = "\033[91m\033[1m" + "spare_train(): Model fit failed." + "\033[0m\n"
            logger.error(err)

            err += str(e)
            err += (
                "\n\nPlease consider ignoring (-iv/--ignore_vars) any "
                + "variables that might not be needed for the training of "
                + "the model, as they could be causing problems.\n\n\n"
            )
            print(err)
            raise Exception(err)

        return result

    def apply_model(self, df: pd.DataFrame) -> Any:

        logger = logging_basic_config(self.verbose, content_only=True)
        result = None
        try:
            result = self.model.predict(
                df[self.predictors]
            )  # if self.model_type in ['SVM','MLPTorch'] else self.model.mdl.predict(df[self.predictors])
        except Exception as e:
            logger.info(
                "\033[91m"
                + "\033[1m"
                + "\n\n\nspare_test(): Model prediction failed."
                + "\033[0m"
            )
            print(e)
            print(
                "Please consider ignoring (-iv/--ignore_vars) any variables "
                + "that might not be needed for the training of the model, as "
                + "they could be causing problems.\n\n\n"
            )
        return result


@dataclass
class MetaData:
    """
     Stores training information on its paired SPARE model

    :param mdl_type: Type of model to be used.
    :type mdl_type: str
    :param mdl_task: Task of the model to be used.
    :type mdl_task: str
    :param kernel: Kernel used for SVM.
    :type kernel: str
    :param predictors: List of predictors used for modeling.
    :type predictors: list
    :param to_predict: Target variable for modeling.
    :type to_predict: str
    :param key_var: Key variable for modeling.
    :type key_var: str
    """

    mdl_type: str
    mdl_task: str
    kernel: str
    predictors: list
    to_predict: str
    key_var: str
    params: Any = None
    stats: Any = None
    cv_folds: Any = None
    scaler: Any = None
    cv_results: Any = None
