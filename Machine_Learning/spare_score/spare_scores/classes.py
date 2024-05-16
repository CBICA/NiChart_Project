from dataclasses import dataclass

from spare_scores.data_prep import logging_basic_config
from spare_scores.mlp import MLPModel
from spare_scores.svm import SVMModel
from spare_scores.mlp_torch import MLPTorchModel


class SpareModel:
    """
    A class for managing different spare models.

    Static attributes:
        model_type (str): Type of model to be used.
        predictors (list): List of predictors used for modeling.
        target (str): Target variable for modeling.
        model: The initialized model object.
        verbose (int): Verbosity level.
        parameters (dict): Additional parameters for the model.

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
    """
    def __init__(self, 
                 model_type, 
                 predictors, 
                 target, 
                 key_var,
                 verbose=1,
                 parameters={},
                 **kwargs):
        
        logger = logging_basic_config(verbose, content_only=True)

        self.model_type = model_type
        self.model = None
        self.predictors = predictors
        self.target = target
        self.key_var = key_var
        self.verbose = verbose
        self.parameters = parameters

        if self.model_type == 'SVM':
            self.model = SVMModel(predictors,
                                  target,
                                  key_var,
                                  verbose,
                                  **parameters,)
        
        elif self.model_type == 'MLP':
            self.model = MLPModel(predictors,
                                   target,
                                   key_var,
                                   verbose,
                                   **parameters)
        elif self.model_type == 'MLPTorch':
            self.model = MLPTorchModel(predictors,
                                   target,
                                   key_var,
                                   verbose,
                                   **parameters,
                                   **kwargs)
        else:
            logger.err(f"Model type {self.model_type} not supported.")
            raise NotImplementedError("Only SVM is supported currently.")
    
    def set_parameters(self, **parameters):
        self.parameters = parameters
        self.__dict__.update(parameters)
        self.model.set_parameters(**parameters)
    
    def get_parameters(self):
        return self.__dict__.copy()
        

    def train_model(self, df, **kwargs):

        logger = logging_basic_config(self.verbose, content_only=True)

        try:
            result = self.model.fit(df[self.predictors 
                                       + [self.key_var]
                                       + [self.target]], 
                                    self.verbose,
                                    **kwargs)
        except Exception as e:
            err = '\033[91m\033[1m' \
                + 'spare_train(): Model fit failed.'\
                + '\033[0m\n'
            logger.error(err)

            err += str(e)
            err += "\n\nPlease consider ignoring (-iv/--ignore_vars) any " \
                + "variables that might not be needed for the training of " \
                + "the model, as they could be causing problems.\n\n\n"
            print(err)
            raise Exception(err)
        
        return result

    def apply_model(self, df):

        logger = logging_basic_config(self.verbose, content_only=True)
        result = None
        try:
            result = self.model.predict(df[self.predictors]) #if self.model_type in ['SVM','MLPTorch'] else self.model.mdl.predict(df[self.predictors])
        except Exception as e:
            logger.info('\033[91m' + '\033[1m'
                        + '\n\n\nspare_test(): Model prediction failed.'
                        + '\033[0m')
            print(e)
            print("Please consider ignoring (-iv/--ignore_vars) any variables "
                + "that might not be needed for the training of the model, as "
                + "they could be causing problems.\n\n\n")
        return result

@dataclass
class MetaData:
    """Stores training information on its paired SPARE model"""
    """
    Attributes:
        mdl_type (str): Type of model to be used.
        mdl_task (str): Task of the model to be used.
        kernel (str): Kernel used for SVM.
        predictors (list): List of predictors used for modeling.
        to_predict (str): Target variable for modeling.
        key_var (str): Key variable for modeling.
    """
    mdl_type: str
    mdl_task: str
    kernel: str
    predictors: list
    to_predict: str
    key_var: str
