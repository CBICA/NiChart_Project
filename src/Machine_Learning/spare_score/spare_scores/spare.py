import logging
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

from .classes import MetaData, SpareModel
from .data_prep import (
    check_test,
    check_train,
    convert_cat_variables,
    logging_basic_config,
)
from .util import (
    check_file_exists,
    is_unique_identifier,
    load_df,
    load_model,
    save_file,
)


def spare_train(
    df: Union[pd.DataFrame, str],
    to_predict: str,
    model_type: str = "SVM",
    pos_group: str = "",
    key_var: str = "",
    data_vars: list = [],
    ignore_vars: list = [],
    kernel: str = "linear",
    output: str = "",
    verbose: int = 1,
    logs: str = "",
    **kwargs: Any,
) -> dict:
    """
    Trains a SPARE model, either classification or regression

    :param df: either a pandas dataframe or a path to a saved csv
                containing training data.
    :type df: pandas.DataFrame
    :param to_predict: variable to predict. Binary for classification and
                continuous for regression. Must be one of the columnes in
                df.
    :type to_predict: str
    :param pos_group: group to assign a positive SPARE score (only for
                classification).
    :type pos_group: str
    :param key_var: The key variable to be used for training. If not
                given, the first column of the dataset is considered the
                primary key of the dataset.
    :type key_var: str
    :param data_vars:  a list of predictors for the training. All must be present
                in columns of df.
    :type data_vars: list
    :param ignore_vars:The list of predictors to be ignored for training. Can be
                a listkey_var, or empty.
    :type ignore_vars: list
    :param kernel: 'linear' or 'rbf' (only linear is supported currently in
                regression).
    :type kernel: str
    :param output: path to save the trained model. '.pkl.gz' file extension
                optional. If None is given, no model will be saved.
    :type output: str
    :param verbose:    Verbosity. Int, higher is more verbose. [0,1,2]
    :type verbose: int
    :param logs: Where to save log file. If not given, logs will only be printed out.
    :type logs: str

    :return: A dictionary with three keys, 'status_code', 'status' and 'data'.
             'status' is either'OK' or the error message. 'data' is a dictionary
             containing the trained model and metadata if successful, or
             None / error object if unsuccessful. 'status_code' is either 0, 1 or 2.
             0 is success, 1 is warning, 2 is error.
    :rtype: dict

    """
    res = {"status_code": int, "status": Any, "data": Any}

    logger = logging_basic_config(verbose=verbose, filename=logs)

    # Make sure that no overwrites happen:
    if check_file_exists(output, logger):
        res["status"] = check_file_exists(output, logger)
        res["status_code"] = 2
        return res

    # Load the data
    df = load_df(df)

    # Assume key_variable (if not given)
    if key_var == "" or key_var is None:
        key_var = df.columns[0]
        if not is_unique_identifier(df, [key_var]):
            logging.info(
                "Assumed primary key is not capable of uniquely "
                + "identifying each row of the dataset. Assumed pkey: "
                + key_var
            )

    # Assume predictors (if not given)
    if data_vars == [] or data_vars is None:
        # Predictors = all_vars - key_var - ignore_vars - to_predict
        if ignore_vars == [] or ignore_vars is None:
            data_vars = list(set(list(df)) - set([key_var]) - set([to_predict]))
        else:
            data_vars = list(
                set(list(df)) - set([key_var]) - set(ignore_vars) - set([to_predict])
            )
    predictors = data_vars

    # Check if it contains any errors.
    try:
        df, predictors, mdl_task = check_train(  # type: ignore
            df, predictors, to_predict, verbose, pos_group
        )
    except Exception as e:
        err = "Dataset check failed before training was initiated."
        logger.error(err)
        print(e)
        res["status"] = err
        res["status_code"] = 2
        return res

    # Create meta data
    meta_data = MetaData(model_type, mdl_task, kernel, predictors, to_predict, key_var)
    meta_data.key_var = key_var

    # Convert categorical variables

    if len(df[to_predict].value_counts().keys()) == 2:
        if set(df[to_predict].value_counts().keys()) != set([0, 1]):
            df[to_predict] = df[to_predict].apply(lambda x: 1 if x == pos_group else 0)

    try:
        df, meta_data = convert_cat_variables(df, predictors + [to_predict], meta_data)
    except ValueError:
        err = (
            "Categorical variables could not be converted, because "
            + "they were not binary."
        )
        logger.error(err)
        res["status"] = err
        res["status_code"] = 2
        return res

    # Create the model
    try:
        spare_model = SpareModel(
            model_type,
            predictors,
            to_predict,
            key_var,
            verbose=1,
            parameters={
                "kernel": kernel,
                "k": 5,
                "n_repeats": 1,
                "task": mdl_task,
                "param_grid": None,
            },
            **kwargs,
        )
    except NotImplementedError:
        err = "SPARE model " + model_type + " is not implemented yet."
        logger.error(err)
        res["status"] = err
        res["status_code"] = 2
        return res
    except ValueError as e:
        logger.error(e)
        print(e)
        res["status"] = e
        res["status_code"] = 2
        return res

    # Train the model
    try:
        trained = spare_model.train_model(df, pos_group=pos_group)
    except Exception as e:
        logger.critical(e)
        print(e)
        res["status"] = e
        res["status_code"] = 2
        return res

    # Save the results
    if trained is None:
        err = "No training output was produced."
        logger.critical(err)
        res["status"] = err
        res["status_code"] = 2
        return res

    df["predicted"] = trained["predicted"]
    model = trained["model"]
    meta_data.params = trained["best_params"]
    meta_data.stats = trained["stats"]
    meta_data.cv_folds = trained["CV_folds"]
    meta_data.scaler = trained["scaler"] if "scaler" in trained.keys() else None

    meta_data.cv_results = df[list(dict.fromkeys([key_var, to_predict, "predicted"]))]
    result = model, vars(meta_data)

    # Save model
    if output != "" and output is not None:
        save_file(result, output, "train", logger)

    res["status"] = "OK"
    res["data"] = result
    res["status_code"] = 0
    return res


def spare_test(
    df: Union[pd.DataFrame, str],
    mdl_path: Union[str, Tuple[dict, dict]],
    key_var: str = "",
    output: str = "",
    spare_var: str = "SPARE_score",
    verbose: int = 1,
    logs: str = "",
) -> pd.DataFrame:
    """
    Applies a trained SPARE model on a test dataset

    :param df:  either a pandas dataframe or a path to a saved csv
                containing the test sample.
    :type df: pandas.DataFrame
    :param mdl_path: either a path to a saved SPARE model ('.pkl.gz' file
                extension expected) or a tuple of SPARE model and
                meta_data.
    :type mdl_path: str
    :param key_var: The of key variable to be used for training. If not
                given, and the saved model does not contain it,the first
                column of the dataset is considered the primary key of the
                dataset.
    :type key_var: str
    :param output: path to save the calculated scores. '.csv' file extension
                optional. If None is given, no data will be saved.
    :type output: str
    :param spare_var: The name of the variable to be predicted. If not given,
                the name 'SPARE_score' will be used.
    :type spare_var: str
    :param verbose: Verbosity. Int, higher is more verbose. [0,1,2]
    :type verbose: int
    :param logs: Where to save log file. If not given, logs will only be
                printed out.
    :type logs: str

    :return: A dictionary with three keys, 'status_code', 'status' and 'data'.
             'status' is either 'OK' or the error message. 'data' is the pandas
             dataframe  containing predicted SPARE scores, or  None / error object
             if  unsuccessful. 'status_code' is either 0, 1 or 2.
             0 is success, 1 is warning, 2 is error.
    :rtype: dict

    """
    res = {"status_code": int, "status": Any, "data": Any}

    logger = logging_basic_config(verbose=verbose, filename=logs)

    # Make sure that no overwrites happen:
    if check_file_exists(output, logger):
        res["status"] = check_file_exists(output, logger)
        res["status_code"] = 2
        return res

    df = load_df(df)

    # Load & check for errors / compatibility the trained SPARE model
    mdl, meta_data = load_model(mdl_path) if isinstance(mdl_path, str) else mdl_path

    try:
        check, cols = check_test(df, meta_data)
    except Exception as e:
        logger.error(e)
        print(e)
        res["status"] = e
        res["status_code"] = 2
        return res

    if cols is not None and cols != []:
        print(check)
        logger.error(check)
        res["status"] = check
        res["data"] = cols
        res["status_code"] = 1
        return res

    # Assume key_variable (if not given)
    if key_var == "" or key_var is None:
        key_var = df.columns[0]
        if not is_unique_identifier(df, [key_var]):
            logging.info(
                "Assumed primary key(s) are not capable of uniquely "
                + "identifying each row of the dataset. Assumed "
                + "primary key(s) are: "
                + key_var
            )

    # Convert categorical variables
    for var, map_dict in meta_data.get("categorical_var_map", {}).items():
        if not isinstance(map_dict, dict):
            continue
        if df[var].isin(map_dict.keys()).any():
            df[var] = df[var].map(map_dict)
        else:
            expected_var = list(map_dict.keys())
            err = (
                f'Column "{var}" expected {expected_var}, but '
                + f"received {list(df[var].unique())}"
            )
            logger.error(err)
            res["status"] = err
            res["data"] = list(df[var].unique())
            res["status_code"] = 1
            return res

    # TODO: Output model description
    n = len(meta_data["cv_results"].index)
    if "Age" in meta_data["cv_results"].keys():
        a1 = int(np.floor(np.min((meta_data["cv_results"]["Age"]))))
        a2 = int(np.ceil(np.max((meta_data["cv_results"]["Age"]))))
    else:
        a1 = None
        a2 = None
    stats_metric = list(meta_data["stats"].keys())[0]
    stats = "{:.3f}".format(np.mean(meta_data["stats"][stats_metric]))
    logger.info(
        f"Model Info: training N = {n} / ages = {a1} - {a2} / "
        + f"expected {stats_metric} = {stats}"
    )

    # Figure out model type and task:
    if "mdl_task" not in meta_data.keys():  # Backwards compatibility
        model_task = (
            "Classification"
            if "Classification" in meta_data["mdl_type"]
            else "Regression"
        )

        if "SVM" in meta_data["mdl_type"]:
            model_type = "SVM"
        elif "MLP" in meta_data["mdl_type"]:
            model_type = "MLP"
        else:
            model_type = "MLPTorch"
    else:
        model_task = meta_data["mdl_task"]
        model_type = meta_data["mdl_type"]

    # Create model instance based on saved model:
    predictors = meta_data["predictors"]
    target = meta_data["to_predict"]
    params = meta_data["params"]
    spare_model = SpareModel(model_type, predictors, target, key_var, verbose=verbose)

    # Set the model attributes to the ones that were saved to the instance
    # during training:
    try:
        spare_model.set_parameters(**params)
        spare_model.set_parameters(
            **{
                "mdl": mdl,
                "task": model_task,
                **{
                    key: meta_data[key]
                    for key in meta_data.keys()
                    if key not in ["mdl", "task"]
                },
            }
        )
    except Exception as e:
        logger.critical(e)
        print(e)
        res["status"] = e
        res["status_code"] = 2
        return res

    # Predict
    try:
        predicted = spare_model.apply_model(df)
    except Exception as e:
        logger.critical(e)
        print(e)
        res["status"] = e
        res["status_code"] = 2
        return res

    # Save the results
    if predicted is None:
        err = "No testing output was produced."
        logger.critical(err)
        res["status"] = err
        res["status_code"] = 2
        return res

    d = {}
    d[key_var] = df[key_var]
    d[spare_var] = predicted
    out_df = pd.DataFrame(data=d)

    if output != "" and output is not None:
        save_file(out_df, output, "test", logger)

    res["status"] = "OK"
    res["data"] = out_df
    res["status_code"] = 0
    return res
