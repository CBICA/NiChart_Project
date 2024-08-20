import gzip
import logging
import os
import pickle
from typing import Any, Union

import numpy as np
import pandas as pd
import pkg_resources  # type: ignore


def expspace(span: list) -> np.ndarray:
    return np.exp(np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1))


def load_df(df: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Fast loader for dataframes

    :param df: Either pd.DataFrame or path to the .csv file
    :type df: Union[pd.DataFrame, str]

    :return: The dataframe
    :rtype: pd.DataFrame
    """
    return pd.read_csv(df, low_memory=False) if isinstance(df, str) else df.copy()


def add_file_extension(filename: str, extension: str) -> str:
    """
    Adds file extension to needed file

    :param filename: The path to the file
    :type filename: str
    :param extension: The wanted extension(i.e. .txt, .csv, etc)
    :type extension: str

    :return: The filename
    :rtype: str
    """
    if not filename.endswith(extension):
        filename += extension
    return filename


def check_file_exists(filename: str, logger: Any) -> Any:
    """
    Checks if file exists

    :param filename: The file that will be searched
    :type filename: str
    :param logger: Output logger
    :type logger: logging.basicConfig

    :return: True if file exists, False otherwise
    :rtype: bool
    """
    # Make sure that no overwrites happen:
    if filename is None or filename == "":
        return False
    if os.path.exists(filename):
        err = (
            "The output filename "
            + filename
            + ", corresponds to an "
            + "existing file, interrupting execution to avoid overwrite."
        )
        print(err)
        logger.info(err)
        return err
    return False


def save_file(result: Any, output: str, action: str, logger: Any) -> None:
    """
    Saves the results in a file depending the action

    :param result: The results that will be dumped into the file
    :type result: Either .csv or pandas.DataFrame depending on the action
    :param output: The output filename
    :type output: str
    :param action: Either 'train' or 'test' depending on the action
    :type action: str
    :param logger: Output logger
    :type logger: logging.basicConfig

    """
    # Add the correct extension:
    if action == "train":
        output = add_file_extension(output, ".pkl.gz")
    if action == "test":
        output = add_file_extension(output, ".csv")

    dirname, fname = os.path.split(output)
    # Make directory doesn't exist:
    if not os.path.exists(output):
        try:
            os.mkdir(dirname)
            logger.info("Created directory {dirname}")
        except FileExistsError:
            logger.info("Directory of file already exists.")
        except FileNotFoundError:
            logger.info("Directory couldn't be created")

    # Create the file:
    if action == "train":
        with gzip.open(output, "wb") as f:
            pickle.dump(result, f)
            logger.info(f"Model {fname} saved to {dirname}/{fname}")

    if action == "test":
        try:
            result.to_csv(output, index=False)
        except Exception as e:
            logger.info(e)

        logger.info(f"Spare scores {fname} saved to {dirname}/{fname}")

    return


def is_unique_identifier(df: pd.DataFrame, column_names: list) -> bool:
    """
    Checks if the passed dataframe is a unique identifier

    :param df: The passed dataframe
    :type df: pandas.DataFrame
    :param column_names: The passed column names
    :type column_names: list

    :return: True if the passed data frame is a unique identifier
             False otherwise
    :rtype: bool

    """
    # Check the number of unique combinations
    unique_combinations = df[column_names].drop_duplicates()
    num_unique_combinations = len(unique_combinations)

    # Check the total number of rows
    num_rows = df.shape[0]
    # Return True if the number of unique combinations is equal to the total
    # number of rows
    return True if (num_unique_combinations == num_rows) else False


def load_model(mdl_path: str) -> Any:
    """
    Loads the model from the passed path

    :param mdl_path: the path to the weights of the model
    :type mdl_path: str

    """

    with gzip.open(mdl_path, "rb") as f:
        return pickle.load(f)


def load_examples(file_name: str = "") -> Any:
    """Loads example data and models in the package.

    :param file_name: either name of the example data saved as .csv or
    name of the SPARE model saved as .pkl.gz.
    :type file_name: str

    :return: the resulted dataframe
    :rtype: None or pandas.DataFrame

    """
    pkg_path = pkg_resources.resource_filename("spare_scores", "")
    list_data = os.listdir(f"{pkg_path}/data/")
    list_mdl = os.listdir(f"{pkg_path}/mdl/")
    if file_name in list_data:
        return pd.read_csv(f"{pkg_path}/data/{file_name}")
    elif file_name in list_mdl:
        return load_model(f"{pkg_path}/mdl/{file_name}")
    else:
        logging.info("Available example data:")
        for a in list_data:
            logging.info(f" - {a}")
        logging.info("Available example SPARE models:")
        for a in list_mdl:
            logging.info(f" - {a}")
    return None


def convert_to_number_if_possible(string: str) -> Union[float, str]:
    """
    Converts the the input string to a float if possible

    :param string: the input string
    :type string: str

    :return: float if the string is numeric, the same string if it's not
    :rtype: float or str

    """
    if string.isnumeric():
        return float(string)
    else:
        return string
