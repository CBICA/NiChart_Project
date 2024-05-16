import gzip
import logging
import os
import pickle
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pkg_resources


def expspace(span: list):
    return np.exp(np.linspace(span[0], 
                              span[1], 
                              num=int(span[1])-int(span[0])+1))

def load_df(df: Union[pd.DataFrame, str]) -> pd.DataFrame:
    return pd.read_csv(df, low_memory=False) if isinstance(df, str)\
                                             else df.copy()

def add_file_extension(filename, extension):
    if not filename.endswith(extension):
        filename += extension
    return filename

def check_file_exists(filename, logger):
    # Make sure that no overwrites happen:
    if filename is None or filename == '':
        return False
    if os.path.exists(filename):
        err = "The output filename " + filename + ", corresponds to an " +\
                "existing file, interrupting execution to avoid overwrite."
        print(err)
        logger.info(err)
        return err
    return False

def save_file(result, output, action, logger):
    # Add the correct extension:
    if action == 'train':
        output = add_file_extension(output, '.pkl.gz')
    if action == 'test':
        output = add_file_extension(output, '.csv')

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
    if action == 'train':
        with gzip.open(output, 'wb') as f:
            pickle.dump(result, f)
            logger.info(f'Model {fname} saved to {dirname}/{fname}')
    
    if action == 'test':
        try:
            result.to_csv(output)
        except Exception as e:
            logger.info(e)
        
        logger.info(f'Spare scores {fname} saved to {dirname}/{fname}')
    
    return

def is_unique_identifier(df, column_names):
    # Check the number of unique combinations
    unique_combinations = df[column_names].drop_duplicates()
    num_unique_combinations = len(unique_combinations)

    # Check the total number of rows
    num_rows = df.shape[0]

    # Return True if the number of unique combinations is equal to the total 
    # number of rows
    return num_unique_combinations == num_rows

def load_model(mdl_path: str) -> Tuple[dict, dict]:
  with gzip.open(mdl_path, 'rb') as f:
    return pickle.load(f)

def load_examples(file_name: str=''):
  """Loads example data and models in the package.

  Args:
    file_name: either name of the example data saved as .csv or
      name of the SPARE model saved as .pkl.gz.

  Returns:
    a tuple containing pandas df and 
  """
  pkg_path = pkg_resources.resource_filename('spare_scores','')
  list_data = os.listdir(f'{pkg_path}/data/')
  list_mdl = os.listdir(f'{pkg_path}/mdl/')
  if file_name in list_data:
    return pd.read_csv(f'{pkg_path}/data/{file_name}')
  elif file_name in list_mdl:
    return load_model(f'{pkg_path}/mdl/{file_name}')
  else:
    logging.info('Available example data:')
    [logging.info(f' - {a}') for a in list_data]
    logging.info('Available example SPARE models:')
    [logging.info(f' - {a}') for a in list_mdl]

def convert_to_number_if_possible(string):
    try:
        number = float(string)  # Attempt to convert the string to a float
        return number
    except ValueError:
        return string
    

