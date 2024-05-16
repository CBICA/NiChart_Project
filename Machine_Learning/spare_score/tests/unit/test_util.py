import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spare_scores.util import (add_file_extension, check_file_exists, expspace,
                               is_unique_identifier, load_df, load_examples,
                               load_model, save_file)


def test_load_model(model_fixture):

    # Test case 1: No arguments given:
    no_args = "load_model() missing 1 required positional " + \
                 "argument: 'mdl_path'"
    with pytest.raises(TypeError, match=re.escape(no_args)):
        load_model()

    # Test case 2: Load a model
    filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_model.pkl.gz'
    filepath = str(filepath)
    result = load_model(filepath)
    assert result[1]['mdl_type'] == model_fixture[1]['mdl_type']
    assert result[1]['kernel'] == model_fixture[1]['kernel']
    assert result[1]['predictors'] == model_fixture[1]['predictors']
    assert result[1]['to_predict'] == model_fixture[1]['to_predict']
    assert result[1]['categorical_var_map'] == model_fixture[1]['categorical_var_map']

def test_expspace():
    # Test case 1: span = [0, 2]
    span = [0, 2]
    expected_result = np.array([1., 2.71828183, 7.3890561])
    assert np.allclose(expspace(span), expected_result)

    # Test case 2: span = [1, 5]
    span = [1, 5]
    expected_result = np.array([ 2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591])
    assert np.allclose(expspace(span), expected_result)

    # Test case 3: span = [-2, 1]
    span = [-2, 1]
    expected_result = np.array([0.13533528, 0.36787944, 1., 2.71828183])
    assert np.allclose(expspace(span), expected_result)

def test_check_file_exists():
    pass

def test_save_file():
    pass

def test_is_unique_identifier():
    pass

def test_load_model():
    pass

def test_load_examples():
    pass

def test_load_df():
    # Test case 1: Input is a string (CSV file path)
    filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_data.csv'
    filepath = str(filepath)
    expected_df = pd.read_csv(filepath, low_memory=False)
    assert load_df(filepath).equals(expected_df)

    # Test case 2: Input is already a DataFrame
    input_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    expected_df = input_df.copy()
    assert load_df(input_df).equals(expected_df)

    # Test case 3: Empty DataFrame
    input_df = pd.DataFrame()
    expected_df = input_df.copy()
    assert load_df(input_df).equals(expected_df)

    # Test case 4: Large DataFrame
    input_df = pd.DataFrame({"A": range(100000), "B": range(100000)})
    expected_df = input_df.copy()
    assert load_df(input_df).equals(expected_df)

def test_add_file_extension():
    # Test case 1: File extension already present
    filename = "myfile.txt"
    extension = ".txt"
    assert add_file_extension(filename, extension) == "myfile.txt"

    # Test case 2: File extension not present
    filename = "myfile"
    extension = ".txt"
    assert add_file_extension(filename, extension) == "myfile.txt"

    # Test case 3: Different extension
    filename = "document"
    extension = ".docx"
    assert add_file_extension(filename, extension) == "document.docx"

    # Test case 4: Empty filename
    filename = ""
    extension = ".txt"
    assert add_file_extension(filename, extension) == ".txt"

    # Test case 5: Empty extension
    filename = "myfile"
    extension = ""
    assert add_file_extension(filename, extension) == "myfile"

    # Test case 6: Multiple extension dots in filename
    filename = "file.tar.gz"
    extension = ".gz"
    assert add_file_extension(filename, extension) == "file.tar.gz"