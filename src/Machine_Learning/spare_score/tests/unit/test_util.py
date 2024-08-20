import logging
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from spare_scores.util import (
    add_file_extension,
    check_file_exists,
    convert_to_number_if_possible,
    expspace,
    is_unique_identifier,
    load_df,
    load_examples,
    load_model,
    save_file,
)


class CheckSpareScoresUtil(unittest.TestCase):
    def test_load_model(self):
        self.model_fixture = load_model("../../tests/fixtures/sample_model.pkl.gz")

        # Test case 1: Load a model
        filepath = (
            Path(__file__).resolve().parent.parent / "fixtures" / "sample_model.pkl.gz"
        )
        filepath = str(filepath)
        result = load_model(filepath)
        self.assertTrue(result[1]["mdl_type"] == self.model_fixture[1]["mdl_type"])
        self.assertTrue(result[1]["kernel"] == self.model_fixture[1]["kernel"])
        self.assertTrue(result[1]["predictors"] == self.model_fixture[1]["predictors"])
        self.assertTrue(result[1]["to_predict"] == self.model_fixture[1]["to_predict"])
        self.assertTrue(
            result[1]["categorical_var_map"]
            == self.model_fixture[1]["categorical_var_map"]
        )

    def test_expspace(self):
        # Test case 1: span = [0, 2]
        span = [0, 2]
        expected_result = np.array([1.0, 2.71828183, 7.3890561])
        self.assertTrue(np.allclose(expspace(span), expected_result))

        # Test case 2: span = [1, 5]
        span = [1, 5]
        expected_result = np.array(
            [2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591]
        )
        self.assertTrue(np.allclose(expspace(span), expected_result))

        # Test case 3: span = [-2, 1]
        span = [-2, 1]
        expected_result = np.array([0.13533528, 0.36787944, 1.0, 2.71828183])
        self.assertTrue(np.allclose(expspace(span), expected_result))

    def test_check_file_exists(self):
        # test case 1: filename=None
        logger = logging.getLogger(__name__)
        result = check_file_exists(None, logger)
        self.assertFalse(result)

        # test case 2: filename=''
        result = check_file_exists("", logger)
        self.assertFalse(result)

        # test case 3: filename exists
        result = check_file_exists("test_util.py", logger)
        err_msg = "The output filename test_util.py, corresponds to an existing file, interrupting execution to avoid overwrite."
        self.assertTrue(result == err_msg)

    def test_save_file(self):
        # test case 1: testing training  output file that don't exist
        result = pd.DataFrame(
            data={
                "Var1": [10, 20, 30, 40],
                "Var2": [20, 30, 40, 50],
                "Var3": [30, 40, 50, 60],
            }
        )
        output = "test_file"
        action = "train"
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=output)
        save_file(result, output, action, logger)
        self.assertTrue(os.path.exists(output + ".pkl.gz"))
        os.remove(output + ".pkl.gz")

        # test case 2: testing testing output file that don't exist
        output = "test_file"
        action = "test"
        save_file(result, output, action, logger)
        self.assertTrue(os.path.exists(output + ".csv"))
        os.remove(output + ".csv")

    def test_is_unique_identifier(self):
        # test case 1: testing with a unique identifier
        df = {
            "ID": [0, 1, 2, 3, 4],
            "Var1": [10, 20, 30, 40, 50],
            "Var2": [22, 23, 24, 25, 26],
        }

        self.df_fixture = pd.DataFrame(data=df)
        self.assertTrue(is_unique_identifier(self.df_fixture, ["Var1"]))
        self.assertTrue(is_unique_identifier(self.df_fixture, ["Var1", "Var2"]))
        self.assertTrue(is_unique_identifier(self.df_fixture, ["ID", "Var1", "Var2"]))

        # test case 2: testing with a non unique identifier
        df = {
            "ID": [0, 1, 2, 0, 4],
            "Var1": [10, 20, 30, 10, 50],
            "Var2": [10, 22, 33, 10, 50],
        }
        self.df_fixture = pd.DataFrame(data=df)
        self.assertFalse(is_unique_identifier(self.df_fixture, ["Var1", "Var2"]))

    def test_load_examples(self):
        # test case 1: testing loading example csv
        file_name = "example_data.csv"
        result = load_examples(file_name)
        self.assertTrue(isinstance(result, pd.DataFrame))

        # test case 2: testing loading model
        file_name = "../../spare_scores/mdl/mdl_SPARE_BA_hMUSE_single.pkl.gz"
        result = load_examples(file_name)
        self.assertFalse(result is None and isinstance(result, pd.DataFrame))

        # test case 3: testing with non existant filename
        file_name = "non_existant"
        result = load_examples(file_name)
        self.assertTrue(result is None)

    def test_convert_to_number_if_possible(self):
        # test case 1: valid convertion to integer
        num = "254"
        self.assertTrue(convert_to_number_if_possible(num) == 254)

        # test case 2: non valid convertion to integer
        num = "CBICA"
        self.assertTrue(convert_to_number_if_possible(num) == num)

    def test_load_df(self):
        # Test case 1: Input is a string (CSV file path)
        filepath = (
            Path(__file__).resolve().parent.parent / "fixtures" / "sample_data.csv"
        )
        filepath = str(filepath)
        expected_df = pd.read_csv(filepath, low_memory=False)
        self.assertTrue(load_df(filepath).equals(expected_df))

        # Test case 2: Input is already a DataFrame
        input_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        expected_df = input_df.copy()
        self.assertTrue(load_df(input_df).equals(expected_df))

        # Test case 3: Empty DataFrame
        input_df = pd.DataFrame()
        expected_df = input_df.copy()
        self.assertTrue(load_df(input_df).equals(expected_df))

        # Test case 4: Large DataFrame
        input_df = pd.DataFrame({"A": range(100000), "B": range(100000)})
        expected_df = input_df.copy()
        self.assertTrue(load_df(input_df).equals(expected_df))

    def test_add_file_extension(self):
        # Test case 1: File extension already present
        filename = "myfile.txt"
        extension = ".txt"
        self.assertTrue(add_file_extension(filename, extension) == "myfile.txt")

        # Test case 2: File extension not present
        filename = "myfile"
        extension = ".txt"
        self.assertTrue(add_file_extension(filename, extension) == "myfile.txt")

        # Test case 3: Different extension
        filename = "document"
        extension = ".docx"
        self.assertTrue(add_file_extension(filename, extension) == "document.docx")

        # Test case 4: Empty filename
        filename = ""
        extension = ".txt"
        self.assertTrue(add_file_extension(filename, extension) == ".txt")

        # Test case 5: Empty extension
        filename = "myfile"
        extension = ""
        self.assertTrue(add_file_extension(filename, extension) == "myfile")

        # Test case 6: Multiple extension dots in filename
        filename = "file.tar.gz"
        extension = ".gz"
        self.assertTrue(add_file_extension(filename, extension) == "file.tar.gz")
