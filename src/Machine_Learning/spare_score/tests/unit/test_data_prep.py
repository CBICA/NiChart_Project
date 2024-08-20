import logging
import os
import unittest

import pandas as pd

from spare_scores.data_prep import (  # If updates go through, it can be updated to spare_scores.data_prep
    age_sex_match,
    check_test,
    check_train,
    logging_basic_config,
    smart_unique,
)
from spare_scores.util import load_df


class CheckDataPrep(unittest.TestCase):

    def test_check_train(self):
        # Test case 1: Valid input dataframe and predictors
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        predictors = ["ROI1", "ROI2", "ROI3"]
        to_predict = "Sex"
        pos_group = "M"
        filtered_df, filtered_predictors, mdl_type = check_train(
            self.df_fixture, predictors, to_predict, pos_group=pos_group
        )
        self.assertTrue(
            filtered_df.equals(self.df_fixture)
        )  # Check if filtered dataframe is the same as the input dataframe
        self.assertTrue(
            filtered_predictors == predictors
        )  # Check if filtered predictors are the same as the input predictors
        self.assertTrue(
            mdl_type == "Classification"
        )  # Check if the SPARE model type is correct

        # Test case 2: Missing required columns
        df_missing_columns = pd.DataFrame(
            {"ID": [1, 2, 3], "Var1": [1, 2, 3], "Var2": [4, 5, 6]}
        )
        predictors = ["Var1", "Var2"]
        to_predict = "ToPredict"
        pos_group = "1"
        res = check_train(df_missing_columns, predictors, to_predict, pos_group)
        self.assertTrue(res == "Variable to predict is not in the input dataframe.")

        # Test case 3: Predictor not in input dataframe
        df = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "Age": [30, 40, 50],
                "Sex": ["M", "F", "M"],
                "Var1": [1, 2, 3],
            }
        )
        predictors = ["Var1", "Var2"]  # Var2 is not in the input dataframe
        to_predict = "ToPredict"
        pos_group = "1"
        res = check_train(df, predictors, to_predict, pos_group)
        self.assertTrue(res == "Not all predictors exist in the input dataframe.")

    def test_check_test(self):
        # Test case 1: Valid input dataframe and meta_data
        df = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "Age": [30, 40, 50],
                "Sex": ["M", "F", "M"],
                "Var1": [1, 2, 3],
                "Var2": [4, 5, 6],
            }
        )
        meta_data = {
            "predictors": ["Var1", "Var2"],
            "cv_results": pd.DataFrame(
                {"ID": [1, 2, 3, 4, 5], "Age": [30, 40, 50, 60, 70]}
            ),
        }

        res = check_test(df, meta_data)
        self.assertTrue(
            res[1] is None
        )  # Check if filtered dataframe is the same as the input dataframe

        # Test case 2: Missing predictors in the input dataframe
        df_missing_predictors = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "Age": [30, 40, 50],
                "Sex": ["M", "F", "M"],
                "Var1": [1, 2, 3],
            }
        )
        meta_data = {
            "predictors": ["Var1", "Var2", "Var3"],
            "cv_results": pd.DataFrame(
                {"ID": [1, 2, 3, 4, 5], "Age": [30, 40, 50, 60, 70]}
            ),
        }
        res = check_test(df_missing_predictors, meta_data)
        self.assertTrue(
            res[0]
            == "Not all predictors exist in the input dataframe: ['Var2', 'Var3']"
        )

        # Test case 3: Passing check.
        df_age_outside_range = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "Age": [20, 45, 55],
                "Sex": ["M", "F", "M"],
                "Var1": [1, 2, 3],
                "Var2": [4, 5, 6],
            }
        )
        meta_data = {
            "predictors": ["Var1", "Var2"],
            "cv_results": pd.DataFrame(
                {"ID": [1, 2, 3, 4, 5], "Age": [30, 40, 50, 60, 70]}
            ),
        }
        res = check_test(df_age_outside_range, meta_data)
        self.assertTrue(res[1] == None)

    def test_smart_unique(self):
        # test case 1: testing smart_unique with df2=None, to_predict=None
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        result = smart_unique(self.df_fixture, None)
        err_msg = 'Either provide a second dataframe or provide a column "to_predict"'
        self.assertTrue(result == err_msg)

        # test case 2: testing smart_unique with no variance. df2=None
        df = {
            "Id": [1, 2, 3, 4, 5],
            "ScanID": ["Scan001", "Scan002", "Scan003", "Scan004", "Scan005"],
            "Age": [35, 40, 45, 31, 45],
            "Sex": ["M", "F", "F", "M", "F"],
            "ROI1": [0.64, 0.64, 0.64, 0.64, 0.64],
            "ROI2": [0.73, 0.91, 0.64, 0.76, 0.78],
        }
        self.df_fixture = pd.DataFrame(data=df)
        result = smart_unique(self.df_fixture, None, to_predict="ROI1")
        err_msg = "Variable to predict has no variance."
        self.assertTrue(result == err_msg)

        # test case 3: testing smart_unique with variance and no duplicate ID's. df2=None
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        result = smart_unique(self.df_fixture, None, "ROI1")
        self.assertTrue(result.equals(self.df_fixture))

        # test case 4: testing smart_unique with variance and duplicate ID's. df2=None
        self.df_fixture = pd.DataFrame(data=df)

        new_row = {
            "ID": 5,
            "ScanID": "Scan006",
            "Age": 45,
            "Sex": "F",
            "ROI1": 0.84,
            "ROI2": 0.73,
        }
        self.df_fixture = self.df_fixture._append(new_row, ignore_index=True)
        result = smart_unique(self.df_fixture, None, "ROI1")
        correct_df = {
            "Id": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
            "ScanID": [
                "Scan001",
                "Scan002",
                "Scan003",
                "Scan004",
                "Scan005",
                "Scan006",
            ],
            "Age": [35, 40, 45, 31, 45, 45],
            "Sex": ["M", "F", "F", "M", "F", "F"],
            "ROI1": [0.64, 0.64, 0.64, 0.64, 0.64, 0.84],
            "ROI2": [0.73, 0.91, 0.64, 0.76, 0.78, 0.73],
            "ID": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                5.0,
            ],
        }
        correct_df = pd.DataFrame(data=correct_df)
        self.assertTrue(result.equals(correct_df))

        # test case 5: testing df2 != None and no_df2=False
        df1 = {
            "ID": [0, 1, 2, 3],
            "Var1": [20, 30, 40, 50],
            "Var2": [25, 35, 45, 55],
        }
        df2 = {"ID": [0, 1, 2, 3], "Var1": [22, 23, 24, 25], "Var2": [34, 35, 36, 37]}

        self.df_fixture1 = pd.DataFrame(data=df1)
        self.df_fixture2 = pd.DataFrame(data=df2)

        result = smart_unique(self.df_fixture1, self.df_fixture2, to_predict=None)
        self.assertTrue(result == (self.df_fixture1, self.df_fixture2))

    def test_age_sex_match(self):
        # test case 1: testing df2=None and to_match=None
        self.df_fixture = load_df("../fixtures/sample_data.csv")
        result = age_sex_match(self.df_fixture, None)
        err_msg = 'Either provide a 2nd dataframe or provide a column "to_match"'
        self.assertTrue(result == err_msg)

        # test case 2: testing non-binary variable "to_match"
        df = {"ID": [0, 1, 2, 3], "Var1": [10, 20, 30, 40], "Var2": [0, 0, 1, 2]}
        self.df_fixture = pd.DataFrame(data=df)
        result = age_sex_match(self.df_fixture, None, to_match="Var2")
        err_msg = "Variable to match must be binary"
        self.assertTrue(result == err_msg)

        # test case 3: testing df2!=None and to_match=None with age_out_percentage>=100
        self.df_fixture1 = load_df("../fixtures/sample_data.csv")
        self.df_fixture2 = self.df_fixture1
        result = age_sex_match(
            self.df_fixture1, self.df_fixture2, to_match=None, age_out_percentage=150
        )  # Now no_df2=False
        err_msg = "Age-out-percentage must be between 0 and 100"
        self.assertTrue(result == err_msg)

        # test case 4: testing df2!=None and to_match=None with age_out_percentage having a valid value
        # and 'Sex' column in df1 and df2 having the same value
        df1 = {
            "ID": [0, 1, 2, 3, 4],
            "Sex": ["F", "F", "F", "F", "F"],
            "Age": [40, 45, 35, 60, 70],
            "Var1": [20, 30, 40, 50, 60],
            "Var2": [22, 23, 24, 25, 26],
        }
        self.df_fixture1 = pd.DataFrame(data=df1)
        self.df_fixture2 = self.df_fixture1
        result = age_sex_match(
            self.df_fixture1, self.df_fixture2, to_match=None
        )  # here sex_match=False
        self.assertTrue(result == (self.df_fixture1, self.df_fixture2))

        # test case 5: testing df2!=None and to_match!=None with age_out_percentage having a valid value
        # and Non-matching Sex on both dataframes
        df1 = {
            "ID": [0, 1, 2, 3, 4],
            "Sex": ["F", "F", "M", "F", "M"],
            "Age": [40, 45, 35, 60, 70],
            "Var1": [20, 30, 40, 50, 60],
            "Var2": [22, 23, 24, 25, 26],
        }

        self.df_fixture1 = pd.DataFrame(data=df1)
        result = age_sex_match(self.df_fixture1, None, to_match="Sex")
        correct_df = pd.DataFrame(
            data={
                "ID": [0, 1, 3, 2, 4],
                "Sex": ["F", "F", "F", "M", "M"],
                "Age": [40, 45, 60, 35, 70],
                "Var1": [20, 30, 50, 40, 60],
                "Var2": [22, 23, 25, 24, 26],
            }
        )
        print(result)
        self.assertTrue(result.equals(correct_df))

    def test_logging_basic_config(self):
        logging_level = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: logging.ERROR,
            4: logging.CRITICAL,
        }
        fmt = "%(message)s"
        # test case 1: testing non existing path
        filename = "test_data_prep_notexisting.py"
        result = logging_basic_config(filename=filename, content_only=True)
        logging.basicConfig(
            level=logging_level[1], format=fmt, force=True, filename=filename
        )
        self.assertTrue(os.path.exists("test_data_prep_notexisting.py"))
        self.assertTrue(result == logging.getLogger())
        os.remove(filename)

        # test case 2: testing existing path
        filename = "test_data_prep.py"
        result = logging_basic_config(filename=filename)
        logging.basicConfig(level=logging_level[1], format=fmt, force=True)
        self.assertTrue(os.path.exists("test_data_prep.py"))
        self.assertTrue(result == logging.getLogger())

    def test_convert_cat_variables(self):
        pass
