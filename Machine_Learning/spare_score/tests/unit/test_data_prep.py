import pandas as pd

from spare_scores.data_prep import check_test, check_train


def test_check_train(df_fixture):
    # Test case 1: Valid input dataframe and predictors
    predictors = ['ROI1', 'ROI2', 'ROI3']
    to_predict = 'Sex'
    pos_group = 'M'
    key_var = 'ID'
    filtered_df, filtered_predictors, mdl_type = check_train(df_fixture, 
                                                             predictors, 
                                                             to_predict, 
                                                             key_var,
                                                             pos_group=pos_group)
    assert filtered_df.equals(df_fixture)  # Check if filtered dataframe is the same as the input dataframe
    assert filtered_predictors == predictors  # Check if filtered predictors are the same as the input predictors
    assert mdl_type == 'Classification'  # Check if the SPARE model type is correct

    # Test case 2: Missing required columns
    df_missing_columns = pd.DataFrame({'ID': [1, 2, 3],
                                       'Var1': [1, 2, 3],
                                       'Var2': [4, 5, 6]})
    predictors = ['Var1', 'Var2']
    to_predict = 'ToPredict'
    pos_group = '1'
    res = check_train(df_missing_columns, predictors, to_predict, pos_group)
    assert res == 'Variable to predict is not in the input dataframe.'

    # Test case 3: Predictor not in input dataframe
    df = pd.DataFrame({'ID': [1, 2, 3],
                       'Age': [30, 40, 50],
                       'Sex': ['M', 'F', 'M'],
                       'Var1': [1, 2, 3]})
    predictors = ['Var1', 'Var2']  # Var2 is not in the input dataframe
    to_predict = 'ToPredict'
    pos_group = '1'
    res = check_train(df, predictors, to_predict, pos_group)
    assert res == 'Not all predictors exist in the input dataframe.'

def test_check_test():
    # Test case 1: Valid input dataframe and meta_data
    df = pd.DataFrame({'ID': [1, 2, 3],
                       'Age': [30, 40, 50],
                       'Sex': ['M', 'F', 'M'],
                       'Var1': [1, 2, 3],
                       'Var2': [4, 5, 6]})
    meta_data = {'predictors': ['Var1', 'Var2'],
                 'cv_results': pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                                             'Age': [30, 40, 50, 60, 70]})}

    res = check_test(df, meta_data)
    assert res[1] is None  # Check if filtered dataframe is the same as the input dataframe

    # Test case 2: Missing predictors in the input dataframe
    df_missing_predictors = pd.DataFrame({'ID': [1, 2, 3],
                                          'Age': [30, 40, 50],
                                          'Sex': ['M', 'F', 'M'],
                                          'Var1': [1, 2, 3]})
    meta_data = {'predictors': ['Var1', 'Var2', 'Var3'],
                 'cv_results': pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                                             'Age': [30, 40, 50, 60, 70]})}
    res = check_test(df_missing_predictors, meta_data)
    assert res[0] == "Not all predictors exist in the input dataframe: ['Var2', 'Var3']"

    # Test case 3: Passing check.
    df_age_outside_range = pd.DataFrame({'ID': [1, 2, 3],
                                         'Age': [20, 45, 55],
                                         'Sex': ['M', 'F', 'M'],
                                         'Var1': [1, 2, 3],
                                         'Var2': [4, 5, 6]})
    meta_data = {'predictors': ['Var1', 'Var2'],
                 'cv_results': pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                                             'Age': [30, 40, 50, 60, 70]})}
    res = check_test(df_age_outside_range, meta_data)
    assert res[1] == None

def test_smart_unique():
    pass

def test_age_sex_match():
    pass

def test_logging_basic_config():
    pass

def test_convert_cat_variables():
    pass