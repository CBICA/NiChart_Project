import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spare_scores.spare_scores import spare_test, spare_train


def test_spare_test(df_fixture, model_fixture):

    # Test case 1: No arguments given:
    with pytest.raises(TypeError):
        spare_test()

    # Test case 2: Test with df
    result = spare_test(df_fixture, model_fixture)
    status_code, status, result = result['status_code'], result['status'], result['data']
    assert status == 'OK'
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == df_fixture.shape[0]
    assert 'SPARE_score' in result.columns  # Column name

    # Test case 3: Test with csv file:
    filepath = Path(__file__).resolve().parent.parent / 'fixtures' / 'sample_data.csv'
    filepath = str(filepath)
    result = spare_test(filepath, model_fixture)
    status_code, status, result = result['status_code'], result['status'], result['data']
    assert status == 'OK'
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == df_fixture.shape[0]
    assert 'SPARE_score' in result.columns  # Column name

    # Test case 4: Column required by the model is missing
    df_fixture.drop(columns='ROI1', inplace=True)
    result = spare_test(df_fixture, model_fixture)
    # {'status' : "Not all predictors exist in the input dataframe: ['ROI1']", 
    #  'data'   : ['ROI1']}
    status_code, status, result = result['status_code'], result['status'], result['data']
    assert status == 'Not all predictors exist in the input dataframe: [\'ROI1\']'
    assert result == ['ROI1']


def test_spare_train(df_fixture, model_fixture):

    # Test case 1: No arguments given:
    with pytest.raises(TypeError):
        spare_train()

    # Test case 2: Test with df
    result = spare_train(df_fixture, 
                         'Age',
                         data_vars = ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 
                                      'ROI6', 'ROI7', 'ROI8', 'ROI9', 'ROI10'],
                          )
    status_code, status, result = result['status_code'], result['status'], result['data']
    model, metadata = result[0], result[1]
    assert status == 'OK'
    assert metadata['mdl_type'] == model_fixture[1]['mdl_type']
    assert metadata['kernel'] == model_fixture[1]['kernel']
    assert set(metadata['predictors']) == set(model_fixture[1]['predictors'])
    assert metadata['to_predict'] == model_fixture[1]['to_predict']
    assert metadata['categorical_var_map'] == model_fixture[1]['categorical_var_map']
