import gzip
import pickle
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def df_fixture():
    # Load the sample data from the fixture
    data_path = Path(__file__).resolve().parent / 'fixtures' / 'sample_data.csv'
    data_path = str(data_path)
    return pd.read_csv(data_path)

@pytest.fixture
def model_fixture():
    # Load the sample model from the fixture
    # This model was created using this package based on the above (randomly
    # generated data)
    model_path = Path(__file__).resolve().parent / 'fixtures' / 'sample_model.pkl.gz'
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model