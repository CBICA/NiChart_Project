import pandas as pd

def filter_age(df, age_column, min_age=None, max_age=None):
    if min_age is not None:
        df = df[df[age_column] >= min_age]
    if max_age is not None:
        df = df[df[age_column] <= max_age]
    return df

import glob

for fname in glob.glob('nichart*.csv'):
    print(f'START: Filtering {fname} and saved as filtered_{fname}')
    df = pd.read_csv(fname)
    filtered_df = filter_age(df, 'Age', min_age=40, max_age=95)
    filtered_df.to_csv(f'filtered_{fname}', index=False)
    print(f'Finished filtering {fname} and saved as filtered_{fname}')
