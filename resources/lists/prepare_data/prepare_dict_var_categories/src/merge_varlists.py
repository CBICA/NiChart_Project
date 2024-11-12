import pandas as pd
import json
import os

list_cat = ['Demog', 'MUSE-Primary']

dict_cat = {}
for tmp_cat in list_cat:
    df = pd.read_csv(f'../input/varlist_{tmp_cat}.csv')
    dict_cat[tmp_cat] = df.VarName.tolist()

if not os.path.exists('../output'):
    os.makedirs('../output')

if not os.path.exists('../output/dict_var_categories.json'):
    with open('../output/dict_var_categories.json', 'w') as f:
        json.dump(dict_cat, f)
