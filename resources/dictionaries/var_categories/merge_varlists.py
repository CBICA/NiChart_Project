import pandas as pd
import json

list_cat = ['Demog', 'MUSE-Primary']

dict_cat = {}
for tmp_cat in list_cat:
    df = pd.read_csv(f'varlist_{tmp_cat}.csv')
    dict_cat[tmp_cat] = df.VarName.tolist()
    
with open('dict_var_categories.json', 'w') as f:
    json.dump(dict_cat, f)
