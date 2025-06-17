import json
import os

import pandas as pd

list_cat = [
    "Demographic",
    "Diagnosis",
    "Cognitive",
    "Scan",
    "MUSE-TopPicks",
    "MUSE-SingleROIs",
    "MUSE-CompositeROIs",
    "SPARE",
    "SurrealGAN",
]

dict_cat = {}
for tmp_cat in list_cat:
    in_list = os.path.join("..", "in", f"varlist_{tmp_cat}.csv")
    try:
        df = pd.read_csv(in_list)
        dict_cat[tmp_cat] = df.VarName.tolist()
        print(f"Extracted list {in_list}")
    except:
        print(f"Error reading input list {in_list}")

out_list = os.path.join("..", "out", "dict_var_categories.json")
with open(out_list, "w") as f:
    json.dump(dict_cat, f, indent=4)
    print(f"Out file is {out_list}")
