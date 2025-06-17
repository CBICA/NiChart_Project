import json
import os

import pandas as pd

list_cat = [
    "Demog",
    "Scan",
    "Diagnosis",
    "MUSE-Essential",
    "MUSE-Single",
    "MUSE-Composite",
    "SPARE",
    "SurrealGAN",
]

dict_cat = {}
for tmp_cat in list_cat:
    df = pd.read_csv(f"../input/varlist_{tmp_cat}.csv")
    dict_cat[tmp_cat] = df.VarName.tolist()

if not os.path.exists("../output"):
    os.makedirs("../output")

# Convert the dictionary to a JSON string with indentation
out_json = json.dumps(dict_cat, indent=4)

# Write the JSON string to a file
if not os.path.exists("../output/dict_var_categories.json"):
    with open("../output/dict_var_categories.json", "w") as f:
        f.write(out_json)
