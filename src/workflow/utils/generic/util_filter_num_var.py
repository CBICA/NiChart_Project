import argparse
import json
import sys

import pandas as pd


def filter_num_var(in_csv, var_name, min_val, max_val, out_csv):
    """
    Filters data based values in a single column
    """

    # Read input files
    df = pd.read_csv(in_csv)

    # Filter data
    df_out = df[df[var_name] >= min_val]
    df_out = df_out[df_out[var_name] <= max_val]

    # Write out file
    df_out.to_csv(out_csv, index=False)


if __name__ == "__main__":
    # Access arguments from command line using sys.argv
    if len(sys.argv) != 6:
        print("Error: Please provide all required arguments")
        print(
            "Usage: python filter_num_var.py in_csv.csv var_name min_val max_val out_csv.csv"
        )
        sys.exit(1)

    in_csv = sys.argv[1]
    var_name = sys.argv[2]
    min_val = int(sys.argv[3])
    max_val = int(sys.argv[4])
    out_csv = sys.argv[5]

    # Call the function
    filter_num_var(in_csv, var_name, min_val, max_val, out_csv)

    print("Variable filtering complete! Output file:", out_csv)
