import argparse
import json
import sys

import pandas as pd


def rename_df_columns(in_csv, in_dict, var_from, var_to, out_csv):
    """
    Rename columns of input csv using the dictionary
    """

    # Read input files
    df = pd.read_csv(in_csv, dtype={"MRID": str})
    dfd = pd.read_csv(in_dict)

    # Convert columns of dataframe to str
    df.columns = df.columns.astype(str)

    # Create dictionary for variables
    dfd.Index = dfd.Index.astype(str)
    vdict = dfd.set_index(var_from)[var_to].to_dict()

    # Rename ROIs
    df_out = df.rename(columns=vdict)

    # Write out file
    df_out.to_csv(out_csv, index=False)


if __name__ == "__main__":
    # Access arguments from command line using sys.argv
    if len(sys.argv) != 6:
        print("Error: Please provide all required arguments")
        print("Usage: python rename_df_columns.py in_csv.csv in_dict.csv out_csv.csv")
        sys.exit(1)

    in_csv = sys.argv[1]
    in_dict = sys.argv[2]
    var_from = sys.argv[3]
    var_to = sys.argv[4]
    out_csv = sys.argv[5]

    # Print run command
    print("About to run: " + " ".join(sys.argv))

    # Call the function
    rename_df_columns(in_csv, in_dict, var_from, var_to, out_csv)

    print("Renaming of columns complete! Output file:", out_csv)
