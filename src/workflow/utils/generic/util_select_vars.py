import argparse
import json
import sys

import pandas as pd


def select_vars(in_csv, dict_csv, dict_var, vars_list, out_csv):
    """
    Select variables from data file
    """

    # Read input files
    df = pd.read_csv(in_csv, dtype={"MRID": str})
    dfd = pd.read_csv(dict_csv)

    # Convert columns of dataframe to str (to handle numeric ROI indices)
    df.columns = df.columns.astype(str)

    # Get variable lists (input var list + rois)
    vars_list = vars_list.split(",")
    dict_vars = dfd[dict_var].astype(str).tolist()

    # Remove duplicate vars (in case a variable is both in roi list and input var list)
    vars_list = [x for x in vars_list if x not in dict_vars]

    # Make a list of selected variables
    sel_vars = vars_list + dict_vars

    # Remove vars that are not in the dataframe
    df_vars = df.columns.tolist()
    sel_vars = [x for x in sel_vars if x in df_vars]

    # Select variables
    df_out = df[sel_vars]

    # FIXME: temp hack to rename MUSE_702 to DLICV
    df_out = df_out.rename(columns={"MUSE_702": "DLICV"})

    # Write out file
    df_out.to_csv(out_csv, index=False)


if __name__ == "__main__":
    # Access arguments from command line using sys.argv
    if len(sys.argv) != 6:
        print("Error: Please provide all required arguments")
        print("Usage: python select_vars.py in_csv dict_csv dict_var vars_list out_csv")
        sys.exit(1)

    in_csv = sys.argv[1]
    dict_csv = sys.argv[2]
    dict_var = sys.argv[3]
    vars_list = sys.argv[4]
    out_csv = sys.argv[5]

    # Print run command
    print("About to run: " + " ".join(sys.argv))

    # Call the function
    select_vars(in_csv, dict_csv, dict_var, vars_list, out_csv)

    print("Selection of variables complete! Output file:", out_csv)
