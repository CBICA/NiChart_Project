import csv as csv
import os
from typing import Any

import pandas as pd


def rename_df_columns(
    in_csv: str, in_dict: str, var_from: str, var_to: str, out_csv: str
) -> None:
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


def corr_icv(
    in_csv: str,
    corr_type: str,
    icv_var: str,
    exclude_vars: str,
    suffix: str,
    out_csv: str,
) -> None:
    """
    Calculates ICV corrected values
    """

    # Read input file
    df = pd.read_csv(in_csv)

    # Get var groups
    list_exclude = exclude_vars.split(",") + [icv_var]
    list_target = df.columns[
        df.columns.isin(list_exclude) if not df.columns.isin(list_exclude) else None
    ]
    df_p1 = df[list_exclude]
    df_p2 = df[list_target]
    val_icv = df[icv_var]

    corr_val: int = 0
    # Set correction factor
    if corr_type == "percICV":
        corr_val = 100
    if corr_type == "normICV":
        corr_val = 1430000  # Average ICV

    # Add suffix to corrected variables
    if suffix != "NONE":
        df_p2 = df_p2.add_suffix(suffix)

    # Correct ICV
    df_p2 = df_p2.div(val_icv, axis=0) * corr_val

    # Combine vars
    df_out = pd.concat([df_p1, df_p2], axis=1)

    # Write out file
    df_out.to_csv(out_csv, index=False)


def merge_dataframes(in_csv1: str, in_csv2: str, key_var: str, out_csv: str) -> None:
    """
    Merge two input data files
    Using an inner merge
    """

    df1 = pd.read_csv(in_csv1, dtype={"MRID": str})
    df2 = pd.read_csv(in_csv2, dtype={"MRID": str})

    df_out = df1.merge(df2, on=key_var)

    # Write out file
    df_out.to_csv(out_csv, index=False)


def select_vars(
    in_csv: str, dict_csv: str, dict_var: str, vars_list: Any, out_csv: str
) -> None:
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


def filter_num_var(
    in_csv: str, var_name: str, min_val: float, max_val: float, out_csv: str
) -> None:
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


def apply_combat(
    in_csv: str, in_mdl: str, key_var: str, suffix: str, out_csv: str
) -> None:
    """
    Select combat output variables and prepare out file without combat suffix
    """
    out_tmp = f"{out_csv[0:-4]}_tmpinit.csv"
    os.system(f"neuroharm -a apply -i {in_csv} -m {in_mdl} -u {out_tmp}")

    # Read input files
    df = pd.read_csv(out_tmp, dtype={"MRID": str})

    # Convert columns of dataframe to str (to handle numeric ROI indices)
    df.columns = df.columns.astype(str)

    # Get variables (key + harmonized rois)
    sel_vars = [key_var] + df.columns[df.columns.str.contains(suffix)].tolist()

    # Select variables
    df_out = df[sel_vars]

    # Remove combat suffix
    df_out.columns = df_out.columns.str.replace(suffix, "")

    # Write out file
    df_out.to_csv(out_csv, index=False)


def combine_rois(
    in_csv: str,
    dict_csv: str,
    out_csv: str,
    key_var: str = "MRID",
    roi_prefix: str = "MUSE_",
) -> None:
    """
    Calculates a dataframe with the volumes of derived + single rois.
    - Derived ROIs are calculated by adding composing single ROIs, read from the dictionary
    - If a derived ROI already exists in input data, it will not be updated
    """

    # Read input files
    df_in = pd.read_csv(in_csv, dtype={"MRID": str})

    # Read derived roi map file to a dictionary
    dict_roi = {}
    with open(dict_csv) as roi_map:
        reader = csv.reader(roi_map, delimiter=",")
        for row in reader:
            key = roi_prefix + str(row[0])
            val = [roi_prefix + str(x) for x in row[2:]]
            dict_roi[key] = val

    # Create derived roi df
    label_names = list(dict_roi.keys())
    df_out = df_in[[key_var]].copy()
    df_out = df_out.reindex(columns=[key_var] + label_names)

    # Calculate volumes for derived rois
    for i, key in enumerate(dict_roi):
        key_vals = dict_roi[key]
        try:
            if key in df_in.columns:
                df_out[key] = df_in[
                    key
                ]  # If ROI is already in df, do not calculate it from parts
                print("WARNING:  Skip derived ROI, already in data: " + key)
            else:
                df_out[key] = df_in[key_vals].sum(
                    axis=1
                )  # If not, calculate it (sum of single ROIs in the dict)
        except:
            df_out = df_out.drop(columns=key)
            print("WARNING:  Skip derived ROI with missing components: " + key)

    # Save df_out
    df_out.to_csv(out_csv, index=False)


def apply_spare(in_csv: str, in_mdl: str, stype: str, out_csv: str) -> None:

    # Apply spare test
    out_tmp = f"{out_csv[0:-4]}_tmpinit.csv"
    os.system(f"spare_score -a test -i {in_csv} -m {in_mdl} -o {out_tmp}")

    # Change column name, select first two columns
    df = pd.read_csv(out_tmp)
    df = df[df.columns[0:2]]
    df.columns = ["MRID", f"SPARE{stype}"]
    df.to_csv(out_csv, index=False)

    # sed "s/SPARE_score/SPARE${stype}/g" ${out_csv%.csv}_tmpout.csv | cut -d, -f1,2 > $out_csv
    # rm -rf ${out_csv%.csv}_tmpout.csv


def merge_dataframes_multi(out_csv: str, key_var: str, list_in_csv: Any) -> None:
    """
    Merge multiple input data files
    Output data includes an inner merge
    """

    df_out = pd.read_csv(list_in_csv[0], dtype={"MRID": str})
    for i, in_csv in enumerate(list_in_csv[1:]):
        # Read csv files
        df_tmp = pd.read_csv(in_csv)
        df_tmp = df_tmp[
            df_tmp[key_var].isna() if df_tmp[key_var].isna() is False else None
        ]

        # Merge DataFrames
        df_out = df_out.merge(df_tmp, on=key_var, suffixes=["", "_tmpremovedupl"])

        # Remove duplicate columns
        df_out = df_out[
            df_out.columns[
                (
                    df_out.columns.str.contains("_tmpremovedupl")
                    if df_out.columns.str.contains("_tmpremovedupl") is False
                    else None
                )
            ]
        ]

    # Write out file
    df_out.to_csv(out_csv, index=False)


def combine_all(out_csv: str, list_in_csv: Any) -> None:
    """
    Combines final output files
    """

    df_demog = pd.read_csv(list_in_csv[0])

    df_roi = pd.read_csv(list_in_csv[1])
    df_roi = df_roi[df_roi.Name != "ICV"]

    df_data = pd.read_csv(list_in_csv[2])
    df_norm = pd.read_csv(list_in_csv[3])
    df_harm = pd.read_csv(list_in_csv[4])
    df_spare = pd.read_csv(list_in_csv[5])

    df_icv = df_data[["MRID", "MUSE_702"]]
    df_icv.columns = ["MRID", "ICV"]

    # print([['MRID'] +  df_roi.Code.to_list()])
    # input()

    df_data = df_data.rename(columns=dict(zip(df_roi.Code, df_roi.Name)))
    df_data = df_data[["MRID"] + df_roi.Name.to_list()]

    df_norm = df_norm.rename(columns=dict(zip(df_roi.Code, df_roi.Name)))
    df_norm = df_norm[["MRID"] + df_roi.Name.to_list()]

    df_harm = df_harm.rename(columns=dict(zip(df_roi.Code, df_roi.Name)))
    df_harm = df_harm[["MRID"] + df_roi.Name.to_list()]

    df_out = df_icv.merge(df_data, on="MRID")
    df_out = df_data.merge(df_norm, on="MRID", suffixes=["", "_normICV"])
    df_out = df_out.merge(df_harm, on="MRID", suffixes=["", "_COMBAT"])
    df_out = df_out.merge(df_spare, on="MRID")

    df_out = df_demog.merge(df_out, on="MRID")

    # Write out file
    df_out.to_csv(out_csv, index=False)
