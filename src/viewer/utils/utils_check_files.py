import os
import shutil
import tkinter as tk
import zipfile
from tkinter import filedialog
from typing import Any, BinaryIO, List, Optional

import pandas as pd
import streamlit as st

#####################################################################
# Dataframe to keep parts of hard-coded paths for output file names
# - Files will be created as:
#     ${out_dir}/${std}/${df.fdir}/${std}_${df.fpref}.csv
#   where ${std} can be one of the input datasets, or "pooled" for combined data
# - Edit the data frame to add new file types
ftypes = [
    "unified",
    "mapped",
    "visits",
    "imputed_in",
    "imputed_out",
    "merged",
    "filtered",
    "final",
]
fdirs = [
    "workingdir",
    "workingdir",
    "workingdir",
    "workingdir",
    "workingdir",
    "workingdir",
    "workingdir",
    "",
]
fprefs = [
    ["clinical_unified", "muse_unified"],
    ["clinical_mapped"],
    ["participants", "visits"],
    ["clinical_mapped", "visits"],
    ["clinical_visits_imputed"],
    ["merged"],
    ["filtered"],
    ["participants_final", "clinical_final", "muse_final", "visits_final"],
]
df_files = pd.DataFrame({"ftype": ftypes, "fdir": fdirs, "fpref": fprefs})
#####################################################################


def get_file_names(std, ftype):

    df_sel = df_files[df_files.ftype == ftype]

    if df_sel.shape[0] != 1:
        return []

    dout = os.path.join(st.session_state.paths["outdir"], std, df_sel.fdir.values[0])

    fpaths = [
        os.path.join(dout, std + "_" + x + ".csv") for x in df_sel.fpref.values[0]
    ]

    print()

    df = pd.DataFrame({"FileType": df_sel.fpref.values[0], "FileName": fpaths})
    df["FileExists"] = df["FileName"].apply(lambda x: 1 if os.path.exists(x) else 0)

    return df


def check_files_exist(std, ftype):
    df_tmp = get_file_names(std, ftype)

    if df_tmp.FileExists.min() == 1:
        return True
    return False


def delete_files(flist):
    for fname in flist:
        try:
            os.remove(fname)
            st.success(f"Removed file: {fname}")
        except:
            st.error(f"Could not delete file: {fname}")
