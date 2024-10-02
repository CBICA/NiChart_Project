from typing import Any

import pandas as pd


def read_derived_roi_list(list_sel_rois: list, list_derived: list) -> Any:
    """
    Create a dictionary from derived roi list
    """

    # Read list
    df_sel = pd.read_csv(list_sel_rois)
    df = pd.read_csv(list_derived, header=None)

    # Keep only selected ROIs
    df = df[df[0].isin(df_sel.Index)]

    # Create dict of roi names and indices
    dict_roi = dict(zip(df[1], df[0]))

    # Create dict of roi indices and derived indices
    dict_derived = {}
    for i, tmp_ind in enumerate(df[0].values):
        df_tmp = df[df[0] == tmp_ind].drop([0, 1], axis=1)
        sel_vals = df_tmp.T.dropna().astype(int).values.flatten()
        dict_derived[tmp_ind] = sel_vals

    return dict_roi, dict_derived
