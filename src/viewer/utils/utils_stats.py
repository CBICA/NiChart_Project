# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st


@st.cache_data()  # type:ignore
def linreg_model(df: pd.DataFrame, xvar: str, yvar: str, hvar: str) -> Any:
    '''
    Fit linear regression model to data
    '''
    dft = df[[xvar, yvar, hvar]].dropna().sort_values(xvar)

    # Add traces for the fit
    dict_out = {}
    for hname, dfh in dft.groupby(hvar):
        x_ext = sm.add_constant(dfh[xvar])
        model = sm.OLS(dfh[yvar], x_ext).fit()
        pred = model.get_prediction(x_ext)
        dict_mdl = {
            "x_hat": dfh[xvar],
            "y_hat": model.fittedvalues,
            "conf_int": pred.conf_int(),
        }
        dict_out[hname] = dict_mdl
    return dict_out

# @st.cache_data()  # type:ignore
def lowess_model(
    df: pd.DataFrame, xvar: str, yvar: str, hvar: str, lowess_s: float
) -> Any:
    if hvar == "":
        dft = df[[xvar, yvar]].sort_values(xvar)
        hvar = "All"
        dft["All"] = "Data"
    else:
        dft = df[[xvar, yvar, hvar]].sort_values(xvar)

    # Add traces for the fit
    dict_out = {}
    for hname, dfh in dft.groupby(hvar):
        lowess = sm.nonparametric.lowess
        pred = lowess(dfh[yvar], dfh[xvar], frac=lowess_s)
        dict_mdl = {"x_hat": pred[:, 0], "y_hat": pred[:, 1], "conf_int": []}
        dict_out[hname] = dict_mdl
    return dict_out


def calc_subject_centiles(df_subj: pd.DataFrame, df_cent: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate subject specific centile values
    """

    # Filter centiles to subject's age
    tmp_ind = (df_cent.Age - df_subj.Age[0]).abs().idxmin()
    sel_age = df_cent.loc[tmp_ind, "Age"]
    df_cent_sel = df_cent[df_cent.Age == sel_age]

    # Find ROIs in subj data that are included in the centiles file
    sel_vars = df_subj.columns[df_subj.columns.isin(df_cent_sel.ROI.unique())].tolist()
    df_cent_sel = df_cent_sel[df_cent_sel.ROI.isin(sel_vars)].drop(
        ["ROI", "Age"], axis=1
    )

    cent = df_cent_sel.columns.str.replace("centile_", "").astype(int).values
    vals_cent = df_cent_sel.values
    vals_subj = df_subj.loc[0, sel_vars]

    cent_subj = np.zeros(vals_subj.shape[0])
    for i, sval in enumerate(vals_subj):
        # Find nearest x values
        ind1 = np.where(vals_subj[i] < vals_cent[i, :])[0][0] - 1
        ind2 = ind1 + 1

        # Calculate slope
        slope = (cent[ind2] - cent[ind1]) / (vals_cent[i, ind2] - vals_cent[i, ind1])

        # Estimate subj centile
        cent_subj[i] = cent[ind1] + slope * (vals_subj[i] - vals_cent[i, ind1])

    df_out = pd.DataFrame(dict(ROI=sel_vars, Centiles=cent_subj))
    return df_out
