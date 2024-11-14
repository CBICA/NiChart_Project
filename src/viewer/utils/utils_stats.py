# -*- coding: utf-8 -*-
from typing import Any

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

@st.cache_data()
def linreg_model(df, xvar, yvar, hvar):
    if hvar == '':
        dft = df[[xvar, yvar]].sort_values(xvar)
        hvar = 'All'
        dft['All'] = 'Data'
    else:
        dft = df[[xvar, yvar, hvar]].sort_values(xvar)

    # Add traces for the fit
    dict_out = {}
    for hname, dfh in dft.groupby(hvar):
        x_ext = sm.add_constant(dfh[xvar])
        model = sm.OLS(dfh[yvar], x_ext).fit()
        pred = model.get_prediction(x_ext)
        dict_mdl = {
          'x_hat': dfh[xvar],
          'y_hat': model.fittedvalues,
          'conf_int': pred.conf_int()
        }
        dict_out[hname] = dict_mdl
    return dict_out

# ## from: https://james-brennan.github.io/posts/lowess_conf
# def lowess_with_conf(x, y, f=1./3.):
#     """
#     Basic LOWESS smoother with uncertainty.
#     Note:
#         - Not robust (so no iteration) and
#             only normally distributed errors.
#         - No higher order polynomials d=1
#             so linear smoother.
#     """
#     # get some paras
#     xwidth = f*(x.max()-x.min()) # effective width after reduction factor
#     N = len(x) # number of obs
#     # Don't assume the data is sorted
#     order = np.argsort(x)
#     # storage
#     y_sm = np.zeros_like(y)
#     y_stderr = np.zeros_like(y)
#     # define the weigthing function -- clipping too!
#     tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
#     # run the regression for each observation i
#     for i in range(N):
#         dist = np.abs((x[order][i]-x[order]))/xwidth
#         w = tricube(dist)
#         # form linear system with the weights
#         A = np.stack([w, x[order]*w]).T
#         b = w * y[order]
#         ATA = A.T.dot(A)
#         ATb = A.T.dot(b)
#         # solve the syste
#         sol = np.linalg.solve(ATA, ATb)
#         # predict for the observation only
#         yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
#         place = order[i]
#         y_sm[place]=yest
#         sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
#         # Calculate the standard error
#         y_stderr[place] = np.sqrt(sigma2 *
#                                 A[i].dot(np.linalg.inv(ATA)
#                                                     ).dot(A[i]))
#     return y_sm, y_stderr

# @st.cache_data()
# def lowess_model(df, xvar, yvar, hvar, lowess_s):
#     if hvar == 'None':
#         dft = df[[xvar, yvar]].sort_values(xvar)
#         hvar = 'All'
#         dft['All'] = 'Data'
#     else:
#         dft = df[[xvar, yvar, hvar]].sort_values(xvar)
#
#     # Add traces for the fit
#     dict_out = {}
#     for hname, dfh in dft.groupby(hvar):
#         y_hat, y_std = lowess_with_conf(np.array(dfh[xvar]), np.array(dfh[yvar]), f=1./3.)
#         conf_int = np.column_stack((y_hat - 1.96*y_std, y_hat + 1.96*y_std))
#         dict_mdl = {
#           'x_hat': dfh[xvar],
#           'y_hat': y_hat,
#           'conf_int': conf_int
#         }
#         dict_out[hname] = dict_mdl
#
#     return dict_out

@st.cache_data()
def lowess_model(df, xvar, yvar, hvar, lowess_s):
    if hvar == '':
        dft = df[[xvar, yvar]].sort_values(xvar)
        hvar = 'All'
        dft['All'] = 'Data'
    else:
        dft = df[[xvar, yvar, hvar]].sort_values(xvar)

    # Add traces for the fit
    dict_out = {}
    for hname, dfh in dft.groupby(hvar):
        lowess = sm.nonparametric.lowess
        pred = lowess(dfh[yvar], dfh[xvar], frac = lowess_s)
        x_hat = pred[:, 0]
        y_hat = pred[:, 1]
        dict_mdl = {
          'x_hat': pred[:, 0],
          'y_hat': pred[:, 1],
          'conf_int': []
        }
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

        print(ind1)

        # Calculate slope
        slope = (cent[ind2] - cent[ind1]) / (vals_cent[i, ind2] - vals_cent[i, ind1])

        # Estimate subj centile
        cent_subj[i] = cent[ind1] + slope * (vals_subj[i] - vals_cent[i, ind1])

    df_out = pd.DataFrame(dict(ROI=sel_vars, Centiles=cent_subj))
    return df_out


