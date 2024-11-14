# -*- coding: utf-8 -*-
from typing import Any

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import utils.utils_stats as utilstat
import plotly.figure_factory as ff

def dist_plot(
    df,
    xvar,
    hvar,
    hvals,
    traces,
    binnum,
    hide_legend,
):
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == '':
        hvar = 'All'
        dft['All'] = 'Data'
    if hvals == []:
        hvals = dft[hvar].unique().tolist()

    data = []
    for hname, dfh in dft.groupby(hvar):
        if hname in hvals:
            dtmp = dfh[xvar]
            drange = dtmp.max() - dtmp.min()
            bin_size = drange / binnum
            data.append(dfh[xvar])

    show_hist = 'histogram' in traces
    show_curve = 'density' in traces
    show_rug = 'rug' in traces

    fig = ff.create_distplot(
        data,
        hvals,
        histnorm='',
        bin_size=bin_size,
        show_hist=show_hist,
        show_rug=show_rug,
        show_curve=show_curve
    )

    return fig


def scatter_trace(
    df,
    xvar,
    yvar,
    hvar,
    hvals,
    traces,
    hide_legend,
    fig
):
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == '':
        hvar = 'All'
        dft['All'] = 'data'
    if hvals == []:
        hvals = dft[hvar].unique().tolist()

    if 'data' in traces:
        for hname, dfh in dft.groupby(hvar):
            if hname in hvals:
                trace = go.Scatter(
                    x=dfh[xvar],
                    y=dfh[yvar],
                    mode='markers',
                    name=hname,
                    legendgroup=hname,
                    showlegend=not hide_legend
                )
                fig.add_trace(trace)

def linreg_trace(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    hvar: str,
    hvals: Any,
    traces: Any,
    hide_legend,
    fig: Any
) -> Any:
    '''
    Add linear fit and confidence interval
    '''
    dict_fit = utilstat.linreg_model(df, xvar, yvar, hvar)

    # Add traces for the fit and confidence intervals
    if 'lin_fit' in traces:
        for hname in dict_fit.keys():
            x_hat = dict_fit[hname]['x_hat']
            y_hat = dict_fit[hname]['y_hat']
            conf_int = dict_fit[hname]['conf_int']
            trace = go.Scatter(
                x=x_hat,
                y=y_hat,
                # showlegend=False,
                mode="lines",
                name=f'lin_{hname}',
                legendgroup=hname,
                showlegend=not hide_legend
            )
            fig.add_trace(trace)

    if 'conf_95%' in traces:
        for hname in dict_fit.keys():
            x_hat = dict_fit[hname]['x_hat']
            y_hat = dict_fit[hname]['y_hat']
            conf_int = dict_fit[hname]['conf_int']
            trace = go.Scatter(
                x=np.concatenate([x_hat, x_hat[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name=f'lin_conf95_{hname}',
                legendgroup=hname,
                showlegend=not hide_legend
            )
            fig.add_trace(trace)

    return fig

def lowess_trace(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    hvar: str,
    hvals: str,
    lowess_s: float,
    hide_legend,
    fig: Any
) -> Any:

    dict_fit = utilstat.lowess_model(df, xvar, yvar, hvar, lowess_s)

    # Add traces for the fit and confidence intervals
    for hname in dict_fit.keys():
        x_hat = dict_fit[hname]['x_hat']
        y_hat = dict_fit[hname]['y_hat']
        trace = go.Scatter(
            x=x_hat,
            y=y_hat,
            # showlegend=False,
            mode="lines",
            name=f'lin_{hname}',
            legendgroup=hname,
            showlegend=not hide_legend
        )
        fig.add_trace(trace)

def dot_trace(
    df: pd.DataFrame,
    sel_mrid: str,
    xvar: str,
    yvar: str,
    hide_legend,
    fig: Any
) -> Any:
    df_tmp = df[df.MRID == sel_mrid]
    trace = go.Scatter(
        x=df_tmp[xvar],
        y=df_tmp[yvar],
        mode='markers',
        name='Selected',
        marker=dict(
            color='rgba(250, 50, 50, 0.5)',
            size=12,
            line=dict(color='Red', width=3)
        ),
        showlegend=not hide_legend
    )
    fig.add_trace(trace)

def percentile_trace(df: pd.DataFrame, xvar: str, yvar: str, fig: Any) -> Any:

    cline = [
        "rgba(255, 255, 255, 0.8)",
        "rgba(255, 225, 225, 0.8)",
        "rgba(255, 187, 187, 0.8)",
        "rgba(255, 0, 0, 0.8)",
        "rgba(255, 187, 187, 0.8)",
        "rgba(255, 225, 225, 0.8)",
        "rgba(255, 255, 255, 0.8)",
    ]

    cfan = [
        "rgba(255, 225, 225, 0.3)",
        "rgba(255, 225, 225, 0.3)",
        "rgba(255, 187, 187, 0.3)",
        "rgba(255, 0, 0, 0.3)",
        "rgba(255, 0, 0, 0.3)",
        "rgba(255, 187, 187, 0.3)",
        "rgba(255, 225, 225, 0.3)",
    ]

    df_tmp = df[df.ROI == yvar]

    # Create line traces
    for i, cvar in enumerate(df.columns[2:]):
        if i == 0:
            ctrace = go.Scatter(
                x=df_tmp[xvar],
                y=df_tmp[cvar],
                mode="lines",
                name=cvar,
                line=dict(color=cline[i]),
            )
        else:
            ctrace = go.Scatter(
                x=df_tmp[xvar],
                y=df_tmp[cvar],
                mode="lines",
                name=cvar,
                line=dict(color=cline[i]),
                fill="tonexty",
                fillcolor=cfan[i],
            )

        fig.append_trace(ctrace, 1, 1)  # plot in first row

    return fig


def dots_trace(df: pd.DataFrame, xvar: str, yvar: str) -> Any:
    trace = go.Scatter(
        x=df[xvar],
        y=df[yvar],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    return trace


