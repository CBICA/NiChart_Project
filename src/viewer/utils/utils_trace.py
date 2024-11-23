# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import utils.utils_stats as utilstat
import streamlit as st

def dist_plot(
    df: pd.DataFrame,
    xvar: str,
    hvar: str,
    hvals: list,
    traces: list,
    binnum: int,
    hide_legend: bool,
) -> Any:
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "Data"
    if hvals == []:
        hvals = dft[hvar].unique().tolist()

    data = []
    for hname, dfh in hvals:
        dtmp = dfh[xvar]
        drange = dtmp.max() - dtmp.min()
        bin_size = drange / binnum
        data.append(dfh[xvar])

    show_hist = "histogram" in traces
    show_curve = "density" in traces
    show_rug = "rug" in traces

    fig = ff.create_distplot(
        data,
        hvals,
        histnorm="",
        bin_size=bin_size,
        show_hist=show_hist,
        show_rug=show_rug,
        show_curve=show_curve,
        #hide_legend=hide_legend  ## THIS IS NOT AVAILABLE IN FF
    )

    return fig

def scatter_trace(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    hvar: str,
    hvals: list,
    traces: list,
    hide_legend: bool,
    fig: Any,
) -> None:
    # Set colormap
    colors = st.session_state.plot_colors['data']
    
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    if "data" in traces:
        for hname in hvals:
            col_ind = vals_hue_all.index(hname)     # Select index of colour for the category
            dfh = dft[dft[hvar]==hname]
            trace = go.Scatter(
                x=dfh[xvar],
                y=dfh[yvar],
                mode="markers",
                marker = {'color' : colors[col_ind]},
                name=hname,
                legendgroup=hname,
                showlegend=not hide_legend,
            )
            fig.add_trace(trace)

def linreg_trace(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    hvar: str,
    hvals: list,
    traces: list,
    hide_legend: bool,
    fig: Any,
) -> Any:
    """
    Add linear fit and confidence interval
    """
    # Set colormap
    colors = st.session_state.plot_colors['data']
    
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    # Calculate fit
    dict_fit = utilstat.linreg_model(dft, xvar, yvar, hvar)

    # Add traces for the fit and confidence intervals
    if "lin_fit" in traces:
        for hname in hvals:
            col_ind = vals_hue_all.index(hname)     # Select index of colour for the category            
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            trace = go.Scatter(
                x=x_hat,
                y=y_hat,
                mode="lines",
                line = {'color' : colors[col_ind]},
                name=f"lin_{hname}",
                legendgroup=hname,
                showlegend=not hide_legend,
            )
            fig.add_trace(trace)

    if "conf_95%" in traces:
        for hname in hvals:
            col_ind = vals_hue_all.index(hname)     # Select index of colour for the category            
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            conf_int = dict_fit[hname]["conf_int"]
            trace = go.Scatter(
                x=np.concatenate([x_hat, x_hat[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill="toself",
                fillcolor=f"rgba({colors[col_ind][4:-1]},0.2)",       # Add alpha channel
                line=dict(color=f"rgba({colors[col_ind][4:-1]},0)"),
                hoverinfo="skip",
                name=f"lin_conf95_{hname}",
                legendgroup=hname,
                showlegend=not hide_legend,
            )
            fig.add_trace(trace)
    
    return fig


def lowess_trace(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    hvar: str,
    hvals: list,
    lowess_s: float,
    hide_legend: bool,
    fig: Any,
) -> Any:
    # Set colormap
    colors = st.session_state.plot_colors['data']
    
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    dict_fit = utilstat.lowess_model(df, xvar, yvar, hvar, lowess_s)

    # Add traces for the fit and confidence intervals
    for hname in hvals:
        col_ind = vals_hue_all.index(hname)     # Select index of colour for the category
        x_hat = dict_fit[hname]["x_hat"]
        y_hat = dict_fit[hname]["y_hat"]
        trace = go.Scatter(
            x=x_hat,
            y=y_hat,
            # showlegend=False,
            mode="lines",
            line = {'color' : colors[col_ind]},            
            name=f"lowess_{hname}",
            legendgroup=hname,
            showlegend=not hide_legend,
        )
        fig.add_trace(trace)


def dot_trace(
    df: pd.DataFrame, sel_mrid: str, xvar: str, yvar: str, hide_legend: bool, fig: Any
) -> Any:
    df_tmp = df[df.MRID == sel_mrid]
    trace = go.Scatter(
        x=df_tmp[xvar],
        y=df_tmp[yvar],
        mode="markers",
        name="Selected",
        marker=dict(
            color="rgba(250, 50, 50, 0.5)", size=12, line=dict(color="Red", width=3)
        ),
        showlegend=not hide_legend,
    )
    fig.add_trace(trace)


def percentile_trace(df: pd.DataFrame, xvar: str, yvar: str, fig: Any) -> Any:
    # Set colormap
    colors = st.session_state.plot_colors['centile']

    # Get centile values for the selected roi
    df_tmp = df[df.VarName == yvar]

    # Create line traces
    for i, cvar in enumerate(df_tmp.columns[2:]):
        #if i == 0:
            #ctrace = go.Scatter(
                #x=df_tmp[xvar],
                #y=df_tmp[cvar],
                #mode="lines",
                #name=cvar,
                #line=dict(color=cline[i]),
            #)
        #else:
            #ctrace = go.Scatter(
                #x=df_tmp[xvar],
                #y=df_tmp[cvar],
                #mode="lines",
                #name=cvar,
                #line=dict(color=cline[i]),
                #fill="tonexty",
                #fillcolor=cfan[i],
            #)
        ctrace = go.Scatter(
            x=df_tmp[xvar],
            y=df_tmp[cvar],
            mode="lines",
            name=cvar,
            legendgroup='centiles',
            line=dict(color=colors[i]),
        )

        fig.add_trace(ctrace)  # plot in first row

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
