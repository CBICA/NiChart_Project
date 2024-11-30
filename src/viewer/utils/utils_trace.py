# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit as st
import utils.utils_stats as utilstat


def dist_plot(
    df: pd.DataFrame,
    xvar: str,
    hvar: str,
    hvals: list,
    traces: list,
    binnum: int,
    hide_legend: bool,
) -> Any:
    # Set colormap
    colors = st.session_state.plot_colors["data"]
    
    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "Data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    data = []
    bin_sizes = []
    colors_sel = []
    for hname in hvals:
        col_ind = vals_hue_all.index(
            hname
        )  # Select index of colour for the category
        dfh = dft[dft[hvar] == hname]
        x_tmp = dfh[xvar]
        x_range = x_tmp.max() - x_tmp.min()
        bin_size = x_range / binnum
        bin_sizes.append(bin_size)
        data.append(x_tmp)
        colors_sel.append(colors[col_ind])

    show_hist = "histogram" in traces
    show_curve = "density" in traces
    show_rug = "rug" in traces

    fig = ff.create_distplot(
        data,
        hvals,
        histnorm="",
        bin_size=bin_sizes,
        colors=colors_sel,
        show_hist=show_hist,
        show_rug=show_rug,
        show_curve=show_curve,
        # hide_legend=hide_legend  ## THIS IS NOT AVAILABLE IN FF
    )

    return fig


def scatter_trace(
    df: pd.DataFrame,
    xvar: str,
    xmin: float,
    xmax: float,
    yvar: str,
    ymin: float,
    ymax: float,
    hvar: str,
    hvals: list,
    traces: list,
    hide_legend: bool,
    fig: Any,
) -> None:
    # Set colormap
    colors = st.session_state.plot_colors["data"]

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
            col_ind = vals_hue_all.index(
                hname
            )  # Select index of colour for the category
            dfh = dft[dft[hvar] == hname]
            trace = go.Scatter(
                x=dfh[xvar],
                y=dfh[yvar],
                mode="markers",
                marker={"color": colors[col_ind]},
                name=hname,
                legendgroup=hname,
                showlegend=not hide_legend,
            )
            fig.add_trace(trace)
        fig.update_layout(xaxis_range=[xmin, xmax])
        fig.update_layout(yaxis_range=[ymin, ymax])


def linreg_trace(
    df: pd.DataFrame,
    xvar: str,
    xmin: float,
    xmax: float,
    yvar: str,
    ymin: float,
    ymax: float,
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
    colors = st.session_state.plot_colors["data"]

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
            col_ind = vals_hue_all.index(
                hname
            )  # Select index of colour for the category
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            trace = go.Scatter(
                x=x_hat,
                y=y_hat,
                mode="lines",
                line={"color": colors[col_ind]},
                name=f"lin_{hname}",
                legendgroup=hname,
                showlegend=not hide_legend,
            )
            fig.add_trace(trace)

    if "conf_95%" in traces:
        for hname in hvals:
            col_ind = vals_hue_all.index(
                hname
            )  # Select index of colour for the category
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            conf_int = dict_fit[hname]["conf_int"]
            trace = go.Scatter(
                x=np.concatenate([x_hat, x_hat[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill="toself",
                fillcolor=f"rgba({colors[col_ind][4:-1]}, 0.2)",  # Add alpha channel
                line=dict(color=f"rgba({colors[col_ind][4:-1]}, 0)"),
                hoverinfo="skip",
                name=f"lin_conf95_{hname}",
                legendgroup=hname,
                showlegend=not hide_legend,
            )
            fig.add_trace(trace)
    fig.update_layout(xaxis_range=[xmin, xmax])
    fig.update_layout(yaxis_range=[ymin, ymax])
    return fig


def lowess_trace(
    df: pd.DataFrame,
    xvar: str,
    xmin: float,
    xmax: float,
    yvar: str,
    ymin: float,
    ymax: float,
    hvar: str,
    hvals: list,
    lowess_s: float,
    hide_legend: bool,
    fig: Any,
) -> Any:
    # Set colormap
    colors = st.session_state.plot_colors["data"]

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    dict_fit = utilstat.lowess_model(dft, xvar, yvar, hvar, lowess_s)

    # Add traces for the fit and confidence intervals
    for hname in hvals:
        col_ind = vals_hue_all.index(hname)  # Select index of colour for the category
        x_hat = dict_fit[hname]["x_hat"]
        y_hat = dict_fit[hname]["y_hat"]
        trace = go.Scatter(
            x=x_hat,
            y=y_hat,
            # showlegend=False,
            mode="lines",
            line={"color": colors[col_ind]},
            name=f"lowess_{hname}",
            legendgroup=hname,
            showlegend=not hide_legend,
        )
        fig.add_trace(trace)

    fig.update_layout(xaxis_range=[xmin, xmax])
    fig.update_layout(yaxis_range=[ymin, ymax])

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


def percentile_trace(
    df: pd.DataFrame,
    xvar: str,
    xmin: float,
    xmax: float,
    yvar: str,
    ymin: float,
    ymax: float,
    traces: list,
    hide_legend: bool,
    fig: Any
) -> Any:

    # Set colormap
    colors = st.session_state.plot_colors["centile"]

    # Get centile values for the selected roi
    df_tmp = df[df.VarName == yvar]

    # Create line traces
    for i, cvar in enumerate(df_tmp.columns[2:]):
        if cvar in traces:
            ctrace = go.Scatter(
                x=df_tmp[xvar],
                y=df_tmp[cvar],
                mode="lines",
                name=cvar,
                legendgroup="centiles",
                line=dict(color=colors[i]),
                showlegend=not hide_legend,
            )

            fig.add_trace(ctrace)  # plot in first row

    fig.update_layout(xaxis_range=[xmin, xmax])
    fig.update_layout(yaxis_range=[ymin, ymax])

    return fig


def dots_trace(
    df: pd.DataFrame,
    xvar: str,
    yvar: str
) -> Any:
    trace = go.Scatter(
        x=df[xvar],
        y=df[yvar],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    return trace
