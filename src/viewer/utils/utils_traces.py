import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_stats as utilstat

import plotly.graph_objs as go
import plotly.figure_factory as ff

###################################################################
# Traces
def add_trace_scatter(
    df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any
) -> None:    
    # Check data
    if df is None:
        return fig
    if df.shape[0] == 0:
        return fig

    # Set colormap
    colors = plot_settings['cmaps']['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']    
    if hvar is None or hvar == 'None':
        hvar = 'grouping_var'
    if hvals is None:
        hvals = df[hvar].dropna().sort_values().unique().tolist()

    if "data" in plot_params['traces']:
        for hname in hvals:
            col_ind = hvals.index(hname)  # Select index of colour for the category
            dfh = df[df[hvar] == hname]
            trace = go.Scatter(
                x=dfh[plot_params['xvar']],
                y=dfh[plot_params['yvar']],
                mode="markers",
                marker={"color": colors[col_ind]},
                name=hname,
                legendgroup=hname,
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(trace)

        #fig.update_layout(xaxis_range=[xmin, xmax])
        #fig.update_layout(yaxis_range=[ymin, ymax])

def add_trace_linreg(
    df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any
) -> None:
    """
    Add linear fit and confidence interval
    """
    # Set colormap
    colors = plot_settings['cmaps']['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']
    traces = plot_params['traces']

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar is None:
        hvar = "grouping_var"
        hvals = None        
        dft["grouping_var"] = "Data"
        
    dft = dft.dropna(subset = hvar)
    vals_hue_all = dft[hvar].sort_values().unique().tolist()
    
    if hvals is None:
        hvals = vals_hue_all

    print(hvar)
    print(hvals)

    # Calculate fit
    dict_fit = utilstat.linreg_model(
        dft, plot_params['xvar'], plot_params['yvar'], hvar
    )

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
                showlegend=not plot_params['hide_legend'],
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
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(trace)

    # fig.update_layout(xaxis_range=[xmin, xmax])
    # fig.update_layout(yaxis_range=[ymin, ymax])
    return fig

def add_trace_lowess(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    # Set colormap
    colors = plot_settings['cmaps']['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']

    # Add a tmp column if group var is not set
    dft = df.copy()
    if hvar == "":
        hvar = "All"
        dft["All"] = "data"
        vals_hue_all = ["All"]

    vals_hue_all = sorted(dft[hvar].unique())
    if hvals == []:
        hvals = vals_hue_all

    dict_fit = utilstat.lowess_model(
        dft, plot_params['xvar'], plot_params['yvar'], hvar, lowess_s
    )

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
            showlegend=not plot_params['hide_legend'],
        )
        fig.add_trace(trace)

    # fig.update_layout(xaxis_range=[xmin, xmax])
    # fig.update_layout(yaxis_range=[ymin, ymax])

def add_trace_dot(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    df_tmp = df[df.MRID == sel_mrid]
    trace = go.Scatter(
        x=df_tmp[plot_params['xvar']],
        y=df_tmp[plot_params['yvar']],
        mode="markers",
        name="Selected",
        marker=dict(
            color="rgba(250, 50, 50, 0.5)", size=12, line=dict(color="Red", width=3)
        ),
        showlegend=not plot_params['hide_legend'],
    )
    fig.add_trace(trace)

def add_trace_centile(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    # Check centile traces
    if plot_params['traces'] is None:
        return fig
    
    if not any("centile" in s for s in plot_params['traces']):
        return fig

    # Set colormap
    colors = plot_settings['cmaps']['centile']

    # Get centile values for the selected roi
    df_tmp = df[df.VarName == plot_params['yvar']]

    # Max centile value for normalization
    flag_norm = plot_params['flag_norm_centiles']
    
    if flag_norm:
        #norm_val = df_tmp[df_tmp.columns[df_tmp.columns.str.contains('centile')]].max().max()
        norm_val = df_tmp['centile_50'].max()

    # Create line traces
    list_tr = [s for s in plot_params['traces'] if "centile" in s]
    for i, cvar in enumerate(list_tr):
        yvals = df_tmp[cvar]
        if flag_norm:
            yvals = yvals * 100 / norm_val
        
        if cvar in df_tmp.columns[2:]:
            ctrace = go.Scatter(
                x=df_tmp[plot_params['xvar']],
                y=yvals,
                mode="lines",
                name=cvar,
                legendgroup="centiles",
                line=dict(color=colors[i]),
                showlegend=not plot_params['hide_legend'],
            )
            fig.add_trace(ctrace)  # plot in first row


    # Update min/max
    #fig.update_layout(xaxis_range=[xmin, xmax])
    #fig.update_layout(yaxis_range=[ymin, ymax])

    return fig

def add_trace_dots(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    trace = go.Scatter(
        x=df[plot_params['xvar']],
        y=df[plot_params['yvar']],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    return trace

