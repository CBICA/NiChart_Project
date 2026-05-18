import os
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import gui.utils_stats as utilstat

import plotly.graph_objs as go

###################################################################
# Traces

def add_lines(
    flag_xline, flag_yline, xcoor, ycoor, fig
):
    '''
    Add lines to show cursor position
    '''
    if flag_xline:
        if ycoor is not None:
            fig.add_hline(
                y = ycoor, line_width=2, line_color="green"
            )
    if flag_yline:

        if xcoor is not None:
            fig.add_vline(
                x = xcoor, line_width=2, line_color="green"
            )

def add_bg_dots(
    plot_params, fig
):
    '''
    Add a ghost trace to handle clicks
    '''
    x_min, x_max = plot_params['xmin'], plot_params['xmax']
    y_min, y_max = plot_params['ymin'], plot_params['ymax']
    xs, ys = np.meshgrid(
        np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20)
    )
    fig.add_trace(go.Scatter(
        x=xs.flatten(), y=ys.flatten(),
        mode="markers",
        marker=dict(opacity=0.0, size=50),  # large + invisible
        # hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_layout(clickmode="event+select")

def add_trace_scatter(
    df: pd.DataFrame, plot_params: dict,
    cmaps: dict, alphas: dict, flag_show_legend: bool,
    fig: Any
) -> None:
    '''
    Add trace with data points
    '''

    # Check data
    if df is None:
        return fig
    if df.shape[0] == 0:
        return fig

    if plot_params['xvar'] not in df or plot_params['yvar'] not in df:
        return fig

    # Set colormap
    colors = cmaps['data']
    alpha = alphas['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']
    if hvar not in df:
        hvar = 'grouping_var'
    if hvals is None:
        hvals = df[hvar].dropna().sort_values().unique().tolist()

    if plot_params['traces'] is not None and "data" in plot_params['traces']:

        for hname in hvals:
            c_ind = hvals.index(hname)  # Select index of colour for the category
            c = colors[f'd{c_ind+1}']
            c_txt = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
            dfh = df[df[hvar] == hname]
            trace = go.Scatter(
                x=dfh[plot_params['xvar']],
                y=dfh[plot_params['yvar']],
                mode="markers",
                marker={"color": c_txt},
                name=hname,
                legendgroup=hname,
                showlegend=flag_show_legend,
            )
            fig.add_trace(trace)

def add_trace_linreg(
    df: pd.DataFrame, plot_params: dict,
    cmaps: dict, alphas: dict, w_fit: float, flag_show_legend: bool,
    fig: Any
) -> None:
    '''
    Add trace for linear fit and confidence interval
    '''
    # Check data
    if plot_params['xvar'] == plot_params['yvar']:
        return fig

    if plot_params['xvar'] not in df or plot_params['yvar'] not in df:
        return fig

    # Set colormap
    colors = cmaps['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']
    if hvar not in df:
        hvar = 'grouping_var'
    if hvals is None or hvals == []:
        hvals = df[hvar].dropna().sort_values().unique().tolist()

    traces = plot_params['traces']
    if traces is None:
        traces = []

    # Calculate fit
    dict_fit = utilstat.linreg_model(
        df, plot_params['xvar'], plot_params['yvar'], hvar
    )

    # Add traces for the fit and confidence intervals
    if "lin_fit" in traces:
        alpha = alphas['lin_fit']

        for i, hname in enumerate(hvals):
            c_ind = hvals.index(hname)  # Select index of colour for the category
            c = colors[f'd{c_ind+1}']
            c_txt = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            line = {"color": c_txt, 'width': w_fit}
            trace = go.Scatter(
                x=x_hat,
                y=y_hat,
                mode="lines",
                line=line,
                name=f"lin_{hname}",
                #legendgroup=hname,
                showlegend=flag_show_legend,
            )
            fig.add_trace(trace)

    if "conf_95%" in traces:
        alpha = alphas['conf_95%']
        for hname in hvals:
            c_ind = hvals.index(hname)  # Select index of colour for the category
            c = colors[f'd{c_ind+1}']
            c_txt = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            conf_int = dict_fit[hname]["conf_int"]
            trace = go.Scatter(
                x=np.concatenate([x_hat, x_hat[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill="toself",
                fillcolor=c_txt,
                line=dict(color=c_txt),
                hoverinfo="skip",
                name=f"lin_conf95_{hname}",
                #legendgroup=hname,
                showlegend=flag_show_legend,
            )
            fig.add_trace(trace)

    return fig

def add_trace_lowess(
    df: pd.DataFrame, plot_params: dict,
    cmaps: dict, alphas: dict, w_fit: float, flag_show_legend: bool,
    fig: Any
) -> None:
    '''
    Add trace for non-linear fit
    '''
    # Check data
    if plot_params['xvar'] not in df or plot_params['yvar'] not in df:
        return fig

    # Check trace
    traces = plot_params['traces']
    if 'lowess' not in traces:
        return fig

    # Set colormap
    colors = cmaps['data']
    alpha = alphas['lowess']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']
    if hvar not in df:
        hvar = 'grouping_var'
    if hvals is None:
        hvals = df[hvar].dropna().sort_values().unique().tolist()

    lowess_s = plot_params['lowess_s']

    dict_fit = utilstat.lowess_model(
        df, plot_params['xvar'], plot_params['yvar'], hvar, lowess_s
    )

    # Add traces for the fit and confidence intervals
    for hname in hvals:
        c_ind = hvals.index(hname)  # Select index of colour for the category
        c = colors[f'd{c_ind+1}']
        c_txt = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
        x_hat = dict_fit[hname]["x_hat"]
        y_hat = dict_fit[hname]["y_hat"]
        line = {"color": c_txt, 'width': w_fit}
        trace = go.Scatter(
            x=x_hat,
            y=y_hat,
            mode="lines",
            line=line,
            name=f"smooth_{hname}",
            #legendgroup=hname,
            showlegend=flag_show_legend,
        )
        fig.add_trace(trace)

def add_trace_dot(
    df: pd.DataFrame, sel_mrid: str, plot_params: dict,
    flag_show_legend: bool,
    fig: Any
) -> None:
    '''
    Add trace for a single dot
    '''
    # Check data
    if plot_params['xvar'] not in df or plot_params['yvar'] not in df:
        return fig

    df_tmp = df[df.MRID == sel_mrid]
    if df_tmp.shape[0] == 0:
        return fig

    trace = go.Scatter(
        x=df_tmp[plot_params['xvar']],
        y=df_tmp[plot_params['yvar']],
        mode="markers",
        name="Selected",
        marker=dict(
            color="rgba(250, 50, 50, 0.5)", size=12, line=dict(color="Red", width=3)
        ),
        showlegend=flag_show_legend,
    )
    fig.add_trace(trace)

def add_trace_centile(
    df: pd.DataFrame, plot_params: dict,
    cmaps: dict, alphas: dict, w_centile: float, flag_show_legend: bool,
    centile_trace_types: list,
    fig: Any
) -> None:
    '''
    Add trace for centile curves
    '''
    # Check data
    if plot_params['xvar'] not in df:
        st.warning(f'X variable {plot_params["xvar"]} not found in centile data!')
        return fig

    if plot_params['yvar'] not in df.VarName.unique():
        st.warning(f'Y variable {plot_params["yvar"]} not found in centile data!')
        # st.dataframe(df.VarName.unique())
        return fig

    cvals = centile_trace_types

    # Check centile traces
    if plot_params['traces'] is None:
        return fig

    if not any("centile" in s for s in plot_params['traces']):
        return fig

    # Set colormap
    colors = cmaps['centiles']
    alpha = alphas['centiles']

    # Get centile values for the selected roi

    df_tmp = df[df.VarName == plot_params['yvar']].sort_values('Age')

    # Max centile value for normalization
    flag_norm = plot_params.get('flag_norm_centiles', False)

    if flag_norm:
        #norm_val = df_tmp[df_tmp.columns[df_tmp.columns.str.contains('centile')]].max().max()
        norm_val = df_tmp['centile_50'].max()

    # Create line traces
    list_tr = [s for s in plot_params['traces'] if "centile" in s]
    for i, cvar in enumerate(cvals):
        if cvar in plot_params['traces']:
            if cvar in df_tmp.columns[2:]:
                yvals = df_tmp[cvar]
                if flag_norm:
                    yvals = yvals * 100 / norm_val

                c_ind = cvals.index(cvar)  # Select index for the centile
                c = colors[cvar]
                c_txt = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'

                ctrace = go.Scatter(
                    x=df_tmp[plot_params['xvar']],
                    y=yvals,
                    mode="markers",
                    name=cvar,
                    marker=dict(opacity=0.0),
                    showlegend=False,
                )
                fig.add_trace(ctrace)  # plot in first row

                ctrace = go.Scatter(
                    x=df_tmp[plot_params['xvar']],
                    y=yvals,
                    mode="lines",
                    name=cvar,
                    legendgroup="centiles",
                    line=dict(color=c_txt, width=w_centile),
                    showlegend=flag_show_legend,
                )
                fig.add_trace(ctrace)  # plot in first row


    # Update min/max
    xmin = plot_params['xmin']
    xmax = plot_params['xmax']
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    fig.update_layout(xaxis_range=[xmin, xmax])
    fig.update_layout(yaxis_range=[ymin, ymax])

    return fig

def add_trace_dots(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    '''
    Add trace for multiple dots
    '''
    # Check data
    if plot_params['xvar'] not in df or plot_params['yvar'] not in df:
        return

    trace = go.Scatter(
        x=df[plot_params['xvar']],
        y=df[plot_params['yvar']],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    fig.add_trace(trace)
