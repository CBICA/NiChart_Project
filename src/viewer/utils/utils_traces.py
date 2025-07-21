import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_stats as utilstat

import plotly.graph_objs as go
import plotly.figure_factory as ff

###################################################################
# Traces
def add_trace_scatter(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    '''
    Add trace with data points
    '''
    # Check data
    if df is None:
        return fig
    if df.shape[0] == 0:
        return fig

    # Set colormap
    colors = plot_settings['cmaps']['data']
    alpha = plot_settings['alphas']['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']    
    if hvar is None or hvar == 'None':
        hvar = 'grouping_var'
    if hvals is None:
        hvals = df[hvar].dropna().sort_values().unique().tolist()

    if "data" in plot_params['traces']:
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
                showlegend = plot_settings['flag_hide_legend'] == 'Show',
            )
            fig.add_trace(trace)

def add_trace_linreg(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    '''
    Add trace for linear fit and confidence interval
    '''
    # Set colormap
    colors = plot_settings['cmaps']['data']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']
    if hvar is None or hvar == 'None':
        hvar = 'grouping_var'
    if hvals is None:
        hvals = df[hvar].dropna().sort_values().unique().tolist()

    traces = plot_params['traces']
     
    if plot_params['xvar'] == plot_params['yvar']:
        return fig
     
    # Calculate fit
    dict_fit = utilstat.linreg_model(
        df, plot_params['xvar'], plot_params['yvar'], hvar
    )

    # Add traces for the fit and confidence intervals
    if "lin_fit" in traces:
        alpha = plot_settings['alphas']['lin_fit']
        w = plot_settings['w_fit']

        for i, hname in enumerate(hvals):
            c_ind = hvals.index(hname)  # Select index of colour for the category
            c = colors[f'd{c_ind+1}']
            c_txt = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
            x_hat = dict_fit[hname]["x_hat"]
            y_hat = dict_fit[hname]["y_hat"]
            line = {"color": c_txt, 'width': w}
            trace = go.Scatter(
                x=x_hat,
                y=y_hat,
                mode="lines",
                line=line,
                name=f"lin_{hname}",
                #legendgroup=hname,
                showlegend = plot_settings['flag_hide_legend'] == 'Show',
            )
            fig.add_trace(trace)

    if "conf_95%" in traces:
        alpha = plot_settings['alphas']['conf_95%']
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
                line=dict(color = c_txt),
                hoverinfo="skip",
                name=f"lin_conf95_{hname}",
                #legendgroup=hname,
                showlegend = plot_settings['flag_hide_legend'] == 'Show',
            )
            fig.add_trace(trace)

    return fig

def add_trace_lowess(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    '''
    Add trace for non-linear fit
    '''
    # Check trace
    traces = plot_params['traces']
    if 'lowess' not in traces:
        return fig

    # Set colormap
    colors = plot_settings['cmaps']['data']
    alpha = plot_settings['alphas']['lowess']
    w = plot_settings['w_fit']

    # Get hue params
    hvar = plot_params['hvar']
    hvals = plot_params['hvals']
    if hvar is None or hvar == 'None':
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
        line = {"color": c_txt, 'width': w}
        trace = go.Scatter(
            x=x_hat,
            y=y_hat,
            mode="lines",
            line = line,
            name=f"lowess_{hname}",
            #legendgroup=hname,
            showlegend = plot_settings['flag_hide_legend'] == 'Show'
        )
        fig.add_trace(trace)

def add_trace_dot(
    df: pd.DataFrame, sel_mrid: str, plot_params: dict, plot_settings: dict, fig: Any
) -> None:
    '''
    Add trace for a single dot
    '''
    df_tmp = df[df.MRID == sel_mrid]
    if df_tmp.shape[0] == 0:
        return fig

    print('aab')
    print(plot_settings['flag_hide_legend'])

    trace = go.Scatter(
        x=df_tmp[plot_params['xvar']],
        y=df_tmp[plot_params['yvar']],
        mode="markers",
        name="Selected",
        marker=dict(
            color="rgba(250, 50, 50, 0.5)", size=12, line=dict(color="Red", width=3)
        ),
        showlegend = plot_settings['flag_hide_legend'] == 'Show'
    )
    fig.add_trace(trace)

def add_trace_centile(df: pd.DataFrame, plot_params: dict, plot_settings: dict, fig: Any) -> None:
    '''
    Add trace for centile curves
    '''
    cvals = st.session_state.plot_settings['centile_trace_types']

    # Check centile traces
    if plot_params['traces'] is None:
        return fig
    
    if not any("centile" in s for s in plot_params['traces']):
        return fig

    # Set colormap
    colors = plot_settings['cmaps']['centiles']
    alpha = plot_settings['alphas']['centiles']
    w = plot_settings['w_centile']

    # Get centile values for the selected roi
    df_tmp = df[df.VarName == plot_params['yvar']].sort_values('Age')

    # Max centile value for normalization
    flag_norm = plot_params['flag_norm_centiles']
    
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
                    mode="lines",
                    name=cvar,
                    #legendgroup="centiles",
                    line=dict(color=c_txt, width = w),
                    showlegend = plot_settings['flag_hide_legend']=='Show',
                )
                fig.add_trace(ctrace)  # plot in first row


    # Update min/max
    #fig.update_layout(xaxis_range=[xmin, xmax])
    #fig.update_layout(yaxis_range=[ymin, ymax])

    return fig

def add_trace_dots(df: pd.DataFrame, plot_params: dict, fig: Any) -> None:
    '''
    Add trace for multiple dots
    '''
    trace = go.Scatter(
        x=df[plot_params['xvar']],
        y=df[plot_params['yvar']],
        showlegend=False,
        mode="markers",
        name="datapoint",
        line=dict(color="rgb(0,160,250)"),
    )
    return trace

